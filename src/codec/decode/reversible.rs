//! Reversible (lossless) decode path.
//!
//! Reference: `zfp/src/template/revdecode{1-4}.c`,
//!            `zfp/src/template/revdecodef.c`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)] // i32→u32 and i64→u64 for BFP reconstruction

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::bitplane::{PlaneBlock, decode_ints};
use crate::codec::decode::core::{inv_cast_f32, inv_cast_f64};
use crate::codec::decode::core::{inv_order_i32, inv_order_i64};
use crate::codec::encode::core::{
    Budget, EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, PBITS_32, PBITS_64, PERM_1, PERM_2, PERM_3,
    PERM_4, skip_to, with_maxbits,
};
use crate::codec::transform::rev_inv_xform;
use crate::config::{ZfpConfig, ZfpRounding};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Two's-complement sign-magnitude mask for f32 (all bits except sign).
const TCMASK_F32: u32 = 0x7fff_ffff;
/// Two's-complement sign-magnitude mask for f64.
const TCMASK_F64: u64 = 0x7fff_ffff_ffff_ffff;

/// Reversible blocks are decoded without `ZFP_ROUND_LAST`'s bias. C applies it
/// inside `decode_ints`, which its reversible decoder shares with the lossy
/// one, so a C build with `ZFP_ROUND_LAST` is lossy in reversible mode. The
/// bias centres quantization error, and a reversible block has none: its
/// planes below the coded precision are exactly zero.
pub(crate) const NO_ROUNDING: ZfpRounding = ZfpRounding::Never;

// ---------------------------------------------------------------------------
// rev_decode_int_block_u32 / u64: shared integer reversible-decode helpers
// ---------------------------------------------------------------------------

/// Generate a helper that decodes a reversibly-encoded integer block into
/// `iblock`, and returns the bits it reads.
///
/// Callers skip the whole block's padding to `min_bits`, which keeps it out of
/// the plane loop. A macro rather than a generic function, as for
/// `decode_int_block!`.
macro_rules! rev_decode_int_block {
    ($name:ident, $int:ty, $uint:ty, $pbits:ident, $inv_order:ident) => {
        fn $name<const N: usize>(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            maxbits: u32,
            iblock: &mut [$int; N],
            perm: &[u8; N],
        ) -> u32
        where
            [$uint; N]: PlaneBlock,
        {
            // Read prec-1, then prec = bits+1
            let prec_minus_1 = bs.read_bits($pbits) as u32;
            let prec = prec_minus_1 + 1;

            // A budget that cannot bind, the usual case, is passed as a
            // constant, which lets the plane coder drop its budget checks.
            // Each arm reorders its own block, as merging the two blocks would
            // copy them.
            let maxbits = maxbits.saturating_sub($pbits);
            let ubits = if with_maxbits(maxbits, prec, <[$uint; N]>::SIZE) {
                let (ublock, ubits) = decode_bounded::<[$uint; N]>(bs, maxbits, prec);
                $inv_order(&ublock, iblock, perm);
                ubits
            } else {
                let (ublock, ubits, _) =
                    decode_ints::<[$uint; N], false>(bs, u32::MAX, prec, NO_ROUNDING);
                $inv_order(&ublock, iblock, perm);
                ubits
            };
            $pbits + ubits
        }
    };
}

rev_decode_int_block!(rev_decode_int_block_u32, i32, u32, PBITS_32, inv_order_i32);
rev_decode_int_block!(rev_decode_int_block_u64, i64, u64, PBITS_64, inv_order_i64);

/// [`decode_ints`] under a budget that binds. Out of line, so that the
/// unbounded call stays inlined.
#[inline(never)]
fn decode_bounded<B: PlaneBlock>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    maxbits: u32,
    prec: u32,
) -> (B, u32) {
    let (ublock, ubits, _) = decode_ints::<B, true>(bs, maxbits, prec, NO_ROUNDING);
    (ublock, ubits)
}

// ---------------------------------------------------------------------------
// rev_inv_reinterpret: two's-complement → sign-magnitude → float
// ---------------------------------------------------------------------------

pub(crate) fn rev_inv_reinterpret_f32(iblock: &[i32], fblock: &mut [f32]) {
    for (&i, f) in iblock.iter().zip(fblock.iter_mut()) {
        let x = if i < 0 {
            (i as u32) ^ TCMASK_F32
        } else {
            i as u32
        };
        *f = f32::from_bits(x);
    }
}

pub(crate) fn rev_inv_reinterpret_f64(iblock: &[i64], fblock: &mut [f64]) {
    for (&i, f) in iblock.iter().zip(fblock.iter_mut()) {
        let x = if i < 0 {
            (i as u64) ^ TCMASK_F64
        } else {
            i as u64
        };
        *f = f64::from_bits(x);
    }
}

// ---------------------------------------------------------------------------
// rev_decode_float_block / rev_decode_double_block
//
// The closure `rev_decode_int` receives the mutable iblock and the remaining
// `maxbits`. It decodes the integers and applies the inverse transform.
// ---------------------------------------------------------------------------

/// Generate the reversible float decoder for one float type.
macro_rules! rev_decode_float_block {
    (
        $name:ident, $float:ty, $int:ty, $ebits:ident, $ebias:ident, $inv_reinterpret:ident,
        $inv_cast:ident $(,)?
    ) => {
        fn $name<B: ZfpBitStreamOps + ?Sized, const N: usize>(
            bs: &mut B,
            fblock: &mut [$float; N],
            budget: Budget,
            rev_decode_int: impl FnOnce(&mut B, &mut [$int; N], u32) -> u32,
        ) -> usize {
            // Read 1 bit: is block non-zero?
            let nonzero = bs.read_bits(1);
            let mut bits = 1u32;

            if nonzero == 0 {
                // All-zero block
                for v in fblock.iter_mut() {
                    *v = 0.0;
                }
                return skip_to(bs, 1, budget.min);
            }

            // Read 1 more bit: BFP path (0) or reinterpret path (1)?
            let reinterpret = bs.read_bits(1);
            bits += 1;

            let mut iblock: [$int; N] = [0; N];

            if reinterpret != 0 {
                // Reinterpret path ("11" header)
                bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
                $inv_reinterpret(&iblock, fblock);
            } else {
                // BFP path ("01" header): read EBITS exponent
                let e_raw = bs.read_bits($ebits) as i32;
                bits += $ebits;
                let emax = e_raw - $ebias;
                bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
                $inv_cast(&iblock, fblock, emax);
            }

            skip_to(bs, bits, budget.min)
        }
    };
}

rev_decode_float_block!(
    rev_decode_float_block,
    f32,
    i32,
    EBITS_F32,
    EBIAS_F32,
    rev_inv_reinterpret_f32,
    inv_cast_f32,
);
rev_decode_float_block!(
    rev_decode_double_block,
    f64,
    i64,
    EBITS_F64,
    EBIAS_F64,
    rev_inv_reinterpret_f64,
    inv_cast_f64,
);

// ---------------------------------------------------------------------------
// Public API: 16 reversible decode functions
// ---------------------------------------------------------------------------

/// Generate the reversible decoders of one dimensionality: floats through the
/// BFP path, integers directly.
macro_rules! reversible_decoders {
    (
        $d:literal, $n:literal, $perm:ident:
        [$i32:ident, $i64:ident, $f32:ident, $f64:ident $(,)?]
    ) => {
        reversible_decoders!(@int $i32, i32, rev_decode_int_block_u32, $d, $n, $perm);
        reversible_decoders!(@int $i64, i64, rev_decode_int_block_u64, $d, $n, $perm);
        reversible_decoders!(@float $f32, f32, rev_decode_float_block, rev_decode_int_block_u32, $d, $n, $perm);
        reversible_decoders!(@float $f64, f64, rev_decode_double_block, rev_decode_int_block_u64, $d, $n, $perm);
    };
    (@int $name:ident, $ty:ty, $int_block:ident, $d:literal, $n:literal, $perm:ident) => {
        #[doc = concat!(
            "Reversible decode of a ", $d, "-D block of `", stringify!($ty),
            "` values; return bits read."
        )]
        pub fn $name(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            block: &mut [$ty; $n],
            config: &ZfpConfig,
        ) -> usize {
            let budget = Budget::of(config);
            let bits = $int_block(bs, budget.max, block, &$perm);
            let bits = skip_to(bs, bits, budget.min);
            rev_inv_xform(block);
            bits
        }
    };
    (
        @float $name:ident, $ty:ty, $float_block:ident, $int_block:ident, $d:literal,
        $n:literal, $perm:ident
    ) => {
        #[doc = concat!(
            "Reversible decode of a ", $d, "-D block of `", stringify!($ty),
            "` values; return bits read."
        )]
        pub fn $name(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            block: &mut [$ty; $n],
            config: &ZfpConfig,
        ) -> usize {
            $float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
                let bits = $int_block(bs, maxbits, iblock, &$perm);
                rev_inv_xform(iblock);
                bits
            })
        }
    };
}

reversible_decoders!("1", 4, PERM_1: [
    decode_block_reversible_1d_i32,
    decode_block_reversible_1d_i64,
    decode_block_reversible_1d_f32,
    decode_block_reversible_1d_f64,
]);
reversible_decoders!("2", 16, PERM_2: [
    decode_block_reversible_2d_i32,
    decode_block_reversible_2d_i64,
    decode_block_reversible_2d_f32,
    decode_block_reversible_2d_f64,
]);
reversible_decoders!("3", 64, PERM_3: [
    decode_block_reversible_3d_i32,
    decode_block_reversible_3d_i64,
    decode_block_reversible_3d_f32,
    decode_block_reversible_3d_f64,
]);
reversible_decoders!("4", 256, PERM_4: [
    decode_block_reversible_4d_i32,
    decode_block_reversible_4d_i64,
    decode_block_reversible_4d_f32,
    decode_block_reversible_4d_f64,
]);
