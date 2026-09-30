//! Reversible (lossless) encode path.
//!
//! Reference: `zfp/src/template/revencode{1-4}.c`,
//!            `zfp/src/template/revencodef.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::bitplane::{PlaneBlock, encode_ints};
use crate::codec::encode::core::{
    Budget, EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, MIN_CAST_EMAX_F32, MIN_CAST_EMAX_F64,
    PERM_1, PERM_2, PERM_3, PERM_4, exponent_block_f32, exponent_block_f64, fwd_cast_f32,
    fwd_cast_f64, fwd_order_i32, fwd_order_i64, pad_to, with_maxbits,
};
use crate::codec::transform::rev_fwd_xform;
use crate::config::ZfpConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Bits used to encode `prec - 1` for f32/i32 blocks (5 bits → max prec 31).
const PBITS_32: u32 = 5;
/// Bits used to encode `prec - 1` for f64/i64 blocks (6 bits → max prec 63).
const PBITS_64: u32 = 6;

// Two's-complement sign-magnitude mask (all bits set except sign for the
// purposes of the reversibility check in rev_fwd_cast).
const TCMASK_F32: u32 = 0x7fff_ffff;
const TCMASK_F64: u64 = 0x7fff_ffff_ffff_ffff;

// ---------------------------------------------------------------------------
// rev_precision: count significant bits across a u32/u64 block
// ---------------------------------------------------------------------------

/// Compute the number of bit planes from the MSB needed to represent all values in `data`.
///
/// Equivalent to `INTPREC - trailing_zeros(OR of all values)`.
/// Matches the C `rev_precision` algorithm: for value `3 = 0b11`, returns 32 (all 32 planes
/// must be encoded because bit 0 is set); for `0x80000000`, returns 1 (only the top plane).
#[inline]
fn rev_precision_u32(data: &[u32]) -> u32 {
    let or: u32 = data.iter().fold(0u32, |acc, &v| acc | v);
    if or == 0 { 0 } else { 32 - or.trailing_zeros() }
}

/// Same as `rev_precision_u32` but for 64-bit values.
#[inline]
fn rev_precision_u64(data: &[u64]) -> u32 {
    let or: u64 = data.iter().fold(0u64, |acc, &v| acc | v);
    if or == 0 { 0 } else { 64 - or.trailing_zeros() }
}

// ---------------------------------------------------------------------------
// rev_inv_cast: reconstruct floats from integer block (for reversibility test)
// ---------------------------------------------------------------------------

/// Inverse block-floating-point: recover f32 values from i32 block at exponent `emax`.
/// Returns true only if every reconstructed value equals the original float bits.
#[expect(clippy::cast_precision_loss)]
fn rev_inv_cast_f32(iblock: &[i32], fblock: &[f32], emax: i32) -> bool {
    // s = 2^(emax - 30)
    let s = libm::ldexpf(1.0f32, emax - 30);
    for (&i, &f) in iblock.iter().zip(fblock.iter()) {
        let reconstructed = s * (i as f32);
        if reconstructed.to_bits() != f.to_bits() {
            return false;
        }
    }
    true
}

/// Same as `rev_inv_cast_f32` but for f64/i64.
#[expect(clippy::cast_precision_loss)]
fn rev_inv_cast_f64(iblock: &[i64], fblock: &[f64], emax: i32) -> bool {
    let s = libm::ldexp(1.0f64, emax - 62);
    for (&i, &f) in iblock.iter().zip(fblock.iter()) {
        let reconstructed = s * (i as f64);
        if reconstructed.to_bits() != f.to_bits() {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// rev_encode_int_block_u32 / u64
// ---------------------------------------------------------------------------

/// Generate a shared integer reversible-encode helper, which applies
/// `fwd_order`, then writes the precision header and encodes the integers.
///
/// It returns the bits it writes. Callers pad the whole block to `min_bits`,
/// which keeps it out of the plane loop. A macro rather than a generic
/// function, as for `decode_int_block!`.
macro_rules! rev_encode_int_block {
    ($name:ident, $int:ty, $uint:ty, $pbits:ident, $fwd_order:ident, $rev_precision:ident) => {
        fn $name<const N: usize>(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            iblock: &[$int; N],
            maxbits: u32,
            maxprec: u32,
            perm: &[u8; N],
        ) -> u32
        where
            [$uint; N]: PlaneBlock,
        {
            let mut ublock: [$uint; N] = [0; N];
            $fwd_order(&mut ublock, iblock, perm);

            // As C: at most `maxprec` bit planes, and always at least one.
            let prec = $rev_precision(&ublock).min(maxprec).max(1);

            bs.write_bits(u64::from(prec - 1), $pbits);

            // A budget that cannot bind, the usual case, is passed as a
            // constant, which lets the plane coder drop its budget checks.
            let maxbits = maxbits.saturating_sub($pbits);
            $pbits
                + if with_maxbits(maxbits, prec, <[$uint; N]>::SIZE) {
                    encode_bounded(bs, maxbits, prec, &ublock)
                } else {
                    encode_ints::<_, false>(bs, u32::MAX, prec, &ublock)
                }
        }
    };
}

rev_encode_int_block!(
    rev_encode_int_block_u32,
    i32,
    u32,
    PBITS_32,
    fwd_order_i32,
    rev_precision_u32
);
rev_encode_int_block!(
    rev_encode_int_block_u64,
    i64,
    u64,
    PBITS_64,
    fwd_order_i64,
    rev_precision_u64
);

/// [`encode_ints`] under a budget that binds. Out of line, so that the
/// unbounded call stays inlined.
#[inline(never)]
fn encode_bounded<B: PlaneBlock>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    maxbits: u32,
    prec: u32,
    block: &B,
) -> u32 {
    encode_ints::<_, true>(bs, maxbits, prec, block)
}

// ---------------------------------------------------------------------------
// rev_encode_float_block / rev_encode_double_block
//
// Reversible float path:
//   1. Compute emax.
//   2. If emax == -EBIAS (all zero): write single 0 bit.
//   3. Try BFP cast (fwd_cast) + reversibility check (rev_inv_cast).
//   4. If reversible: write "01" (2 bits), write emax+EBIAS as EBITS bits,
//      call rev_encode_int_block on i-block.
//   5. If not reversible: reinterpret float bits as two's complement (sign flip
//      on negative values using TCMASK), write "11" (2 bits),
//      call rev_encode_int_block on reinterpreted i-block.
//   6. Pad the block to minbits. C does not pad an all-zero block, yet its
//      decoder skips to minbits, so C misdecodes its own streams whenever
//      minbits > 1; padding keeps the two in step.
//
// The closure `rev_encode_int` receives the mutable iblock and the remaining
// `maxbits`. It applies the forward transform and encodes the integers.
// ---------------------------------------------------------------------------

/// Generate the reversible float path above for one float type.
macro_rules! rev_encode_float_block {
    (
        $name:ident, $float:ty, $int:ty, $sign_shift:literal, $ebits:ident, $ebias:ident,
        $min_cast_emax:ident, $tcmask:ident, $exponent_block:ident, $fwd_cast:ident,
        $rev_inv_cast:ident $(,)?
    ) => {
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // unsigned→signed for reversible encoding
        fn $name<B: ZfpBitStreamMutOps + ?Sized, const N: usize>(
            bs: &mut B,
            fblock: &[$float; N],
            budget: Budget,
            rev_encode_int: impl FnOnce(&mut B, &mut [$int; N], u32) -> u32,
        ) -> usize {
            let emax = $exponent_block(fblock);
            let e = (emax + $ebias) as u32;

            // C's cast of a nonzero block below `MIN_CAST_EMAX_F32` (or `_F64`)
            // overflows (see `fwd_cast_f32`), and gives integers that
            // reconstruct to about `±2^(emax+1)`, or zero once the inverse
            // scale underflows. Neither is the block's largest value, which is
            // nonzero and below `2^emax`, so C always reinterprets such a
            // block. Do the same without casting, since the cast no longer
            // overflows.
            let tiny = e != 0 && emax < $min_cast_emax;

            // BFP cast: if emax == -EBIAS, all iblock[i] = 0
            let mut iblock: [$int; N] = [0; N];
            if e != 0 && !tiny {
                $fwd_cast(&mut iblock, fblock, emax);
            }

            let bits = if !tiny && $rev_inv_cast(&iblock, fblock, emax) {
                // BFP path is reversible
                if e != 0 {
                    // Non-zero block: write "01" header + EBITS exponent
                    bs.write_bits(0b01, 2);
                    bs.write_bits(u64::from(e), $ebits);
                    let header_bits = 2 + $ebits;
                    header_bits + rev_encode_int(bs, &mut iblock, budget.after(header_bits).max)
                } else {
                    // Genuinely all-zero block: write single "0" bit
                    bs.put_bit(0);
                    1
                }
            } else {
                // BFP is not reversible; reinterpret float bits as two's
                // complement integers
                let mut iblock_tc = fblock.map(|f| {
                    let bits = f.to_bits();
                    if bits >> $sign_shift != 0 {
                        (bits ^ $tcmask) as $int
                    } else {
                        bits as $int
                    }
                });
                // Write "11" header
                bs.write_bits(0b11, 2);
                2 + rev_encode_int(bs, &mut iblock_tc, budget.after(2).max)
            };
            pad_to(bs, bits, budget.min)
        }
    };
}

rev_encode_float_block!(
    rev_encode_float_block,
    f32,
    i32,
    31,
    EBITS_F32,
    EBIAS_F32,
    MIN_CAST_EMAX_F32,
    TCMASK_F32,
    exponent_block_f32,
    fwd_cast_f32,
    rev_inv_cast_f32,
);
rev_encode_float_block!(
    rev_encode_double_block,
    f64,
    i64,
    63,
    EBITS_F64,
    EBIAS_F64,
    MIN_CAST_EMAX_F64,
    TCMASK_F64,
    exponent_block_f64,
    fwd_cast_f64,
    rev_inv_cast_f64,
);

// ---------------------------------------------------------------------------
// Public API: 16 reversible encode functions
// ---------------------------------------------------------------------------

/// Generate the reversible encoders of one dimensionality: floats through the
/// BFP path, integers directly.
macro_rules! reversible_encoders {
    (
        $d:literal, $n:literal, $perm:ident:
        [$i32:ident, $i64:ident, $f32:ident, $f64:ident $(,)?]
    ) => {
        reversible_encoders!(@int $i32, i32, rev_encode_int_block_u32, $d, $n, $perm);
        reversible_encoders!(@int $i64, i64, rev_encode_int_block_u64, $d, $n, $perm);
        reversible_encoders!(@float $f32, f32, rev_encode_float_block, rev_encode_int_block_u32, $d, $n, $perm);
        reversible_encoders!(@float $f64, f64, rev_encode_double_block, rev_encode_int_block_u64, $d, $n, $perm);
    };
    (@int $name:ident, $ty:ty, $int_block:ident, $d:literal, $n:literal, $perm:ident) => {
        #[doc = concat!(
            "Reversible encode of a ", $d, "-D block of `", stringify!($ty),
            "` values, which it transforms in place; return bits written."
        )]
        pub fn $name(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            block: &mut [$ty; $n],
            config: &ZfpConfig,
        ) -> usize {
            rev_fwd_xform(block);
            let budget = Budget::of(config);
            let bits = $int_block(bs, block, budget.max, config.max_prec(), &$perm);
            pad_to(bs, bits, budget.min)
        }
    };
    (
        @float $name:ident, $ty:ty, $float_block:ident, $int_block:ident, $d:literal,
        $n:literal, $perm:ident
    ) => {
        #[doc = concat!(
            "Reversible encode of a ", $d, "-D block of `", stringify!($ty),
            "` values; return bits written."
        )]
        pub fn $name(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            block: &[$ty; $n],
            config: &ZfpConfig,
        ) -> usize {
            $float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
                rev_fwd_xform(iblock);
                $int_block(bs, iblock, maxbits, config.max_prec(), &$perm)
            })
        }
    };
}

reversible_encoders!("1", 4, PERM_1: [
    encode_block_reversible_1d_i32,
    encode_block_reversible_1d_i64,
    encode_block_reversible_1d_f32,
    encode_block_reversible_1d_f64,
]);
reversible_encoders!("2", 16, PERM_2: [
    encode_block_reversible_2d_i32,
    encode_block_reversible_2d_i64,
    encode_block_reversible_2d_f32,
    encode_block_reversible_2d_f64,
]);
reversible_encoders!("3", 64, PERM_3: [
    encode_block_reversible_3d_i32,
    encode_block_reversible_3d_i64,
    encode_block_reversible_3d_f32,
    encode_block_reversible_3d_f64,
]);
reversible_encoders!("4", 256, PERM_4: [
    encode_block_reversible_4d_i32,
    encode_block_reversible_4d_i64,
    encode_block_reversible_4d_f32,
    encode_block_reversible_4d_f64,
]);
