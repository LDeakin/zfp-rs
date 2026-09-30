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
    Budget, EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, PERM_1, PERM_2, PERM_3, PERM_4,
    with_maxbits,
};
use crate::codec::transform::rev_inv_xform;
use crate::config::{ZfpConfig, ZfpRounding};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PBITS_32: u32 = 5;
const PBITS_64: u32 = 6;

/// Two's-complement sign-magnitude mask for f32 (all bits except sign).
const TCMASK_F32: u32 = 0x7fff_ffff;
/// Two's-complement sign-magnitude mask for f64.
const TCMASK_F64: u64 = 0x7fff_ffff_ffff_ffff;

/// Reversible blocks are decoded without `ZFP_ROUND_LAST`'s bias. C applies it
/// inside `decode_ints`, which its reversible decoder shares with the lossy
/// one, so a C build with `ZFP_ROUND_LAST` is lossy in reversible mode. The
/// bias centres quantization error, and a reversible block has none: its
/// planes below the coded precision are exactly zero.
const NO_ROUNDING: ZfpRounding = ZfpRounding::Never;

// ---------------------------------------------------------------------------
// rev_decode_int_block: shared integer reversible-decode helper
// ---------------------------------------------------------------------------

/// Decode a reversibly-encoded integer block into `iblock`.
/// Returns bits read. Callers skip the whole block's padding to `min_bits`,
/// which keeps it out of the plane loop.
fn rev_decode_int_block_u32<const N: usize>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    maxbits: u32,
    iblock: &mut [i32; N],
    perm: &[u8; N],
) -> u32
where
    [u32; N]: PlaneBlock,
{
    // Read prec-1 (PBITS_32 bits), then prec = bits+1
    let prec_minus_1 = bs.read_bits(PBITS_32) as u32;
    let prec = prec_minus_1 + 1;

    // A budget that cannot bind, the usual case, is passed as a constant,
    // which lets the plane coder drop its budget checks. Each arm reorders its
    // own block, as merging the two blocks would copy them.
    let maxbits = maxbits.saturating_sub(PBITS_32);
    let ubits = if with_maxbits(maxbits, prec, <[u32; N]>::SIZE) {
        let (ublock, ubits) = decode_bounded::<[u32; N]>(bs, maxbits, prec);
        inv_order_i32(&ublock, iblock, perm);
        ubits
    } else {
        let (ublock, ubits, _) = decode_ints::<[u32; N], false>(bs, u32::MAX, prec, NO_ROUNDING);
        inv_order_i32(&ublock, iblock, perm);
        ubits
    };
    PBITS_32 + ubits
}

fn rev_decode_int_block_u64<const N: usize>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    maxbits: u32,
    iblock: &mut [i64; N],
    perm: &[u8; N],
) -> u32
where
    [u64; N]: PlaneBlock,
{
    let prec_minus_1 = bs.read_bits(PBITS_64) as u32;
    let prec = prec_minus_1 + 1;

    let maxbits = maxbits.saturating_sub(PBITS_64);
    let ubits = if with_maxbits(maxbits, prec, <[u64; N]>::SIZE) {
        let (ublock, ubits) = decode_bounded::<[u64; N]>(bs, maxbits, prec);
        inv_order_i64(&ublock, iblock, perm);
        ubits
    } else {
        let (ublock, ubits, _) = decode_ints::<[u64; N], false>(bs, u32::MAX, prec, NO_ROUNDING);
        inv_order_i64(&ublock, iblock, perm);
        ubits
    };
    PBITS_64 + ubits
}

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

/// Skip a block of `bits` bits to `minbits`; return its size.
fn skip_to(bs: &mut (impl ZfpBitStreamOps + ?Sized), bits: u32, minbits: u32) -> usize {
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
        minbits as usize
    } else {
        bits as usize
    }
}

// ---------------------------------------------------------------------------
// rev_inv_reinterpret: two's-complement → sign-magnitude → float
// ---------------------------------------------------------------------------

fn rev_inv_reinterpret_f32(iblock: &[i32], fblock: &mut [f32]) {
    for (&i, f) in iblock.iter().zip(fblock.iter_mut()) {
        let x = if i < 0 {
            (i as u32) ^ TCMASK_F32
        } else {
            i as u32
        };
        *f = f32::from_bits(x);
    }
}

fn rev_inv_reinterpret_f64(iblock: &[i64], fblock: &mut [f64]) {
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
// `maxbits`. It is responsible for:
//   - decoding the integers
//   - converting the slice to a fixed-size array (the size is determined by
//     the caller's block parameter, so this is guaranteed to succeed)
//   - applying the inverse transform
// ---------------------------------------------------------------------------

fn rev_decode_float_block<B: ZfpBitStreamOps + ?Sized, const N: usize>(
    bs: &mut B,
    fblock: &mut [f32; N],
    budget: Budget,
    rev_decode_int: impl FnOnce(&mut B, &mut [i32; N], u32) -> u32,
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

    let mut iblock = [0i32; N];

    if reinterpret != 0 {
        // Reinterpret path ("11" header)
        bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
        rev_inv_reinterpret_f32(&iblock, fblock);
    } else {
        // BFP path ("01" header): read EBITS exponent
        let e_raw = bs.read_bits(EBITS_F32) as i32;
        bits += EBITS_F32;
        let emax = e_raw - EBIAS_F32;
        bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
        inv_cast_f32(&iblock, fblock, emax);
    }

    skip_to(bs, bits, budget.min)
}

fn rev_decode_double_block<B: ZfpBitStreamOps + ?Sized, const N: usize>(
    bs: &mut B,
    fblock: &mut [f64; N],
    budget: Budget,
    rev_decode_int: impl FnOnce(&mut B, &mut [i64; N], u32) -> u32,
) -> usize {
    let nonzero = bs.read_bits(1);
    let mut bits = 1u32;

    if nonzero == 0 {
        for v in fblock.iter_mut() {
            *v = 0.0;
        }
        return skip_to(bs, 1, budget.min);
    }

    let reinterpret = bs.read_bits(1);
    bits += 1;

    let mut iblock = [0i64; N];

    if reinterpret != 0 {
        bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
        rev_inv_reinterpret_f64(&iblock, fblock);
    } else {
        let e_raw = bs.read_bits(EBITS_F64) as i32;
        bits += EBITS_F64;
        let emax = e_raw - EBIAS_F64;
        bits += rev_decode_int(bs, &mut iblock, budget.after(bits).max);
        inv_cast_f64(&iblock, fblock, emax);
    }

    skip_to(bs, bits, budget.min)
}

// ---------------------------------------------------------------------------
// Public API: 8 reversible decode functions
// ---------------------------------------------------------------------------

/// Reversible decode of a 1-D block of `f32` values; return bits read.
pub fn decode_block_reversible_1d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f32; 4],
    config: &ZfpConfig,
) -> usize {
    rev_decode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_1);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 1-D block of `f64` values; return bits read.
pub fn decode_block_reversible_1d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f64; 4],
    config: &ZfpConfig,
) -> usize {
    rev_decode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_1);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 2-D block of `f32` values; return bits read.
pub fn decode_block_reversible_2d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f32; 16],
    config: &ZfpConfig,
) -> usize {
    rev_decode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_2);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 2-D block of `f64` values; return bits read.
pub fn decode_block_reversible_2d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f64; 16],
    config: &ZfpConfig,
) -> usize {
    rev_decode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_2);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 3-D block of `f32` values; return bits read.
pub fn decode_block_reversible_3d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f32; 64],
    config: &ZfpConfig,
) -> usize {
    rev_decode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_3);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 3-D block of `f64` values; return bits read.
pub fn decode_block_reversible_3d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f64; 64],
    config: &ZfpConfig,
) -> usize {
    rev_decode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_3);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 4-D block of `f32` values; return bits read.
pub fn decode_block_reversible_4d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f32; 256],
    config: &ZfpConfig,
) -> usize {
    rev_decode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_4);
        rev_inv_xform(iblock);
        bits
    })
}

/// Reversible decode of a 4-D block of `f64` values; return bits read.
pub fn decode_block_reversible_4d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f64; 256],
    config: &ZfpConfig,
) -> usize {
    rev_decode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_4);
        rev_inv_xform(iblock);
        bits
    })
}

// ---------------------------------------------------------------------------
// Reversible decode for integers: direct (no BFP step)
// ---------------------------------------------------------------------------

/// Reversible decode of a 1-D block of `i32` values; return bits read.
pub fn decode_block_reversible_1d_i32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i32; 4],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i32; 4] = [0; 4];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u32(bs, budget.max, &mut iblock, &PERM_1);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 1-D block of `i64` values; return bits read.
pub fn decode_block_reversible_1d_i64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i64; 4],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i64; 4] = [0; 4];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u64(bs, budget.max, &mut iblock, &PERM_1);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 2-D block of `i32` values; return bits read.
pub fn decode_block_reversible_2d_i32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i32; 16],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i32; 16] = [0; 16];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u32(bs, budget.max, &mut iblock, &PERM_2);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 2-D block of `i64` values; return bits read.
pub fn decode_block_reversible_2d_i64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i64; 16],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i64; 16] = [0; 16];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u64(bs, budget.max, &mut iblock, &PERM_2);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 3-D block of `i32` values; return bits read.
pub fn decode_block_reversible_3d_i32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i32; 64],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i32; 64] = [0; 64];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u32(bs, budget.max, &mut iblock, &PERM_3);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 3-D block of `i64` values; return bits read.
pub fn decode_block_reversible_3d_i64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i64; 64],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i64; 64] = [0; 64];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u64(bs, budget.max, &mut iblock, &PERM_3);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 4-D block of `i32` values; return bits read.
pub fn decode_block_reversible_4d_i32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i32; 256],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i32; 256] = [0; 256];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u32(bs, budget.max, &mut iblock, &PERM_4);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 4-D block of `i64` values; return bits read.
pub fn decode_block_reversible_4d_i64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i64; 256],
    config: &ZfpConfig,
) -> usize {
    let mut iblock: [i64; 256] = [0; 256];
    let budget = Budget::of(config);
    let bits = rev_decode_int_block_u64(bs, budget.max, &mut iblock, &PERM_4);
    let bits = skip_to(bs, bits, budget.min);
    crate::codec::transform::rev_inv_xform(&mut iblock);
    *block = iblock;
    bits
}
