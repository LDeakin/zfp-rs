//! Reversible (lossless) decode path.
//!
//! Reference: `zfp/src/template/revdecode{1-4}.c`,
//!            `zfp/src/template/revdecodef.c`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)] // i32→u32 and i64→u64 for BFP reconstruction

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::{decode_ints_u32, decode_ints_u64, inv_order_i32, inv_order_i64};
use crate::codec::decode::core::{inv_cast_f32, inv_cast_f64};
use crate::codec::encode::core::{PERM_1, PERM_2, PERM_3, PERM_4};
use crate::codec::transform::{
    rev_inv_xform_1d, rev_inv_xform_1d_i64, rev_inv_xform_2d, rev_inv_xform_2d_i64,
    rev_inv_xform_3d, rev_inv_xform_3d_i64, rev_inv_xform_4d, rev_inv_xform_4d_i64,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PBITS_32: u32 = 5;
const PBITS_64: u32 = 6;

const EBITS_F32: u32 = 8;
const EBITS_F64: u32 = 11;
const EBIAS_F32: i32 = 127;
const EBIAS_F64: i32 = 1023;

/// Two's-complement sign-magnitude mask for f32 (all bits except sign).
const TCMASK_F32: u32 = 0x7fff_ffff;
/// Two's-complement sign-magnitude mask for f64.
const TCMASK_F64: u64 = 0x7fff_ffff_ffff_ffff;

// ---------------------------------------------------------------------------
// rev_decode_int_block: shared integer reversible-decode helper
// ---------------------------------------------------------------------------

/// Decode a reversibly-encoded integer block into `iblock`.
/// Returns bits read.
fn rev_decode_int_block_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    iblock: &mut [i32],
    perm: &[u8],
) -> usize {
    let n = iblock.len();
    // Read prec-1 (PBITS_32 bits), then prec = bits+1
    let prec_minus_1 = bs.read_bits(PBITS_32) as u32;
    let prec = prec_minus_1 + 1;
    let mut bits = PBITS_32 as usize;

    let mut ublock = vec![0u32; n];
    let remaining = maxbits.saturating_sub(PBITS_32);
    bits += decode_ints_u32(bs, remaining, prec, &mut ublock) as usize;

    inv_order_i32(&ublock, iblock, perm);
    bits
}

fn rev_decode_int_block_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    iblock: &mut [i64],
    perm: &[u8],
) -> usize {
    let n = iblock.len();
    let prec_minus_1 = bs.read_bits(PBITS_64) as u32;
    let prec = prec_minus_1 + 1;
    let mut bits = PBITS_64 as usize;

    let mut ublock = vec![0u64; n];
    let remaining = maxbits.saturating_sub(PBITS_64);
    bits += decode_ints_u64(bs, remaining, prec, &mut ublock) as usize;

    inv_order_i64(&ublock, iblock, perm);
    bits
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
// bit budget. It is responsible for:
//   - decoding the integers
//   - converting the slice to a fixed-size array (the size is determined by
//     the caller's block parameter, so this is guaranteed to succeed)
//   - applying the inverse transform
// ---------------------------------------------------------------------------

fn rev_decode_float_block<const N: usize>(
    bs: &mut dyn ZfpBitStreamOps,
    fblock: &mut [f32; N],
    maxbits: u32,
    rev_decode_int: impl FnOnce(&mut dyn ZfpBitStreamOps, &mut [i32; N], u32) -> usize,
) -> usize {
    // Read 1 bit: is block non-zero?
    let nonzero = bs.read_bits(1);
    let mut bits = 1usize;

    if nonzero == 0 {
        // All-zero block
        for v in fblock.iter_mut() {
            *v = 0.0;
        }
        return bits;
    }

    // Read 1 more bit: BFP path (0) or reinterpret path (1)?
    let reinterpret = bs.read_bits(1);
    bits += 1;

    let mut iblock = [0i32; N];

    if reinterpret != 0 {
        // Reinterpret path ("11" header)
        let remaining = maxbits.saturating_sub(bits as u32);
        bits += rev_decode_int(bs, &mut iblock, remaining);
        rev_inv_reinterpret_f32(&iblock, fblock);
    } else {
        // BFP path ("01" header): read EBITS exponent
        let e_raw = bs.read_bits(EBITS_F32) as i32;
        bits += EBITS_F32 as usize;
        let emax = e_raw - EBIAS_F32;
        let remaining = maxbits.saturating_sub(bits as u32);
        bits += rev_decode_int(bs, &mut iblock, remaining);
        inv_cast_f32(&iblock, fblock, emax);
    }

    bits
}

fn rev_decode_double_block<const N: usize>(
    bs: &mut dyn ZfpBitStreamOps,
    fblock: &mut [f64; N],
    maxbits: u32,
    rev_decode_int: impl FnOnce(&mut dyn ZfpBitStreamOps, &mut [i64; N], u32) -> usize,
) -> usize {
    let nonzero = bs.read_bits(1);
    let mut bits = 1usize;

    if nonzero == 0 {
        for v in fblock.iter_mut() {
            *v = 0.0;
        }
        return bits;
    }

    let reinterpret = bs.read_bits(1);
    bits += 1;

    let mut iblock = [0i64; N];

    if reinterpret != 0 {
        let remaining = maxbits.saturating_sub(bits as u32);
        bits += rev_decode_int(bs, &mut iblock, remaining);
        rev_inv_reinterpret_f64(&iblock, fblock);
    } else {
        let e_raw = bs.read_bits(EBITS_F64) as i32;
        bits += EBITS_F64 as usize;
        let emax = e_raw - EBIAS_F64;
        let remaining = maxbits.saturating_sub(bits as u32);
        bits += rev_decode_int(bs, &mut iblock, remaining);
        inv_cast_f64(&iblock, fblock, emax);
    }

    bits
}

// ---------------------------------------------------------------------------
// Public API: 8 reversible decode functions
// ---------------------------------------------------------------------------

/// Reversible decode of a 1-D block of `f32` values; return bits read.
pub fn decode_block_reversible_1d_f32(bs: &mut dyn ZfpBitStreamOps, block: &mut [f32; 4]) -> usize {
    rev_decode_float_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_1);
        rev_inv_xform_1d(iblock);
        bits
    })
}

/// Reversible decode of a 1-D block of `f64` values; return bits read.
pub fn decode_block_reversible_1d_f64(bs: &mut dyn ZfpBitStreamOps, block: &mut [f64; 4]) -> usize {
    rev_decode_double_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_1);
        rev_inv_xform_1d_i64(iblock);
        bits
    })
}

/// Reversible decode of a 2-D block of `f32` values; return bits read.
pub fn decode_block_reversible_2d_f32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f32; 16],
) -> usize {
    rev_decode_float_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_2);
        rev_inv_xform_2d(iblock);
        bits
    })
}

/// Reversible decode of a 2-D block of `f64` values; return bits read.
pub fn decode_block_reversible_2d_f64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f64; 16],
) -> usize {
    rev_decode_double_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_2);
        rev_inv_xform_2d_i64(iblock);
        bits
    })
}

/// Reversible decode of a 3-D block of `f32` values; return bits read.
pub fn decode_block_reversible_3d_f32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f32; 64],
) -> usize {
    rev_decode_float_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_3);
        rev_inv_xform_3d(iblock);
        bits
    })
}

/// Reversible decode of a 3-D block of `f64` values; return bits read.
pub fn decode_block_reversible_3d_f64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f64; 64],
) -> usize {
    rev_decode_double_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_3);
        rev_inv_xform_3d_i64(iblock);
        bits
    })
}

/// Reversible decode of a 4-D block of `f32` values; return bits read.
pub fn decode_block_reversible_4d_f32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f32; 256],
) -> usize {
    rev_decode_float_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u32(bs, maxbits, iblock, &PERM_4);
        rev_inv_xform_4d(iblock);
        bits
    })
}

/// Reversible decode of a 4-D block of `f64` values; return bits read.
pub fn decode_block_reversible_4d_f64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [f64; 256],
) -> usize {
    rev_decode_double_block(bs, block, u32::MAX, |bs, iblock, maxbits| {
        let bits = rev_decode_int_block_u64(bs, maxbits, iblock, &PERM_4);
        rev_inv_xform_4d_i64(iblock);
        bits
    })
}

// ---------------------------------------------------------------------------
// Reversible decode for integers: direct (no BFP step)
// ---------------------------------------------------------------------------

/// Reversible decode of a 1-D block of `i32` values; return bits read.
pub fn decode_block_reversible_1d_i32(bs: &mut dyn ZfpBitStreamOps, block: &mut [i32; 4]) -> usize {
    let mut iblock: [i32; 4] = [0; 4];
    let bits = rev_decode_int_block_u32(bs, u32::MAX, &mut iblock, &PERM_1);
    crate::codec::transform::rev_inv_xform_1d(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 1-D block of `i64` values; return bits read.
pub fn decode_block_reversible_1d_i64(bs: &mut dyn ZfpBitStreamOps, block: &mut [i64; 4]) -> usize {
    let mut iblock: [i64; 4] = [0; 4];
    let bits = rev_decode_int_block_u64(bs, u32::MAX, &mut iblock, &PERM_1);
    crate::codec::transform::rev_inv_xform_1d_i64(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 2-D block of `i32` values; return bits read.
pub fn decode_block_reversible_2d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i32; 16],
) -> usize {
    let mut iblock: [i32; 16] = [0; 16];
    let bits = rev_decode_int_block_u32(bs, u32::MAX, &mut iblock, &PERM_2);
    crate::codec::transform::rev_inv_xform_2d(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 2-D block of `i64` values; return bits read.
pub fn decode_block_reversible_2d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i64; 16],
) -> usize {
    let mut iblock: [i64; 16] = [0; 16];
    let bits = rev_decode_int_block_u64(bs, u32::MAX, &mut iblock, &PERM_2);
    crate::codec::transform::rev_inv_xform_2d_i64(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 3-D block of `i32` values; return bits read.
pub fn decode_block_reversible_3d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i32; 64],
) -> usize {
    let mut iblock: [i32; 64] = [0; 64];
    let bits = rev_decode_int_block_u32(bs, u32::MAX, &mut iblock, &PERM_3);
    crate::codec::transform::rev_inv_xform_3d(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 3-D block of `i64` values; return bits read.
pub fn decode_block_reversible_3d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i64; 64],
) -> usize {
    let mut iblock: [i64; 64] = [0; 64];
    let bits = rev_decode_int_block_u64(bs, u32::MAX, &mut iblock, &PERM_3);
    crate::codec::transform::rev_inv_xform_3d_i64(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 4-D block of `i32` values; return bits read.
pub fn decode_block_reversible_4d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i32; 256],
) -> usize {
    let mut iblock: [i32; 256] = [0; 256];
    let bits = rev_decode_int_block_u32(bs, u32::MAX, &mut iblock, &PERM_4);
    crate::codec::transform::rev_inv_xform_4d(&mut iblock);
    *block = iblock;
    bits
}

/// Reversible decode of a 4-D block of `i64` values; return bits read.
pub fn decode_block_reversible_4d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    block: &mut [i64; 256],
) -> usize {
    let mut iblock: [i64; 256] = [0; 256];
    let bits = rev_decode_int_block_u64(bs, u32::MAX, &mut iblock, &PERM_4);
    crate::codec::transform::rev_inv_xform_4d_i64(&mut iblock);
    *block = iblock;
    bits
}
