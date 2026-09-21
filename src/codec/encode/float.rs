#![allow(clippy::cast_sign_loss)] // i32→u32 for exponent encoding
#![allow(clippy::cast_precision_loss)] // i32→f32 and i64→f64 for reconstruction (intentional loss)
//! Floating-point-specific encode path (exponent extraction + significand coding).
//!
//! Reference: `zfp/src/template/encodef.c`, `codecf.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::{
    exponent_block_f32, exponent_block_f64, fwd_cast_f32, fwd_cast_f64, precision_f,
};
use crate::codec::encode::integer::{
    encode_block_1d_i32, encode_block_1d_i64, encode_block_2d_i32, encode_block_2d_i64,
    encode_block_3d_i32, encode_block_3d_i64, encode_block_4d_i32, encode_block_4d_i64,
};
use crate::config::ZfpRounding;

// Number of exponent bits: 8 for f32, 11 for f64.
const EBITS_F32: u32 = 8;
const EBITS_F64: u32 = 11;
const EBIAS_F32: i32 = 127;
const EBIAS_F64: i32 = 1023;

// ---------------------------------------------------------------------------
// Shared encode helpers (generic over block size N)
// ---------------------------------------------------------------------------

/// Generic f32-block encode: exponent header + integer block encode.
fn encode_float_block<const N: usize, F>(
    bs: &mut dyn ZfpBitStreamMutOps,
    fblock: &[f32; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
    encode_int: F,
) -> usize
where
    F: FnOnce(&mut [i32; N], &mut dyn ZfpBitStreamMutOps, u32, u32, u32) -> usize,
{
    // Compute the number of dimensions from the block size N (4=1D, 16=2D, 64=3D, 256=4D).
    // SAFETY: N is always 4, 16, 64, or 256 (powers of 4), so trailing_zeros is even and ≥ 2.
    #[allow(clippy::cast_possible_truncation)]
    let dims = N.trailing_zeros() / 2;
    let emax = exponent_block_f32(fblock);
    let prec = precision_f(emax, maxprec, minexp, dims, rounding.tight_error());
    let e = if prec != 0 {
        (emax + EBIAS_F32) as u32
    } else {
        0
    };

    if e != 0 {
        let header_bits = EBITS_F32 + 1;
        bs.write_bits(2 * u64::from(e) + 1, header_bits);
        let mut iblock = [0i32; N];
        fwd_cast_f32(&mut iblock, fblock, emax);
        let remaining_min = minbits.saturating_sub(header_bits);
        let remaining_max = maxbits.saturating_sub(header_bits);
        header_bits as usize + encode_int(&mut iblock, bs, remaining_min, remaining_max, prec)
    } else {
        bs.write_bit(0);
        let bits = 1u32;
        if bits < minbits {
            bs.pad((minbits - bits) as usize);
            minbits as usize
        } else {
            bits as usize
        }
    }
}

/// Generic f64-block encode: exponent header + integer block encode.
fn encode_double_block<const N: usize, F>(
    bs: &mut dyn ZfpBitStreamMutOps,
    fblock: &[f64; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
    encode_int: F,
) -> usize
where
    F: FnOnce(&mut [i64; N], &mut dyn ZfpBitStreamMutOps, u32, u32, u32) -> usize,
{
    // Compute the number of dimensions from the block size N (4=1D, 16=2D, 64=3D, 256=4D).
    // SAFETY: N is always 4, 16, 64, or 256 (powers of 4), so trailing_zeros is even and ≥ 2.
    #[allow(clippy::cast_possible_truncation)]
    let dims = N.trailing_zeros() / 2;
    let emax = exponent_block_f64(fblock);
    let prec = precision_f(emax, maxprec, minexp, dims, rounding.tight_error());
    let e = if prec != 0 {
        (emax + EBIAS_F64) as u32
    } else {
        0
    };

    if e != 0 {
        let header_bits = EBITS_F64 + 1;
        bs.write_bits(2 * u64::from(e) + 1, header_bits);
        let mut iblock = [0i64; N];
        fwd_cast_f64(&mut iblock, fblock, emax);
        let remaining_min = minbits.saturating_sub(header_bits);
        let remaining_max = maxbits.saturating_sub(header_bits);
        header_bits as usize + encode_int(&mut iblock, bs, remaining_min, remaining_max, prec)
    } else {
        bs.write_bit(0);
        let bits = 1u32;
        if bits < minbits {
            bs.pad((minbits - bits) as usize);
            minbits as usize
        } else {
            bits as usize
        }
    }
}

// ---------------------------------------------------------------------------
// 1-D
// ---------------------------------------------------------------------------

/// Encode a 1-D block of 4 `f32` values; returns bits written.
pub fn encode_block_1d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f32; 4],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_float_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_1d_i32(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 1-D block of 4 `f64` values; returns bits written.
pub fn encode_block_1d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f64; 4],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_double_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_1d_i64(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 2-D block of 16 `f32` values; returns bits written.
pub fn encode_block_2d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f32; 16],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_float_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_2d_i32(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 2-D block of 16 `f64` values; returns bits written.
pub fn encode_block_2d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f64; 16],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_double_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_2d_i64(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 3-D block of 64 `f32` values; returns bits written.
pub fn encode_block_3d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f32; 64],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_float_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_3d_i32(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 3-D block of 64 `f64` values; returns bits written.
pub fn encode_block_3d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f64; 64],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_double_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_3d_i64(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 4-D block of 256 `f32` values; returns bits written.
pub fn encode_block_4d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f32; 256],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_float_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_4d_i32(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}

/// Encode a 4-D block of 256 `f64` values; returns bits written.
pub fn encode_block_4d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    block: &[f64; 256],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    rounding: ZfpRounding,
) -> usize {
    encode_double_block(
        bs,
        block,
        minbits,
        maxbits,
        maxprec,
        minexp,
        rounding,
        |iblock, bs, minbits, maxbits, maxprec| {
            encode_block_4d_i64(bs, iblock, minbits, maxbits, maxprec, rounding)
        },
    )
}
