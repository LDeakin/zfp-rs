//! 2-D block encode (operates on a 4×4 block).
//!
//! Reference: `zfp/src/template/encode2.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::pad_strided;
use crate::codec::encode::core::strided_encode_wrappers;
use crate::codec::encode::float::{encode_block_2d_f32, encode_block_2d_f64};
use crate::codec::encode::integer::{encode_block_2d_i32, encode_block_2d_i64};

const MINBITS: u32 = 0;
const INT32_MAXBITS: u32 = 32 * 16 + 1;
const INT64_MAXBITS: u32 = 64 * 16 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 16;
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 16;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;
const FLOAT_MINEXP: i32 = -149;
const DOUBLE_MINEXP: i32 = -1074;

// ---------------------------------------------------------------------------
// Generic strided gather helpers (type-parameterised)
// ---------------------------------------------------------------------------

/// Gather a 4×4 block from a strided 2-D array.
///
/// # Safety
/// The caller must ensure the array spans at least 4 elements in each
/// dimension with the given strides.
unsafe fn gather_2d<T: Copy>(data: &[T], sx: isize, sy: isize) -> [T; 16] {
    // SAFETY: all elements are immediately written before being read.
    let mut block: [T; 16] = unsafe { std::mem::zeroed() };
    let p = data.as_ptr();
    let mut q = 0usize;
    for y in 0isize..4 {
        for x in 0isize..4 {
            // SAFETY: caller guarantees valid strides
            block[q] = unsafe { *p.offset(y * sy + x * sx) };
            q += 1;
        }
    }
    block
}

/// Gather a partial 2-D block (nx, ny ≤ 4) and pad to 4×4.
///
/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
unsafe fn gather_partial_2d<T: Copy + Default>(
    data: &[T],
    nx: usize,
    ny: usize,
    sx: isize,
    sy: isize,
) -> [T; 16] {
    let mut block = [T::default(); 16];
    let p = data.as_ptr();
    for y in 0..ny {
        for x in 0..nx {
            // SAFETY: caller guarantees valid strides
            block[4 * y + x] = unsafe { *p.offset(y.cast_signed() * sy + x.cast_signed() * sx) };
        }
        pad_strided!(block, 4 * y, nx, 1, T::default());
    }
    for x in 0..4usize {
        pad_strided!(block, x, ny, 4, T::default());
    }
    block
}

// ---------------------------------------------------------------------------
// Public API: contiguous
// ---------------------------------------------------------------------------

/// Encode a contiguous 2-D block of 16 `i32` values; return bits written.
pub fn encode_block_2d_i32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i32; 16]) -> usize {
    encode_block_2d_i32(bs, block, MINBITS, INT32_MAXBITS, INT32_MAXPREC)
}

/// Encode a contiguous 2-D block of 16 `i64` values; return bits written.
pub fn encode_block_2d_i64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i64; 16]) -> usize {
    encode_block_2d_i64(bs, block, MINBITS, INT64_MAXBITS, INT64_MAXPREC)
}

/// Encode a contiguous 2-D block of 16 `f32` values; return bits written.
pub fn encode_block_2d_f32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f32; 16]) -> usize {
    encode_block_2d_f32(
        bs,
        block,
        MINBITS,
        FLOAT_MAXBITS,
        FLOAT_MAXPREC,
        FLOAT_MINEXP,
    )
}

/// Encode a contiguous 2-D block of 16 `f64` values; return bits written.
pub fn encode_block_2d_f64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f64; 16]) -> usize {
    encode_block_2d_f64(
        bs,
        block,
        MINBITS,
        DOUBLE_MAXBITS,
        DOUBLE_MAXPREC,
        DOUBLE_MINEXP,
    )
}

// ---------------------------------------------------------------------------
// Strided block encode (generated)
// ---------------------------------------------------------------------------

strided_encode_wrappers! {
    ty: f64,
    gather: gather_2d,
    gather_partial: gather_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: encode_block_strided_2d_f64,
    partial: encode_partial_block_strided_2d_f64,
    full_rate: encode_block_strided_2d_f64_rate,
    partial_rate: encode_partial_block_strided_2d_f64_rate,
    encode: encode_block_2d_f64,
    encode_default: encode_block_2d_f64_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_encode_wrappers! {
    ty: f32,
    gather: gather_2d,
    gather_partial: gather_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: encode_block_strided_2d_f32,
    partial: encode_partial_block_strided_2d_f32,
    full_rate: encode_block_strided_2d_f32_rate,
    partial_rate: encode_partial_block_strided_2d_f32_rate,
    encode: encode_block_2d_f32,
    encode_default: encode_block_2d_f32_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_encode_wrappers! {
    ty: i32,
    gather: gather_2d,
    gather_partial: gather_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: encode_block_strided_2d_i32,
    partial: encode_partial_block_strided_2d_i32,
    full_rate: encode_block_strided_2d_i32_rate,
    partial_rate: encode_partial_block_strided_2d_i32_rate,
    encode: encode_block_2d_i32,
    encode_default: encode_block_2d_i32_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}

strided_encode_wrappers! {
    ty: i64,
    gather: gather_2d,
    gather_partial: gather_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: encode_block_strided_2d_i64,
    partial: encode_partial_block_strided_2d_i64,
    full_rate: encode_block_strided_2d_i64_rate,
    partial_rate: encode_partial_block_strided_2d_i64_rate,
    encode: encode_block_2d_i64,
    encode_default: encode_block_2d_i64_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}
