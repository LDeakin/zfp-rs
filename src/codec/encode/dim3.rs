//! 3-D block encode (operates on a 4×4×4 block).
//!
//! Reference: `zfp/src/template/encode3.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::pad_strided;
use crate::codec::encode::core::strided_encode_wrappers;
use crate::codec::encode::float::{encode_block_3d_f32, encode_block_3d_f64};
use crate::codec::encode::integer::{encode_block_3d_i32, encode_block_3d_i64};

const MINBITS: u32 = 0;
const INT32_MAXBITS: u32 = 32 * 64 + 1;
const INT64_MAXBITS: u32 = 64 * 64 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 64;
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 64;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;
const FLOAT_MINEXP: i32 = -149;
const DOUBLE_MINEXP: i32 = -1074;

// ---------------------------------------------------------------------------
// Generic strided gather helpers (type-parameterised)
// ---------------------------------------------------------------------------

/// Gather a 4×4×4 block from a strided 3-D array.
///
/// # Safety
/// The caller must ensure the array spans at least 4 elements in each
/// dimension with the given strides.
fn gather_3d<T: Copy>(data: &[T], sx: isize, sy: isize, sz: isize) -> [T; 64] {
    // SAFETY: all elements are immediately written before being read.
    let mut block: [T; 64] = unsafe { std::mem::zeroed() };
    let p = data.as_ptr();
    let mut q = 0usize;
    for z in 0isize..4 {
        for y in 0isize..4 {
            for x in 0isize..4 {
                // SAFETY: caller guarantees valid strides
                block[q] = unsafe { *p.offset(z * sz + y * sy + x * sx) };
                q += 1;
            }
        }
    }
    block
}

/// Gather a partial 3-D block (nx, ny, nz ≤ 4) and pad to 4×4×4.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn gather_partial_3d<T: Copy + Default>(
    data: &[T],
    nx: usize,
    ny: usize,
    nz: usize,
    sx: isize,
    sy: isize,
    sz: isize,
) -> [T; 64] {
    let mut block = [T::default(); 64];
    let p = data.as_ptr();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                // SAFETY: caller guarantees valid strides
                block[16 * z + 4 * y + x] = unsafe {
                    *p.offset(z.cast_signed() * sz + y.cast_signed() * sy + x.cast_signed() * sx)
                };
            }
            pad_strided!(block, 16 * z + 4 * y, nx, 1, T::default());
        }
        for x in 0..4usize {
            pad_strided!(block, 16 * z + x, ny, 4, T::default());
        }
    }
    for y in 0..4usize {
        for x in 0..4usize {
            pad_strided!(block, 4 * y + x, nz, 16, T::default());
        }
    }
    block
}

// ---------------------------------------------------------------------------
// Public API: contiguous
// ---------------------------------------------------------------------------

pub fn encode_block_3d_i32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i32; 64]) -> usize {
    encode_block_3d_i32(bs, block, MINBITS, INT32_MAXBITS, INT32_MAXPREC)
}

pub fn encode_block_3d_i64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i64; 64]) -> usize {
    encode_block_3d_i64(bs, block, MINBITS, INT64_MAXBITS, INT64_MAXPREC)
}

pub fn encode_block_3d_f32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f32; 64]) -> usize {
    encode_block_3d_f32(
        bs,
        block,
        MINBITS,
        FLOAT_MAXBITS,
        FLOAT_MAXPREC,
        FLOAT_MINEXP,
    )
}

pub fn encode_block_3d_f64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f64; 64]) -> usize {
    encode_block_3d_f64(
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
    gather: gather_3d,
    gather_partial: gather_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: encode_block_strided_3d_f64,
    partial: encode_partial_block_strided_3d_f64,
    full_rate: encode_block_strided_3d_f64_rate,
    partial_rate: encode_partial_block_strided_3d_f64_rate,
    encode: encode_block_3d_f64,
    encode_default: encode_block_3d_f64_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_encode_wrappers! {
    ty: f32,
    gather: gather_3d,
    gather_partial: gather_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: encode_block_strided_3d_f32,
    partial: encode_partial_block_strided_3d_f32,
    full_rate: encode_block_strided_3d_f32_rate,
    partial_rate: encode_partial_block_strided_3d_f32_rate,
    encode: encode_block_3d_f32,
    encode_default: encode_block_3d_f32_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_encode_wrappers! {
    ty: i32,
    gather: gather_3d,
    gather_partial: gather_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: encode_block_strided_3d_i32,
    partial: encode_partial_block_strided_3d_i32,
    full_rate: encode_block_strided_3d_i32_rate,
    partial_rate: encode_partial_block_strided_3d_i32_rate,
    encode: encode_block_3d_i32,
    encode_default: encode_block_3d_i32_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}

strided_encode_wrappers! {
    ty: i64,
    gather: gather_3d,
    gather_partial: gather_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: encode_block_strided_3d_i64,
    partial: encode_partial_block_strided_3d_i64,
    full_rate: encode_block_strided_3d_i64_rate,
    partial_rate: encode_partial_block_strided_3d_i64_rate,
    encode: encode_block_3d_i64,
    encode_default: encode_block_3d_i64_default,
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}
