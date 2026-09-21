//! 3-D block decode (operates on a 4×4×4 block).
//!
//! Reference: `zfp/src/template/decode3.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::strided_decode_wrappers;
use crate::codec::decode::float::{
    DOUBLE_MINEXP, FLOAT_MINEXP, decode_block_3d_f32, decode_block_3d_f64,
};
use crate::codec::decode::integer::{decode_block_3d_i32, decode_block_3d_i64};

const MINBITS: u32 = 0;
const INT32_MAXBITS: u32 = 32 * 64 + 1;
const INT64_MAXBITS: u32 = 64 * 64 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 64;
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 64;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;

/// # Safety
/// `data` must be valid for every offset the strides generate.
unsafe fn scatter_3d<T: Copy>(block: &[T; 64], data: &mut [T], sx: isize, sy: isize, sz: isize) {
    let p = data.as_mut_ptr();
    let mut q = 0usize;
    for z in 0isize..4 {
        for y in 0isize..4 {
            for x in 0isize..4 {
                // SAFETY: caller guarantees data spans 4^3 elements with strides sx, sy, sz
                unsafe { *p.offset(z * sz + y * sy + x * sx) = block[q] };
                q += 1;
            }
        }
    }
}

/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize→isize for pointer offset
unsafe fn scatter_partial_3d<T: Copy>(
    block: &[T; 64],
    data: &mut [T],
    nx: usize,
    ny: usize,
    nz: usize,
    sx: isize,
    sy: isize,
    sz: isize,
) {
    let p = data.as_mut_ptr();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                // SAFETY: caller guarantees data spans nx*ny*nz elements with strides
                unsafe {
                    *p.offset(z.cast_signed() * sz + y.cast_signed() * sy + x.cast_signed() * sx) =
                        block[16 * z + 4 * y + x];
                }
            }
        }
    }
}

// Contiguous
pub fn decode_block_3d_i32_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [i32; 64]) -> usize {
    let before = bs.read_pos();
    *block = decode_block_3d_i32(bs, MINBITS, INT32_MAXBITS, INT32_MAXPREC);
    (bs.read_pos() - before) as usize
}
pub fn decode_block_3d_i64_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [i64; 64]) -> usize {
    let before = bs.read_pos();
    *block = decode_block_3d_i64(bs, MINBITS, INT64_MAXBITS, INT64_MAXPREC);
    (bs.read_pos() - before) as usize
}
pub fn decode_block_3d_f32_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [f32; 64]) -> usize {
    let before = bs.read_pos();
    *block = decode_block_3d_f32(bs, MINBITS, FLOAT_MAXBITS, FLOAT_MAXPREC, FLOAT_MINEXP);
    (bs.read_pos() - before) as usize
}
pub fn decode_block_3d_f64_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [f64; 64]) -> usize {
    let before = bs.read_pos();
    *block = decode_block_3d_f64(bs, MINBITS, DOUBLE_MAXBITS, DOUBLE_MAXPREC, DOUBLE_MINEXP);
    (bs.read_pos() - before) as usize
}

// ---------------------------------------------------------------------------
// Strided block decode (generated)
// ---------------------------------------------------------------------------

strided_decode_wrappers! {
    ty: f64,
    scatter: scatter_3d,
    scatter_partial: scatter_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: decode_block_strided_3d_f64,
    partial: decode_partial_block_strided_3d_f64,
    full_rate: decode_block_strided_3d_f64_rate,
    partial_rate: decode_partial_block_strided_3d_f64_rate,
    decode: decode_block_3d_f64,
    defaults: [MINBITS, DOUBLE_MAXBITS, DOUBLE_MAXPREC, DOUBLE_MINEXP],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_decode_wrappers! {
    ty: f32,
    scatter: scatter_3d,
    scatter_partial: scatter_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: decode_block_strided_3d_f32,
    partial: decode_partial_block_strided_3d_f32,
    full_rate: decode_block_strided_3d_f32_rate,
    partial_rate: decode_partial_block_strided_3d_f32_rate,
    decode: decode_block_3d_f32,
    defaults: [MINBITS, FLOAT_MAXBITS, FLOAT_MAXPREC, FLOAT_MINEXP],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32],
}

strided_decode_wrappers! {
    ty: i32,
    scatter: scatter_3d,
    scatter_partial: scatter_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: decode_block_strided_3d_i32,
    partial: decode_partial_block_strided_3d_i32,
    full_rate: decode_block_strided_3d_i32_rate,
    partial_rate: decode_partial_block_strided_3d_i32_rate,
    decode: decode_block_3d_i32,
    defaults: [MINBITS, INT32_MAXBITS, INT32_MAXPREC],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}

strided_decode_wrappers! {
    ty: i64,
    scatter: scatter_3d,
    scatter_partial: scatter_partial_3d,
    strides: [sx, sy, sz],
    lengths: [nx, ny, nz],
    full: decode_block_strided_3d_i64,
    partial: decode_partial_block_strided_3d_i64,
    full_rate: decode_block_strided_3d_i64_rate,
    partial_rate: decode_partial_block_strided_3d_i64_rate,
    decode: decode_block_3d_i64,
    defaults: [MINBITS, INT64_MAXBITS, INT64_MAXPREC],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32],
}
