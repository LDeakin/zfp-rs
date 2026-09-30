//! 2-D block decode (operates on a 4×4 block).
//!
//! Reference: `zfp/src/template/decode2.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::{strided_decode_wrappers, write_row};
use crate::codec::decode::float::{decode_block_2d_f32, decode_block_2d_f64};
use crate::codec::decode::integer::{decode_block_2d_i32, decode_block_2d_i64};
#[cfg(feature = "internals")]
use crate::config::ZfpConfig;
#[cfg(feature = "internals")]
use crate::types::{ZfpDimensionality, ZfpScalarType};

/// # Safety
/// `data` must be valid for every offset the strides generate.
unsafe fn scatter_2d<T: Copy>(block: &[T; 16], data: *mut T, sx: isize, sy: isize) {
    for (y, row) in (0isize..).zip(block.as_chunks::<4>().0) {
        // SAFETY: caller guarantees valid strides
        unsafe { write_row(data.offset(y * sy), sx, row) };
    }
}

/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize→isize for pointer offset
unsafe fn scatter_partial_2d<T: Copy>(
    block: &[T; 16],
    data: *mut T,
    nx: usize,
    ny: usize,
    sx: isize,
    sy: isize,
) {
    for y in 0..ny {
        for x in 0..nx {
            unsafe { *data.offset(y.cast_signed() * sy + x.cast_signed() * sx) = block[4 * y + x] };
        }
    }
}

// ---------------------------------------------------------------------------
// Strided block decode (generated)
// ---------------------------------------------------------------------------

strided_decode_wrappers! {
    ty: f64,
    scatter: scatter_2d,
    scatter_partial: scatter_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: decode_block_strided_2d_f64,
    partial: decode_partial_block_strided_2d_f64,
    full_rate: decode_block_strided_2d_f64_rate,
    partial_rate: decode_partial_block_strided_2d_f64_rate,
    decode: decode_block_2d_f64,
    default: ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D2),
}

strided_decode_wrappers! {
    ty: f32,
    scatter: scatter_2d,
    scatter_partial: scatter_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: decode_block_strided_2d_f32,
    partial: decode_partial_block_strided_2d_f32,
    full_rate: decode_block_strided_2d_f32_rate,
    partial_rate: decode_partial_block_strided_2d_f32_rate,
    decode: decode_block_2d_f32,
    default: ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D2),
}

strided_decode_wrappers! {
    ty: i32,
    scatter: scatter_2d,
    scatter_partial: scatter_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: decode_block_strided_2d_i32,
    partial: decode_partial_block_strided_2d_i32,
    full_rate: decode_block_strided_2d_i32_rate,
    partial_rate: decode_partial_block_strided_2d_i32_rate,
    decode: decode_block_2d_i32,
    default: ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D2),
}

strided_decode_wrappers! {
    ty: i64,
    scatter: scatter_2d,
    scatter_partial: scatter_partial_2d,
    strides: [sx, sy],
    lengths: [nx, ny],
    full: decode_block_strided_2d_i64,
    partial: decode_partial_block_strided_2d_i64,
    full_rate: decode_block_strided_2d_i64_rate,
    partial_rate: decode_partial_block_strided_2d_i64_rate,
    decode: decode_block_2d_i64,
    default: ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D2),
}
