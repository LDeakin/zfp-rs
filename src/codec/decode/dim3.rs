//! 3-D block decode (operates on a 4×4×4 block).
//!
//! Reference: `zfp/src/template/decode3.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::{strided_decode_wrappers, write_row};
use crate::codec::decode::float::{decode_block_3d_f32, decode_block_3d_f64};
use crate::codec::decode::integer::{decode_block_3d_i32, decode_block_3d_i64};
#[cfg(feature = "internals")]
use crate::config::ZfpConfig;
#[cfg(feature = "internals")]
use crate::types::{ZfpDimensionality, ZfpScalarType};

/// # Safety
/// `data` must be valid for every offset the strides generate.
pub(crate) unsafe fn scatter_3d<T: Copy>(
    block: &[T; 64],
    data: *mut T,
    sx: isize,
    sy: isize,
    sz: isize,
) {
    for (z, plane) in (0isize..).zip(block.as_chunks::<16>().0) {
        for (y, row) in (0isize..).zip(plane.as_chunks::<4>().0) {
            // SAFETY: caller guarantees data spans 4^3 elements with strides sx, sy, sz
            unsafe { write_row(data.offset(z * sz + y * sy), sx, row) };
        }
    }
}

/// # Safety
/// `data` must be valid for every offset the strides generate.
pub(crate) unsafe fn scatter_partial_3d<T: Copy>(
    block: &[T; 64],
    data: *mut T,
    nx: usize,
    ny: usize,
    nz: usize,
    sx: isize,
    sy: isize,
    sz: isize,
) {
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                // SAFETY: caller guarantees data spans nx*ny*nz elements with strides
                unsafe {
                    *data.offset(
                        z.cast_signed() * sz + y.cast_signed() * sy + x.cast_signed() * sx,
                    ) = block[16 * z + 4 * y + x];
                }
            }
        }
    }
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
    default: ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D3),
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
    default: ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D3),
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
    default: ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D3),
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
    default: ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D3),
}
