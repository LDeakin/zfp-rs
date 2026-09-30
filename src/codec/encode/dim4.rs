//! 4-D block encode (operates on a 4×4×4×4 block).
//!
//! Reference: `zfp/src/template/encode4.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::pad_strided;
use crate::codec::encode::core::strided_encode_wrappers;
use crate::codec::encode::float::{encode_block_4d_f32, encode_block_4d_f64};
use crate::codec::encode::integer::{encode_block_4d_i32, encode_block_4d_i64};
#[cfg(feature = "internals")]
use crate::config::ZfpConfig;
#[cfg(feature = "internals")]
use crate::types::{ZfpDimensionality, ZfpScalarType};

// ---------------------------------------------------------------------------
// Generic strided gather helpers (type-parameterised)
// ---------------------------------------------------------------------------

/// Gather a 4×4×4×4 block from a strided 4-D array.
///
/// # Safety
/// The caller must ensure the array spans at least 4 elements in each
/// dimension with the given strides.
pub(crate) unsafe fn gather_4d<T: Copy + Default>(
    data: *const T,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [T; 256] {
    let mut block = [T::default(); 256];
    let mut q = 0usize;
    for w in 0isize..4 {
        for z in 0isize..4 {
            for y in 0isize..4 {
                for x in 0isize..4 {
                    // SAFETY: caller guarantees valid strides
                    block[q] = unsafe { *data.offset(w * sw + z * sz + y * sy + x * sx) };
                    q += 1;
                }
            }
        }
    }
    block
}

/// Gather a partial 4-D block (nx, ny, nz, nw ≤ 4) and pad to 4×4×4×4.
///
/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub(crate) unsafe fn gather_partial_4d<T: Copy + Default>(
    data: *const T,
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [T; 256] {
    let mut block = [T::default(); 256];
    for w in 0..nw {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    // SAFETY: caller guarantees valid strides
                    block[64 * w + 16 * z + 4 * y + x] = unsafe {
                        *data.offset(
                            w.cast_signed() * sw
                                + z.cast_signed() * sz
                                + y.cast_signed() * sy
                                + x.cast_signed() * sx,
                        )
                    };
                }
                pad_strided!(block, 64 * w + 16 * z + 4 * y, nx, 1, T::default());
            }
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 16 * z + x, ny, 4, T::default());
            }
        }
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 4 * y + x, nz, 16, T::default());
            }
        }
    }
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 16 * z + 4 * y + x, nw, 64, T::default());
            }
        }
    }
    block
}

// ---------------------------------------------------------------------------
// Strided block encode (generated)
// ---------------------------------------------------------------------------

strided_encode_wrappers! {
    ty: f64,
    gather: gather_4d,
    gather_partial: gather_partial_4d,
    strides: [sx, sy, sz, sw],
    lengths: [nx, ny, nz, nw],
    full: encode_block_strided_4d_f64,
    partial: encode_partial_block_strided_4d_f64,
    full_rate: encode_block_strided_4d_f64_rate,
    partial_rate: encode_partial_block_strided_4d_f64_rate,
    encode: encode_block_4d_f64,
    default: ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D4),
}

strided_encode_wrappers! {
    ty: f32,
    gather: gather_4d,
    gather_partial: gather_partial_4d,
    strides: [sx, sy, sz, sw],
    lengths: [nx, ny, nz, nw],
    full: encode_block_strided_4d_f32,
    partial: encode_partial_block_strided_4d_f32,
    full_rate: encode_block_strided_4d_f32_rate,
    partial_rate: encode_partial_block_strided_4d_f32_rate,
    encode: encode_block_4d_f32,
    default: ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D4),
}

strided_encode_wrappers! {
    ty: i32,
    gather: gather_4d,
    gather_partial: gather_partial_4d,
    strides: [sx, sy, sz, sw],
    lengths: [nx, ny, nz, nw],
    full: encode_block_strided_4d_i32,
    partial: encode_partial_block_strided_4d_i32,
    full_rate: encode_block_strided_4d_i32_rate,
    partial_rate: encode_partial_block_strided_4d_i32_rate,
    encode: encode_block_4d_i32,
    default: ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D4),
}

strided_encode_wrappers! {
    ty: i64,
    gather: gather_4d,
    gather_partial: gather_partial_4d,
    strides: [sx, sy, sz, sw],
    lengths: [nx, ny, nz, nw],
    full: encode_block_strided_4d_i64,
    partial: encode_partial_block_strided_4d_i64,
    full_rate: encode_block_strided_4d_i64_rate,
    partial_rate: encode_partial_block_strided_4d_i64_rate,
    encode: encode_block_4d_i64,
    default: ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D4),
}
