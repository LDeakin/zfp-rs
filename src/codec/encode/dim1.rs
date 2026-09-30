//! 1-D block encode (operates on a 4-element block).
//!
//! Reference: `zfp/src/template/encode1.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::strided_encode_wrappers;
use crate::codec::encode::float::{encode_block_1d_f32, encode_block_1d_f64};
use crate::codec::encode::integer::{encode_block_1d_i32, encode_block_1d_i64};
#[cfg(feature = "internals")]
use crate::config::ZfpConfig;
#[cfg(feature = "internals")]
use crate::types::{ZfpDimensionality, ZfpScalarType};

// ---------------------------------------------------------------------------
// Generic gather helpers (strides → contiguous block)
// ---------------------------------------------------------------------------

/// Gather 4 values with stride `sx` into a contiguous array.
///
/// # Safety
/// Caller must ensure `data` spans at least 4 elements with stride `sx`.
unsafe fn gather_1d<T: Copy + Default>(data: *const T, sx: isize) -> [T; 4] {
    let mut block = [T::default(); 4];
    for (dst, x) in block.iter_mut().zip(0isize..4) {
        // SAFETY: caller guarantees valid strides
        *dst = unsafe { *data.offset(x * sx) };
    }
    block
}

/// Gather a partial 1-D block (nx ≤ 4) and pad to 4 elements.
///
/// Mirrors C's `pad_block`: position 3 always comes from position 0; positions
/// `nx..3` come from position `nx-1`; positions `0..nx` are the real data.
///
/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub(crate) unsafe fn gather_partial_1d<T: Copy + Default>(
    data: *const T,
    nx: usize,
    sx: isize,
) -> [T; 4] {
    let mut block = [T::default(); 4];
    for (dst, x) in block[..nx].iter_mut().zip(0isize..) {
        *dst = unsafe { *data.offset(x * sx) };
    }
    // Pad remaining positions: mirrors C pad_block fall-through.
    match nx {
        0 => {
            // block[0] = block[0];
            block[1] = block[0];
            block[2] = block[1];
            block[3] = block[0];
        }
        1 => {
            block[1] = block[0];
            block[2] = block[1];
            block[3] = block[0];
        }
        2 => {
            block[2] = block[1];
            block[3] = block[0];
        }
        3 => {
            block[3] = block[0];
        }
        _ => {}
    }
    block
}

// ---------------------------------------------------------------------------
// Contiguous block encode
// ---------------------------------------------------------------------------

/// Encode a contiguous 1-D block of 4 `i32` values; return bits written.
#[cfg(feature = "internals")]
pub fn encode_block_1d_i32_default(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i32; 4],
) -> usize {
    encode_block_1d_i32(
        bs,
        block,
        &ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D1),
    )
}

/// Encode a contiguous 1-D block of 4 `i64` values; return bits written.
#[cfg(feature = "internals")]
pub fn encode_block_1d_i64_default(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i64; 4],
) -> usize {
    encode_block_1d_i64(
        bs,
        block,
        &ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D1),
    )
}

/// Encode a contiguous 1-D block of 4 `f32` values; return bits written.
#[cfg(feature = "internals")]
pub fn encode_block_1d_f32_default(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 4],
) -> usize {
    encode_block_1d_f32(
        bs,
        block,
        &ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D1),
    )
}

/// Encode a contiguous 1-D block of 4 `f64` values; return bits written.
#[cfg(feature = "internals")]
pub fn encode_block_1d_f64_default(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 4],
) -> usize {
    encode_block_1d_f64(
        bs,
        block,
        &ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D1),
    )
}

// ---------------------------------------------------------------------------
// Strided block encode (generated)
// ---------------------------------------------------------------------------

strided_encode_wrappers! {
    ty: f64,
    gather: gather_1d,
    gather_partial: gather_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: encode_block_strided_1d_f64,
    partial: encode_partial_block_strided_1d_f64,
    full_rate: encode_block_strided_1d_f64_rate,
    partial_rate: encode_partial_block_strided_1d_f64_rate,
    encode: encode_block_1d_f64,
    encode_default: encode_block_1d_f64_default,
}

strided_encode_wrappers! {
    ty: f32,
    gather: gather_1d,
    gather_partial: gather_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: encode_block_strided_1d_f32,
    partial: encode_partial_block_strided_1d_f32,
    full_rate: encode_block_strided_1d_f32_rate,
    partial_rate: encode_partial_block_strided_1d_f32_rate,
    encode: encode_block_1d_f32,
    encode_default: encode_block_1d_f32_default,
}

strided_encode_wrappers! {
    ty: i32,
    gather: gather_1d,
    gather_partial: gather_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: encode_block_strided_1d_i32,
    partial: encode_partial_block_strided_1d_i32,
    full_rate: encode_block_strided_1d_i32_rate,
    partial_rate: encode_partial_block_strided_1d_i32_rate,
    encode: encode_block_1d_i32,
    encode_default: encode_block_1d_i32_default,
}

strided_encode_wrappers! {
    ty: i64,
    gather: gather_1d,
    gather_partial: gather_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: encode_block_strided_1d_i64,
    partial: encode_partial_block_strided_1d_i64,
    full_rate: encode_block_strided_1d_i64_rate,
    partial_rate: encode_partial_block_strided_1d_i64_rate,
    encode: encode_block_1d_i64,
    encode_default: encode_block_1d_i64_default,
}
