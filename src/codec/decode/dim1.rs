//! 1-D block decode (operates on a 4-element block).
//!
//! Reference: `zfp/src/template/decode1.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::strided_decode_wrappers;
use crate::codec::decode::float::{decode_block_1d_f32, decode_block_1d_f64};
use crate::codec::decode::integer::{decode_block_1d_i32, decode_block_1d_i64};
#[cfg(feature = "internals")]
use crate::config::ZfpConfig;
#[cfg(feature = "internals")]
use crate::types::{ZfpDimensionality, ZfpScalarType};

// ---------------------------------------------------------------------------
// Scatter helpers
// ---------------------------------------------------------------------------

/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap)] // usize→isize for pointer offset
unsafe fn scatter_1d<T: Copy>(block: &[T; 4], data: *mut T, sx: isize) {
    for (x, &val) in block.iter().enumerate() {
        // SAFETY: caller guarantees data spans 4 elements with stride sx
        unsafe { *data.offset(x as isize * sx) = val };
    }
}

/// # Safety
/// `data` must be valid for every offset the strides generate.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for pointer offset
unsafe fn scatter_partial_1d<T: Copy>(block: &[T; 4], data: *mut T, nx: usize, sx: isize) {
    for (x, &val) in block[..nx].iter().enumerate() {
        unsafe { *data.offset(x as isize * sx) = val };
    }
}

// ---------------------------------------------------------------------------
// Contiguous block decode
// ---------------------------------------------------------------------------

/// Decode a contiguous 1-D block of 4 `i32` values; return bits read.
#[cfg(feature = "internals")]
pub fn decode_block_1d_i32_default(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i32; 4],
) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_i32(
        bs,
        &ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D1),
    );
    *block = decoded;
    bs.read_pos().wrapping_sub(before) as usize
}

/// Decode a contiguous 1-D block of 4 `i64` values; return bits read.
#[cfg(feature = "internals")]
pub fn decode_block_1d_i64_default(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [i64; 4],
) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_i64(
        bs,
        &ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D1),
    );
    *block = decoded;
    bs.read_pos().wrapping_sub(before) as usize
}

/// Decode a contiguous 1-D block of 4 `f32` values; return bits read.
#[cfg(feature = "internals")]
pub fn decode_block_1d_f32_default(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f32; 4],
) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_f32(
        bs,
        &ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D1),
    );
    *block = decoded;
    bs.read_pos().wrapping_sub(before) as usize
}

/// Decode a contiguous 1-D block of 4 `f64` values; return bits read.
#[cfg(feature = "internals")]
pub fn decode_block_1d_f64_default(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    block: &mut [f64; 4],
) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_f64(
        bs,
        &ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D1),
    );
    *block = decoded;
    bs.read_pos().wrapping_sub(before) as usize
}

// ---------------------------------------------------------------------------
// Strided block decode (generated)
// ---------------------------------------------------------------------------

strided_decode_wrappers! {
    ty: f64,
    scatter: scatter_1d,
    scatter_partial: scatter_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: decode_block_strided_1d_f64,
    partial: decode_partial_block_strided_1d_f64,
    full_rate: decode_block_strided_1d_f64_rate,
    partial_rate: decode_partial_block_strided_1d_f64_rate,
    decode: decode_block_1d_f64,
    default: ZfpConfig::block_default(ZfpScalarType::F64, ZfpDimensionality::D1),
}

strided_decode_wrappers! {
    ty: f32,
    scatter: scatter_1d,
    scatter_partial: scatter_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: decode_block_strided_1d_f32,
    partial: decode_partial_block_strided_1d_f32,
    full_rate: decode_block_strided_1d_f32_rate,
    partial_rate: decode_partial_block_strided_1d_f32_rate,
    decode: decode_block_1d_f32,
    default: ZfpConfig::block_default(ZfpScalarType::F32, ZfpDimensionality::D1),
}

strided_decode_wrappers! {
    ty: i32,
    scatter: scatter_1d,
    scatter_partial: scatter_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: decode_block_strided_1d_i32,
    partial: decode_partial_block_strided_1d_i32,
    full_rate: decode_block_strided_1d_i32_rate,
    partial_rate: decode_partial_block_strided_1d_i32_rate,
    decode: decode_block_1d_i32,
    default: ZfpConfig::block_default(ZfpScalarType::I32, ZfpDimensionality::D1),
}

strided_decode_wrappers! {
    ty: i64,
    scatter: scatter_1d,
    scatter_partial: scatter_partial_1d,
    strides: [sx],
    lengths: [nx],
    full: decode_block_strided_1d_i64,
    partial: decode_partial_block_strided_1d_i64,
    full_rate: decode_block_strided_1d_i64_rate,
    partial_rate: decode_partial_block_strided_1d_i64_rate,
    decode: decode_block_1d_i64,
    default: ZfpConfig::block_default(ZfpScalarType::I64, ZfpDimensionality::D1),
}
