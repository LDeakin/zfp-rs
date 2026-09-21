//! 1-D block decode (operates on a 4-element block).
//!
//! Reference: `zfp/src/template/decode1.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::strided_decode_wrappers;
use crate::codec::decode::float::{
    DOUBLE_MINEXP, FLOAT_MINEXP, decode_block_1d_f32, decode_block_1d_f64,
};
use crate::codec::decode::integer::{decode_block_1d_i32, decode_block_1d_i64};
use crate::config::ZfpRounding;

const MINBITS: u32 = 0;
const INT32_MAXBITS: u32 = 32 * 4 + 1;
const INT64_MAXBITS: u32 = 64 * 4 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 4;
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 4;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;

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
pub fn decode_block_1d_i32_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [i32; 4]) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_i32(
        bs,
        MINBITS,
        INT32_MAXBITS,
        INT32_MAXPREC,
        ZfpRounding::Never,
    );
    *block = decoded;
    (bs.read_pos() - before) as usize
}

/// Decode a contiguous 1-D block of 4 `i64` values; return bits read.
pub fn decode_block_1d_i64_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [i64; 4]) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_i64(
        bs,
        MINBITS,
        INT64_MAXBITS,
        INT64_MAXPREC,
        ZfpRounding::Never,
    );
    *block = decoded;
    (bs.read_pos() - before) as usize
}

/// Decode a contiguous 1-D block of 4 `f32` values; return bits read.
pub fn decode_block_1d_f32_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [f32; 4]) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_f32(
        bs,
        MINBITS,
        FLOAT_MAXBITS,
        FLOAT_MAXPREC,
        FLOAT_MINEXP,
        ZfpRounding::Never,
    );
    *block = decoded;
    (bs.read_pos() - before) as usize
}

/// Decode a contiguous 1-D block of 4 `f64` values; return bits read.
pub fn decode_block_1d_f64_default(bs: &mut dyn ZfpBitStreamOps, block: &mut [f64; 4]) -> usize {
    let before = bs.read_pos();
    let decoded = decode_block_1d_f64(
        bs,
        MINBITS,
        DOUBLE_MAXBITS,
        DOUBLE_MAXPREC,
        DOUBLE_MINEXP,
        ZfpRounding::Never,
    );
    *block = decoded;
    (bs.read_pos() - before) as usize
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
    defaults: [MINBITS, DOUBLE_MAXBITS, DOUBLE_MAXPREC, DOUBLE_MINEXP, ZfpRounding::Never],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32, rounding: ZfpRounding],
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
    defaults: [MINBITS, FLOAT_MAXBITS, FLOAT_MAXPREC, FLOAT_MINEXP, ZfpRounding::Never],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, minexp: i32, rounding: ZfpRounding],
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
    defaults: [MINBITS, INT32_MAXBITS, INT32_MAXPREC, ZfpRounding::Never],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, rounding: ZfpRounding],
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
    defaults: [MINBITS, INT64_MAXBITS, INT64_MAXPREC, ZfpRounding::Never],
    rate_params: [minbits: u32, maxbits: u32, maxprec: u32, rounding: ZfpRounding],
}
