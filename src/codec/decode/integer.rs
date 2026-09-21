//! Integer-specific decode path.
//!
//! Reference: `zfp/src/template/decodei.c`, `decode.c`

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::{
    decode_block_1d_i32_core, decode_block_1d_i64_core, decode_block_2d_i32_core,
    decode_block_2d_i64_core, decode_block_3d_i32_core, decode_block_3d_i64_core,
    decode_block_4d_i32_core, decode_block_4d_i64_core,
};
use crate::config::ZfpRounding;

/// Decode a 1-D block of 4 `i32` values; return bits read.
pub fn decode_block_1d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 4] {
    decode_block_1d_i32_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 1-D block of 4 `i64` values; return bits read.
pub fn decode_block_1d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 4] {
    decode_block_1d_i64_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 2-D block of 16 `i32` values; return bits read.
pub fn decode_block_2d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 16] {
    decode_block_2d_i32_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 2-D block of 16 `i64` values; return bits read.
pub fn decode_block_2d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 16] {
    decode_block_2d_i64_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 3-D block of 64 `i32` values; return bits read.
pub fn decode_block_3d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 64] {
    decode_block_3d_i32_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 3-D block of 64 `i64` values; return bits read.
pub fn decode_block_3d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 64] {
    decode_block_3d_i64_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 4-D block of 256 `i32` values; return bits read.
pub fn decode_block_4d_i32(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 256] {
    decode_block_4d_i32_core(bs, minbits, maxbits, maxprec, rounding)
}

/// Decode a 4-D block of 256 `i64` values; return bits read.
pub fn decode_block_4d_i64(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 256] {
    decode_block_4d_i64_core(bs, minbits, maxbits, maxprec, rounding)
}
