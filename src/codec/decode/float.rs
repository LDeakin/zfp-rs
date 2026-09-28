//! Floating-point-specific decode path.
//!
//! Reference: `zfp/src/template/decodef.c`, `codecf.c`

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::decode::core::{decode_double_block, decode_float_block};
use crate::config::ZfpConfig;
use crate::types::ZfpDimensionality;

// ---------------------------------------------------------------------------
// 1-D
// ---------------------------------------------------------------------------

pub fn decode_block_1d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f32; 4] {
    let (block, _) = decode_float_block::<4>(bs, config, ZfpDimensionality::D1);
    block
}

pub fn decode_block_1d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f64; 4] {
    let (block, _) = decode_double_block::<4>(bs, config, ZfpDimensionality::D1);
    block
}

// ---------------------------------------------------------------------------
// 2-D
// ---------------------------------------------------------------------------

pub fn decode_block_2d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f32; 16] {
    let (block, _) = decode_float_block::<16>(bs, config, ZfpDimensionality::D2);
    block
}

pub fn decode_block_2d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f64; 16] {
    let (block, _) = decode_double_block::<16>(bs, config, ZfpDimensionality::D2);
    block
}

// ---------------------------------------------------------------------------
// 3-D
// ---------------------------------------------------------------------------

pub fn decode_block_3d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f32; 64] {
    let (block, _) = decode_float_block::<64>(bs, config, ZfpDimensionality::D3);
    block
}

pub fn decode_block_3d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f64; 64] {
    let (block, _) = decode_double_block::<64>(bs, config, ZfpDimensionality::D3);
    block
}

// ---------------------------------------------------------------------------
// 4-D
// ---------------------------------------------------------------------------

pub fn decode_block_4d_f32(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f32; 256] {
    let (block, _) = decode_float_block::<256>(bs, config, ZfpDimensionality::D4);
    block
}

pub fn decode_block_4d_f64(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
) -> [f64; 256] {
    let (block, _) = decode_double_block::<256>(bs, config, ZfpDimensionality::D4);
    block
}
