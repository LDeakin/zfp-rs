//! 1-D block encode (operates on a 4-element block).
//!
//! Reference: `zfp/src/template/encode1.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::float::{encode_block_1d_f32, encode_block_1d_f64};
use crate::codec::encode::integer::{encode_block_1d_i32, encode_block_1d_i64};

// Default stream params used when the caller doesn't pass ZFP stream config.
// These match "lossless integer" mode: no rate constraint, full precision.
const INT32_MAXBITS: u32 = 32 * 4 + 1; // > block size * intprec
const INT64_MAXBITS: u32 = 64 * 4 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 4; // exponent bits + mantissa bits
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 4;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;
const FLOAT_MINEXP: i32 = -149;
const DOUBLE_MINEXP: i32 = -1074;

// ---------------------------------------------------------------------------
// Generic gather helpers (strides → contiguous block)
// ---------------------------------------------------------------------------

/// Gather 4 values with stride `sx` into a contiguous array.
///
/// # Safety
/// Caller must ensure `data` spans at least 4 elements with stride `sx`.
fn gather_1d<T: Copy>(data: &[T], sx: isize) -> [T; 4] {
    // SAFETY: all elements are immediately written before being read.
    let mut block: [T; 4] = unsafe { std::mem::zeroed() };
    let p = data.as_ptr();
    for (dst, x) in block.iter_mut().zip(0isize..4) {
        // SAFETY: caller guarantees valid strides
        *dst = unsafe { *p.offset(x * sx) };
    }
    block
}

/// Gather a partial 1-D block (nx ≤ 4) and pad to 4 elements.
///
/// Mirrors C's `pad_block`: position 3 always comes from position 0; positions
/// `nx..3` come from position `nx-1`; positions `0..nx` are the real data.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn gather_partial_1d<T: Copy>(data: &[T], nx: usize, sx: isize) -> [T; 4] {
    // SAFETY: all elements are immediately written before being read.
    let mut block: [T; 4] = unsafe { std::mem::zeroed() };
    let p = data.as_ptr();
    for (dst, x) in block[..nx].iter_mut().zip(0isize..) {
        *dst = unsafe { *p.offset(x * sx) };
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
pub fn encode_block_1d_i32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i32; 4]) -> usize {
    encode_block_1d_i32(bs, block, 0, INT32_MAXBITS, INT32_MAXPREC)
}

/// Encode a contiguous 1-D block of 4 `i64` values; return bits written.
pub fn encode_block_1d_i64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i64; 4]) -> usize {
    encode_block_1d_i64(bs, block, 0, INT64_MAXBITS, INT64_MAXPREC)
}

/// Encode a contiguous 1-D block of 4 `f32` values; return bits written.
pub fn encode_block_1d_f32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f32; 4]) -> usize {
    encode_block_1d_f32(bs, block, 0, FLOAT_MAXBITS, FLOAT_MAXPREC, FLOAT_MINEXP)
}

/// Encode a contiguous 1-D block of 4 `f64` values; return bits written.
pub fn encode_block_1d_f64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f64; 4]) -> usize {
    encode_block_1d_f64(bs, block, 0, DOUBLE_MAXBITS, DOUBLE_MAXPREC, DOUBLE_MINEXP)
}

// ---------------------------------------------------------------------------
// Strided block encode (gather then encode)
// ---------------------------------------------------------------------------

/// Encode a strided 1-D block of 4 `f64` values; return bits written.
pub fn encode_block_strided_1d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    sx: isize,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_f64_default(bs, &block)
}

/// Encode a strided 1-D block of 4 `f32` values; return bits written.
pub fn encode_block_strided_1d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    sx: isize,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_f32_default(bs, &block)
}

/// Encode a strided 1-D block of 4 `i32` values; return bits written.
pub fn encode_block_strided_1d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    sx: isize,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_i32_default(bs, &block)
}

/// Encode a strided 1-D block of 4 `i64` values; return bits written.
pub fn encode_block_strided_1d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    sx: isize,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_i64_default(bs, &block)
}

// ---------------------------------------------------------------------------
// Partial strided block encode
// ---------------------------------------------------------------------------

/// Encode a partial strided 1-D block (nx ≤ 4); return bits written.
pub fn encode_partial_block_strided_1d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    nx: usize,
    sx: isize,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_f64_default(bs, &block)
}

/// Encode a partial strided 1-D block (nx ≤ 4); return bits written.
pub fn encode_partial_block_strided_1d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    nx: usize,
    sx: isize,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_f32_default(bs, &block)
}

/// Encode a partial strided 1-D block (nx ≤ 4); return bits written.
pub fn encode_partial_block_strided_1d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    nx: usize,
    sx: isize,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_i32_default(bs, &block)
}

/// Encode a partial strided 1-D block (nx ≤ 4); return bits written.
pub fn encode_partial_block_strided_1d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    nx: usize,
    sx: isize,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_i64_default(bs, &block)
}

// ---------------------------------------------------------------------------
// Rate-constrained strided block encode (for checksum tests)
// ---------------------------------------------------------------------------

pub fn encode_block_strided_1d_f64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_f64(bs, &block, minbits, maxbits, maxprec, minexp)
}

pub fn encode_block_strided_1d_f32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_f32(bs, &block, minbits, maxbits, maxprec, minexp)
}

pub fn encode_block_strided_1d_i32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_i32(bs, &block, minbits, maxbits, maxprec)
}

pub fn encode_block_strided_1d_i64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_1d(data, sx);
    encode_block_1d_i64(bs, &block, minbits, maxbits, maxprec)
}

pub fn encode_partial_block_strided_1d_f64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    nx: usize,
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_f64(bs, &block, minbits, maxbits, maxprec, minexp)
}

pub fn encode_partial_block_strided_1d_f32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    nx: usize,
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_f32(bs, &block, minbits, maxbits, maxprec, minexp)
}

pub fn encode_partial_block_strided_1d_i32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    nx: usize,
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_i32(bs, &block, minbits, maxbits, maxprec)
}

pub fn encode_partial_block_strided_1d_i64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    nx: usize,
    sx: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_partial_1d(data, nx, sx);
    encode_block_1d_i64(bs, &block, minbits, maxbits, maxprec)
}
