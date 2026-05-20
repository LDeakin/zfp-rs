//! 4-D block encode (operates on a 4×4×4×4 block).
//!
//! Reference: `zfp/src/template/encode4.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::pad_strided;
use crate::codec::encode::float::{encode_block_4d_f32, encode_block_4d_f64};
use crate::codec::encode::integer::{encode_block_4d_i32, encode_block_4d_i64};

const MINBITS: u32 = 0;
const INT32_MAXBITS: u32 = 32 * 256 + 1;
const INT64_MAXBITS: u32 = 64 * 256 + 1;
const INT32_MAXPREC: u32 = 32;
const INT64_MAXPREC: u32 = 64;
const FLOAT_MAXBITS: u32 = (8 + 1) + 32 * 256;
const DOUBLE_MAXBITS: u32 = (11 + 1) + 64 * 256;
const FLOAT_MAXPREC: u32 = 32;
const DOUBLE_MAXPREC: u32 = 64;
const FLOAT_MINEXP: i32 = -149;
const DOUBLE_MINEXP: i32 = -1074;

// ---------------------------------------------------------------------------
// Generic strided gather helpers (type-parameterised)
// ---------------------------------------------------------------------------

/// Gather a 4×4×4×4 block from a strided 4-D array.
///
/// # Safety
/// The caller must ensure the array spans at least 4 elements in each
/// dimension with the given strides.
fn gather_4d<T: Copy>(data: &[T], sx: isize, sy: isize, sz: isize, sw: isize) -> [T; 256] {
    // SAFETY: all elements are immediately written before being read.
    let mut block: [T; 256] = unsafe { std::mem::zeroed() };
    let p = data.as_ptr();
    let mut q = 0usize;
    for w in 0isize..4 {
        for z in 0isize..4 {
            for y in 0isize..4 {
                for x in 0isize..4 {
                    // SAFETY: caller guarantees valid strides
                    block[q] = unsafe { *p.offset(w * sw + z * sz + y * sy + x * sx) };
                    q += 1;
                }
            }
        }
    }
    block
}

/// Gather a partial 4-D block (nx, ny, nz, nw ≤ 4) and pad to 4×4×4×4.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn gather_partial_4d_f64(
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [f64; 256] {
    let mut block = [0f64; 256];
    let p = data.as_ptr();
    for w in 0..nw {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    block[64 * w + 16 * z + 4 * y + x] = unsafe {
                        *p.offset(
                            w.cast_signed() * sw
                                + z.cast_signed() * sz
                                + y.cast_signed() * sy
                                + x.cast_signed() * sx,
                        )
                    };
                }
                pad_strided!(block, 64 * w + 16 * z + 4 * y, nx, 1, 0.0f64);
            }
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 16 * z + x, ny, 4, 0.0f64);
            }
        }
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 4 * y + x, nz, 16, 0.0f64);
            }
        }
    }
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 16 * z + 4 * y + x, nw, 64, 0.0f64);
            }
        }
    }
    block
}

fn gather_partial_4d_f32(
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [f32; 256] {
    let mut block = [0f32; 256];
    let p = data.as_ptr();
    for w in 0..nw {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    block[64 * w + 16 * z + 4 * y + x] = unsafe {
                        *p.offset(
                            w.cast_signed() * sw
                                + z.cast_signed() * sz
                                + y.cast_signed() * sy
                                + x.cast_signed() * sx,
                        )
                    };
                }
                pad_strided!(block, 64 * w + 16 * z + 4 * y, nx, 1, 0.0f32);
            }
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 16 * z + x, ny, 4, 0.0f32);
            }
        }
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 4 * y + x, nz, 16, 0.0f32);
            }
        }
    }
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 16 * z + 4 * y + x, nw, 64, 0.0f32);
            }
        }
    }
    block
}

fn gather_partial_4d_i32(
    data: &[i32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [i32; 256] {
    let mut block = [0i32; 256];
    let p = data.as_ptr();
    for w in 0..nw {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    block[64 * w + 16 * z + 4 * y + x] = unsafe {
                        *p.offset(
                            w.cast_signed() * sw
                                + z.cast_signed() * sz
                                + y.cast_signed() * sy
                                + x.cast_signed() * sx,
                        )
                    };
                }
                pad_strided!(block, 64 * w + 16 * z + 4 * y, nx, 1, 0i32);
            }
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 16 * z + x, ny, 4, 0i32);
            }
        }
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 4 * y + x, nz, 16, 0i32);
            }
        }
    }
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 16 * z + 4 * y + x, nw, 64, 0i32);
            }
        }
    }
    block
}

fn gather_partial_4d_i64(
    data: &[i64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> [i64; 256] {
    let mut block = [0i64; 256];
    let p = data.as_ptr();
    for w in 0..nw {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    block[64 * w + 16 * z + 4 * y + x] = unsafe {
                        *p.offset(
                            w.cast_signed() * sw
                                + z.cast_signed() * sz
                                + y.cast_signed() * sy
                                + x.cast_signed() * sx,
                        )
                    };
                }
                pad_strided!(block, 64 * w + 16 * z + 4 * y, nx, 1, 0i64);
            }
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 16 * z + x, ny, 4, 0i64);
            }
        }
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 64 * w + 4 * y + x, nz, 16, 0i64);
            }
        }
    }
    for z in 0..4usize {
        for y in 0..4usize {
            for x in 0..4usize {
                pad_strided!(block, 16 * z + 4 * y + x, nw, 64, 0i64);
            }
        }
    }
    block
}

// ---------------------------------------------------------------------------
// Public API: contiguous
// ---------------------------------------------------------------------------

pub fn encode_block_4d_i32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i32; 256]) -> usize {
    encode_block_4d_i32(bs, block, MINBITS, INT32_MAXBITS, INT32_MAXPREC)
}

pub fn encode_block_4d_i64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[i64; 256]) -> usize {
    encode_block_4d_i64(bs, block, MINBITS, INT64_MAXBITS, INT64_MAXPREC)
}

pub fn encode_block_4d_f32_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f32; 256]) -> usize {
    encode_block_4d_f32(
        bs,
        block,
        MINBITS,
        FLOAT_MAXBITS,
        FLOAT_MAXPREC,
        FLOAT_MINEXP,
    )
}

pub fn encode_block_4d_f64_default(bs: &mut dyn ZfpBitStreamMutOps, block: &[f64; 256]) -> usize {
    encode_block_4d_f64(
        bs,
        block,
        MINBITS,
        DOUBLE_MAXBITS,
        DOUBLE_MAXPREC,
        DOUBLE_MINEXP,
    )
}

// ---------------------------------------------------------------------------
// Public API: strided
// ---------------------------------------------------------------------------

pub fn encode_block_strided_4d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_f64_default(bs, &block)
}

pub fn encode_block_strided_4d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_f32_default(bs, &block)
}

pub fn encode_block_strided_4d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_i32_default(bs, &block)
}

pub fn encode_block_strided_4d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_i64_default(bs, &block)
}

// ---------------------------------------------------------------------------
// Public API: partial strided
// ---------------------------------------------------------------------------

pub fn encode_partial_block_strided_4d_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_partial_4d_f64(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_f64_default(bs, &block)
}

pub fn encode_partial_block_strided_4d_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_partial_4d_f32(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_f32_default(bs, &block)
}

pub fn encode_partial_block_strided_4d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_partial_4d_i32(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_i32_default(bs, &block)
}

pub fn encode_partial_block_strided_4d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) -> usize {
    let block = gather_partial_4d_i64(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_i64_default(bs, &block)
}

// ---------------------------------------------------------------------------
// Rate-constrained strided block encode (for checksum tests)
// ---------------------------------------------------------------------------

pub fn encode_block_strided_4d_f64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_f64(bs, &block, minbits, maxbits, maxprec, minexp)
}
pub fn encode_block_strided_4d_f32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_f32(bs, &block, minbits, maxbits, maxprec, minexp)
}
pub fn encode_block_strided_4d_i32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_i32(bs, &block, minbits, maxbits, maxprec)
}
pub fn encode_block_strided_4d_i64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_4d(data, sx, sy, sz, sw);
    encode_block_4d_i64(bs, &block, minbits, maxbits, maxprec)
}
pub fn encode_partial_block_strided_4d_f64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_partial_4d_f64(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_f64(bs, &block, minbits, maxbits, maxprec, minexp)
}
pub fn encode_partial_block_strided_4d_f32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> usize {
    let block = gather_partial_4d_f32(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_f32(bs, &block, minbits, maxbits, maxprec, minexp)
}
pub fn encode_partial_block_strided_4d_i32_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i32],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_partial_4d_i32(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_i32(bs, &block, minbits, maxbits, maxprec)
}
pub fn encode_partial_block_strided_4d_i64_rate(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[i64],
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> usize {
    let block = gather_partial_4d_i64(data, nx, ny, nz, nw, sx, sy, sz, sw);
    encode_block_4d_i64(bs, &block, minbits, maxbits, maxprec)
}
