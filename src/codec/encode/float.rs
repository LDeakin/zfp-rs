#![allow(clippy::cast_sign_loss)] // i32→u32 for exponent encoding
//! Floating-point-specific encode path (exponent extraction + significand coding).
//!
//! Reference: `zfp/src/template/encodef.c`, `codecf.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::bitplane::PlaneBlock;
use crate::codec::encode::core::{
    EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, exponent_block_f32, exponent_block_f64,
    fwd_cast_f32, fwd_cast_f64, precision_f,
};
use crate::codec::encode::integer::{
    Dim1i32, Dim1i64, Dim2i32, Dim2i64, Dim3i32, Dim3i64, Dim4i32, Dim4i64, Transform32,
    Transform64, encode_int_block_32, encode_int_block_64,
};
use crate::config::ZfpConfig;

// ---------------------------------------------------------------------------
// Shared encode helpers (generic over block size N)
// ---------------------------------------------------------------------------

/// Generic f32-block encode: exponent header + integer block encode.
fn encode_float_block<T: Transform32<N>, const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    fblock: &[f32; N],
    config: &ZfpConfig,
) -> usize
where
    [u32; N]: PlaneBlock,
{
    let minbits = config.min_bits();
    // Compute the number of dimensions from the block size N (4=1D, 16=2D, 64=3D, 256=4D).
    // SAFETY: N is always 4, 16, 64, or 256 (powers of 4), so trailing_zeros is even and ≥ 2.
    let dims = N.trailing_zeros() / 2;
    let emax = exponent_block_f32(fblock);
    let prec = precision_f(
        emax,
        config.max_prec(),
        config.min_exp(),
        dims,
        config.rounding().tight_error(),
    );
    let e = if prec != 0 {
        (emax + EBIAS_F32) as u32
    } else {
        0
    };

    if e != 0 {
        let header_bits = EBITS_F32 + 1;
        bs.write_bits(2 * u64::from(e) + 1, header_bits);
        let mut iblock = [0i32; N];
        fwd_cast_f32(&mut iblock, fblock, emax);
        header_bits as usize
            + encode_int_block_32::<T, N>(
                bs,
                &iblock,
                minbits.saturating_sub(header_bits),
                config.max_bits().saturating_sub(header_bits),
                prec,
                config.rounding(),
            )
    } else {
        bs.put_bit(0);
        let bits = 1u32;
        if bits < minbits {
            bs.pad(u64::from(minbits - bits));
            minbits as usize
        } else {
            bits as usize
        }
    }
}

/// Generic f64-block encode: exponent header + integer block encode.
fn encode_double_block<T: Transform64<N>, const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    fblock: &[f64; N],
    config: &ZfpConfig,
) -> usize
where
    [u64; N]: PlaneBlock,
{
    let minbits = config.min_bits();
    // Compute the number of dimensions from the block size N (4=1D, 16=2D, 64=3D, 256=4D).
    // SAFETY: N is always 4, 16, 64, or 256 (powers of 4), so trailing_zeros is even and ≥ 2.
    let dims = N.trailing_zeros() / 2;
    let emax = exponent_block_f64(fblock);
    let prec = precision_f(
        emax,
        config.max_prec(),
        config.min_exp(),
        dims,
        config.rounding().tight_error(),
    );
    let e = if prec != 0 {
        (emax + EBIAS_F64) as u32
    } else {
        0
    };

    if e != 0 {
        let header_bits = EBITS_F64 + 1;
        bs.write_bits(2 * u64::from(e) + 1, header_bits);
        let mut iblock = [0i64; N];
        fwd_cast_f64(&mut iblock, fblock, emax);
        header_bits as usize
            + encode_int_block_64::<T, N>(
                bs,
                &iblock,
                minbits.saturating_sub(header_bits),
                config.max_bits().saturating_sub(header_bits),
                prec,
                config.rounding(),
            )
    } else {
        bs.put_bit(0);
        let bits = 1u32;
        if bits < minbits {
            bs.pad(u64::from(minbits - bits));
            minbits as usize
        } else {
            bits as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: 8 contiguous block encoders
// ---------------------------------------------------------------------------

/// Encode a 1-D block of 4 `f32` values; returns bits written.
pub fn encode_block_1d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 4],
    config: &ZfpConfig,
) -> usize {
    encode_float_block::<Dim1i32, 4>(bs, block, config)
}

/// Encode a 1-D block of 4 `f64` values; returns bits written.
pub fn encode_block_1d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 4],
    config: &ZfpConfig,
) -> usize {
    encode_double_block::<Dim1i64, 4>(bs, block, config)
}

/// Encode a 2-D block of 16 `f32` values; returns bits written.
pub fn encode_block_2d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 16],
    config: &ZfpConfig,
) -> usize {
    encode_float_block::<Dim2i32, 16>(bs, block, config)
}

/// Encode a 2-D block of 16 `f64` values; returns bits written.
pub fn encode_block_2d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 16],
    config: &ZfpConfig,
) -> usize {
    encode_double_block::<Dim2i64, 16>(bs, block, config)
}

/// Encode a 3-D block of 64 `f32` values; returns bits written.
pub fn encode_block_3d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 64],
    config: &ZfpConfig,
) -> usize {
    encode_float_block::<Dim3i32, 64>(bs, block, config)
}

/// Encode a 3-D block of 64 `f64` values; returns bits written.
pub fn encode_block_3d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 64],
    config: &ZfpConfig,
) -> usize {
    encode_double_block::<Dim3i64, 64>(bs, block, config)
}

/// Encode a 4-D block of 256 `f32` values; returns bits written.
pub fn encode_block_4d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 256],
    config: &ZfpConfig,
) -> usize {
    encode_float_block::<Dim4i32, 256>(bs, block, config)
}

/// Encode a 4-D block of 256 `f64` values; returns bits written.
pub fn encode_block_4d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 256],
    config: &ZfpConfig,
) -> usize {
    encode_double_block::<Dim4i64, 256>(bs, block, config)
}
