//! Integer-specific encode path.
//!
//! Reference: `zfp/src/template/encodei.c`, `encode.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::encode::core::{
    encode_ints_u32, encode_ints_u64, fwd_order_i32, fwd_order_i64, fwd_round_i32, fwd_round_i64,
};
use crate::codec::transform::{
    fwd_xform_1d, fwd_xform_1d_i64, fwd_xform_2d, fwd_xform_2d_i64, fwd_xform_3d, fwd_xform_3d_i64,
    fwd_xform_4d, fwd_xform_4d_i64,
};
use crate::config::ZfpRounding;

// ---------------------------------------------------------------------------
// Transform trait: selects dimension-specific transform + permutation
// ---------------------------------------------------------------------------

/// Encapsulates the forward transform and permutation table for a given
/// dimension and integer bit-width.
pub(crate) trait Transform32<const N: usize> {
    fn transform(block: &mut [i32; N]);
    fn perm() -> &'static [u8; N];
}

pub(crate) trait Transform64<const N: usize> {
    fn transform(block: &mut [i64; N]);
    fn perm() -> &'static [u8; N];
}

// -- 1-D -------------------------------------------------------------------

struct Dim1i32;
impl Transform32<4> for Dim1i32 {
    fn transform(block: &mut [i32; 4]) {
        fwd_xform_1d(block);
    }
    fn perm() -> &'static [u8; 4] {
        &crate::codec::encode::core::PERM_1
    }
}

struct Dim1i64;
impl Transform64<4> for Dim1i64 {
    fn transform(block: &mut [i64; 4]) {
        fwd_xform_1d_i64(block);
    }
    fn perm() -> &'static [u8; 4] {
        &crate::codec::encode::core::PERM_1
    }
}

// -- 2-D -------------------------------------------------------------------

struct Dim2i32;
impl Transform32<16> for Dim2i32 {
    fn transform(block: &mut [i32; 16]) {
        fwd_xform_2d(block);
    }
    fn perm() -> &'static [u8; 16] {
        &crate::codec::encode::core::PERM_2
    }
}

struct Dim2i64;
impl Transform64<16> for Dim2i64 {
    fn transform(block: &mut [i64; 16]) {
        fwd_xform_2d_i64(block);
    }
    fn perm() -> &'static [u8; 16] {
        &crate::codec::encode::core::PERM_2
    }
}

// -- 3-D -------------------------------------------------------------------

struct Dim3i32;
impl Transform32<64> for Dim3i32 {
    fn transform(block: &mut [i32; 64]) {
        fwd_xform_3d(block);
    }
    fn perm() -> &'static [u8; 64] {
        &crate::codec::encode::core::PERM_3
    }
}

struct Dim3i64;
impl Transform64<64> for Dim3i64 {
    fn transform(block: &mut [i64; 64]) {
        fwd_xform_3d_i64(block);
    }
    fn perm() -> &'static [u8; 64] {
        &crate::codec::encode::core::PERM_3
    }
}

// -- 4-D -------------------------------------------------------------------

struct Dim4i32;
impl Transform32<256> for Dim4i32 {
    fn transform(block: &mut [i32; 256]) {
        fwd_xform_4d(block);
    }
    fn perm() -> &'static [u8; 256] {
        &crate::codec::encode::core::PERM_4
    }
}

struct Dim4i64;
impl Transform64<256> for Dim4i64 {
    fn transform(block: &mut [i64; 256]) {
        fwd_xform_4d_i64(block);
    }
    fn perm() -> &'static [u8; 256] {
        &crate::codec::encode::core::PERM_4
    }
}

// ---------------------------------------------------------------------------
// Shared integer encode helpers (generic over block size N)
// ---------------------------------------------------------------------------

/// Generic integer encode for 32-bit values.
///
/// Applies the forward transform (via `T`), reorders via the permutation
/// table, converts to negabinary, then encodes bit-planes.
fn encode_int_block_32<T: Transform32<N>, const N: usize>(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i32; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    let mut block = *iblock;
    T::transform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i32(&mut block, maxprec);
    }
    let mut ublock = [0u32; N];
    #[allow(clippy::cast_sign_loss)] // i32→u32 for negabinary encoding
    fwd_order_i32(&mut ublock, &block, T::perm());
    let bits = encode_ints_u32(bs, maxbits, maxprec, &ublock);
    let bits = if bits < minbits {
        bs.pad((minbits - bits) as usize);
        minbits
    } else {
        bits
    };
    bits as usize
}

/// Generic integer encode for 64-bit values.
fn encode_int_block_64<T: Transform64<N>, const N: usize>(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i64; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    let mut block = *iblock;
    T::transform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i64(&mut block, maxprec);
    }
    let mut ublock = [0u64; N];
    fwd_order_i64(&mut ublock, &block, T::perm());
    let bits = encode_ints_u64(bs, maxbits, maxprec, &ublock);
    let bits = if bits < minbits {
        bs.pad((minbits - bits) as usize);
        minbits
    } else {
        bits
    };
    bits as usize
}

// ---------------------------------------------------------------------------
// Public API: 8 contiguous block encoders
// ---------------------------------------------------------------------------

/// Encode a decorrelated 1-D block of 4 `i32` values; returns bits written.
///
/// Applies the forward transform, reorders via `PERM_1`, then encodes bit-planes.
pub fn encode_block_1d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i32; 4],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_32::<Dim1i32, 4>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 1-D block of 4 `i64` values; returns bits written.
pub fn encode_block_1d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i64; 4],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_64::<Dim1i64, 4>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 2-D block of 16 `i32` values; returns bits written.
pub fn encode_block_2d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i32; 16],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_32::<Dim2i32, 16>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 2-D block of 16 `i64` values; returns bits written.
pub fn encode_block_2d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i64; 16],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_64::<Dim2i64, 16>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 3-D block of 64 `i32` values; returns bits written.
pub fn encode_block_3d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i32; 64],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_32::<Dim3i32, 64>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 3-D block of 64 `i64` values; returns bits written.
pub fn encode_block_3d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i64; 64],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_64::<Dim3i64, 64>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 4-D block of 256 `i32` values; returns bits written.
pub fn encode_block_4d_i32(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i32; 256],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_32::<Dim4i32, 256>(bs, iblock, minbits, maxbits, maxprec, rounding)
}

/// Encode a decorrelated 4-D block of 256 `i64` values; returns bits written.
pub fn encode_block_4d_i64(
    bs: &mut dyn ZfpBitStreamMutOps,
    iblock: &[i64; 256],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize {
    encode_int_block_64::<Dim4i64, 256>(bs, iblock, minbits, maxbits, maxprec, rounding)
}
