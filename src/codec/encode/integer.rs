//! Integer-specific encode path.
//!
//! Reference: `zfp/src/template/encodei.c`, `encode.c`

#![allow(
    clippy::inline_always,
    reason = "LLVM declines to inline the transform into the block encoder; the call costs a store-forwarding stall"
)]

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::bitplane::{PlaneBlock, encode_ints};
use crate::codec::encode::core::{fwd_order_i32, fwd_order_i64, fwd_round_i32, fwd_round_i64};
use crate::codec::transform::fwd_xform;
use crate::config::{ZfpConfig, ZfpRounding};

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

pub(crate) struct Dim1i32;
impl Transform32<4> for Dim1i32 {
    #[inline(always)]
    fn transform(block: &mut [i32; 4]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 4] {
        &crate::codec::encode::core::PERM_1
    }
}

pub(crate) struct Dim1i64;
impl Transform64<4> for Dim1i64 {
    #[inline(always)]
    fn transform(block: &mut [i64; 4]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 4] {
        &crate::codec::encode::core::PERM_1
    }
}

// -- 2-D -------------------------------------------------------------------

pub(crate) struct Dim2i32;
impl Transform32<16> for Dim2i32 {
    #[inline(always)]
    fn transform(block: &mut [i32; 16]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 16] {
        &crate::codec::encode::core::PERM_2
    }
}

pub(crate) struct Dim2i64;
impl Transform64<16> for Dim2i64 {
    #[inline(always)]
    fn transform(block: &mut [i64; 16]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 16] {
        &crate::codec::encode::core::PERM_2
    }
}

// -- 3-D -------------------------------------------------------------------

pub(crate) struct Dim3i32;
impl Transform32<64> for Dim3i32 {
    #[inline(always)]
    fn transform(block: &mut [i32; 64]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 64] {
        &crate::codec::encode::core::PERM_3
    }
}

pub(crate) struct Dim3i64;
impl Transform64<64> for Dim3i64 {
    #[inline(always)]
    fn transform(block: &mut [i64; 64]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 64] {
        &crate::codec::encode::core::PERM_3
    }
}

// -- 4-D -------------------------------------------------------------------

pub(crate) struct Dim4i32;
impl Transform32<256> for Dim4i32 {
    #[inline(always)]
    fn transform(block: &mut [i32; 256]) {
        fwd_xform(block);
    }
    fn perm() -> &'static [u8; 256] {
        &crate::codec::encode::core::PERM_4
    }
}

pub(crate) struct Dim4i64;
impl Transform64<256> for Dim4i64 {
    #[inline(always)]
    fn transform(block: &mut [i64; 256]) {
        fwd_xform(block);
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
pub(crate) fn encode_int_block_32<T: Transform32<N>, const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize
where
    [u32; N]: PlaneBlock,
{
    let mut block = *iblock;
    T::transform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i32(&mut block, maxprec);
    }
    let mut ublock = [0u32; N];
    fwd_order_i32(&mut ublock, &block, T::perm());
    let bits = encode_ints::<_, true>(bs, maxbits, maxprec, &ublock);
    let bits = if bits < minbits {
        bs.pad(u64::from(minbits - bits));
        minbits
    } else {
        bits
    };
    bits as usize
}

/// Generic integer encode for 64-bit values.
pub(crate) fn encode_int_block_64<T: Transform64<N>, const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; N],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> usize
where
    [u64; N]: PlaneBlock,
{
    let mut block = *iblock;
    T::transform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i64(&mut block, maxprec);
    }
    let mut ublock = [0u64; N];
    fwd_order_i64(&mut ublock, &block, T::perm());
    let bits = encode_ints::<_, true>(bs, maxbits, maxprec, &ublock);
    let bits = if bits < minbits {
        bs.pad(u64::from(minbits - bits));
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
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; 4],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_32::<Dim1i32, 4>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 1-D block of 4 `i64` values; returns bits written.
pub fn encode_block_1d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; 4],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_64::<Dim1i64, 4>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 2-D block of 16 `i32` values; returns bits written.
pub fn encode_block_2d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; 16],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_32::<Dim2i32, 16>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 2-D block of 16 `i64` values; returns bits written.
pub fn encode_block_2d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; 16],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_64::<Dim2i64, 16>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 3-D block of 64 `i32` values; returns bits written.
pub fn encode_block_3d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; 64],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_32::<Dim3i32, 64>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 3-D block of 64 `i64` values; returns bits written.
pub fn encode_block_3d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; 64],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_64::<Dim3i64, 64>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 4-D block of 256 `i32` values; returns bits written.
pub fn encode_block_4d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; 256],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_32::<Dim4i32, 256>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}

/// Encode a decorrelated 4-D block of 256 `i64` values; returns bits written.
pub fn encode_block_4d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; 256],
    config: &ZfpConfig,
) -> usize {
    encode_int_block_64::<Dim4i64, 256>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}
