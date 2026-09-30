//! Integer-specific encode path.
//!
//! Reference: `zfp/src/template/encodei.c`, `encode.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::bitplane::{PlaneBlock, encode_ints};
use crate::codec::encode::core::{
    PERM_1, PERM_2, PERM_3, PERM_4, fwd_order_i32, fwd_order_i64, fwd_round_i32, fwd_round_i64,
    pad_to,
};
use crate::codec::transform::fwd_xform;
use crate::config::{ZfpConfig, ZfpRounding};

/// The coefficient order of a 4^d block of `N` values, as a type, so that each
/// encoder instance reorders by a constant table.
pub(crate) trait Perm<const N: usize> {
    const PERM: &'static [u8; N];
}

/// 1-D blocks.
pub(crate) struct Dim1;
/// 2-D blocks.
pub(crate) struct Dim2;
/// 3-D blocks.
pub(crate) struct Dim3;
/// 4-D blocks.
pub(crate) struct Dim4;

impl Perm<4> for Dim1 {
    const PERM: &'static [u8; 4] = &PERM_1;
}
impl Perm<16> for Dim2 {
    const PERM: &'static [u8; 16] = &PERM_2;
}
impl Perm<64> for Dim3 {
    const PERM: &'static [u8; 64] = &PERM_3;
}
impl Perm<256> for Dim4 {
    const PERM: &'static [u8; 256] = &PERM_4;
}

// ---------------------------------------------------------------------------
// Shared integer encode helpers (generic over block size N)
// ---------------------------------------------------------------------------

/// Generic integer encode for 32-bit values.
///
/// Applies the forward transform, reorders by `P`'s permutation table,
/// converts to negabinary, then encodes bit-planes.
pub(crate) fn encode_int_block_32<P: Perm<N>, const N: usize>(
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
    fwd_xform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i32(&mut block, maxprec);
    }
    let mut ublock = [0u32; N];
    fwd_order_i32(&mut ublock, &block, P::PERM);
    let bits = encode_ints::<_, true>(bs, maxbits, maxprec, &ublock);
    pad_to(bs, bits, minbits)
}

/// Generic integer encode for 64-bit values.
pub(crate) fn encode_int_block_64<P: Perm<N>, const N: usize>(
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
    fwd_xform(&mut block);
    if matches!(rounding, ZfpRounding::First { .. }) {
        fwd_round_i64(&mut block, maxprec);
    }
    let mut ublock = [0u64; N];
    fwd_order_i64(&mut ublock, &block, P::PERM);
    let bits = encode_ints::<_, true>(bs, maxbits, maxprec, &ublock);
    pad_to(bs, bits, minbits)
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
    encode_int_block_32::<Dim1, _>(
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
    encode_int_block_64::<Dim1, _>(
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
    encode_int_block_32::<Dim2, _>(
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
    encode_int_block_64::<Dim2, _>(
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
    encode_int_block_32::<Dim3, _>(
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
    encode_int_block_64::<Dim3, _>(
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
    encode_int_block_32::<Dim4, _>(
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
    encode_int_block_64::<Dim4, _>(
        bs,
        iblock,
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.rounding(),
    )
}
