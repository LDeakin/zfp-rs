//! Reversible (lossless) encode path.
//!
//! Reference: `zfp/src/template/revencode{1-4}.c`,
//!            `zfp/src/template/revencodef.c`

use crate::bitstream::ZfpBitStreamMutOps;
use crate::codec::bitplane::{PlaneBlock, encode_ints};
use crate::codec::encode::core::{
    Budget, PERM_1, PERM_2, PERM_3, PERM_4, exponent_block_f32, exponent_block_f64, fwd_cast_f32,
    fwd_cast_f64, fwd_order_i32, fwd_order_i64, with_maxbits,
};
use crate::codec::transform::rev_fwd_xform;
use crate::config::ZfpConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Bits used to encode `prec - 1` for f32/i32 blocks (5 bits → max prec 31).
const PBITS_32: u32 = 5;
/// Bits used to encode `prec - 1` for f64/i64 blocks (6 bits → max prec 63).
const PBITS_64: u32 = 6;

const EBITS_F32: u32 = 8;
const EBITS_F64: u32 = 11;
const EBIAS_F32: i32 = 127;
const EBIAS_F64: i32 = 1023;

// Two's-complement sign-magnitude mask (all bits set except sign for the
// purposes of the reversibility check in rev_fwd_cast).
const TCMASK_F32: u32 = 0x7fff_ffff;
const TCMASK_F64: u64 = 0x7fff_ffff_ffff_ffff;

// ---------------------------------------------------------------------------
// rev_precision: count significant bits across a u32/u64 block
// ---------------------------------------------------------------------------

/// Compute the number of bit planes from the MSB needed to represent all values in `data`.
///
/// Equivalent to `INTPREC - trailing_zeros(OR of all values)`.
/// Matches the C `rev_precision` algorithm: for value `3 = 0b11`, returns 32 (all 32 planes
/// must be encoded because bit 0 is set); for `0x80000000`, returns 1 (only the top plane).
#[inline]
fn rev_precision_u32(data: &[u32]) -> u32 {
    let or: u32 = data.iter().fold(0u32, |acc, &v| acc | v);
    if or == 0 { 0 } else { 32 - or.trailing_zeros() }
}

/// Same as `rev_precision_u32` but for 64-bit values.
#[inline]
fn rev_precision_u64(data: &[u64]) -> u32 {
    let or: u64 = data.iter().fold(0u64, |acc, &v| acc | v);
    if or == 0 { 0 } else { 64 - or.trailing_zeros() }
}

// ---------------------------------------------------------------------------
// rev_inv_cast: reconstruct floats from integer block (for reversibility test)
// ---------------------------------------------------------------------------

/// Inverse block-floating-point: recover f32 values from i32 block at exponent `emax`.
/// Returns true only if every reconstructed value equals the original float bits.
#[expect(clippy::cast_precision_loss)]
fn rev_inv_cast_f32(iblock: &[i32], fblock: &[f32], emax: i32) -> bool {
    // s = 2^(emax - 30)
    let s = libm::ldexpf(1.0f32, emax - 30);
    for (&i, &f) in iblock.iter().zip(fblock.iter()) {
        let reconstructed = s * (i as f32);
        if reconstructed.to_bits() != f.to_bits() {
            return false;
        }
    }
    true
}

/// Same as `rev_inv_cast_f32` but for f64/i64.
#[expect(clippy::cast_precision_loss)]
fn rev_inv_cast_f64(iblock: &[i64], fblock: &[f64], emax: i32) -> bool {
    let s = libm::ldexp(1.0f64, emax - 62);
    for (&i, &f) in iblock.iter().zip(fblock.iter()) {
        let reconstructed = s * (i as f64);
        if reconstructed.to_bits() != f.to_bits() {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// rev_encode_int_block_i32 / i64
//
// Shared integer reversible-encode helper.
// Applies fwd_order then writes prec header + encodes integers.
// Returns bits written (excluding any bits written before this call). Callers
// pad the whole block to `min_bits`, which keeps it out of the plane loop.
// ---------------------------------------------------------------------------

fn rev_encode_int_block_u32<const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i32; N],
    maxbits: u32,
    maxprec: u32,
    perm: &[u8; N],
) -> u32
where
    [u32; N]: PlaneBlock,
{
    let mut ublock = [0u32; N];
    fwd_order_i32(&mut ublock, iblock, perm);

    // As C: at most `maxprec` bit planes, and always at least one.
    let prec = rev_precision_u32(&ublock).min(maxprec).max(1);

    bs.write_bits(u64::from(prec - 1), PBITS_32);

    // A budget that cannot bind, the usual case, is passed as a constant,
    // which lets the plane coder drop its budget checks.
    let maxbits = maxbits.saturating_sub(PBITS_32);
    PBITS_32
        + if with_maxbits(maxbits, prec, <[u32; N]>::SIZE) {
            encode_bounded(bs, maxbits, prec, &ublock)
        } else {
            encode_ints::<_, false>(bs, u32::MAX, prec, &ublock)
        }
}

fn rev_encode_int_block_u64<const N: usize>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    iblock: &[i64; N],
    maxbits: u32,
    maxprec: u32,
    perm: &[u8; N],
) -> u32
where
    [u64; N]: PlaneBlock,
{
    let mut ublock = [0u64; N];
    fwd_order_i64(&mut ublock, iblock, perm);

    let prec = rev_precision_u64(&ublock).min(maxprec).max(1);

    bs.write_bits(u64::from(prec - 1), PBITS_64);

    let maxbits = maxbits.saturating_sub(PBITS_64);
    PBITS_64
        + if with_maxbits(maxbits, prec, <[u64; N]>::SIZE) {
            encode_bounded(bs, maxbits, prec, &ublock)
        } else {
            encode_ints::<_, false>(bs, u32::MAX, prec, &ublock)
        }
}

/// [`encode_ints`] under a budget that binds. Out of line, so that the
/// unbounded call stays inlined.
#[inline(never)]
fn encode_bounded<B: PlaneBlock>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    maxbits: u32,
    prec: u32,
    block: &B,
) -> u32 {
    encode_ints::<_, true>(bs, maxbits, prec, block)
}

/// Pad a block of `bits` bits with zeros to `minbits`; return its size.
fn pad_to(bs: &mut (impl ZfpBitStreamMutOps + ?Sized), bits: u32, minbits: u32) -> usize {
    if bits < minbits {
        bs.pad(u64::from(minbits - bits));
        minbits as usize
    } else {
        bits as usize
    }
}

// ---------------------------------------------------------------------------
// rev_encode_float_block / rev_encode_double_block
//
// Reversible float path:
//   1. Compute emax.
//   2. If emax == -EBIAS (all zero): write single 0 bit.
//   3. Try BFP cast (fwd_cast) + reversibility check (rev_inv_cast).
//   4. If reversible: write "01" (2 bits), write emax+EBIAS as EBITS bits,
//      call rev_encode_int_block on i-block.
//   5. If not reversible: reinterpret float bits as two's complement (sign flip
//      on negative values using TCMASK), write "11" (2 bits),
//      call rev_encode_int_block on reinterpreted i-block.
//   6. Pad the block to minbits. C does not pad an all-zero block, yet its
//      decoder skips to minbits, so C misdecodes its own streams whenever
//      minbits > 1; padding keeps the two in step.
//
// The closure `rev_encode_int` receives the mutable iblock and the remaining
// `maxbits`. It is responsible for:
//   - converting the slice to a fixed-size array (the size is determined by
//     the caller's block parameter, so this is guaranteed to succeed)
//   - applying the inverse transform
//   - encoding the integers
// ---------------------------------------------------------------------------

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // u32→i32 for reversible encoding
fn rev_encode_float_block<B: ZfpBitStreamMutOps + ?Sized, const N: usize>(
    bs: &mut B,
    fblock: &[f32; N],
    budget: Budget,
    rev_encode_int: impl FnOnce(&mut B, &mut [i32; N], u32) -> u32,
) -> usize {
    let emax = exponent_block_f32(fblock);
    let e = (emax + EBIAS_F32) as u32;

    // BFP cast: if emax == -EBIAS, all iblock[i] = 0
    let mut iblock = [0i32; N];
    if e != 0 {
        fwd_cast_f32(&mut iblock, fblock, emax);
    }

    let bits = if rev_inv_cast_f32(&iblock, fblock, emax) {
        // BFP path is reversible
        if e != 0 {
            // Non-zero block: write "01" header + EBITS exponent
            bs.write_bits(0b01, 2);
            bs.write_bits(u64::from(e), EBITS_F32);
            let header_bits = 2 + EBITS_F32;
            header_bits + rev_encode_int(bs, &mut iblock, budget.after(header_bits).max)
        } else {
            // Genuinely all-zero block: write single "0" bit
            bs.put_bit(0);
            1
        }
    } else {
        // BFP is not reversible; reinterpret float bits as two's complement integers
        let mut iblock_tc = [0i32; N];
        for i in 0..N {
            let bits = fblock[i].to_bits();
            iblock_tc[i] = if bits >> 31 != 0 {
                (bits ^ TCMASK_F32) as i32
            } else {
                bits as i32
            };
        }
        // Write "11" header
        bs.write_bits(0b11, 2);
        2 + rev_encode_int(bs, &mut iblock_tc, budget.after(2).max)
    };
    pad_to(bs, bits, budget.min)
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // u64→i64 for reversible encoding
fn rev_encode_double_block<B: ZfpBitStreamMutOps + ?Sized, const N: usize>(
    bs: &mut B,
    fblock: &[f64; N],
    budget: Budget,
    rev_encode_int: impl FnOnce(&mut B, &mut [i64; N], u32) -> u32,
) -> usize {
    let emax = exponent_block_f64(fblock);
    let e = (emax + EBIAS_F64) as u32;

    let mut iblock = [0i64; N];
    if e != 0 {
        fwd_cast_f64(&mut iblock, fblock, emax);
    }

    let bits = if rev_inv_cast_f64(&iblock, fblock, emax) {
        if e != 0 {
            bs.write_bits(0b01, 2);
            bs.write_bits(u64::from(e), EBITS_F64);
            let header_bits = 2 + EBITS_F64;
            header_bits + rev_encode_int(bs, &mut iblock, budget.after(header_bits).max)
        } else {
            bs.put_bit(0);
            1
        }
    } else {
        let mut iblock_tc = [0i64; N];
        for i in 0..N {
            let bits = fblock[i].to_bits();
            iblock_tc[i] = if bits >> 63 != 0 {
                (bits ^ TCMASK_F64) as i64
            } else {
                bits as i64
            };
        }
        bs.write_bits(0b11, 2);
        2 + rev_encode_int(bs, &mut iblock_tc, budget.after(2).max)
    };
    pad_to(bs, bits, budget.min)
}

// ---------------------------------------------------------------------------
// Public API: 8 reversible encode functions
// ---------------------------------------------------------------------------

/// Reversible encode of a 1-D block of `f32` values; return bits written.
pub fn encode_block_reversible_1d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 4],
    config: &ZfpConfig,
) -> usize {
    rev_encode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u32(bs, iblock, maxbits, config.max_prec(), &PERM_1)
    })
}

/// Reversible encode of a 1-D block of `f64` values; return bits written.
pub fn encode_block_reversible_1d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 4],
    config: &ZfpConfig,
) -> usize {
    rev_encode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u64(bs, iblock, maxbits, config.max_prec(), &PERM_1)
    })
}

/// Reversible encode of a 2-D block of `f32` values; return bits written.
pub fn encode_block_reversible_2d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 16],
    config: &ZfpConfig,
) -> usize {
    rev_encode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u32(bs, iblock, maxbits, config.max_prec(), &PERM_2)
    })
}

/// Reversible encode of a 2-D block of `f64` values; return bits written.
pub fn encode_block_reversible_2d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 16],
    config: &ZfpConfig,
) -> usize {
    rev_encode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u64(bs, iblock, maxbits, config.max_prec(), &PERM_2)
    })
}

/// Reversible encode of a 3-D block of `f32` values; return bits written.
pub fn encode_block_reversible_3d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 64],
    config: &ZfpConfig,
) -> usize {
    rev_encode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u32(bs, iblock, maxbits, config.max_prec(), &PERM_3)
    })
}

/// Reversible encode of a 3-D block of `f64` values; return bits written.
pub fn encode_block_reversible_3d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 64],
    config: &ZfpConfig,
) -> usize {
    rev_encode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u64(bs, iblock, maxbits, config.max_prec(), &PERM_3)
    })
}

/// Reversible encode of a 4-D block of `f32` values; return bits written.
pub fn encode_block_reversible_4d_f32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f32; 256],
    config: &ZfpConfig,
) -> usize {
    rev_encode_float_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u32(bs, iblock, maxbits, config.max_prec(), &PERM_4)
    })
}

/// Reversible encode of a 4-D block of `f64` values; return bits written.
pub fn encode_block_reversible_4d_f64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[f64; 256],
    config: &ZfpConfig,
) -> usize {
    rev_encode_double_block(bs, block, Budget::of(config), |bs, iblock, maxbits| {
        rev_fwd_xform(iblock);
        rev_encode_int_block_u64(bs, iblock, maxbits, config.max_prec(), &PERM_4)
    })
}

// ---------------------------------------------------------------------------
// Reversible encode for integers: direct (no BFP step)
// ---------------------------------------------------------------------------

/// Reversible encode of a 1-D block of `i32` values; return bits written.
pub fn encode_block_reversible_1d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i32; 4],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u32(bs, &b, budget.max, config.max_prec(), &PERM_1);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 1-D block of `i64` values; return bits written.
pub fn encode_block_reversible_1d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i64; 4],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u64(bs, &b, budget.max, config.max_prec(), &PERM_1);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 2-D block of `i32` values; return bits written.
pub fn encode_block_reversible_2d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i32; 16],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u32(bs, &b, budget.max, config.max_prec(), &PERM_2);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 2-D block of `i64` values; return bits written.
pub fn encode_block_reversible_2d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i64; 16],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u64(bs, &b, budget.max, config.max_prec(), &PERM_2);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 3-D block of `i32` values; return bits written.
pub fn encode_block_reversible_3d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i32; 64],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u32(bs, &b, budget.max, config.max_prec(), &PERM_3);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 3-D block of `i64` values; return bits written.
pub fn encode_block_reversible_3d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i64; 64],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u64(bs, &b, budget.max, config.max_prec(), &PERM_3);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 4-D block of `i32` values; return bits written.
pub fn encode_block_reversible_4d_i32(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i32; 256],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u32(bs, &b, budget.max, config.max_prec(), &PERM_4);
    pad_to(bs, bits, budget.min)
}

/// Reversible encode of a 4-D block of `i64` values; return bits written.
pub fn encode_block_reversible_4d_i64(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    block: &[i64; 256],
    config: &ZfpConfig,
) -> usize {
    let mut b = *block;
    crate::codec::transform::rev_fwd_xform(&mut b);
    let budget = Budget::of(config);
    let bits = rev_encode_int_block_u64(bs, &b, budget.max, config.max_prec(), &PERM_4);
    pad_to(bs, bits, budget.min)
}
