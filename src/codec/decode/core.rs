//! Shared decode utilities: `uint2int`, `inv_order`, `inv_round`, block decode.
//!
//! Reference: `zfp/src/template/decode.c`, `codecf.c`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)] // i32→f32 and i64→f64 for reconstruction (intentional loss)

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::bitplane::decode_ints;
use crate::codec::encode::core::{
    NBMASK_U32, NBMASK_U64, PERM_1, PERM_2, PERM_3, PERM_4, precision_f,
};
use crate::config::{ZfpConfig, ZfpRounding};
use crate::types::ZfpDimensionality;

// ---------------------------------------------------------------------------
// Negabinary (uint2int) conversion: inverse of int2uint
// ---------------------------------------------------------------------------

/// Map negabinary `u32` → two's-complement `i32`.
///
/// Formula: `(x ^ NBMASK).wrapping_sub(NBMASK) as i32`
#[inline]
pub(crate) fn uint2int_u32(x: u32) -> i32 {
    (x ^ NBMASK_U32).wrapping_sub(NBMASK_U32).cast_signed()
}

/// Map negabinary `u64` → two's-complement `i64`.
#[inline]
pub(crate) fn uint2int_u64(x: u64) -> i64 {
    (x ^ NBMASK_U64).wrapping_sub(NBMASK_U64).cast_signed()
}

// ---------------------------------------------------------------------------
// Inverse order: reorder + uint2int
// ---------------------------------------------------------------------------

/// Reorder `ublock` by `perm` and convert each element to two's-complement `i32`.
pub(crate) fn inv_order_i32(ublock: &[u32], iblock: &mut [i32], perm: &[u8]) {
    for (&u, &p) in ublock.iter().zip(perm.iter()) {
        iblock[p as usize] = uint2int_u32(u);
    }
}

/// Reorder `ublock` by `perm` and convert each element to two's-complement `i64`.
pub(crate) fn inv_order_i64(ublock: &[u64], iblock: &mut [i64], perm: &[u8]) {
    for (&u, &p) in ublock.iter().zip(perm.iter()) {
        iblock[p as usize] = uint2int_u64(u);
    }
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// Bias coefficients so truncation rounds to nearest (`ZFP_ROUND_LAST`).
///
/// Adds 1/6 ulp to unbias errors; the first `m` values carry one extra bit of
/// precision. `NBMASK` is unsigned upstream, so both shifts are logical.
/// Reference: `zfp/src/template/decode.c`.
macro_rules! inv_round {
    ($name:ident, $u:ty, $nbmask:expr) => {
        #[inline]
        pub(crate) fn $name(data: &mut [$u], m: u32, prec: u32) {
            if prec < <$u>::BITS - 1 {
                let m = (m as usize).min(data.len());
                let (head, tail) = data.split_at_mut(m);
                let hi = ($nbmask >> 2) >> prec;
                let lo = ($nbmask >> 1) >> prec;
                for v in head {
                    *v = v.wrapping_add(hi);
                }
                for v in tail {
                    *v = v.wrapping_add(lo);
                }
            }
        }
    };
}
inv_round!(inv_round_u32, u32, NBMASK_U32);
inv_round!(inv_round_u64, u64, NBMASK_U64);

// ---------------------------------------------------------------------------
// Block decode: integer (matches C `decode_block_Int_DIMS`)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn decode_block_1d_i32_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 4] {
    let (ublock, bits, zero) = decode_ints::<[u32; 4], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i32; 4];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i32(&ublock, &mut iblock, &PERM_1);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

#[inline]
pub(crate) fn decode_block_1d_i64_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 4] {
    let (ublock, bits, zero) = decode_ints::<[u64; 4], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i64; 4];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i64(&ublock, &mut iblock, &PERM_1);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

#[inline]
pub(crate) fn decode_block_2d_i32_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 16] {
    let (ublock, bits, zero) = decode_ints::<[u32; 16], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i32; 16];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i32(&ublock, &mut iblock, &PERM_2);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

#[inline]
pub(crate) fn decode_block_2d_i64_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 16] {
    let (ublock, bits, zero) = decode_ints::<[u64; 16], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i64; 16];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i64(&ublock, &mut iblock, &PERM_2);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

#[inline]
pub(crate) fn decode_block_3d_i32_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 64] {
    let (ublock, bits, zero) = decode_ints::<[u32; 64], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i32; 64];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i32(&ublock, &mut iblock, &PERM_3);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

#[inline]
pub(crate) fn decode_block_3d_i64_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 64] {
    let (ublock, bits, zero) = decode_ints::<[u64; 64], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i64; 64];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i64(&ublock, &mut iblock, &PERM_3);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

pub(crate) fn decode_block_4d_i32_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i32; 256] {
    let (ublock, bits, zero) = decode_ints::<[u32; 256], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i32; 256];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i32(&ublock, &mut iblock, &PERM_4);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

pub(crate) fn decode_block_4d_i64_core(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> [i64; 256] {
    let (ublock, bits, zero) = decode_ints::<[u64; 256], true>(bs, maxbits, maxprec, rounding);
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
    }
    let mut iblock = [0i64; 256];
    // An all-zero block transforms to zeros.
    if !zero {
        inv_order_i64(&ublock, &mut iblock, &PERM_4);
        crate::codec::transform::inv_xform(&mut iblock);
    }
    iblock
}

// ---------------------------------------------------------------------------
// Strided row access (used by the full-block scatters)
// ---------------------------------------------------------------------------

/// Write `row` to `data[0]`, `data[sx]`, `data[2 * sx]` and `data[3 * sx]`.
///
/// Unit stride, the usual case, is a single store.
///
/// # Safety
/// `data` must be valid for every offset the stride generates.
#[inline]
pub(crate) unsafe fn write_row<T: Copy>(data: *mut T, sx: isize, row: &[T; 4]) {
    if sx == 1 {
        // SAFETY: the row is four contiguous elements, and `[T; 4]` has the
        // alignment of `T`.
        unsafe { data.cast::<[T; 4]>().write(*row) }
    } else {
        for (x, &v) in (0isize..).zip(row) {
            // SAFETY: the caller's contract.
            unsafe { *data.offset(x * sx) = v };
        }
    }
}

// ---------------------------------------------------------------------------
// Float decode helpers
// ---------------------------------------------------------------------------

/// Inverse block-floating-point: dequantize i32 → f32.
pub(crate) fn inv_cast_f32(iblock: &[i32], fblock: &mut [f32], emax: i32) {
    let s = libm::ldexpf(1.0f32, emax - 30);
    for (f, &i) in fblock.iter_mut().zip(iblock.iter()) {
        *f = s * i as f32;
    }
}

/// Inverse block-floating-point: dequantize i64 → f64.
pub(crate) fn inv_cast_f64(iblock: &[i64], fblock: &mut [f64], emax: i32) {
    let s = libm::ldexp(1.0f64, emax - 62);
    for (f, &i) in fblock.iter_mut().zip(iblock.iter()) {
        *f = s * i as f64;
    }
}

/// Decode a float block: read exponent, then integer block, then `inv_cast`.
///
/// Returns (decoded values, bits read).
pub(crate) fn decode_float_block<const N: usize>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
    dims: ZfpDimensionality,
) -> ([f32; N], usize) {
    const EBITS: u32 = 8;
    const EBIAS: i32 = 127;
    let (minbits, maxbits, rounding) = (config.min_bits(), config.max_bits(), config.rounding());
    let mut fblock = [0f32; N];
    let mut bits: u32 = 1;
    if bs.get_bit() != 0 {
        // block has nonzero values
        bits += EBITS;
        let emax = bs.read_bits(EBITS) as i32 - EBIAS;
        let prec = precision_f(
            emax,
            config.max_prec(),
            config.min_exp(),
            u32::from(dims),
            rounding.tight_error(),
        );
        let remaining_min = minbits.saturating_sub(bits);
        let remaining_max = maxbits.saturating_sub(bits);
        let iblock_bits = match dims {
            ZfpDimensionality::D1 => {
                let iblock =
                    decode_block_1d_i32_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D2 => {
                let iblock =
                    decode_block_2d_i32_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D3 => {
                let iblock =
                    decode_block_3d_i32_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D4 => {
                let iblock =
                    decode_block_4d_i32_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
        };
        let _ = iblock_bits;
        bits = maxbits; // consumed up to maxbits
    } else if minbits > bits {
        bs.skip(u64::from(minbits - bits));
        bits = minbits;
    }
    (fblock, bits as usize)
}

/// Decode a double block: read exponent, then integer block, then `inv_cast`.
pub(crate) fn decode_double_block<const N: usize>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
    dims: ZfpDimensionality,
) -> ([f64; N], usize) {
    const EBITS: u32 = 11;
    const EBIAS: i32 = 1023;
    let (minbits, maxbits, rounding) = (config.min_bits(), config.max_bits(), config.rounding());
    let mut fblock = [0f64; N];
    let mut bits: u32 = 1;
    if bs.get_bit() != 0 {
        bits += EBITS;
        let emax = bs.read_bits(EBITS) as i32 - EBIAS;
        let prec = precision_f(
            emax,
            config.max_prec(),
            config.min_exp(),
            u32::from(dims),
            rounding.tight_error(),
        );
        let remaining_min = minbits.saturating_sub(bits);
        let remaining_max = maxbits.saturating_sub(bits);
        match dims {
            ZfpDimensionality::D1 => {
                let iblock =
                    decode_block_1d_i64_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D2 => {
                let iblock =
                    decode_block_2d_i64_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D3 => {
                let iblock =
                    decode_block_3d_i64_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D4 => {
                let iblock =
                    decode_block_4d_i64_core(bs, remaining_min, remaining_max, prec, rounding);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
        }
        bits = maxbits;
    } else if minbits > bits {
        bs.skip(u64::from(minbits - bits));
        bits = minbits;
    }
    (fblock, bits as usize)
}

// ---------------------------------------------------------------------------
// Strided wrapper generation
// ---------------------------------------------------------------------------

/// Generate the four strided decode entry points for one dimensionality and
/// scalar type.
///
/// Each decodes into a contiguous block, then scatters it through the caller's
/// strides. The `_rate` variants serve the whole-field driver; the other two
/// exist for the C ABI and the port tests.
macro_rules! strided_decode_wrappers {
    (
        ty: $ty:ty,
        scatter: $scatter:ident,
        scatter_partial: $scatter_partial:ident,
        strides: [$($s:ident),+],
        lengths: [$($n:ident),+],
        full: $full:ident,
        partial: $partial:ident,
        full_rate: $full_rate:ident,
        partial_rate: $partial_rate:ident,
        decode: $decode:ident,
        default: $default:expr $(,)?
    ) => {
        /// Decode a strided block; return bits read.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        #[cfg(feature = "internals")]
        pub unsafe fn $full(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            data: *mut $ty,
            $($s: isize,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, &$default);
            unsafe { $scatter(&block, data, $($s),+) };
            (bs.read_pos() - before) as usize
        }

        /// Decode a partial (boundary) strided block; return bits read.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        #[cfg(feature = "internals")]
        pub unsafe fn $partial(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            data: *mut $ty,
            $($n: usize,)+
            $($s: isize,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, &$default);
            unsafe { $scatter_partial(&block, data, $($n,)+ $($s),+) };
            (bs.read_pos() - before) as usize
        }

        /// Decode a strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        pub unsafe fn $full_rate(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            data: *mut $ty,
            $($s: isize,)+
            config: &$crate::config::ZfpConfig,
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, config);
            unsafe { $scatter(&block, data, $($s),+) };
            (bs.read_pos() - before) as usize
        }

        /// Decode a partial strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        pub unsafe fn $partial_rate(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            data: *mut $ty,
            $($n: usize,)+
            $($s: isize,)+
            config: &$crate::config::ZfpConfig,
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, config);
            unsafe { $scatter_partial(&block, data, $($n,)+ $($s),+) };
            (bs.read_pos() - before) as usize
        }
    };
}
pub(crate) use strided_decode_wrappers;

#[cfg(test)]
mod tests {
    use super::{inv_round_u32, inv_round_u64};

    #[test]
    fn inv_round_is_a_no_op_at_full_precision() {
        let mut d = [1u32, 2, 3];
        inv_round_u32(&mut d, 1, 31);
        assert_eq!(d, [1, 2, 3]);
        let mut d = [1u64, 2, 3];
        inv_round_u64(&mut d, 1, 63);
        assert_eq!(d, [1, 2, 3]);
    }

    #[test]
    fn inv_round_gives_the_first_m_values_the_smaller_bias() {
        // hi = (NBMASK >> 2) >> prec for the first m, lo = (NBMASK >> 1) >> prec after.
        let mut d = [0u32; 4];
        inv_round_u32(&mut d, 2, 4);
        let (hi, lo) = (0x2aaa_aaaau32 >> 4, 0x5555_5555u32 >> 4);
        assert_eq!(d, [hi, hi, lo, lo]);
    }

    #[test]
    fn inv_round_handles_m_at_both_extremes() {
        let lo = 0x5555_5555u32 >> 8;
        let hi = 0x2aaa_aaaau32 >> 8;
        let mut d = [0u32; 3];
        inv_round_u32(&mut d, 0, 8);
        assert_eq!(d, [lo; 3]);
        // m == size (set by a negative group test) biases everything as `hi`.
        let mut d = [0u32; 3];
        inv_round_u32(&mut d, 3, 8);
        assert_eq!(d, [hi; 3]);
        // m past the end is clamped rather than panicking.
        let mut d = [0u32; 3];
        inv_round_u32(&mut d, 99, 8);
        assert_eq!(d, [hi; 3]);
    }

    #[test]
    fn inv_round_wraps_instead_of_overflowing() {
        let mut d = [u32::MAX];
        inv_round_u32(&mut d, 0, 8);
        assert_eq!(d[0], u32::MAX.wrapping_add(0x5555_5555u32 >> 8));
    }
}
