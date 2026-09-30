//! Shared decode utilities: `uint2int`, `inv_order`, `inv_round`, block decode.
//!
//! Reference: `zfp/src/template/decode.c`, `codecf.c`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)] // i32→f32 and i64→f64 for reconstruction (intentional loss)

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::bitplane::decode_ints;
use crate::codec::encode::core::{
    EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, NBMASK_U32, NBMASK_U64, PERM_1, PERM_2, PERM_3,
    PERM_4, precision_f,
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

/// Define a per-dimension integer block decoder, as C's `decode_block_Int_DIMS`:
/// `decode_ints`, the skip to `minbits`, then the inverse permutation and
/// transform.
///
/// These are eight functions rather than one generic over the block size and
/// word width, because that changed what LLVM inlines around them. Decoding
/// took up to 15% more cycles, most for 1-D fields, with each function's
/// callers inlining a different amount of `decode_ints`. Generated from one
/// definition, each compiles as when written out by hand.
macro_rules! decode_int_block {
    (
        $(#[$attr:meta])*
        $name:ident, $int:ty, $uint:ty, $n:literal, $perm:ident, $inv_order:ident
    ) => {
        $(#[$attr])*
        pub(crate) fn $name(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            minbits: u32,
            maxbits: u32,
            maxprec: u32,
            rounding: ZfpRounding,
        ) -> [$int; $n] {
            let (ublock, bits, zero) =
                decode_ints::<[$uint; $n], true>(bs, maxbits, maxprec, rounding);
            if bits < minbits {
                bs.skip(u64::from(minbits - bits));
            }
            let mut iblock = [0; $n];
            // An all-zero block transforms to zeros.
            if !zero {
                $inv_order(&ublock, &mut iblock, &$perm);
                crate::codec::transform::inv_xform(&mut iblock);
            }
            iblock
        }
    };
}

decode_int_block! { #[inline] decode_block_1d_i32_core, i32, u32, 4, PERM_1, inv_order_i32 }
decode_int_block! { #[inline] decode_block_1d_i64_core, i64, u64, 4, PERM_1, inv_order_i64 }
decode_int_block! { #[inline] decode_block_2d_i32_core, i32, u32, 16, PERM_2, inv_order_i32 }
decode_int_block! { #[inline] decode_block_2d_i64_core, i64, u64, 16, PERM_2, inv_order_i64 }
decode_int_block! { #[inline] decode_block_3d_i32_core, i32, u32, 64, PERM_3, inv_order_i32 }
decode_int_block! { #[inline] decode_block_3d_i64_core, i64, u64, 64, PERM_3, inv_order_i64 }
decode_int_block! { decode_block_4d_i32_core, i32, u32, 256, PERM_4, inv_order_i32 }
decode_int_block! { decode_block_4d_i64_core, i64, u64, 256, PERM_4, inv_order_i64 }

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

/// Generate a float block decoder: read the exponent, then the integer block,
/// then `inv_cast`.
///
/// One definition serves `f32` and `f64`, and is a macro rather than a generic
/// function for the reason `decode_int_block!` is. It takes the type's
/// exponent width and bias, its dequantization, and its integer core for each
/// dimensionality.
macro_rules! decode_float_block {
    (
        $(#[$attr:meta])*
        $name:ident, $float:ty, $ebits:expr, $ebias:expr, $inv_cast:ident,
        [$core1:ident, $core2:ident, $core3:ident, $core4:ident $(,)?]
    ) => {
        $(#[$attr])*
        pub(crate) fn $name<const N: usize>(
            bs: &mut (impl ZfpBitStreamOps + ?Sized),
            config: &ZfpConfig,
            dims: ZfpDimensionality,
        ) -> [$float; N] {
            const EBITS: u32 = $ebits;
            const EBIAS: i32 = $ebias;
            let (minbits, maxbits, rounding) =
                (config.min_bits(), config.max_bits(), config.rounding());
            let mut fblock: [$float; N] = [0.0; N];
            if bs.get_bit() != 0 {
                // The block has nonzero values, and a header of a bit and the
                // exponent.
                let bits = 1 + EBITS;
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
                        let iblock = $core1(bs, remaining_min, remaining_max, prec, rounding);
                        $inv_cast(&iblock, &mut fblock, emax);
                    }
                    ZfpDimensionality::D2 => {
                        let iblock = $core2(bs, remaining_min, remaining_max, prec, rounding);
                        $inv_cast(&iblock, &mut fblock, emax);
                    }
                    ZfpDimensionality::D3 => {
                        let iblock = $core3(bs, remaining_min, remaining_max, prec, rounding);
                        $inv_cast(&iblock, &mut fblock, emax);
                    }
                    ZfpDimensionality::D4 => {
                        let iblock = $core4(bs, remaining_min, remaining_max, prec, rounding);
                        $inv_cast(&iblock, &mut fblock, emax);
                    }
                }
            } else if minbits > 1 {
                // An all-zero block is its one-bit header, padded to `minbits`.
                bs.skip(u64::from(minbits - 1));
            }
            fblock
        }
    };
}

decode_float_block! {
    /// Decode a float block: read exponent, then integer block, then `inv_cast`.
    decode_float_block, f32, EBITS_F32, EBIAS_F32, inv_cast_f32,
    [
        decode_block_1d_i32_core,
        decode_block_2d_i32_core,
        decode_block_3d_i32_core,
        decode_block_4d_i32_core,
    ]
}

decode_float_block! {
    /// Decode a double block: read exponent, then integer block, then `inv_cast`.
    decode_double_block, f64, EBITS_F64, EBIAS_F64, inv_cast_f64,
    [
        decode_block_1d_i64_core,
        decode_block_2d_i64_core,
        decode_block_3d_i64_core,
        decode_block_4d_i64_core,
    ]
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
            bs.read_pos().wrapping_sub(before) as usize
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
            bs.read_pos().wrapping_sub(before) as usize
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
            bs.read_pos().wrapping_sub(before) as usize
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
            bs.read_pos().wrapping_sub(before) as usize
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
