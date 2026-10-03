//! Shared encode utilities: permutation tables, int2uint, rate predicate.
//!
//! Reference: `zfp/src/template/codec{1-4}.c`, `encode.c`, `codecf.c`

#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// Permutation tables (reorder block coefficients by frequency / L1+L2 norm)
// ---------------------------------------------------------------------------

/// 1-D permutation (trivial order).
pub(crate) const PERM_1: [u8; 4] = [0, 1, 2, 3];

/// 2-D permutation; `index(i,j) = i + 4*j`.
pub(crate) const PERM_2: [u8; 16] = [0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15];

/// 3-D permutation; `index(i,j,k) = i + 4*j + 16*k`.
pub(crate) const PERM_3: [u8; 64] = [
    0, 1, 4, 16, 20, 17, 5, 2, 8, 32, 21, 6, 18, 24, 9, 33, 36, 3, 12, 48, 22, 25, 37, 40, 34, 10,
    7, 19, 28, 13, 49, 52, 41, 38, 26, 23, 29, 53, 11, 35, 44, 14, 50, 56, 42, 27, 39, 45, 30, 54,
    57, 60, 51, 15, 43, 46, 58, 61, 55, 31, 62, 59, 47, 63,
];

/// 4-D permutation; `index(i,j,k,l) = i + 4*j + 16*k + 64*l`.
pub(crate) const PERM_4: [u8; 256] = [
    0, 1, 4, 16, 64, 5, 80, 17, 68, 65, 20, 2, 8, 32, 128, 84, 81, 69, 21, 6, 18, 66, 24, 72, 9,
    96, 33, 36, 129, 132, 144, 3, 12, 48, 192, 85, 82, 70, 22, 73, 25, 88, 37, 100, 97, 148, 145,
    133, 10, 160, 34, 136, 130, 40, 7, 19, 67, 28, 76, 13, 112, 49, 52, 193, 196, 208, 86, 89, 101,
    149, 161, 137, 41, 134, 38, 164, 26, 152, 146, 104, 98, 74, 83, 71, 23, 77, 29, 92, 53, 116,
    113, 212, 209, 197, 11, 35, 131, 44, 140, 14, 176, 50, 56, 194, 200, 224, 90, 165, 102, 153,
    150, 105, 168, 162, 138, 42, 87, 93, 117, 213, 27, 75, 99, 39, 135, 147, 108, 45, 141, 156, 30,
    78, 177, 180, 54, 114, 120, 57, 198, 210, 216, 201, 225, 228, 15, 240, 51, 204, 195, 60, 169,
    166, 154, 106, 91, 103, 151, 109, 157, 94, 181, 118, 121, 214, 217, 229, 163, 139, 43, 142, 46,
    172, 58, 184, 178, 232, 226, 202, 241, 205, 61, 199, 55, 244, 31, 220, 211, 124, 115, 79, 170,
    167, 155, 107, 158, 110, 173, 122, 185, 182, 233, 230, 218, 95, 245, 119, 221, 215, 125, 242,
    206, 62, 203, 59, 248, 47, 236, 227, 188, 179, 143, 171, 174, 186, 234, 246, 222, 126, 219,
    123, 249, 111, 237, 231, 189, 183, 159, 252, 243, 207, 63, 175, 250, 187, 238, 235, 190, 253,
    247, 223, 127, 254, 251, 239, 191, 255,
];

// ---------------------------------------------------------------------------
// Negabinary (int2uint) conversion
// ---------------------------------------------------------------------------

/// Negabinary mask (`0xaaaa...`), also the basis of the rounding biases.
pub(crate) const NBMASK_U32: u32 = 0xaaaa_aaaa;
pub(crate) const NBMASK_U64: u64 = 0xaaaa_aaaa_aaaa_aaaa;

/// Bias coefficients so truncation rounds to nearest (`ZFP_ROUND_FIRST`).
///
/// Adds or subtracts 1/6 ulp to unbias errors. `NBMASK` is unsigned upstream,
/// so both shifts are logical. Reference: `zfp/src/template/encode.c`.
macro_rules! fwd_round {
    ($name:ident, $i:ty, $u:ty, $nbmask:expr) => {
        #[inline]
        #[allow(clippy::cast_possible_wrap)] // bias < 2^(BITS - 2)
        pub(crate) fn $name(iblock: &mut [$i], maxprec: u32) {
            if maxprec < <$u>::BITS {
                let bias = (($nbmask >> 2) >> maxprec) as $i;
                if maxprec & 1 == 1 {
                    for v in iblock {
                        *v = v.wrapping_add(bias);
                    }
                } else {
                    for v in iblock {
                        *v = v.wrapping_sub(bias);
                    }
                }
            }
        }
    };
}
fwd_round!(fwd_round_i32, i32, u32, NBMASK_U32);
fwd_round!(fwd_round_i64, i64, u64, NBMASK_U64);

/// Map two's-complement `i32` → negabinary `u32`.
///
/// Formula: `((x as u32).wrapping_add(NBMASK)) ^ NBMASK`
/// where `NBMASK = 0xaaaaaaaau`.
#[inline]
#[allow(clippy::cast_sign_loss)] // i32→u32 for negabinary encoding
pub(crate) fn int2uint_i32(x: i32) -> u32 {
    (x as u32).wrapping_add(NBMASK_U32) ^ NBMASK_U32
}

/// Map two's-complement `i64` → negabinary `u64`.
#[inline]
#[allow(clippy::cast_sign_loss)] // i64→u64 for negabinary encoding
pub(crate) fn int2uint_i64(x: i64) -> u64 {
    (x as u64).wrapping_add(NBMASK_U64) ^ NBMASK_U64
}

// ---------------------------------------------------------------------------
// Forward order: reorder + int2uint
// ---------------------------------------------------------------------------

/// Reorder `iblock` by `perm` and convert each element to negabinary `u32`.
pub(crate) fn fwd_order_i32(ublock: &mut [u32], iblock: &[i32], perm: &[u8]) {
    for (u, &p) in ublock.iter_mut().zip(perm.iter()) {
        *u = int2uint_i32(iblock[p as usize]);
    }
}

/// Reorder `iblock` by `perm` and convert each element to negabinary `u64`.
pub(crate) fn fwd_order_i64(ublock: &mut [u64], iblock: &[i64], perm: &[u8]) {
    for (u, &p) in ublock.iter_mut().zip(perm.iter()) {
        *u = int2uint_i64(iblock[p as usize]);
    }
}

// ---------------------------------------------------------------------------
// Block bit budget
// ---------------------------------------------------------------------------

/// The bits a block may take: at least `min`, and at most `max` once its
/// headers leave room.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Budget {
    pub(crate) min: u32,
    pub(crate) max: u32,
}

impl Budget {
    #[inline]
    pub(crate) fn of(config: &crate::config::ZfpConfig) -> Self {
        Self {
            min: config.min_bits(),
            max: config.max_bits(),
        }
    }

    /// What is left after `bits` header bits.
    ///
    /// Saturates where C subtracts unsigned integers: a `max_bits` smaller
    /// than the headers wraps around in C, leaving the block unbounded.
    #[inline]
    #[must_use]
    pub(crate) fn after(self, bits: u32) -> Self {
        Self {
            min: self.min.saturating_sub(bits),
            max: self.max.saturating_sub(bits),
        }
    }
}

/// Pad a block of `bits` bits with zeros to `minbits`; return its size.
pub(crate) fn pad_to(
    bs: &mut (impl crate::bitstream::ZfpBitStreamMutOps + ?Sized),
    bits: u32,
    minbits: u32,
) -> usize {
    if bits < minbits {
        bs.pad(u64::from(minbits - bits));
        minbits as usize
    } else {
        bits as usize
    }
}

/// Skip a block of `bits` bits to `minbits`; return its size.
pub(crate) fn skip_to(
    bs: &mut (impl crate::bitstream::ZfpBitStreamOps + ?Sized),
    bits: u32,
    minbits: u32,
) -> usize {
    if bits < minbits {
        bs.skip(u64::from(minbits - bits));
        minbits as usize
    } else {
        bits as usize
    }
}

// ---------------------------------------------------------------------------
// Rate-constraint predicate
// ---------------------------------------------------------------------------

/// True when the maximum possible bit count for this block exceeds `maxbits`
/// (i.e. rate-constrained encoding is required).
///
/// `(maxprec + 1) * size - 1 > maxbits`
#[inline]
pub(crate) fn with_maxbits(maxbits: u32, maxprec: u32, size: u32) -> bool {
    (u64::from(maxprec) + 1) * u64::from(size) > u64::from(maxbits) + 1
}

// ---------------------------------------------------------------------------
// Strided pad helpers (used by multi-D partial gather)
// ---------------------------------------------------------------------------

/// Pad a 4-element run in a flat array, where elements are spaced `stride` apart.
///
/// Matches C `pad_block(p, n, s)`: fills positions `[n*s..3*s]` from existing values.
macro_rules! pad_strided {
    ($block:expr, $offset:expr, $n:expr, $stride:expr, $zero:expr) => {{
        let (b, off, s) = (&mut $block, $offset, $stride);
        match $n {
            0 => {
                b[off] = $zero;
                b[off + s] = b[off];
                b[off + 2 * s] = b[off + s];
                b[off + 3 * s] = b[off];
            }
            1 => {
                b[off + s] = b[off];
                b[off + 2 * s] = b[off + s];
                b[off + 3 * s] = b[off];
            }
            2 => {
                b[off + 2 * s] = b[off + s];
                b[off + 3 * s] = b[off];
            }
            3 => {
                b[off + 3 * s] = b[off];
            }
            _ => {}
        }
    }};
}
pub(crate) use pad_strided;

// ---------------------------------------------------------------------------
// Floating-point utilities
// ---------------------------------------------------------------------------

/// Maximum number of bit planes to encode for a floating-point block.
///
/// `MIN(maxprec, MAX(0, maxexp - minexp + 2*dims + 2))`, or `+ 1` under
/// `ZFP_WITH_TIGHT_ERROR`. `dims` is the number of spatial dimensions (1–4).
#[inline]
pub(crate) fn precision_f(
    maxexp: i32,
    maxprec: u32,
    minexp: i32,
    dims: u32,
    tight_error: bool,
) -> u32 {
    let slack = if tight_error { 1 } else { 2 };
    // Both exponents may span the full i32 range. Clamp before converting
    // back to u32 so neither subtraction nor conversion wraps.
    let raw = i64::from(maxexp) - i64::from(minexp) + 2 * i64::from(dims) + i64::from(slack);
    // The clamp bounds `raw` by `maxprec`, so the conversion cannot fail.
    u32::try_from(raw.clamp(0, i64::from(maxprec))).unwrap_or(maxprec)
}

/// Largest `|x|` in a block; NaNs are skipped, as C's `max < f` test never
/// takes them.
///
/// With the NaNs zeroed, `if a > b { a } else { b }` is exactly `maxps`, and a
/// tree reduction keeps the dependency chain short.
macro_rules! max_abs {
    ($data:expr, $t:ty) => {{
        let mut keys = [0.0; N];
        for (key, x) in keys.iter_mut().zip($data) {
            let abs = x.abs();
            *key = if abs <= <$t>::INFINITY { abs } else { 0.0 };
        }
        let mut n = keys.len();
        while n > 1 {
            n /= 2;
            for i in 0..n {
                let (a, b) = (keys[i], keys[i + n]);
                keys[i] = if a > b { a } else { b };
            }
        }
        keys[0]
    }};
}

/// Return the maximum floating-point exponent in an f32 block.
///
/// Uses `frexp` semantics: returns the exponent `e` such that `|x| = m * 2^e`
/// with `0.5 ≤ m < 1`, clamped to `1 - EBIAS` for subnormals. Returns
/// `-EBIAS = -127` when all values are zero.
#[allow(
    clippy::inline_always,
    reason = "a block stage, inlined into the float block encoders as the transforms are"
)]
#[inline(always)]
pub(crate) fn exponent_block_f32<const N: usize>(data: &[f32; N]) -> i32 {
    let max = max_abs!(data, f32).to_bits();
    match max >> 23 {
        _ if max == 0 => -EBIAS_F32,
        // frexpf leaves the exponent of infinity at zero.
        0xff => 0,
        // A normal number's `frexpf` exponent. A subnormal one's biased
        // exponent is zero, which gives the clamp.
        biased => biased.cast_signed() - (EBIAS_F32 - 1),
    }
}

/// Return the maximum floating-point exponent in an f64 block.
///
/// As [`exponent_block_f32`].
#[allow(
    clippy::inline_always,
    reason = "a block stage, inlined into the float block encoders as the transforms are"
)]
#[inline(always)]
pub(crate) fn exponent_block_f64<const N: usize>(data: &[f64; N]) -> i32 {
    let max = max_abs!(data, f64).to_bits();
    match max >> 52 {
        _ if max == 0 => -EBIAS_F64,
        0x7ff => 0,
        biased => biased as i32 - (EBIAS_F64 - 1),
    }
}

/// Width of an `f32` block's biased exponent.
pub(crate) const EBITS_F32: u32 = crate::types::ZfpScalarType::F32.exponent_bits();
/// Width of an `f64` block's biased exponent.
pub(crate) const EBITS_F64: u32 = crate::types::ZfpScalarType::F64.exponent_bits();
/// Bias of an `f32` block's exponent.
pub(crate) const EBIAS_F32: i32 = 127;
/// Bias of an `f64` block's exponent.
pub(crate) const EBIAS_F64: i32 = 1023;
/// Bits used to encode `prec - 1` for f32/i32 reversible blocks (5 bits → max prec 31).
pub(crate) const PBITS_32: u32 = 5;
/// Bits used to encode `prec - 1` for f64/i64 reversible blocks (6 bits → max prec 63).
pub(crate) const PBITS_64: u32 = 6;

/// The smallest `emax` whose scale `2^(30 - emax)` is finite as an `f32`.
pub(crate) const MIN_CAST_EMAX_F32: i32 = -97;
/// The smallest `emax` whose scale `2^(62 - emax)` is finite as an `f64`.
pub(crate) const MIN_CAST_EMAX_F64: i32 = -961;

/// Forward block-floating-point transform: quantize f32 → i32 relative to exponent `emax`.
///
/// C computes the scale `2^(30 - emax)` as an `f32`, which overflows to
/// infinity below [`MIN_CAST_EMAX_F32`], so every value of a block smaller
/// than `2^-98` casts to `i32::MIN` (zfp issue #119). Such blocks are scaled
/// in two steps instead, which is exact, so their integers are what C's
/// would be if its scale did not overflow.
pub(crate) fn fwd_cast_f32(iblock: &mut [i32], fblock: &[f32], emax: i32) {
    if emax < MIN_CAST_EMAX_F32 {
        fwd_cast_tiny_f32(iblock, fblock, emax);
        return;
    }
    let s = libm::ldexpf(1.0f32, 30 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        *i = truncate_f32(s * f);
    }
}

/// [`fwd_cast_f32`] for `emax < MIN_CAST_EMAX_F32`.
///
/// `f * 2^64` is normal, even for a subnormal `f`, and the second factor is
/// at most `2^93`, so neither product rounds or overflows.
#[cold]
#[inline(never)]
fn fwd_cast_tiny_f32(iblock: &mut [i32], fblock: &[f32], emax: i32) {
    let s = libm::ldexpf(1.0f32, 30 - 64 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        *i = truncate_f32(f * TWO_POW_64_F32 * s);
    }
}

/// Forward block-floating-point transform: quantize f64 → i64 relative to exponent `emax`.
///
/// As [`fwd_cast_f32`], with blocks smaller than `2^-962` scaled in two steps.
pub(crate) fn fwd_cast_f64(iblock: &mut [i64], fblock: &[f64], emax: i32) {
    if emax < MIN_CAST_EMAX_F64 {
        fwd_cast_tiny_f64(iblock, fblock, emax);
        return;
    }
    let s = libm::ldexp(1.0f64, 62 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        *i = truncate_f64(s * f);
    }
}

/// [`fwd_cast_f64`] for `emax < MIN_CAST_EMAX_F64`; as [`fwd_cast_tiny_f32`].
#[cold]
#[inline(never)]
fn fwd_cast_tiny_f64(iblock: &mut [i64], fblock: &[f64], emax: i32) {
    let s = libm::ldexp(1.0f64, 62 - 64 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        *i = truncate_f64(f * TWO_POW_64_F64 * s);
    }
}

const TWO_POW_64_F32: f32 = 18_446_744_073_709_551_616.0;
const TWO_POW_64_F64: f64 = 18_446_744_073_709_551_616.0;

/// `v` truncated to `i32`, or `i32::MIN` if out of range, as x86's `cvttss2si`.
///
/// Only blocks with NaNs or infinities, which the lossy modes do not support,
/// give values out of range. C's cast of them is undefined; this gives
/// x86-64's result.
#[inline]
fn truncate_f32(v: f32) -> i32 {
    if (-2_147_483_648.0_f32..2_147_483_648.0_f32).contains(&v) {
        // SAFETY: `v` is in range, so this truncates exactly as `as`
        // would, without the saturation that keeps `as` from vectorizing.
        unsafe { v.to_int_unchecked() }
    } else {
        i32::MIN
    }
}

/// `v` truncated to `i64`, or `i64::MIN` if out of range, as x86's `cvttsd2si`.
///
/// As [`truncate_f32`].
#[inline]
fn truncate_f64(v: f64) -> i64 {
    // The instruction itself, since SSE2 cannot convert `f64` lanes to `i64`
    // and the portable form costs two compares and a select per value.
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_cvttsd_si64, _mm_set_sd};
        // SAFETY: SSE2 is part of the x86-64 baseline.
        unsafe { _mm_cvttsd_si64(_mm_set_sd(v)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if (-9_223_372_036_854_775_808.0_f64..9_223_372_036_854_775_808.0_f64).contains(&v) {
            // SAFETY: as in `truncate_f32`.
            unsafe { v.to_int_unchecked() }
        } else {
            i64::MIN
        }
    }
}

// ---------------------------------------------------------------------------
// Strided wrapper generation
// ---------------------------------------------------------------------------

/// Generate the four strided encode entry points for one dimensionality and
/// scalar type.
///
/// Each gathers a block through the caller's strides, then defers to the
/// contiguous encoder. The `_rate` variants serve the whole-field driver; the
/// other two exist for the C ABI and the port tests.
macro_rules! strided_encode_wrappers {
    (
        ty: $ty:ty,
        gather: $gather:ident,
        gather_partial: $gather_partial:ident,
        strides: [$($s:ident),+],
        lengths: [$($n:ident),+],
        full: $full:ident,
        partial: $partial:ident,
        full_rate: $full_rate:ident,
        partial_rate: $partial_rate:ident,
        encode: $encode:ident,
        default: $default:expr $(,)?
    ) => {
        /// Encode a strided block; return bits written.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// `codec::block::strided`.
        #[cfg(feature = "internals")]
        pub unsafe fn $full(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            data: *const $ty,
            $($s: isize,)+
        ) -> usize {
            let block = unsafe { $gather(data, $($s),+) };
            $encode(bs, &block, &$default)
        }

        /// Encode a partial (boundary) strided block; return bits written.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// `codec::block::strided`.
        #[cfg(feature = "internals")]
        pub unsafe fn $partial(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            data: *const $ty,
            $($n: usize,)+
            $($s: isize,)+
        ) -> usize {
            let block = unsafe { $gather_partial(data, $($n,)+ $($s),+) };
            $encode(bs, &block, &$default)
        }

        /// Encode a strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// `codec::block::strided`.
        pub unsafe fn $full_rate(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            data: *const $ty,
            $($s: isize,)+
            config: &$crate::config::ZfpConfig,
        ) -> usize {
            let block = unsafe { $gather(data, $($s),+) };
            $encode(bs, &block, config)
        }

        /// Encode a partial strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// `codec::block::strided`.
        pub unsafe fn $partial_rate(
            bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
            data: *const $ty,
            $($n: usize,)+
            $($s: isize,)+
            config: &$crate::config::ZfpConfig,
        ) -> usize {
            let block = unsafe { $gather_partial(data, $($n,)+ $($s),+) };
            $encode(bs, &block, config)
        }
    };
}
pub(crate) use strided_encode_wrappers;

#[cfg(test)]
mod tests {
    use super::{
        MIN_CAST_EMAX_F32, MIN_CAST_EMAX_F64, exponent_block_f32, exponent_block_f64, fwd_cast_f32,
        fwd_cast_f64, fwd_round_i32, fwd_round_i64, precision_f, truncate_f64,
    };

    #[test]
    fn truncate_f64_gives_i64_min_out_of_range() {
        let two63 = 9_223_372_036_854_775_808.0_f64;
        for (v, expect) in [
            (0.0, 0),
            (-0.0, 0),
            (1.9, 1),
            (-1.9, -1),
            (-two63, i64::MIN),
            (two63, i64::MIN),
            (f64::MAX, i64::MIN),
            (f64::INFINITY, i64::MIN),
            (f64::NEG_INFINITY, i64::MIN),
            (f64::NAN, i64::MIN),
            (4_611_686_018_427_387_904.5, 4_611_686_018_427_387_904),
        ] {
            assert_eq!(truncate_f64(v), expect, "{v}");
        }
    }

    /// C's `exponent_block`: `frexp` of the largest `|x|`, NaNs skipped.
    fn frexp_exponent_f32(data: &[f32]) -> i32 {
        let max = data
            .iter()
            .fold(0.0f32, |max, x| if max < x.abs() { x.abs() } else { max });
        if max > 0.0 {
            libm::frexpf(max).1.max(-126)
        } else {
            -127
        }
    }

    fn frexp_exponent_f64(data: &[f64]) -> i32 {
        let max = data
            .iter()
            .fold(0.0f64, |max, x| if max < x.abs() { x.abs() } else { max });
        if max > 0.0 {
            libm::frexp(max).1.max(-1022)
        } else {
            -1023
        }
    }

    #[test]
    fn exponent_block_matches_frexp() {
        let specials32 = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE / 3.0,
            f32::from_bits(1),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fff_ffff),
            1.0,
            0.75,
            -3.5e-20,
        ];
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for &a in &specials32 {
            for &b in &specials32 {
                let block = [a, b, 0.5 * a, -b];
                assert_eq!(exponent_block_f32(&block), frexp_exponent_f32(&block));
                let block = [f64::from(a), f64::from(b), 1e-310, -f64::from(b)];
                assert_eq!(exponent_block_f64(&block), frexp_exponent_f64(&block));
            }
        }
        // As glibc's `frexp` gives, and musl's does not.
        assert_eq!(exponent_block_f32(&[f32::INFINITY, 1.0, 0.0, -2.0]), 0);
        assert_eq!(exponent_block_f64(&[f64::NEG_INFINITY, 1.0, 0.0, -2.0]), 0);
        for _ in 0..10_000 {
            let block: [f32; 4] = std::array::from_fn(|_| f32::from_bits(next() as u32));
            assert_eq!(exponent_block_f32(&block), frexp_exponent_f32(&block));
            let block: [f64; 4] = std::array::from_fn(|_| f64::from_bits(next()));
            assert_eq!(exponent_block_f64(&block), frexp_exponent_f64(&block));
        }
    }

    /// Biased exponents of a block, each at most `top`, or `None` for zeros.
    /// Exponent zero gives subnormals.
    fn block_below<const N: usize>(next: &mut impl FnMut() -> u64, top: u64) -> [Option<u64>; N] {
        std::array::from_fn(|_| {
            let r = next();
            (r >> 61 != 0).then(|| (r >> 16) % (top + 1))
        })
    }

    /// `fwd_cast` truncates `f * 2^(30 - emax)` (`2^(62 - emax)` for `f64`)
    /// computed exactly, including for blocks too small for C's scale.
    #[test]
    fn fwd_cast_scales_exactly() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut tiny = [0usize; 2];
        for top in 0..0xff {
            for _ in 0..200 {
                let exps = block_below::<4>(&mut next, top);
                let block = exps.map(|e| {
                    e.map_or(0.0, |e| {
                        f32::from_bits((next() as u32 & 0x807f_ffff) | ((e as u32) << 23))
                    })
                });
                let emax = exponent_block_f32(&block);
                if emax == -127 {
                    continue;
                }
                tiny[0] += usize::from(emax < MIN_CAST_EMAX_F32);
                let mut got = [0i32; 4];
                fwd_cast_f32(&mut got, &block, emax);
                let want = block.map(|f| (f64::from(f) * libm::ldexp(1.0, 30 - emax)) as i32);
                assert_eq!(got, want, "{block:?}");
            }
        }
        for top in 0..0x7ff {
            for _ in 0..20 {
                let exps = block_below::<4>(&mut next, top);
                let block = exps.map(|e| {
                    e.map_or(0.0, |e| {
                        f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | (e << 52))
                    })
                });
                let emax = exponent_block_f64(&block);
                if emax == -1023 {
                    continue;
                }
                tiny[1] += usize::from(emax < MIN_CAST_EMAX_F64);
                let mut got = [0i64; 4];
                fwd_cast_f64(&mut got, &block, emax);
                let want = block.map(|f| truncate_f64(libm::ldexp(f, 62 - emax)));
                assert_eq!(got, want, "{block:?}");
            }
        }
        assert!(tiny.iter().all(|&n| n > 1000), "{tiny:?}");
    }

    /// Below `MIN_CAST_EMAX_*`, C's scale overflows. The boundary is where it
    /// does.
    #[test]
    fn min_cast_emax_is_where_the_scale_overflows() {
        assert!(libm::ldexpf(1.0, 30 - MIN_CAST_EMAX_F32).is_finite());
        assert!(libm::ldexpf(1.0, 30 - (MIN_CAST_EMAX_F32 - 1)).is_infinite());
        assert!(libm::ldexp(1.0, 62 - MIN_CAST_EMAX_F64).is_finite());
        assert!(libm::ldexp(1.0, 62 - (MIN_CAST_EMAX_F64 - 1)).is_infinite());
    }

    /// NaN still casts to the minimum integer in a tiny block, as with C's
    /// overflowing scale.
    #[test]
    fn fwd_cast_of_nan_in_a_tiny_block() {
        let block = [3.0e-31f32, f32::NAN, 0.0, -3.0e-31];
        let mut got = [0i32; 4];
        fwd_cast_f32(&mut got, &block, exponent_block_f32(&block));
        assert_eq!(got[1], i32::MIN);
        assert_eq!(got[2], 0);
        assert_eq!(got[0], -got[3]);
        assert!(got[0] > 0);
    }

    #[test]
    fn fwd_round_is_a_no_op_at_full_precision() {
        let mut b = [1i32, -2, 3, -4];
        fwd_round_i32(&mut b, 32);
        assert_eq!(b, [1, -2, 3, -4]);
        let mut b = [1i64, -2, 3, -4];
        fwd_round_i64(&mut b, 64);
        assert_eq!(b, [1, -2, 3, -4]);
    }

    #[test]
    fn fwd_round_bias_sign_follows_maxprec_parity() {
        // bias = (NBMASK >> 2) >> maxprec; added when maxprec is odd.
        let mut odd = [0i32; 2];
        fwd_round_i32(&mut odd, 3);
        assert_eq!(odd, [0x2aaa_aaaa >> 3; 2]);
        let mut even = [0i32; 2];
        fwd_round_i32(&mut even, 4);
        assert_eq!(even, [-(0x2aaa_aaaa >> 4); 2]);
    }

    #[test]
    fn fwd_round_wraps_instead_of_overflowing() {
        let mut b = [i32::MAX];
        fwd_round_i32(&mut b, 1); // odd maxprec: adds the bias
        assert_eq!(b[0], i32::MAX.wrapping_add(0x2aaa_aaaa >> 1));
    }

    #[test]
    fn precision_tight_error_drops_one_bit_plane() {
        // maxprec does not clamp here, so the +2/+1 difference shows through.
        assert_eq!(precision_f(-120, 64, -149, 1, false), 33);
        assert_eq!(precision_f(-120, 64, -149, 1, true), 32);
        // clamped by maxprec: both agree
        assert_eq!(precision_f(0, 12, -149, 3, false), 12);
        assert_eq!(precision_f(0, 12, -149, 3, true), 12);
        // clamped at zero
        assert_eq!(precision_f(-149, 64, 0, 1, true), 0);
    }

    #[test]
    fn precision_handles_exponent_extremes() {
        for tight_error in [false, true] {
            assert_eq!(precision_f(-332, 64, i32::MAX, 1, tight_error), 0);
            assert_eq!(precision_f(-332, 64, i32::MIN, 1, tight_error), 64);
            assert_eq!(precision_f(i32::MAX, 64, i32::MIN, 4, tight_error), 64);
            assert_eq!(precision_f(i32::MIN, 64, i32::MAX, 4, tight_error), 0);
        }
    }
}
