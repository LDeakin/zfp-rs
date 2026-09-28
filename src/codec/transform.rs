//! Forward and inverse decorrelating transforms.
//!
//! Implements the lifting (wavelet-like) transform and the orthogonal transform
//! used by the ZFP codec, and the reversible (Lorenzo) transform used by
//! lossless coding.
//!
//! A transform lifts every line of the 4^d block along x, then y, and so on;
//! the inverse runs the axes in reverse. The lines along one axis are disjoint,
//! so rather than lift them one at a time as C does, each axis is lifted as
//! vectors: along an axis of stride `s`, the lines start at `s` consecutive
//! elements, which form the lanes.
//!
//! Reference: `zfp/src/template/encode.c` (`fwd_lift`),
//!            `zfp/src/template/decode.c` (`inv_lift`),
//!            `zfp/src/template/encode{1-4}.c` (`fwd_xform`),
//!            `zfp/src/template/decode{1-4}.c` (`inv_xform`),
//!            `zfp/src/template/revencode.c`, `revdecode.c`.

#![allow(clippy::many_single_char_names)]
#![allow(
    clippy::inline_always,
    reason = "LLVM declines to inline these generic helpers into every block codec, which measured up to 75% slower"
)]

/// A block coefficient, with C's two's-complement wraparound.
pub(crate) trait Coeff: Copy {
    /// Whether vector arithmetic shifts are cheap on the baseline target.
    const FAST_SHIFT: bool;
    #[must_use]
    fn add(self, other: Self) -> Self;
    #[must_use]
    fn sub(self, other: Self) -> Self;
    /// Arithmetic shift right by one.
    #[must_use]
    fn half(self) -> Self;
}

macro_rules! impl_coeff {
    ($($t:ty),*) => {$(
        impl Coeff for $t {
            const FAST_SHIFT: bool = <$t>::BITS == 32;

            #[inline]
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }

            #[inline]
            fn half(self) -> Self {
                self >> 1
            }
        }
    )*};
}
impl_coeff!(i32, i64);

type Line<T> = (T, T, T, T);

/// Forward lifting step on one line `(x, y, z, w)`.
///
/// Non-orthogonal transform:
/// ```text
///        ( 4  4  4  4) (x)
/// 1/16 * ( 5  1 -1 -5) (y)
///        (-4  4  4 -4) (z)
///        (-2  6 -6  2) (w)
/// ```
#[inline(always)]
fn fwd_lift<T: Coeff>((mut x, mut y, mut z, mut w): Line<T>) -> Line<T> {
    x = x.add(w).half();
    w = w.sub(x);
    z = z.add(y).half();
    y = y.sub(z);
    x = x.add(z).half();
    z = z.sub(x);
    w = w.add(y).half();
    y = y.sub(w);
    w = w.add(y.half());
    y = y.sub(w.half());
    (x, y, z, w)
}

/// Inverse lifting step on one line `(x, y, z, w)`.
///
/// Non-orthogonal transform:
/// ```text
///       ( 4  6 -4 -1) (x)
/// 1/4 * ( 4  2  4  5) (y)
///       ( 4 -2  4 -5) (z)
///       ( 4 -6 -4  1) (w)
/// ```
#[inline(always)]
fn inv_lift<T: Coeff>((mut x, mut y, mut z, mut w): Line<T>) -> Line<T> {
    y = y.add(w.half());
    w = w.sub(y.half());
    y = y.add(w);
    w = w.sub(y.sub(w));
    z = z.add(x);
    x = x.sub(z.sub(x));
    y = y.add(z);
    z = z.sub(y.sub(z));
    w = w.add(x);
    x = x.sub(w.sub(x));
    (x, y, z, w)
}

/// Reversible forward lifting step (Lorenzo difference) on one line.
///
/// ```text
/// w -= z; z -= y; y -= x; w -= z; z -= y; w -= z;
/// ```
#[inline(always)]
fn rev_fwd_lift<T: Coeff>((x, mut y, mut z, mut w): Line<T>) -> Line<T> {
    w = w.sub(z);
    z = z.sub(y);
    y = y.sub(x);
    w = w.sub(z);
    z = z.sub(y);
    w = w.sub(z);
    (x, y, z, w)
}

/// Reversible inverse lifting step (inverse Lorenzo, P4 Pascal matrix).
///
/// ```text
/// w += z; z += y; w += z; y += x; z += y; w += z;
/// ```
#[inline(always)]
fn rev_inv_lift<T: Coeff>((x, mut y, mut z, mut w): Line<T>) -> Line<T> {
    w = w.add(z);
    z = z.add(y);
    w = w.add(z);
    y = y.add(x);
    z = z.add(y);
    w = w.add(z);
    (x, y, z, w)
}

/// Lift every line along the axis of stride `s` of a 4^d block.
///
/// The lines run through `4s`-element groups, each starting at one of the
/// group's first `s` elements. With `lanes`, a group's lines are lifted
/// together, which LLVM vectorizes. Otherwise they go in rows of at most four,
/// as C nests its loops, which LLVM leaves scalar.
#[inline(always)]
fn lift_axis<T: Coeff, const N: usize>(
    p: &mut [T; N],
    s: usize,
    lanes: bool,
    lift: impl Fn(Line<T>) -> Line<T>,
) {
    for group in (0..N).step_by(4 * s) {
        // Lines `i..i + len` of the group at a time.
        let len = if lanes { s } else { s.min(4) };
        for row in (group..group + s).step_by(len) {
            for i in row..row + len {
                let (x, y, z, w) = lift((p[i], p[i + s], p[i + 2 * s], p[i + 3 * s]));
                p[i] = x;
                p[i + s] = y;
                p[i + 2 * s] = z;
                p[i + 3 * s] = w;
            }
        }
    }
}

/// Lift along every axis of a 4^d block (`N = 4^d`), from x up.
#[inline(always)]
fn fwd_axes<T: Coeff, const N: usize>(
    p: &mut [T; N],
    lanes: bool,
    lift: impl Fn(Line<T>) -> Line<T> + Copy,
) {
    lift_axis(p, 1, lanes, lift);
    if N >= 16 {
        lift_axis(p, 4, lanes, lift);
    }
    if N >= 64 {
        lift_axis(p, 16, lanes, lift);
    }
    if N >= 256 {
        lift_axis(p, 64, lanes, lift);
    }
}

/// Lift along every axis of a 4^d block (`N = 4^d`), from the last down to x.
#[inline(always)]
fn inv_axes<T: Coeff, const N: usize>(
    p: &mut [T; N],
    lanes: bool,
    lift: impl Fn(Line<T>) -> Line<T> + Copy,
) {
    if N >= 256 {
        lift_axis(p, 64, lanes, lift);
    }
    if N >= 64 {
        lift_axis(p, 16, lanes, lift);
    }
    if N >= 16 {
        lift_axis(p, 4, lanes, lift);
    }
    lift_axis(p, 1, lanes, lift);
}

/// Forward decorrelating transform of a 4^d block (C `fwd_xform`).
#[inline]
pub(crate) fn fwd_xform<T: Coeff, const N: usize>(p: &mut [T; N]) {
    // The forward lift shifts six times, and SSE2 has no 64-bit arithmetic
    // shift, so vectorized `i64` lanes are slower than scalar code.
    fwd_axes(p, T::FAST_SHIFT, fwd_lift);
}

/// Inverse decorrelating transform of a 4^d block (C `inv_xform`).
#[inline]
pub(crate) fn inv_xform<T: Coeff, const N: usize>(p: &mut [T; N]) {
    inv_axes(p, true, inv_lift);
}

/// Reversible forward transform of a 4^d block (C `rev_fwd_xform`).
#[inline]
pub(crate) fn rev_fwd_xform<T: Coeff, const N: usize>(p: &mut [T; N]) {
    fwd_axes(p, true, rev_fwd_lift);
}

/// Reversible inverse transform of a 4^d block (C `rev_inv_xform`).
#[inline]
pub(crate) fn rev_inv_xform<T: Coeff, const N: usize>(p: &mut [T; N]) {
    inv_axes(p, true, rev_inv_lift);
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "test data is deliberately truncated or small"
    )]

    use super::*;

    /// C's loops: one line at a time, each axis in turn.
    fn by_line<T: Coeff, const N: usize>(
        p: &mut [T; N],
        forward: bool,
        lift: fn(Line<T>) -> Line<T>,
    ) {
        let mut strides: Vec<usize> = [1, 4, 16, 64].into_iter().filter(|&s| s < N).collect();
        if !forward {
            strides.reverse();
        }
        for s in strides {
            for base in (0..N).filter(|&b| (b / s) % 4 == 0) {
                let (x, y, z, w) = lift((p[base], p[base + s], p[base + 2 * s], p[base + 3 * s]));
                p[base] = x;
                p[base + s] = y;
                p[base + 2 * s] = z;
                p[base + 3 * s] = w;
            }
        }
    }

    fn check<const N: usize>(seed: u64) {
        let mut x = seed;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        // Full-range values, so the lifts wrap.
        let block64: [i64; N] = std::array::from_fn(|_| next().cast_signed());
        let block32: [i32; N] = std::array::from_fn(|_| next() as i32);

        macro_rules! case {
            ($block:expr, $xform:ident, $lift:ident, $forward:expr) => {{
                let mut got = $block;
                $xform(&mut got);
                let mut expect = $block;
                by_line(&mut expect, $forward, $lift);
                assert_eq!(got, expect, "{} N={N}", stringify!($xform));
            }};
        }
        case!(block32, fwd_xform, fwd_lift, true);
        case!(block64, fwd_xform, fwd_lift, true);
        case!(block32, inv_xform, inv_lift, false);
        case!(block64, inv_xform, inv_lift, false);
        case!(block32, rev_fwd_xform, rev_fwd_lift, true);
        case!(block64, rev_fwd_xform, rev_fwd_lift, true);
        case!(block32, rev_inv_xform, rev_inv_lift, false);
        case!(block64, rev_inv_xform, rev_inv_lift, false);
    }

    #[test]
    fn transforms_match_line_at_a_time() {
        for seed in [1, 0x9e37_79b9_7f4a_7c15, 12345] {
            check::<4>(seed);
            check::<16>(seed);
            check::<64>(seed);
            check::<256>(seed);
        }
    }

    #[test]
    fn reversible_transforms_invert() {
        let block: [i32; 64] = std::array::from_fn(|i| (i as i32 - 20).pow(3));
        let mut p = block;
        rev_fwd_xform(&mut p);
        rev_inv_xform(&mut p);
        assert_eq!(p, block);
    }
}
