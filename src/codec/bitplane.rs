//! Embedded bit-plane coder, shared by every encode and decode path.
//!
//! Produces exactly the bits of C's `encode_ints`/`decode_ints`
//! (`zfp/src/template/encode.c`, `decode.c`), but not one bit at a time:
//!
//! - A block is transposed into bit planes up front (and back after
//!   decoding) with a recursive bit-matrix transpose, rather than gathering
//!   plane `k` from every coefficient in turn.
//! - Each group test is coded in one go. A positive test and the run of zeros
//!   up to the next one-bit form a single code word, and the decoder finds that
//!   one-bit with `trailing_zeros` on 64 bits of lookahead.
//! - The stream cursor is held in a [`BitReader`] or [`BitWriter`] for the
//!   length of the block.

#![allow(clippy::cast_possible_truncation)]
#![allow(
    clippy::inline_always,
    reason = "LLVM declines to inline these generic helpers into every block codec, which measured up to 75% slower"
)]

use crate::bitstream::{BitReader, BitWriter, ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::codec::decode::core::{inv_round_u32, inv_round_u64};
use crate::codec::encode::core::with_maxbits;
use crate::config::ZfpRounding;

// ---------------------------------------------------------------------------
// Bit-matrix transpose
// ---------------------------------------------------------------------------

/// `MASKS[s]` selects the low `2^s` bits of every `2^(s+1)`-bit group.
const MASKS: [u64; 6] = [
    0x5555_5555_5555_5555,
    0x3333_3333_3333_3333,
    0x0f0f_0f0f_0f0f_0f0f,
    0x00ff_00ff_00ff_00ff,
    0x0000_ffff_0000_ffff,
    0x0000_0000_ffff_ffff,
];

/// A word of bit-matrix rows.
trait Word:
    Copy
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::BitXorAssign
    + std::ops::Shl<usize, Output = Self>
    + std::ops::Shr<usize, Output = Self>
{
    /// `MASKS[s]`, truncated to the word.
    fn mask(s: usize) -> Self;
}

impl Word for u32 {
    #[inline]
    fn mask(s: usize) -> Self {
        MASKS[s] as u32
    }
}

impl Word for u64 {
    #[inline]
    fn mask(s: usize) -> Self {
        MASKS[s]
    }
}

/// Transpose the `R`×`R` bit matrix held in every `R`-bit lane of `a`, where
/// row `i` is the lane in `a[i]` and column `c` is bit `c` of the lane.
///
/// Each step swaps the off-diagonal quadrants of every `2j`×`2j` sub-matrix
/// (Hacker's Delight, 7-3). The lanes are independent, so this transposes
/// one matrix per lane at once.
#[inline(always)]
#[allow(clippy::many_single_char_names)] // as in C
fn transpose<W: Word, const R: usize>(a: &mut [W; R]) {
    let mut j = R / 2;
    let mut s = R.trailing_zeros() as usize;
    while j != 0 {
        s -= 1;
        let m = W::mask(s);
        let mut k = 0;
        while k < R {
            for i in k..k + j {
                let t = ((a[i] >> j) ^ a[i + j]) & m;
                a[i] ^= t << j;
                a[i + j] ^= t;
            }
            k += 2 * j;
        }
        j /= 2;
    }
}

// ---------------------------------------------------------------------------
// Bit planes
// ---------------------------------------------------------------------------

/// One bit plane: bit `i` is bit `k` of coefficient `i`.
pub(crate) trait Plane: Copy {
    /// Position of the lowest one-bit at or above `n`.
    fn next_one(&self, n: u32) -> Option<u32>;
    /// Set bit `n`.
    fn set(&mut self, n: u32);
    /// Write bits `0..m`.
    fn write_low(&self, w: &mut BitWriter, m: u32);
    /// Read bits `0..m`; the rest are zero.
    fn read_low(r: &mut BitReader, m: u32) -> Self;
    /// Word `c`, covering coefficients `64*c..64*(c+1)`.
    fn word(&self, c: usize) -> u64;
    /// The plane whose word `c` is `word(c)`.
    fn from_words(word: impl Fn(usize) -> u64) -> Self;
    #[cfg(test)]
    fn bit(&self, i: usize) -> bool;
}

/// Mask of the low `n <= 64` bits.
#[inline]
fn low_mask(n: u32) -> u64 {
    if n < 64 { (1u64 << n) - 1 } else { u64::MAX }
}

/// A plane of at most 64 coefficients.
impl Plane for u64 {
    #[inline]
    fn next_one(&self, n: u32) -> Option<u32> {
        let rest = *self >> n;
        (rest != 0).then(|| n + rest.trailing_zeros())
    }

    #[inline]
    fn set(&mut self, n: u32) {
        *self |= 1 << n;
    }

    #[inline]
    fn write_low(&self, w: &mut BitWriter, m: u32) {
        w.put(*self & low_mask(m), m);
    }

    #[inline]
    fn read_low(r: &mut BitReader, m: u32) -> Self {
        r.read(m)
    }

    #[inline]
    fn word(&self, _c: usize) -> u64 {
        *self
    }

    #[inline]
    fn from_words(word: impl Fn(usize) -> u64) -> Self {
        word(0)
    }

    #[cfg(test)]
    fn bit(&self, i: usize) -> bool {
        (self >> i) & 1 == 1
    }
}

/// A plane of 256 coefficients (4-D blocks).
impl Plane for [u64; 4] {
    #[inline]
    fn next_one(&self, n: u32) -> Option<u32> {
        let i = (n / 64) as usize;
        let rest = self[i] >> (n % 64);
        if rest != 0 {
            return Some(n + rest.trailing_zeros());
        }
        (i + 1..4)
            .find(|&j| self[j] != 0)
            .map(|j| 64 * j as u32 + self[j].trailing_zeros())
    }

    #[inline]
    fn set(&mut self, n: u32) {
        self[(n / 64) as usize] |= 1 << (n % 64);
    }

    #[inline]
    fn write_low(&self, w: &mut BitWriter, m: u32) {
        let (full, rest) = ((m / 64) as usize, m % 64);
        for &word in &self[..full] {
            w.put(word, 64);
        }
        if rest != 0 {
            w.put(self[full] & low_mask(rest), rest);
        }
    }

    #[inline]
    fn read_low(r: &mut BitReader, m: u32) -> Self {
        let (full, rest) = ((m / 64) as usize, m % 64);
        let mut x = [0; 4];
        for word in &mut x[..full] {
            *word = r.read(64);
        }
        if rest != 0 {
            x[full] = r.read(rest);
        }
        x
    }

    #[inline]
    fn word(&self, c: usize) -> u64 {
        self[c]
    }

    #[inline]
    fn from_words(word: impl Fn(usize) -> u64) -> Self {
        std::array::from_fn(word)
    }

    #[cfg(test)]
    fn bit(&self, i: usize) -> bool {
        (self[i / 64] >> (i % 64)) & 1 == 1
    }
}

/// A block of negabinary coefficients, coded one bit plane at a time.
///
/// The coder reads and writes planes through [`Self::Planes`], a transposed
/// block, and only for the planes it codes.
pub(crate) trait PlaneBlock: Sized {
    type Plane: Plane;
    /// The block in bit-plane order.
    type Planes;
    /// Bits per coefficient (C `intprec`).
    const INTPREC: u32;
    /// Coefficients per block.
    const SIZE: u32;

    /// Number of planes up to and including the highest one-bit.
    fn top(&self) -> u32;
    /// Transpose planes `kmin..INTPREC`; lower planes are unspecified.
    fn to_planes(&self, kmin: u32) -> Self::Planes;
    /// Plane `k` of a block from [`Self::to_planes`].
    fn plane(planes: &Self::Planes, k: u32) -> Self::Plane;

    /// A block with every plane zero, for [`Self::set_plane`].
    fn zero_planes() -> Self::Planes;
    /// Store plane `k`, at most once, into [`Self::zero_planes`].
    fn set_plane(planes: &mut Self::Planes, k: u32, plane: Self::Plane);
    /// Transpose back, given that every plane below `kmin` is zero.
    fn from_planes(planes: &Self::Planes, kmin: u32) -> Self;
    /// The all-zero block.
    fn zero() -> Self;

    /// Bias for `ZFP_ROUND_LAST`; see `inv_round_u32`.
    fn inv_round(&mut self, m: u32, prec: u32);
}

/// Implement [`PlaneBlock`] for a block of `$n <= 16` coefficients, as
/// transposes in `$n`-bit lanes of the coefficients themselves: plane `k` is
/// lane `k / $n` of word `k % $n`. Transposing every plane is cheap enough at
/// this size.
macro_rules! plane_block_lanes {
    ($u:ty, $n:literal, $inv_round:ident) => {
        impl PlaneBlock for [$u; $n] {
            type Plane = u64;
            type Planes = [$u; $n];
            const INTPREC: u32 = <$u>::BITS;
            const SIZE: u32 = $n;

            #[inline(always)]
            fn top(&self) -> u32 {
                <$u>::BITS - self.iter().fold(0, |acc, &v| acc | v).leading_zeros()
            }

            #[inline(always)]
            fn to_planes(&self, _kmin: u32) -> Self::Planes {
                let mut a = *self;
                transpose(&mut a);
                a
            }

            #[inline(always)]
            fn plane(planes: &Self::Planes, k: u32) -> u64 {
                u64::from(planes[(k % $n) as usize] >> ($n * (k / $n))) & low_mask($n)
            }

            #[inline(always)]
            fn zero_planes() -> Self::Planes {
                [0; $n]
            }

            #[inline(always)]
            fn set_plane(planes: &mut Self::Planes, k: u32, plane: u64) {
                planes[(k % $n) as usize] |= (plane as $u) << ($n * (k / $n));
            }

            #[inline(always)]
            fn from_planes(planes: &Self::Planes, _kmin: u32) -> Self {
                let mut a = *planes;
                transpose(&mut a);
                a
            }

            #[inline]
            fn zero() -> Self {
                [0; $n]
            }

            #[inline]
            fn inv_round(&mut self, m: u32, prec: u32) {
                $inv_round(self, m, prec);
            }
        }
    };
}

plane_block_lanes!(u32, 4, inv_round_u32);
plane_block_lanes!(u64, 4, inv_round_u64);
plane_block_lanes!(u32, 16, inv_round_u32);
plane_block_lanes!(u64, 16, inv_round_u64);

/// Transpose planes `kmin..P` of 64 `P`-bit coefficients into `planes`.
///
/// Only the top 16, 32 or 64 planes are transposed, whichever covers `kmin`:
/// `R` planes pack into `R` rows of `64 / R` lanes, and lane `l` of row `c`
/// ends up holding coefficients `R*l..R*(l+1)` of plane `P - R + c`, which is
/// exactly row `c` read as one 64-bit plane.
#[inline(always)]
fn chunk_to_planes<const P: usize>(coeff: impl Fn(usize) -> u64, kmin: u32, planes: &mut [u64; P]) {
    let need = P as u32 - kmin;
    if need <= 16 {
        top_to_planes::<16>(coeff, planes);
    } else if P == 32 || need <= 32 {
        top_to_planes::<32>(coeff, planes);
    } else {
        top_to_planes::<64>(coeff, planes);
    }
}

#[inline(always)]
fn top_to_planes<const R: usize>(coeff: impl Fn(usize) -> u64, planes: &mut [u64]) {
    let shift = planes.len() - R;
    let mut rows: [u64; R] = std::array::from_fn(|c| {
        (0..64 / R).fold(0, |row, l| {
            row | (((coeff(R * l + c) >> shift) & low_mask(R as u32)) << (R * l))
        })
    });
    transpose(&mut rows);
    planes[shift..].copy_from_slice(&rows);
}

/// Inverse of [`chunk_to_planes`], given that every plane below `kmin` is zero.
#[inline(always)]
fn chunk_from_planes<const P: usize>(planes: &[u64; P], kmin: u32, store: impl FnMut(usize, u64)) {
    let need = P as u32 - kmin.min(P as u32);
    if need <= 16 {
        top_from_planes::<16>(planes, store);
    } else if P == 32 || need <= 32 {
        top_from_planes::<32>(planes, store);
    } else {
        top_from_planes::<64>(planes, store);
    }
}

#[inline(always)]
fn top_from_planes<const R: usize>(planes: &[u64], mut store: impl FnMut(usize, u64)) {
    let shift = planes.len() - R;
    let mut rows: [u64; R] = planes[shift..]
        .try_into()
        .unwrap_or_else(|_| unreachable!());
    transpose(&mut rows);
    for (c, &row) in rows.iter().enumerate() {
        for l in 0..64 / R {
            store(R * l + c, ((row >> (R * l)) & low_mask(R as u32)) << shift);
        }
    }
}

/// Implement [`PlaneBlock`] for a block of `64 * $chunks` coefficients, one
/// plane word per 64-coefficient chunk.
macro_rules! plane_block_chunks {
    ($u:ty, $n:literal, $chunks:literal, $plane:ty, $inv_round:ident) => {
        impl PlaneBlock for [$u; $n] {
            type Plane = $plane;
            type Planes = [[u64; <$u>::BITS as usize]; $chunks];
            const INTPREC: u32 = <$u>::BITS;
            const SIZE: u32 = $n;

            #[inline(always)]
            fn top(&self) -> u32 {
                <$u>::BITS - self.iter().fold(0, |acc, &v| acc | v).leading_zeros()
            }

            #[inline(always)]
            fn to_planes(&self, kmin: u32) -> Self::Planes {
                let mut planes = Self::zero_planes();
                for (c, chunk) in planes.iter_mut().enumerate() {
                    chunk_to_planes(|i| u64::from(self[64 * c + i]), kmin, chunk);
                }
                planes
            }

            #[inline(always)]
            fn plane(planes: &Self::Planes, k: u32) -> $plane {
                <$plane>::from_words(|c| planes[c][k as usize])
            }

            #[inline(always)]
            fn zero_planes() -> Self::Planes {
                [[0; <$u>::BITS as usize]; $chunks]
            }

            #[inline(always)]
            fn set_plane(planes: &mut Self::Planes, k: u32, plane: $plane) {
                for (c, chunk) in planes.iter_mut().enumerate() {
                    chunk[k as usize] = plane.word(c);
                }
            }

            #[inline(always)]
            fn from_planes(planes: &Self::Planes, kmin: u32) -> Self {
                let mut block = [0; $n];
                for (c, chunk) in planes.iter().enumerate() {
                    chunk_from_planes(chunk, kmin, |i, v| block[64 * c + i] = v as $u);
                }
                block
            }

            #[inline]
            fn zero() -> Self {
                [0; $n]
            }

            #[inline]
            fn inv_round(&mut self, m: u32, prec: u32) {
                $inv_round(self, m, prec);
            }
        }
    };
}

plane_block_chunks!(u32, 64, 1, u64, inv_round_u32);
plane_block_chunks!(u64, 64, 1, u64, inv_round_u64);
plane_block_chunks!(u32, 256, 4, [u64; 4], inv_round_u32);
plane_block_chunks!(u64, 256, 4, [u64; 4], inv_round_u64);

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Encode `block` one bit plane at a time from the MSB; return bits written.
///
/// C `encode_ints`. `maxbits` only binds when the block could exceed it, as C
/// selects its unconstrained `encode_ints_prec` otherwise.
#[inline]
pub(crate) fn encode_ints<B: PlaneBlock>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    maxbits: u32,
    maxprec: u32,
    block: &B,
) -> u32 {
    let budget = if with_maxbits(maxbits, maxprec, B::SIZE) {
        maxbits
    } else {
        u32::MAX
    };
    let kmin = B::INTPREC.saturating_sub(maxprec);
    let planes = block.to_planes(kmin);
    let mut w = BitWriter::new(bs);
    encode_planes(
        &mut w,
        budget,
        kmin,
        B::INTPREC,
        B::SIZE,
        block.top(),
        |k| B::plane(&planes, k),
    )
}

/// The bit-plane loop of C's `encode_few_ints` and `encode_many_ints`, which
/// differ only in how they store a plane; see [`Plane`]. `top` is
/// [`PlaneBlock::top`].
#[inline(always)]
#[allow(clippy::many_single_char_names)] // as in C
fn encode_planes<P: Plane>(
    w: &mut BitWriter,
    budget: u32,
    kmin: u32,
    intprec: u32,
    size: u32,
    top: u32,
    plane: impl Fn(u32) -> P,
) -> u32 {
    let mut bits = budget;
    let mut n = 0u32;
    // Planes above the highest one-bit are empty, so each is a single
    // negative group test.
    let empty = (intprec - top.max(kmin)).min(bits);
    w.put_zeros(empty);
    bits -= empty;
    let mut k = intprec - empty;
    while bits != 0 && k > kmin {
        if n == size {
            // Every coefficient is significant, so whole planes are stored as
            // is. The last one, which may be cut short, takes the general path.
            while bits > size && k > kmin {
                k -= 1;
                plane(k).write_low(w, size);
                bits -= size;
            }
            if k == kmin {
                break;
            }
        }
        k -= 1;
        let x = &plane(k);
        // Step 1: the first `n` bits are known significant; write them as is.
        let m = n.min(bits);
        bits -= m;
        x.write_low(w, m);
        // Step 2: unary run-length encode the rest. A positive group test is a
        // one, then a zero for every position before the next one-bit, then
        // that one-bit, which is implied at the last position.
        while bits != 0 && n < size {
            let Some(p) = x.next_one(n) else {
                // Negative group test: done with this plane.
                w.put(0, 1);
                bits -= 1;
                break;
            };
            let zeros = p - n;
            let last = p == size - 1;
            let len = zeros + 2 - u32::from(last);
            // The budget may cut the code short.
            let take = len.min(bits);
            if take <= 64 {
                // With 63 or more zeros, the one-bit falls past `take`.
                let code = if last || zeros >= 63 {
                    1
                } else {
                    1 | (2 << zeros)
                };
                w.put(code & low_mask(take), take);
            } else {
                w.put(1, 1);
                w.put_zeros(zeros.min(take - 1));
                if take == len && !last {
                    w.put(1, 1);
                }
            }
            bits -= take;
            n = p + 1;
        }
    }
    budget - bits
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Decode a block one bit plane at a time from the MSB; return it with the
/// bits read.
///
/// C `decode_ints`, including its `ZFP_ROUND_LAST` bias. As in
/// [`encode_ints`], `maxbits` only binds when the block could exceed it.
///
/// Also returns whether the block is all zeros, which lets callers skip the
/// inverse transform, since it maps zeros to zeros.
#[inline]
pub(crate) fn decode_ints<B: PlaneBlock>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    maxbits: u32,
    maxprec: u32,
    rounding: ZfpRounding,
) -> (B, u32, bool) {
    let constrained = with_maxbits(maxbits, maxprec, B::SIZE);
    let budget = if constrained { maxbits } else { u32::MAX };
    let kmin = B::INTPREC.saturating_sub(maxprec);
    let mut planes = B::zero_planes();
    let (bits, m, prec, low) = {
        let mut r = BitReader::new(bs);
        decode_planes(&mut r, budget, kmin, B::INTPREC, B::SIZE, |k, x| {
            B::set_plane(&mut planes, k, x);
        })
    };
    let zero = low == B::INTPREC;
    let mut block = if zero {
        B::zero()
    } else {
        B::from_planes(&planes, low)
    };
    let round = matches!(rounding, ZfpRounding::Last { .. });
    if round {
        let (m, prec) = if constrained {
            (m, prec)
        } else {
            // C's `decode_ints_prec` always exits with `k == kmin - 1` and
            // `m == 0`.
            (0, B::INTPREC.wrapping_sub(kmin.wrapping_sub(1)))
        };
        block.inv_round(m, prec);
    }
    (block, bits, zero && !round)
}

/// The bit-plane loop of C's `decode_few_ints` and `decode_many_ints`.
///
/// Returns `(bits, m, prec, low)`: bits read, the `ZFP_ROUND_LAST` state,
/// and the lowest nonzero plane, or `intprec` if every plane is zero.
#[inline(always)]
#[allow(clippy::many_single_char_names)] // as in C
fn decode_planes<P: Plane>(
    r: &mut BitReader,
    budget: u32,
    kmin: u32,
    intprec: u32,
    size: u32,
    mut store: impl FnMut(u32, P),
) -> (u32, u32, u32, u32) {
    let mut bits = budget;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut k = intprec;
    let mut low = intprec;
    while bits != 0 {
        m = 0;
        if k <= kmin {
            // C decrements k in the loop condition even on this exit.
            k = k.wrapping_sub(1);
            break;
        }
        if n == 0 {
            // Until a coefficient is significant, an empty plane is a single
            // negative group test, so a run of zeros is a run of empty planes.
            let empty = r.peek().trailing_zeros().min(k - kmin).min(bits);
            if empty != 0 {
                r.consume(empty);
                bits -= empty;
                k -= empty;
                m = size;
                continue;
            }
        } else if n == size {
            // Every coefficient is significant, so whole planes are stored as
            // is. The last one, which may be cut short, takes the general path
            // to leave `m` and `k` as C does.
            while bits > size && k > kmin {
                k -= 1;
                store(k, P::read_low(r, size));
                bits -= size;
                low = k;
            }
            if k == kmin {
                k = k.wrapping_sub(1);
                break;
            }
        }
        k -= 1;
        // Step 1: the first `n` bits of the plane are stored as is.
        m = n.min(bits);
        bits -= m;
        let mut x = P::read_low(r, m);
        // Step 2: unary run-length decode the rest.
        while bits != 0 && n < size {
            let ahead = r.peek();
            bits -= 1;
            if ahead & 1 == 0 {
                // Negative group test: done with this plane.
                r.consume(1);
                m = size;
                break;
            }
            // Positive group test: scan up to `limit` positions for the next
            // one-bit, which is implied at the last position, or wherever the
            // budget runs out.
            let limit = (size - 1 - n).min(bits);
            let zeros = (ahead >> 1).trailing_zeros();
            if zeros < limit.min(63) {
                // The one-bit is within the lookahead.
                r.consume(zeros + 2);
                bits -= zeros + 1;
                n += zeros;
            } else if limit <= 63 {
                // The lookahead covers all `limit` positions, and they are zero.
                r.consume(limit + 1);
                bits -= limit;
                n += limit;
            } else {
                r.consume(64);
                bits -= 63;
                n += 63;
                scan_zeros(r, &mut bits, &mut n, limit - 63);
            }
            x.set(n);
            n += 1;
            m = n;
        }
        store(k, x);
        low = k;
    }
    // `n` only grows when a one-bit is decoded, so while it is zero, every
    // plane is.
    if n == 0 {
        low = intprec;
    }
    (budget - bits, m, intprec.wrapping_sub(k), low)
}

/// Consume zeros up to and including the next one-bit, at most `limit` bits.
/// Only 4-D blocks have runs too long for [`decode_planes`]' lookahead.
#[cold]
fn scan_zeros(r: &mut BitReader, bits: &mut u32, n: &mut u32, mut limit: u32) {
    while limit != 0 {
        let zeros = r.peek().trailing_zeros();
        let span = limit.min(64);
        if zeros < span {
            r.consume(zeros + 1);
            *bits -= zeros + 1;
            *n += zeros;
            return;
        }
        r.consume(span);
        *bits -= span;
        *n += span;
        limit -= span;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check every plane against bits gathered one at a time, as C does, and
    /// that the inverse transpose restores the block.
    fn check_round_trip<B, U>(block: &B)
    where
        B: PlaneBlock + AsRef<[U]> + PartialEq + std::fmt::Debug,
        U: Copy + Into<u64>,
    {
        for kmin in 0..B::INTPREC {
            let planes = block.to_planes(kmin);
            let mut stored = B::zero_planes();
            for k in kmin..B::INTPREC {
                let plane = B::plane(&planes, k);
                for (i, &v) in block.as_ref().iter().enumerate() {
                    let bit = (v.into() >> k) & 1 == 1;
                    assert_eq!(plane.bit(i), bit, "plane {k} coefficient {i}");
                }
                B::set_plane(&mut stored, k, plane);
            }
            // The block with the planes below `kmin` cleared.
            let low = B::from_planes(&stored, kmin);
            for (&got, &v) in low.as_ref().iter().zip(block.as_ref()) {
                let v: u64 = v.into();
                assert_eq!(got.into(), (v >> kmin) << kmin, "kmin {kmin}");
            }
        }
    }

    /// A deterministic bit pattern with varied density.
    fn pattern(i: usize, salt: u64) -> u64 {
        let x = (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
        let x = x ^ (x >> 29);
        x.wrapping_mul(0xbf58_476d_1ce4_e5b9) & (x >> (i % 7))
    }

    #[test]
    fn planes_match_naive_extraction() {
        for salt in [0, 1, 0xdead_beef, u64::MAX] {
            check_round_trip::<[u32; 4], _>(&std::array::from_fn(|i| pattern(i, salt) as u32));
            check_round_trip::<[u64; 4], _>(&std::array::from_fn(|i| pattern(i, salt)));
            check_round_trip::<[u32; 16], _>(&std::array::from_fn(|i| pattern(i, salt) as u32));
            check_round_trip::<[u64; 16], _>(&std::array::from_fn(|i| pattern(i, salt)));
            check_round_trip::<[u32; 64], _>(&std::array::from_fn(|i| pattern(i, salt) as u32));
            check_round_trip::<[u64; 64], _>(&std::array::from_fn(|i| pattern(i, salt)));
            check_round_trip::<[u32; 256], _>(&std::array::from_fn(|i| pattern(i, salt) as u32));
            check_round_trip::<[u64; 256], _>(&std::array::from_fn(|i| pattern(i, salt)));
        }
    }
}
