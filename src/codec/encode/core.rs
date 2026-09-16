//! Shared encode utilities: permutation tables, int2uint, bit-plane encoder.
//!
//! Reference: `zfp/src/template/codec{1-4}.c`, `encode.c`, `codecf.c`

#![allow(clippy::cast_possible_truncation)]

use crate::bitstream::ZfpBitStreamMutOps;

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

/// Map two's-complement `i32` → negabinary `u32`.
///
/// Formula: `((x as u32).wrapping_add(NBMASK)) ^ NBMASK`
/// where `NBMASK = 0xaaaaaaaau`.
#[inline]
#[allow(clippy::cast_sign_loss)] // i32→u32 for negabinary encoding
pub(crate) fn int2uint_i32(x: i32) -> u32 {
    const NBMASK: u32 = 0xaaaa_aaaa_u32;
    (x as u32).wrapping_add(NBMASK) ^ NBMASK
}

/// Map two's-complement `i64` → negabinary `u64`.
#[inline]
#[allow(clippy::cast_sign_loss)] // i64→u64 for negabinary encoding
pub(crate) fn int2uint_i64(x: i64) -> u64 {
    const NBMASK: u64 = 0xaaaa_aaaa_aaaa_aaaa;
    (x as u64).wrapping_add(NBMASK) ^ NBMASK
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
// Bit-plane encoders (u32 elements, size ≤ 64)
// ---------------------------------------------------------------------------

/// Encode `size ≤ 64` unsigned 32-bit integers with a rate constraint.
///
/// Returns the number of bits written.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_few_ints_u32(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;
    let mut n: u32 = 0;

    let mut k = intprec;
    while bits != 0 && k > kmin {
        k -= 1;
        // Step 1: extract bit plane k into x (one bit per element)
        let mut x: u64 = 0;
        for (i, &val) in data.iter().enumerate() {
            x |= (u64::from(val >> k) & 1) << i;
        }
        // Step 2: write first n committed bits
        let m = n.min(bits);
        bits -= m;
        x = bs.write_bits(x, m);
        // Step 3: unary RLE for the remaining bits
        // Mirrors C: `for (; bits && n < size; x >>= 1, n++)`
        let mut i = n;
        while bits != 0 && i < size {
            bits -= 1;
            if bs.write_bit(u32::from(x != 0)) != 0 {
                // positive group test: scan for 1-bit
                // Mirrors C inner: `for (; bits && n < size-1; x >>= 1, n++)`
                // Note: on break, C's for-loop increment does NOT run, so x and i
                // are NOT advanced for the breaking iteration.
                while bits != 0 && i < size - 1 {
                    bits -= 1;
                    let bit = (x & 1) as u32;
                    if bs.write_bit(bit) != 0 {
                        break;
                    }
                    x >>= 1;
                    i += 1;
                }
            } else {
                // negative group test: done with bit plane
                break;
            }
            // outer post-increment
            x >>= 1;
            i += 1;
        }
        n = i;
    }
    maxbits - bits
}

/// Encode `size ≤ 64` unsigned 64-bit integers with a rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_few_ints_u64(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;
    let mut n: u32 = 0;

    let mut k = intprec;
    while bits != 0 && k > kmin {
        k -= 1;
        // Step 1: extract bit plane k
        let mut x: u64 = 0;
        for (i, &val) in data.iter().enumerate() {
            x |= ((val >> k) & 1) << i;
        }
        // Step 2: write first n committed bits
        let m = n.min(bits);
        bits -= m;
        x = bs.write_bits(x, m);
        // Step 3: unary RLE
        // Mirrors C: `for (; bits && n < size; x >>= 1, n++)`
        let mut i = n;
        while bits != 0 && i < size {
            bits -= 1;
            if bs.write_bit(u32::from(x != 0)) != 0 {
                // inner scan: `for (; bits && n < size-1; x >>= 1, n++)`
                while bits != 0 && i < size - 1 {
                    bits -= 1;
                    let bit = (x & 1) as u32;
                    if bs.write_bit(bit) != 0 {
                        break;
                    }
                    x >>= 1;
                    i += 1;
                }
            } else {
                break;
            }
            // outer post-increment
            x >>= 1;
            i += 1;
        }
        n = i;
    }
    maxbits - bits
}

// ---------------------------------------------------------------------------
// Bit-plane encoders (u32 elements, size > 64): only for 4D blocks
// ---------------------------------------------------------------------------

/// Encode `size > 64` unsigned 32-bit integers with a rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_many_ints_u32(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;
    let mut n: u32 = 0;

    let mut k = intprec;
    while bits != 0 && k > kmin {
        k -= 1;
        // Step 1: write first n individual bits directly
        let m = n.min(bits);
        bits -= m;
        for &val in &data[..m as usize] {
            bs.write_bit((val >> k) & 1);
        }
        // Step 2: count remaining 1-bits
        let mut c: u32 = 0;
        for &val in &data[m as usize..] {
            c += (val >> k) & 1;
        }
        // Step 3: unary RLE for remainder
        // Mirrors C: `for (; bits && n < size; n++)`
        // Note: `n` is shared between outer and inner loops, matching the C semantics.
        'outer_u32: loop {
            if bits == 0 || n >= size {
                break;
            }
            bits -= 1;
            if bs.write_bit(u32::from(c > 0)) != 0 {
                // positive group test; scan for one-bit
                // C: `for (c--; bits && n < size - 1; n++)`
                c -= 1;
                loop {
                    if bits == 0 || n >= size - 1 {
                        break;
                    }
                    bits -= 1;
                    let bit = (data[n as usize] >> k) & 1;
                    if bs.write_bit(bit) != 0 {
                        break; // inner break: outer n++ still runs
                    }
                    n += 1; // inner n++
                }
            } else {
                // negative group test; done with bit plane (outer break: n++ does NOT run)
                break 'outer_u32;
            }
            n += 1; // outer n++
        }
    }
    maxbits - bits
}

/// Encode `size > 64` unsigned 64-bit integers with a rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_many_ints_u64(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;
    let mut n: u32 = 0;

    let mut k = intprec;
    while bits != 0 && k > kmin {
        k -= 1;
        let m = n.min(bits);
        bits -= m;
        for &val in &data[..m as usize] {
            bs.write_bit(((val >> k) & 1) as u32);
        }
        let mut c: u32 = 0;
        for &val in &data[m as usize..] {
            c += ((val >> k) & 1) as u32;
        }
        // Mirrors C: `for (; bits && n < size; n++)`
        'outer_u64: loop {
            if bits == 0 || n >= size {
                break;
            }
            bits -= 1;
            if bs.write_bit(u32::from(c > 0)) != 0 {
                c -= 1;
                loop {
                    if bits == 0 || n >= size - 1 {
                        break;
                    }
                    bits -= 1;
                    let bit = ((data[n as usize] >> k) & 1) as u32;
                    if bs.write_bit(bit) != 0 {
                        break;
                    }
                    n += 1;
                }
            } else {
                break 'outer_u64;
            }
            n += 1;
        }
    }
    maxbits - bits
}

// ---------------------------------------------------------------------------
// Variable-rate bit-plane encoders (no maxbits constraint)
// ---------------------------------------------------------------------------

/// Encode `size ≤ 64` u32 integers with no rate constraint; returns bits written.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_few_ints_prec_u32(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxprec: u32,
    data: &[u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.write_pos();
    let mut n: u32 = 0;

    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Extract bit plane k
        let mut x: u64 = 0;
        for (i, &val) in data.iter().enumerate() {
            x |= (u64::from(val >> k) & 1) << i;
        }
        // Write first n committed bits
        x = bs.write_bits(x, n);
        // Unary RLE for remainder: mirrors C `for (; n < size && write_bit(!!x); x>>=1, n++)`
        while n < size && bs.write_bit(u32::from(x != 0)) != 0 {
            // inner scan: `for (; n < size-1 && !write_bit(x&1); x>>=1, n++)`
            while n < size - 1 && bs.write_bit((x & 1) as u32) == 0 {
                x >>= 1;
                n += 1;
            }
            // outer post-increment
            x >>= 1;
            n += 1;
        }
    }
    (bs.write_pos() - start) as u32
}

/// Encode `size ≤ 64` u64 integers with no rate constraint; returns bits written.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_few_ints_prec_u64(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxprec: u32,
    data: &[u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.write_pos();
    let mut n: u32 = 0;

    let mut k = intprec;
    while k > kmin {
        k -= 1;
        let mut x: u64 = 0;
        for (i, &val) in data.iter().enumerate() {
            x |= ((val >> k) & 1) << i;
        }
        x = bs.write_bits(x, n);
        // Unary RLE for remainder: mirrors C `for (; n < size && write_bit(!!x); x>>=1, n++)`
        while n < size && bs.write_bit(u32::from(x != 0)) != 0 {
            // inner scan: `for (; n < size-1 && !write_bit(x&1); x>>=1, n++)`
            while n < size - 1 && bs.write_bit((x & 1) as u32) == 0 {
                x >>= 1;
                n += 1;
            }
            // outer post-increment
            x >>= 1;
            n += 1;
        }
    }
    (bs.write_pos() - start) as u32
}

/// Encode `size > 64` u32 integers with no rate constraint; returns bits written.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_many_ints_prec_u32(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxprec: u32,
    data: &[u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.write_pos();
    let mut n: u32 = 0;

    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Step 1: write first n bits directly
        for &val in &data[..n as usize] {
            bs.write_bit((val >> k) & 1);
        }
        // Step 2: count remaining 1-bits
        let mut c: u32 = 0;
        for &val in &data[n as usize..] {
            c += (val >> k) & 1;
        }
        // Step 3: unary RLE
        // Mirrors C: `for (; n < size && write_bit(!!c); n++)`
        while n < size && bs.write_bit(u32::from(c > 0)) != 0 {
            // inner: `for (c--; n < size-1 && !write_bit(data[n]>>k&1); n++)`
            c -= 1;
            while n < size - 1 && bs.write_bit((data[n as usize] >> k) & 1) == 0 {
                n += 1;
            }
            // outer post-increment
            n += 1;
        }
    }
    (bs.write_pos() - start) as u32
}

/// Encode `size > 64` u64 integers with no rate constraint; returns bits written.
#[allow(clippy::many_single_char_names)]
pub(crate) fn encode_many_ints_prec_u64(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxprec: u32,
    data: &[u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.write_pos();
    let mut n: u32 = 0;

    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Step 1: write first n bits directly
        for &val in &data[..n as usize] {
            bs.write_bit(((val >> k) & 1) as u32);
        }
        // Step 2: count remaining 1-bits
        let mut c: u32 = 0;
        for &val in &data[n as usize..] {
            c += ((val >> k) & 1) as u32;
        }
        // Step 3: unary RLE
        // Mirrors C: `for (; n < size && write_bit(!!c); n++)`
        while n < size && bs.write_bit(u32::from(c > 0)) != 0 {
            c -= 1;
            while n < size - 1 && bs.write_bit(((data[n as usize] >> k) & 1) as u32) == 0 {
                n += 1;
            }
            n += 1;
        }
    }
    (bs.write_pos() - start) as u32
}

// ---------------------------------------------------------------------------
// Main dispatch: encode_ints
// ---------------------------------------------------------------------------

/// Encode `data.len()` unsigned 32-bit integers; returns bits written.
pub(crate) fn encode_ints_u32(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u32],
) -> u32 {
    let size = data.len() as u32;
    if with_maxbits(maxbits, maxprec, size) {
        if size <= 64 {
            encode_few_ints_u32(bs, maxbits, maxprec, data)
        } else {
            encode_many_ints_u32(bs, maxbits, maxprec, data)
        }
    } else if size <= 64 {
        encode_few_ints_prec_u32(bs, maxprec, data)
    } else {
        encode_many_ints_prec_u32(bs, maxprec, data)
    }
}

/// Encode `data.len()` unsigned 64-bit integers; returns bits written.
pub(crate) fn encode_ints_u64(
    bs: &mut dyn ZfpBitStreamMutOps,
    maxbits: u32,
    maxprec: u32,
    data: &[u64],
) -> u32 {
    let size = data.len() as u32;
    if with_maxbits(maxbits, maxprec, size) {
        if size <= 64 {
            encode_few_ints_u64(bs, maxbits, maxprec, data)
        } else {
            encode_many_ints_u64(bs, maxbits, maxprec, data)
        }
    } else if size <= 64 {
        encode_few_ints_prec_u64(bs, maxprec, data)
    } else {
        encode_many_ints_prec_u64(bs, maxprec, data)
    }
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
/// `precision(maxexp, maxprec, minexp, dims) = MIN(maxprec, MAX(0, maxexp - minexp + 2*dims + 2))`
///
/// `dims` is the number of spatial dimensions (1–4).
#[inline]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // u32→i32 for exponent calc
pub(crate) fn precision_f(maxexp: i32, maxprec: u32, minexp: i32, dims: u32) -> u32 {
    let raw = maxexp - minexp + 2 * dims as i32 + 2;
    maxprec.min(raw.max(0) as u32)
}

/// Return the maximum floating-point exponent in an f32 block.
///
/// Uses `frexp` semantics: returns the exponent `e` such that `|x| = m * 2^e`
/// with `0.5 ≤ m < 1`. Returns `-EBIAS = -127` when all values are zero.
pub(crate) fn exponent_block_f32(data: &[f32]) -> i32 {
    const EBIAS: i32 = 127;
    let max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max > 0.0 {
        // frexpf returns (mantissa in [0.5,1), exponent e) such that x = m * 2^e
        let (_, e) = libm::frexpf(max);
        e.max(1 - EBIAS)
    } else {
        -EBIAS
    }
}

/// Return the maximum floating-point exponent in an f64 block.
pub(crate) fn exponent_block_f64(data: &[f64]) -> i32 {
    const EBIAS: i32 = 1023;
    let max = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max > 0.0 {
        let (_, e) = libm::frexp(max);
        e.max(1 - EBIAS)
    } else {
        -EBIAS
    }
}

/// Forward block-floating-point transform: quantize f32 → i32 relative to exponent `emax`.
pub(crate) fn fwd_cast_f32(iblock: &mut [i32], fblock: &[f32], emax: i32) {
    // s = 2^(30 - emax).  When emax < -97, s overflows f32 to +inf,
    // causing `v = s * f` to be ±inf even for small finite f.
    // In C on x86, (int32_t)(±inf) uses CVTTSS2SI, which returns
    // INT32_MIN (0x80000000) for any out-of-range value; replicate here.
    let s = libm::ldexpf(1.0f32, 30 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        let v = s * f;
        *i = if v.is_finite() && (-2_147_483_648.0_f32..2_147_483_648.0_f32).contains(&v) {
            v as i32
        } else {
            i32::MIN
        };
    }
}

/// Forward block-floating-point transform: quantize f64 → i64 relative to exponent `emax`.
pub(crate) fn fwd_cast_f64(iblock: &mut [i64], fblock: &[f64], emax: i32) {
    // Same as fwd_cast_f32: when emax < -961, s overflows f64 to +inf.
    // In C on x86, (int64_t)(±inf) returns INT64_MIN; replicate here.
    let s = libm::ldexp(1.0f64, 62 - emax);
    for (i, f) in iblock.iter_mut().zip(fblock.iter()) {
        let v = s * f;
        *i = if v.is_finite()
            && (-9_223_372_036_854_775_808.0_f64..9_223_372_036_854_775_808.0_f64).contains(&v)
        {
            v as i64
        } else {
            i64::MIN
        };
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
        encode_default: $encode_default:ident,
        rate_params: [$($p:ident: $pty:ty),+] $(,)?
    ) => {
        /// Encode a strided block; return bits written.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        #[cfg(feature = "ffi")]
        pub unsafe fn $full(
            bs: &mut dyn ZfpBitStreamMutOps,
            data: &[$ty],
            $($s: isize,)+
        ) -> usize {
            let block = unsafe { $gather(data, $($s),+) };
            $encode_default(bs, &block)
        }

        /// Encode a partial (boundary) strided block; return bits written.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        #[cfg(feature = "ffi")]
        pub unsafe fn $partial(
            bs: &mut dyn ZfpBitStreamMutOps,
            data: &[$ty],
            $($n: usize,)+
            $($s: isize,)+
        ) -> usize {
            let block = unsafe { $gather_partial(data, $($n,)+ $($s),+) };
            $encode_default(bs, &block)
        }

        /// Encode a strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        pub unsafe fn $full_rate(
            bs: &mut dyn ZfpBitStreamMutOps,
            data: &[$ty],
            $($s: isize,)+
            $($p: $pty,)+
        ) -> usize {
            let block = unsafe { $gather(data, $($s),+) };
            $encode(bs, &block, $($p),+)
        }

        /// Encode a partial strided block with explicit stream parameters.
        ///
        /// # Safety
        /// `data` must be valid for every offset the strides generate. See
        /// [`crate::codec::block`].
        pub unsafe fn $partial_rate(
            bs: &mut dyn ZfpBitStreamMutOps,
            data: &[$ty],
            $($n: usize,)+
            $($s: isize,)+
            $($p: $pty,)+
        ) -> usize {
            let block = unsafe { $gather_partial(data, $($n,)+ $($s),+) };
            $encode(bs, &block, $($p),+)
        }
    };
}
pub(crate) use strided_encode_wrappers;
