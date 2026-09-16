//! Shared decode utilities: `uint2int`, `inv_order`, bit-plane decoder.
//!
//! Reference: `zfp/src/template/decode.c`, `codecf.c`

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)] // i32→f32 and i64→f64 for reconstruction (intentional loss)

use crate::bitstream::ZfpBitStreamOps;
use crate::codec::encode::core::{PERM_1, PERM_2, PERM_3, PERM_4, precision_f, with_maxbits};
use crate::types::ZfpDimensionality;

// ---------------------------------------------------------------------------
// Negabinary (uint2int) conversion: inverse of int2uint
// ---------------------------------------------------------------------------

/// Map negabinary `u32` → two's-complement `i32`.
///
/// Formula: `(x ^ NBMASK).wrapping_sub(NBMASK) as i32`
#[inline]
pub(crate) fn uint2int_u32(x: u32) -> i32 {
    const NBMASK: u32 = 0xaaaa_aaaa_u32;
    (x ^ NBMASK).wrapping_sub(NBMASK).cast_signed()
}

/// Map negabinary `u64` → two's-complement `i64`.
#[inline]
pub(crate) fn uint2int_u64(x: u64) -> i64 {
    const NBMASK: u64 = 0xaaaa_aaaa_aaaa_aaaa;
    (x ^ NBMASK).wrapping_sub(NBMASK).cast_signed()
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
// Bit-plane decoders (u32, size ≤ 64): rate-constrained
// ---------------------------------------------------------------------------

/// Decode `size ≤ 64` u32 integers from a rate-constrained bitstream.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_few_ints_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut m: u32 = 0;
    let mut k = intprec;
    while bits != 0 {
        m = 0;
        if k <= kmin {
            break;
        }
        k -= 1;
        // Step 1: decode first n bits of bit plane k
        m = n.min(bits);
        bits -= m;
        let mut x = bs.read_bits(m);
        // Step 2: unary RLE decode remainder
        // Mirrors C: `for (; bits && n < size; n++, m=n)`
        while bits != 0 && n < size {
            bits -= 1;
            if bs.read_bit() != 0 {
                // positive group test: scan for next 1-bit
                // Mirrors C inner: `for (; bits && n < size-1; n++) { bits--; if (read_bit()) break; }`
                while bits != 0 && n < size - 1 {
                    bits -= 1;
                    if bs.read_bit() != 0 {
                        break;
                    }
                    n += 1;
                }
                // set bit at found position
                x |= 1u64 << n;
            } else {
                // negative group test: done with bit plane
                m = size;
                break;
            }
            // outer post-increment
            n += 1;
            m = n;
        }
        // Step 3: deposit bit plane from x
        let mut i = 0usize;
        let mut xx = x;
        while xx != 0 {
            data[i] += ((xx & 1) as u32) << k;
            xx >>= 1;
            i += 1;
        }
    }
    let _ = m; // m used only for ROUND_LAST mode
    maxbits - bits
}

/// Decode `size ≤ 64` u64 integers from a rate-constrained bitstream.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_few_ints_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while bits != 0 {
        if k <= kmin {
            break;
        }
        k -= 1;
        let m = n.min(bits);
        bits -= m;
        let mut x = bs.read_bits(m);
        // Mirrors C: `for (; bits && n < size; n++, m=n)`
        while bits != 0 && n < size {
            bits -= 1;
            if bs.read_bit() != 0 {
                // positive group test: scan for next 1-bit
                while bits != 0 && n < size - 1 {
                    bits -= 1;
                    if bs.read_bit() != 0 {
                        break;
                    }
                    n += 1;
                }
                x |= 1u64 << n;
            } else {
                break;
            }
            n += 1;
        }
        let mut i = 0usize;
        let mut xx = x;
        while xx != 0 {
            data[i] += (xx & 1) << k;
            xx >>= 1;
            i += 1;
        }
    }
    maxbits - bits
}

// ---------------------------------------------------------------------------
// Bit-plane decoders (u32, size > 64): rate-constrained
// ---------------------------------------------------------------------------

/// Decode `size > 64` u32 integers from a rate-constrained bitstream.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_many_ints_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while bits != 0 {
        if k <= kmin {
            break;
        }
        k -= 1;
        // Step 1: decode first n individual bits
        let m = n.min(bits);
        bits -= m;
        for d in &mut data[..m as usize] {
            if bs.read_bit() != 0 {
                *d += 1u32 << k;
            }
        }
        // Step 2: unary RLE decode remainder.
        // Mirrors C: `for (; bits && n < size; n++, m = n)`
        // The outer `n++` runs at end of each positive iteration (not on negative break).
        // The inner `for (; bits && n < size - 1; n++)` increments n only if no break.
        'outer_u32: loop {
            if bits == 0 || n >= size {
                break;
            }
            bits -= 1;
            if bs.read_bit() != 0 {
                // positive group test; scan for one-bit
                loop {
                    if bits == 0 || n >= size - 1 {
                        break;
                    }
                    bits -= 1;
                    if bs.read_bit() != 0 {
                        break; // inner break: outer n++ still runs
                    }
                    n += 1; // inner n++ (only if no break)
                }
                data[n as usize] += 1u32 << k;
            } else {
                break 'outer_u32; // negative: outer n++ does NOT run
            }
            n += 1; // outer n++
        }
    }
    maxbits - bits
}

/// Decode `size > 64` u64 integers from a rate-constrained bitstream.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_many_ints_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while bits != 0 {
        if k <= kmin {
            break;
        }
        k -= 1;
        let m = n.min(bits);
        bits -= m;
        for d in &mut data[..m as usize] {
            if bs.read_bit() != 0 {
                *d += 1u64 << k;
            }
        }
        'outer_u64: loop {
            if bits == 0 || n >= size {
                break;
            }
            bits -= 1;
            if bs.read_bit() != 0 {
                loop {
                    if bits == 0 || n >= size - 1 {
                        break;
                    }
                    bits -= 1;
                    if bs.read_bit() != 0 {
                        break;
                    }
                    n += 1;
                }
                data[n as usize] += 1u64 << k;
            } else {
                break 'outer_u64;
            }
            n += 1;
        }
    }
    maxbits - bits
}

// ---------------------------------------------------------------------------
// Variable-rate bit-plane decoders (no maxbits constraint)
// ---------------------------------------------------------------------------

/// Decode `size ≤ 64` u32 integers with no rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_few_ints_prec_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxprec: u32,
    data: &mut [u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.read_pos();

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Step 1: decode first n bits
        let mut x = bs.read_bits(n);
        // Step 2: unary RLE decode remainder
        // Mirrors C: `for (; n < size && read_bit(); x += 1<<n, n++)`
        while n < size && bs.read_bit() != 0 {
            // inner scan: `for (; n < size-1 && !read_bit(); n++)`
            while n < size - 1 && bs.read_bit() == 0 {
                n += 1;
            }
            // outer post-increment: set bit at found position, advance n
            x |= 1u64 << n;
            n += 1;
        }
        // Step 3: deposit bit plane
        let mut i = 0usize;
        let mut xx = x;
        while xx != 0 {
            data[i] += ((xx & 1) as u32) << k;
            xx >>= 1;
            i += 1;
        }
    }
    (bs.read_pos() - start) as u32
}

/// Decode `size ≤ 64` u64 integers with no rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_few_ints_prec_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxprec: u32,
    data: &mut [u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.read_pos();

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while k > kmin {
        k -= 1;
        let mut x = bs.read_bits(n);
        // Mirrors C: `for (; n < size && read_bit(); x += 1<<n, n++)`
        while n < size && bs.read_bit() != 0 {
            // inner scan: `for (; n < size-1 && !read_bit(); n++)`
            while n < size - 1 && bs.read_bit() == 0 {
                n += 1;
            }
            // outer post-increment
            x |= 1u64 << n;
            n += 1;
        }
        let mut i = 0usize;
        let mut xx = x;
        while xx != 0 {
            data[i] += (xx & 1) << k;
            xx >>= 1;
            i += 1;
        }
    }
    (bs.read_pos() - start) as u32
}

/// Decode `size > 64` u32 integers with no rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_many_ints_prec_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxprec: u32,
    data: &mut [u32],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 32;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.read_pos();

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Step 1: decode first n bits directly
        for d in &mut data[..n as usize] {
            if bs.read_bit() != 0 {
                *d += 1u32 << k;
            }
        }
        // Step 2: unary RLE decode
        // Mirrors C: `for (; n < size && read_bit(); data[n] += 1<<k, n++)`
        while n < size && bs.read_bit() != 0 {
            // inner: `for (; n < size-1 && !read_bit(); n++)`
            while n < size - 1 && bs.read_bit() == 0 {
                n += 1;
            }
            // outer post-increment
            data[n as usize] += 1u32 << k;
            n += 1;
        }
    }
    (bs.read_pos() - start) as u32
}

/// Decode `size > 64` u64 integers with no rate constraint.
#[allow(clippy::many_single_char_names)]
pub(crate) fn decode_many_ints_prec_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxprec: u32,
    data: &mut [u64],
) -> u32 {
    let size = data.len() as u32;
    let intprec: u32 = 64;
    let kmin = intprec.saturating_sub(maxprec);
    let start = bs.read_pos();

    for d in data.iter_mut() {
        *d = 0;
    }

    let mut n: u32 = 0;
    let mut k = intprec;
    while k > kmin {
        k -= 1;
        // Step 1: decode first n bits directly
        for d in &mut data[..n as usize] {
            if bs.read_bit() != 0 {
                *d += 1u64 << k;
            }
        }
        // Step 2: unary RLE decode
        // Mirrors C: `for (; n < size && read_bit(); data[n] += 1<<k, n++)`
        while n < size && bs.read_bit() != 0 {
            // inner: `for (; n < size-1 && !read_bit(); n++)`
            while n < size - 1 && bs.read_bit() == 0 {
                n += 1;
            }
            // outer post-increment
            data[n as usize] += 1u64 << k;
            n += 1;
        }
    }
    (bs.read_pos() - start) as u32
}

// ---------------------------------------------------------------------------
// Main dispatch: decode_ints
// ---------------------------------------------------------------------------

/// Decode `data.len()` u32 integers; returns bits read.
pub(crate) fn decode_ints_u32(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u32],
) -> u32 {
    let size = data.len() as u32;
    if with_maxbits(maxbits, maxprec, size) {
        if size <= 64 {
            decode_few_ints_u32(bs, maxbits, maxprec, data)
        } else {
            decode_many_ints_u32(bs, maxbits, maxprec, data)
        }
    } else if size <= 64 {
        decode_few_ints_prec_u32(bs, maxprec, data)
    } else {
        decode_many_ints_prec_u32(bs, maxprec, data)
    }
}

/// Decode `data.len()` u64 integers; returns bits read.
pub(crate) fn decode_ints_u64(
    bs: &mut dyn ZfpBitStreamOps,
    maxbits: u32,
    maxprec: u32,
    data: &mut [u64],
) -> u32 {
    let size = data.len() as u32;
    if with_maxbits(maxbits, maxprec, size) {
        if size <= 64 {
            decode_few_ints_u64(bs, maxbits, maxprec, data)
        } else {
            decode_many_ints_u64(bs, maxbits, maxprec, data)
        }
    } else if size <= 64 {
        decode_few_ints_prec_u64(bs, maxprec, data)
    } else {
        decode_many_ints_prec_u64(bs, maxprec, data)
    }
}

// ---------------------------------------------------------------------------
// Block decode: integer (matches C `decode_block_Int_DIMS`)
// ---------------------------------------------------------------------------

pub(crate) fn decode_block_1d_i32_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i32; 4] {
    let mut ublock = [0u32; 4];
    let bits = decode_ints_u32(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i32; 4];
    inv_order_i32(&ublock, &mut iblock, &PERM_1);
    crate::codec::transform::inv_xform_1d(&mut iblock);
    iblock
}

pub(crate) fn decode_block_1d_i64_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i64; 4] {
    let mut ublock = [0u64; 4];
    let bits = decode_ints_u64(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i64; 4];
    inv_order_i64(&ublock, &mut iblock, &PERM_1);
    crate::codec::transform::inv_xform_1d_i64(&mut iblock);
    iblock
}

pub(crate) fn decode_block_2d_i32_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i32; 16] {
    let mut ublock = [0u32; 16];
    let bits = decode_ints_u32(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i32; 16];
    inv_order_i32(&ublock, &mut iblock, &PERM_2);
    crate::codec::transform::inv_xform_2d(&mut iblock);
    iblock
}

pub(crate) fn decode_block_2d_i64_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i64; 16] {
    let mut ublock = [0u64; 16];
    let bits = decode_ints_u64(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i64; 16];
    inv_order_i64(&ublock, &mut iblock, &PERM_2);
    crate::codec::transform::inv_xform_2d_i64(&mut iblock);
    iblock
}

pub(crate) fn decode_block_3d_i32_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i32; 64] {
    let mut ublock = [0u32; 64];
    let bits = decode_ints_u32(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i32; 64];
    inv_order_i32(&ublock, &mut iblock, &PERM_3);
    crate::codec::transform::inv_xform_3d(&mut iblock);
    iblock
}

pub(crate) fn decode_block_3d_i64_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i64; 64] {
    let mut ublock = [0u64; 64];
    let bits = decode_ints_u64(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i64; 64];
    inv_order_i64(&ublock, &mut iblock, &PERM_3);
    crate::codec::transform::inv_xform_3d_i64(&mut iblock);
    iblock
}

pub(crate) fn decode_block_4d_i32_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i32; 256] {
    let mut ublock = [0u32; 256];
    let bits = decode_ints_u32(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i32; 256];
    inv_order_i32(&ublock, &mut iblock, &PERM_4);
    crate::codec::transform::inv_xform_4d(&mut iblock);
    iblock
}

pub(crate) fn decode_block_4d_i64_core(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
) -> [i64; 256] {
    let mut ublock = [0u64; 256];
    let bits = decode_ints_u64(bs, maxbits, maxprec, &mut ublock);
    if bits < minbits {
        bs.skip((minbits - bits) as usize);
    }
    let mut iblock = [0i64; 256];
    inv_order_i64(&ublock, &mut iblock, &PERM_4);
    crate::codec::transform::inv_xform_4d_i64(&mut iblock);
    iblock
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
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    dims: ZfpDimensionality,
) -> ([f32; N], usize) {
    const EBITS: u32 = 8;
    const EBIAS: i32 = 127;
    let mut fblock = [0f32; N];
    let mut bits: u32 = 1;
    if bs.read_bit() != 0 {
        // block has nonzero values
        bits += EBITS;
        let emax = bs.read_bits(EBITS) as i32 - EBIAS;
        let prec = precision_f(emax, maxprec, minexp, u32::from(dims));
        let remaining_min = minbits.saturating_sub(bits);
        let remaining_max = maxbits.saturating_sub(bits);
        let iblock_bits = match dims {
            ZfpDimensionality::D1 => {
                let iblock = decode_block_1d_i32_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D2 => {
                let iblock = decode_block_2d_i32_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D3 => {
                let iblock = decode_block_3d_i32_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
            ZfpDimensionality::D4 => {
                let iblock = decode_block_4d_i32_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f32(&iblock, &mut fblock, emax);
                iblock.len()
            }
        };
        let _ = iblock_bits;
        bits = maxbits; // consumed up to maxbits
    } else if minbits > bits {
        bs.skip((minbits - bits) as usize);
        bits = minbits;
    }
    (fblock, bits as usize)
}

/// Decode a double block: read exponent, then integer block, then `inv_cast`.
pub(crate) fn decode_double_block<const N: usize>(
    bs: &mut dyn ZfpBitStreamOps,
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
    dims: ZfpDimensionality,
) -> ([f64; N], usize) {
    const EBITS: u32 = 11;
    const EBIAS: i32 = 1023;
    let mut fblock = [0f64; N];
    let mut bits: u32 = 1;
    if bs.read_bit() != 0 {
        bits += EBITS;
        let emax = bs.read_bits(EBITS) as i32 - EBIAS;
        let prec = precision_f(emax, maxprec, minexp, u32::from(dims));
        let remaining_min = minbits.saturating_sub(bits);
        let remaining_max = maxbits.saturating_sub(bits);
        match dims {
            ZfpDimensionality::D1 => {
                let iblock = decode_block_1d_i64_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D2 => {
                let iblock = decode_block_2d_i64_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D3 => {
                let iblock = decode_block_3d_i64_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
            ZfpDimensionality::D4 => {
                let iblock = decode_block_4d_i64_core(bs, remaining_min, remaining_max, prec);
                inv_cast_f64(&iblock, &mut fblock, emax);
            }
        }
        bits = maxbits;
    } else if minbits > bits {
        bs.skip((minbits - bits) as usize);
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
        defaults: [$($d:expr),+],
        rate_params: [$($p:ident: $pty:ty),+] $(,)?
    ) => {
        /// Decode a strided block; return bits read.
        pub fn $full(
            bs: &mut dyn ZfpBitStreamOps,
            data: &mut [$ty],
            $($s: isize,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, $($d),+);
            $scatter(&block, data, $($s),+);
            (bs.read_pos() - before) as usize
        }

        /// Decode a partial (boundary) strided block; return bits read.
        pub fn $partial(
            bs: &mut dyn ZfpBitStreamOps,
            data: &mut [$ty],
            $($n: usize,)+
            $($s: isize,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, $($d),+);
            $scatter_partial(&block, data, $($n,)+ $($s),+);
            (bs.read_pos() - before) as usize
        }

        /// Decode a strided block with explicit stream parameters.
        pub fn $full_rate(
            bs: &mut dyn ZfpBitStreamOps,
            data: &mut [$ty],
            $($s: isize,)+
            $($p: $pty,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, $($p),+);
            $scatter(&block, data, $($s),+);
            (bs.read_pos() - before) as usize
        }

        /// Decode a partial strided block with explicit stream parameters.
        pub fn $partial_rate(
            bs: &mut dyn ZfpBitStreamOps,
            data: &mut [$ty],
            $($n: usize,)+
            $($s: isize,)+
            $($p: $pty,)+
        ) -> usize {
            let before = bs.read_pos();
            let block = $decode(bs, $($p),+);
            $scatter_partial(&block, data, $($n,)+ $($s),+);
            (bs.read_pos() - before) as usize
        }
    };
}
pub(crate) use strided_decode_wrappers;
