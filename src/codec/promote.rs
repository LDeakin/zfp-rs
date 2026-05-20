//! Integer promotion and demotion.
//!
//! Lifts narrow integer types (i8, u8, i16, u16) into the internal i32
//! fixed-point representation used by the codec, and back again.
//!
//! Reference: `zfp/src/zfp.c` (`zfp_promote_*` / `zfp_demote_*` functions).
//!
//! Block element count = 1 << (2 * dims).

use crate::types::ZfpDimensionality;

/// Number of elements in a block of the given dimensionality.
#[inline]
fn block_len(dims: ZfpDimensionality) -> usize {
    1 << (2 * u32::from(dims))
}

/// Promote a block of `i8` values to `i32`.
/// Each value is left-shifted by 23 to occupy the high bits of the i32.
pub fn promote_i8_to_i32(dst: &mut [i32], src: &[i8], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        dst[i] = i32::from(src[i]) << 23;
    }
}

/// Promote a block of `u8` values to `i32`.
/// The unsigned range [0, 255] is mapped to signed [-128, 127] << 23.
pub fn promote_u8_to_i32(dst: &mut [i32], src: &[u8], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        dst[i] = (i32::from(src[i]) - 0x80) << 23;
    }
}

/// Promote a block of `i16` values to `i32`.
/// Each value is left-shifted by 15.
pub fn promote_i16_to_i32(dst: &mut [i32], src: &[i16], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        dst[i] = i32::from(src[i]) << 15;
    }
}

/// Promote a block of `u16` values to `i32`.
/// The unsigned range [0, 65535] is mapped to signed [-32768, 32767] << 15.
pub fn promote_u16_to_i32(dst: &mut [i32], src: &[u16], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        dst[i] = (i32::from(src[i]) - 0x8000) << 15;
    }
}

/// Demote a block of `i32` values back to `i8` (with clamping).
#[allow(clippy::cast_possible_truncation)] // clamped to [-0x80, 0x7f]
pub fn demote_i32_to_i8(dst: &mut [i8], src: &[i32], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        let v = src[i] >> 23;
        dst[i] = v.clamp(-0x80, 0x7f) as i8;
    }
}

/// Demote a block of `i32` values back to `u8` (with clamping).
#[allow(clippy::cast_sign_loss)] // clamped to [0x00, 0xff]
pub fn demote_i32_to_u8(dst: &mut [u8], src: &[i32], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        let v = (src[i] >> 23) + 0x80;
        dst[i] = v.clamp(0x00, 0xff) as u8;
    }
}

/// Demote a block of `i32` values back to `i16` (with clamping).
#[allow(clippy::cast_possible_truncation)] // clamped to [-0x8000, 0x7fff]
pub fn demote_i32_to_i16(dst: &mut [i16], src: &[i32], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        let v = src[i] >> 15;
        dst[i] = v.clamp(-0x8000, 0x7fff) as i16;
    }
}

/// Demote a block of `i32` values back to `u16` (with clamping).
#[allow(clippy::cast_sign_loss)] // clamped to [0x0000, 0xffff]
pub fn demote_i32_to_u16(dst: &mut [u16], src: &[i32], dims: ZfpDimensionality) {
    let count = block_len(dims);
    for i in 0..count {
        let v = (src[i] >> 15) + 0x8000;
        dst[i] = v.clamp(0x0000, 0xffff) as u16;
    }
}
