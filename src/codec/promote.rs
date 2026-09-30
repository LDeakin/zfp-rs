//! Integer promotion and demotion.
//!
//! Lifts narrow integer types (i8, u8, i16, u16) into the internal i32
//! fixed-point representation used by the codec, and back again.
//!
//! Reference: `zfp/src/zfp.c` (`zfp_promote_*` / `zfp_demote_*` functions).
//!
//! Block element count = 1 << (2 * dims).

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::types::{ZfpBlockError, ZfpDimensionality};

/// Map the first `dims.block_size()` values of `src` into `dst` with `f`.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if either slice is shorter than a block, leaving
/// `dst` unchanged.
fn map_block<S: Copy, D>(
    dst: &mut [D],
    src: &[S],
    dims: ZfpDimensionality,
    f: impl Fn(S) -> D,
) -> Result<(), ZfpBlockError> {
    let n = dims.block_size();
    let (Some(dst), Some(src)) = (dst.get_mut(..n), src.get(..n)) else {
        return Err(ZfpBlockError);
    };
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f(s);
    }
    Ok(())
}

/// Promote a block of `i8` values to `i32`.
/// Each value is left-shifted by 23 to occupy the high bits of the i32.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
pub fn promote_i8_to_i32(
    dst: &mut [i32],
    src: &[i8],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| i32::from(x) << 23)
}

/// Promote a block of `u8` values to `i32`.
/// The unsigned range [0, 255] is mapped to signed [-128, 127] << 23.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the values are 8 or 16 bits, shifted within 32"
)]
pub fn promote_u8_to_i32(
    dst: &mut [i32],
    src: &[u8],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| (i32::from(x) - 0x80) << 23)
}

/// Promote a block of `i16` values to `i32`.
/// Each value is left-shifted by 15.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
pub fn promote_i16_to_i32(
    dst: &mut [i32],
    src: &[i16],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| i32::from(x) << 15)
}

/// Promote a block of `u16` values to `i32`.
/// The unsigned range [0, 65535] is mapped to signed [-32768, 32767] << 15.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the values are 8 or 16 bits, shifted within 32"
)]
pub fn promote_u16_to_i32(
    dst: &mut [i32],
    src: &[u16],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| (i32::from(x) - 0x8000) << 15)
}

/// Demote a block of `i32` values back to `i8` (with clamping).
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[allow(clippy::cast_possible_truncation)] // clamped to [-0x80, 0x7f]
pub fn demote_i32_to_i8(
    dst: &mut [i8],
    src: &[i32],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| (x >> 23).clamp(-0x80, 0x7f) as i8)
}

/// Demote a block of `i32` values back to `u8` (with clamping).
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // clamped to [0x00, 0xff]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the values are 8 or 16 bits, shifted within 32"
)]
pub fn demote_i32_to_u8(
    dst: &mut [u8],
    src: &[i32],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| {
        ((x >> 23) + 0x80).clamp(0x00, 0xff) as u8
    })
}

/// Demote a block of `i32` values back to `i16` (with clamping).
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[allow(clippy::cast_possible_truncation)] // clamped to [-0x8000, 0x7fff]
pub fn demote_i32_to_i16(
    dst: &mut [i16],
    src: &[i32],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| (x >> 15).clamp(-0x8000, 0x7fff) as i16)
}

/// Demote a block of `i32` values back to `u16` (with clamping).
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `dst` or `src` is shorter than a block.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // clamped to [0x0000, 0xffff]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the values are 8 or 16 bits, shifted within 32"
)]
pub fn demote_i32_to_u16(
    dst: &mut [u16],
    src: &[i32],
    dims: ZfpDimensionality,
) -> Result<(), ZfpBlockError> {
    map_block(dst, src, dims, |x| {
        ((x >> 15) + 0x8000).clamp(0x0000, 0xffff) as u16
    })
}

#[cfg(test)]
mod tests {
    use super::{demote_i32_to_u16, promote_i8_to_i32};
    use crate::types::{ZfpBlockError, ZfpDimensionality};

    /// Slices shorter than a block were indexed out of bounds.
    #[test]
    fn short_slices_are_rejected_and_left_unchanged() {
        let mut dst = [7i32; 15];
        assert_eq!(
            promote_i8_to_i32(&mut dst, &[1i8; 16], ZfpDimensionality::D2),
            Err(ZfpBlockError)
        );
        assert_eq!(dst, [7; 15]);
        let mut dst = [7u16; 16];
        assert_eq!(
            demote_i32_to_u16(&mut dst, &[0i32; 3], ZfpDimensionality::D2),
            Err(ZfpBlockError)
        );
        assert_eq!(dst, [7; 16]);
        // Longer slices are fine: only the first block is converted.
        let mut dst = [7i32; 5];
        assert_eq!(
            promote_i8_to_i32(&mut dst, &[1i8; 4], ZfpDimensionality::D1),
            Ok(())
        );
        assert_eq!(dst, [1 << 23, 1 << 23, 1 << 23, 1 << 23, 7]);
    }
}
