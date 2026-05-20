//! Promote and demote utilities: C-level wrappers for type conversion.
//!
//! Implements: `zfp_promote_int8_to_int32`, `zfp_promote_uint8_to_int32`,
//! `zfp_promote_int16_to_int32`, `zfp_promote_uint16_to_int32`,
//! `zfp_demote_int32_to_int8`, `zfp_demote_int32_to_uint8`,
//! `zfp_demote_int32_to_int16`, `zfp_demote_int32_to_uint16`.

use crate::abi::uint;
use zfp_rs::ZfpDimensionality;

/// Block element count for the given dimensionality (4^dims).
fn block_len(dims: ZfpDimensionality) -> usize {
    1usize << (2 * usize::from(dims))
}

macro_rules! impl_promote {
    ($fn_name:ident, $src:ty, $dst:ty, $fn:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(oblock: *mut $dst, iblock: *const $src, dims: uint) {
            let Ok(dims) = ZfpDimensionality::try_from(dims) else {
                return;
            };
            if oblock.is_null() || iblock.is_null() {
                return;
            }
            let count = block_len(dims);
            // SAFETY: caller guarantees oblock and iblock point to `count` elements
            let out = std::slice::from_raw_parts_mut(oblock, count);
            let inp = std::slice::from_raw_parts(iblock, count);
            zfp_rs::codec::promote::$fn(out, inp, dims);
        }
    };
}

macro_rules! impl_demote {
    ($fn_name:ident, $dst:ty, $src:ty, $fn:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(oblock: *mut $dst, iblock: *const $src, dims: uint) {
            let Ok(dims) = ZfpDimensionality::try_from(dims) else {
                return;
            };
            if oblock.is_null() || iblock.is_null() {
                return;
            }
            let count = block_len(dims);
            // SAFETY: caller guarantees oblock and iblock point to `count` elements
            let out = std::slice::from_raw_parts_mut(oblock, count);
            let inp = std::slice::from_raw_parts(iblock, count);
            zfp_rs::codec::promote::$fn(out, inp, dims);
        }
    };
}

impl_promote!(zfp_promote_int8_to_int32, i8, i32, promote_i8_to_i32);
impl_promote!(zfp_promote_uint8_to_int32, u8, i32, promote_u8_to_i32);
impl_promote!(zfp_promote_int16_to_int32, i16, i32, promote_i16_to_i32);
impl_promote!(zfp_promote_uint16_to_int32, u16, i32, promote_u16_to_i32);

impl_demote!(zfp_demote_int32_to_int8, i8, i32, demote_i32_to_i8);
impl_demote!(zfp_demote_int32_to_uint8, u8, i32, demote_i32_to_u8);
impl_demote!(zfp_demote_int32_to_int16, i16, i32, demote_i32_to_i16);
impl_demote!(zfp_demote_int32_to_uint16, u16, i32, demote_i32_to_u16);
