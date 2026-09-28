//! Utility functions for FFI pointer validation and type dispatch.

use crate::abi::{
    bitstream, uint, zfp_bool, zfp_false, zfp_mode, zfp_true, zfp_type, zfp_type_zfp_type_double,
    zfp_type_zfp_type_float, zfp_type_zfp_type_int32, zfp_type_zfp_type_int64,
};
use zfp_rs::{ZfpDimensionality, ZfpMode, ZfpScalarType};

/// Check if a `bitstream` pointer is null.
///
/// # Safety
/// `stream` must be a valid pointer or null.
#[inline]
pub unsafe fn is_bitstream_null(stream: *const bitstream) -> zfp_bool {
    if stream.is_null() {
        zfp_false
    } else {
        zfp_true
    }
}

/// Check if a mutable `bitstream` pointer is null.
///
/// # Safety
/// `stream` must be a valid pointer or null.
#[inline]
pub unsafe fn is_bitstream_mut_null(stream: *mut bitstream) -> zfp_bool {
    if stream.is_null() {
        zfp_false
    } else {
        zfp_true
    }
}

/// Convert a `zfp_type` to the corresponding Rust `ZfpScalarType`.
/// Returns `None` for `zfp_type_none` (unsupported).
pub fn zfp_type_to_scalar(ty: zfp_type) -> Option<zfp_rs::ZfpScalarType> {
    match ty {
        zfp_type_zfp_type_int32 => Some(zfp_rs::ZfpScalarType::I32),
        zfp_type_zfp_type_int64 => Some(zfp_rs::ZfpScalarType::I64),
        zfp_type_zfp_type_float => Some(zfp_rs::ZfpScalarType::F32),
        zfp_type_zfp_type_double => Some(zfp_rs::ZfpScalarType::F64),
        _ => None,
    }
}

/// Convert the Rust `ZfpMode` to the C `zfp_mode`.
pub fn rust_mode_to_zfp(mode: ZfpMode) -> zfp_mode {
    match mode {
        ZfpMode::Null => zfp_mode::zfp_mode_null,
        ZfpMode::Expert => zfp_mode::zfp_mode_expert,
        ZfpMode::FixedRate => zfp_mode::zfp_mode_fixed_rate,
        ZfpMode::FixedPrecision => zfp_mode::zfp_mode_fixed_precision,
        ZfpMode::FixedAccuracy => zfp_mode::zfp_mode_fixed_accuracy,
        ZfpMode::Reversible => zfp_mode::zfp_mode_reversible,
    }
}

/// Convert C `zfp_type` to Rust `ZfpScalarType`.
/// Returns `None` for `zfp_type_none` and out-of-range values.
pub fn zfp_type_to_rust_type(ty: zfp_type) -> Option<ZfpScalarType> {
    match ty {
        zfp_type_zfp_type_int32 => Some(ZfpScalarType::I32),
        zfp_type_zfp_type_int64 => Some(ZfpScalarType::I64),
        zfp_type_zfp_type_float => Some(ZfpScalarType::F32),
        zfp_type_zfp_type_double => Some(ZfpScalarType::F64),
        _ => None,
    }
}

/// Convert Rust `ZfpScalarType` to C `zfp_type`.
pub fn rust_type_to_zfp_type(ty: ZfpScalarType) -> zfp_type {
    match ty {
        ZfpScalarType::I32 => zfp_type_zfp_type_int32,
        ZfpScalarType::I64 => zfp_type_zfp_type_int64,
        ZfpScalarType::F32 => zfp_type_zfp_type_float,
        ZfpScalarType::F64 => zfp_type_zfp_type_double,
    }
}

/// Convert C uint dimension to Rust `ZfpDimensionality`.
pub fn c_dims_to_rust(dims: uint) -> Option<ZfpDimensionality> {
    ZfpDimensionality::try_from(dims).ok()
}
