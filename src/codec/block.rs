//! Block-level encode/decode entry points.
//!
//! Dispatches to the appropriate encoder/decoder based on scalar type
//! and dimensionality.
//!
//! # Preconditions for the `*_strided` entry points
//!
//! [`encode_block`] and [`decode_block`] take a slice and validate its length.
//! The `*_strided` variants cannot: they mirror the C API, where `data` is the
//! block *origin* and the gather/scatter helpers index it as
//! `*data.offset(x*sx + y*sy + z*sz + w*sw)`. Non-unit strides step beyond the
//! block's element count and negative strides step backwards from the origin,
//! so the caller must guarantee that every offset the strides generate is in
//! bounds of the allocation `data` points into. That is why every entry point
//! here is `unsafe`.
//!
//! Note that the `&[T]` these functions currently take does not describe the
//! memory they touch: a slice reference carries provenance over its own
//! elements only, so indexing outside it is undefined behaviour even when the
//! allocation extends that far. A later commit replaces the slice with a raw
//! pointer for this reason.
//!
//! Callers that cannot uphold the precondition should use
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress] and
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress], which
//! validate the field's index span and alignment against its buffer.

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::types::{ZfpBlockError, ZfpDimensionality, ZfpScalar, ZfpScalarType};
mod strided;

// Public only with `ffi`, which the C-ABI layer enables. Without it these stay
// crate-internal: they are unsafe, and the safe whole-field API covers every
// use a Rust caller has.
#[cfg(feature = "ffi")]
pub use strided::*;
#[cfg(not(feature = "ffi"))]
pub(crate) use strided::*;

// ---------------------------------------------------------------------------
// Contiguous block encode
// ---------------------------------------------------------------------------

/// Encode a contiguous 4^d block of scalars; return bits written.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn encode_block<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
            // T == i32 via exhaustive match on ZfpScalar::scalar_type().
            let b: &[i32; 4] = as_typed_block_1d::<T, i32>(data)?;
            Ok(dim1::encode_block_1d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
            // T == i64 via exhaustive match on ZfpScalar::scalar_type().
            let b: &[i64; 4] = as_typed_block_1d::<T, i64>(data)?;
            Ok(dim1::encode_block_1d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D1) => {
            // T == f32 via exhaustive match on ZfpScalar::scalar_type().
            let b: &[f32; 4] = as_typed_block_1d::<T, f32>(data)?;
            Ok(dim1::encode_block_1d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            // T == f64 via exhaustive match on ZfpScalar::scalar_type().
            let b: &[f64; 4] = as_typed_block_1d::<T, f64>(data)?;
            Ok(dim1::encode_block_1d_f64_default(bs, b))
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
            let b: &[i32; 16] = as_typed_block_2d::<T, i32>(data)?;
            Ok(dim2::encode_block_2d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
            let b: &[i64; 16] = as_typed_block_2d::<T, i64>(data)?;
            Ok(dim2::encode_block_2d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D2) => {
            let b: &[f32; 16] = as_typed_block_2d::<T, f32>(data)?;
            Ok(dim2::encode_block_2d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            let b: &[f64; 16] = as_typed_block_2d::<T, f64>(data)?;
            Ok(dim2::encode_block_2d_f64_default(bs, b))
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
            let b: &[i32; 64] = as_typed_block_3d::<T, i32>(data)?;
            Ok(dim3::encode_block_3d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
            let b: &[i64; 64] = as_typed_block_3d::<T, i64>(data)?;
            Ok(dim3::encode_block_3d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D3) => {
            let b: &[f32; 64] = as_typed_block_3d::<T, f32>(data)?;
            Ok(dim3::encode_block_3d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            let b: &[f64; 64] = as_typed_block_3d::<T, f64>(data)?;
            Ok(dim3::encode_block_3d_f64_default(bs, b))
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
            let b: &[i32; 256] = as_typed_block_4d::<T, i32>(data)?;
            Ok(dim4::encode_block_4d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
            let b: &[i64; 256] = as_typed_block_4d::<T, i64>(data)?;
            Ok(dim4::encode_block_4d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D4) => {
            let b: &[f32; 256] = as_typed_block_4d::<T, f32>(data)?;
            Ok(dim4::encode_block_4d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            let b: &[f64; 256] = as_typed_block_4d::<T, f64>(data)?;
            Ok(dim4::encode_block_4d_f64_default(bs, b))
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguous block decode
// ---------------------------------------------------------------------------

/// Decode a contiguous 4^d block of scalars; return bits read.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn decode_block<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
            let b: &mut [i32; 4] = as_typed_block_1d_mut::<T, i32>(data)?;
            Ok(dim1::decode_block_1d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
            let b: &mut [i64; 4] = as_typed_block_1d_mut::<T, i64>(data)?;
            Ok(dim1::decode_block_1d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D1) => {
            let b: &mut [f32; 4] = as_typed_block_1d_mut::<T, f32>(data)?;
            Ok(dim1::decode_block_1d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            let b: &mut [f64; 4] = as_typed_block_1d_mut::<T, f64>(data)?;
            Ok(dim1::decode_block_1d_f64_default(bs, b))
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
            let b: &mut [i32; 16] = as_typed_block_2d_mut::<T, i32>(data)?;
            Ok(dim2::decode_block_2d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
            let b: &mut [i64; 16] = as_typed_block_2d_mut::<T, i64>(data)?;
            Ok(dim2::decode_block_2d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D2) => {
            let b: &mut [f32; 16] = as_typed_block_2d_mut::<T, f32>(data)?;
            Ok(dim2::decode_block_2d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            let b: &mut [f64; 16] = as_typed_block_2d_mut::<T, f64>(data)?;
            Ok(dim2::decode_block_2d_f64_default(bs, b))
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
            let b: &mut [i32; 64] = as_typed_block_3d_mut::<T, i32>(data)?;
            Ok(dim3::decode_block_3d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
            let b: &mut [i64; 64] = as_typed_block_3d_mut::<T, i64>(data)?;
            Ok(dim3::decode_block_3d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D3) => {
            let b: &mut [f32; 64] = as_typed_block_3d_mut::<T, f32>(data)?;
            Ok(dim3::decode_block_3d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            let b: &mut [f64; 64] = as_typed_block_3d_mut::<T, f64>(data)?;
            Ok(dim3::decode_block_3d_f64_default(bs, b))
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
            let b: &mut [i32; 256] = as_typed_block_4d_mut::<T, i32>(data)?;
            Ok(dim4::decode_block_4d_i32_default(bs, b))
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
            let b: &mut [i64; 256] = as_typed_block_4d_mut::<T, i64>(data)?;
            Ok(dim4::decode_block_4d_i64_default(bs, b))
        }
        (ZfpScalarType::Float, ZfpDimensionality::D4) => {
            let b: &mut [f32; 256] = as_typed_block_4d_mut::<T, f32>(data)?;
            Ok(dim4::decode_block_4d_f32_default(bs, b))
        }
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            let b: &mut [f64; 256] = as_typed_block_4d_mut::<T, f64>(data)?;
            Ok(dim4::decode_block_4d_f64_default(bs, b))
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible (lossless) block encode / decode: f32 and f64 only
// ---------------------------------------------------------------------------

/// Reversible (lossless) encode of a contiguous 4^d block of `f32`;
/// return bits written.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn encode_block_reversible_f32(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f32],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::encode::reversible::{
        encode_block_reversible_1d_f32, encode_block_reversible_2d_f32,
        encode_block_reversible_3d_f32, encode_block_reversible_4d_f32,
    };
    match dims {
        ZfpDimensionality::D1 => Ok(encode_block_reversible_1d_f32(
            bs,
            as_block_1d::<f32>(data)?,
        )),
        ZfpDimensionality::D2 => Ok(encode_block_reversible_2d_f32(
            bs,
            as_block_2d::<f32>(data)?,
        )),
        ZfpDimensionality::D3 => Ok(encode_block_reversible_3d_f32(
            bs,
            as_block_3d::<f32>(data)?,
        )),
        ZfpDimensionality::D4 => Ok(encode_block_reversible_4d_f32(
            bs,
            as_block_4d::<f32>(data)?,
        )),
    }
}

/// Reversible (lossless) encode of a contiguous 4^d block of `f64`;
/// return bits written.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn encode_block_reversible_f64(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[f64],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::encode::reversible::{
        encode_block_reversible_1d_f64, encode_block_reversible_2d_f64,
        encode_block_reversible_3d_f64, encode_block_reversible_4d_f64,
    };
    match dims {
        ZfpDimensionality::D1 => Ok(encode_block_reversible_1d_f64(
            bs,
            as_block_1d::<f64>(data)?,
        )),
        ZfpDimensionality::D2 => Ok(encode_block_reversible_2d_f64(
            bs,
            as_block_2d::<f64>(data)?,
        )),
        ZfpDimensionality::D3 => Ok(encode_block_reversible_3d_f64(
            bs,
            as_block_3d::<f64>(data)?,
        )),
        ZfpDimensionality::D4 => Ok(encode_block_reversible_4d_f64(
            bs,
            as_block_4d::<f64>(data)?,
        )),
    }
}

/// Reversible (lossless) decode of a contiguous 4^d block of `f32`;
/// return bits read.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn decode_block_reversible_f32(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [f32],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::decode::reversible::{
        decode_block_reversible_1d_f32, decode_block_reversible_2d_f32,
        decode_block_reversible_3d_f32, decode_block_reversible_4d_f32,
    };
    match dims {
        ZfpDimensionality::D1 => Ok(decode_block_reversible_1d_f32(
            bs,
            as_block_1d_mut::<f32>(data)?,
        )),
        ZfpDimensionality::D2 => Ok(decode_block_reversible_2d_f32(
            bs,
            as_block_2d_mut::<f32>(data)?,
        )),
        ZfpDimensionality::D3 => Ok(decode_block_reversible_3d_f32(
            bs,
            as_block_3d_mut::<f32>(data)?,
        )),
        ZfpDimensionality::D4 => Ok(decode_block_reversible_4d_f32(
            bs,
            as_block_4d_mut::<f32>(data)?,
        )),
    }
}

/// Reversible (lossless) decode of a contiguous 4^d block of `f64`;
/// return bits read.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` does not match the expected block
/// size for the given dimensionality.
pub fn decode_block_reversible_f64(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [f64],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    use crate::codec::decode::reversible::{
        decode_block_reversible_1d_f64, decode_block_reversible_2d_f64,
        decode_block_reversible_3d_f64, decode_block_reversible_4d_f64,
    };
    match dims {
        ZfpDimensionality::D1 => Ok(decode_block_reversible_1d_f64(
            bs,
            as_block_1d_mut::<f64>(data)?,
        )),
        ZfpDimensionality::D2 => Ok(decode_block_reversible_2d_f64(
            bs,
            as_block_2d_mut::<f64>(data)?,
        )),
        ZfpDimensionality::D3 => Ok(decode_block_reversible_3d_f64(
            bs,
            as_block_3d_mut::<f64>(data)?,
        )),
        ZfpDimensionality::D4 => Ok(decode_block_reversible_4d_f64(
            bs,
            as_block_4d_mut::<f64>(data)?,
        )),
    }
}

// ---------------------------------------------------------------------------
// Slice-to-array conversion helpers
// ---------------------------------------------------------------------------

/// Convert a slice to a fixed-size array reference for a 1-D block.
///
/// Convert a slice to a fixed-size array reference for a 1-D block.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 4.
#[inline]
pub(crate) fn as_block_1d<T>(data: &[T]) -> Result<&'_ [T; 4], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_block_1d`].
#[inline]
pub(crate) fn as_block_1d_mut<T>(data: &mut [T]) -> Result<&'_ mut [T; 4], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Convert a slice to a fixed-size array reference for a 2-D block.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 16.
#[inline]
pub(crate) fn as_block_2d<T>(data: &[T]) -> Result<&'_ [T; 16], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_block_2d`].
#[inline]
pub(crate) fn as_block_2d_mut<T>(data: &mut [T]) -> Result<&'_ mut [T; 16], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Convert a slice to a fixed-size array reference for a 3-D block.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 64.
#[inline]
pub(crate) fn as_block_3d<T>(data: &[T]) -> Result<&'_ [T; 64], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_block_3d`].
#[inline]
pub(crate) fn as_block_3d_mut<T>(data: &mut [T]) -> Result<&'_ mut [T; 64], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Convert a slice to a fixed-size array reference for a 4-D block.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 256.
#[inline]
pub(crate) fn as_block_4d<T>(data: &[T]) -> Result<&'_ [T; 256], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_block_4d`].
#[inline]
pub(crate) fn as_block_4d_mut<T>(data: &mut [T]) -> Result<&'_ mut [T; 256], ZfpBlockError> {
    data.try_into().map_err(|_| ZfpBlockError)
}

/// Reinterpret a `&[T]` slice as a `[U; 4]` array via bytemuck casting for 1-D blocks.
///
/// Performs two conversions:
/// 1. `bytemuck::cast_slice` reinterprets `&[T]` → `&[U]`
/// 2. `try_into()` converts `&[U]` → `&[U; 4]`
///
/// # Panics
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 4.
#[inline]
pub(crate) fn as_typed_block_1d<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &[T],
) -> Result<&'_ [U; 4], ZfpBlockError> {
    let slice = bytemuck::cast_slice::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_typed_block_1d`].
#[inline]
pub(crate) fn as_typed_block_1d_mut<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &mut [T],
) -> Result<&'_ mut [U; 4], ZfpBlockError> {
    let slice = bytemuck::cast_slice_mut::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Reinterpret a `&[T]` slice as a `[U; 16]` array via bytemuck casting for 2-D blocks.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 16.
#[inline]
pub(crate) fn as_typed_block_2d<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &[T],
) -> Result<&'_ [U; 16], ZfpBlockError> {
    let slice = bytemuck::cast_slice::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_typed_block_2d`].
#[inline]
pub(crate) fn as_typed_block_2d_mut<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &mut [T],
) -> Result<&'_ mut [U; 16], ZfpBlockError> {
    let slice = bytemuck::cast_slice_mut::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Reinterpret a `&[T]` slice as a `[U; 64]` array via bytemuck casting for 3-D blocks.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 64.
#[inline]
pub(crate) fn as_typed_block_3d<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &[T],
) -> Result<&'_ [U; 64], ZfpBlockError> {
    let slice = bytemuck::cast_slice::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_typed_block_3d`].
#[inline]
pub(crate) fn as_typed_block_3d_mut<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &mut [T],
) -> Result<&'_ mut [U; 64], ZfpBlockError> {
    let slice = bytemuck::cast_slice_mut::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Reinterpret a `&[T]` slice as a `[U; 256]` array via bytemuck casting for 4-D blocks.
///
/// Returns [`ZfpBlockError`] if `data.len()` is not 256.
#[inline]
pub(crate) fn as_typed_block_4d<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &[T],
) -> Result<&'_ [U; 256], ZfpBlockError> {
    let slice = bytemuck::cast_slice::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}

/// Mutable variant of [`as_typed_block_4d`].
#[inline]
pub(crate) fn as_typed_block_4d_mut<T: bytemuck::Pod, U: bytemuck::Pod>(
    data: &mut [T],
) -> Result<&'_ mut [U; 256], ZfpBlockError> {
    let slice = bytemuck::cast_slice_mut::<T, U>(data);
    slice.try_into().map_err(|_| ZfpBlockError)
}
