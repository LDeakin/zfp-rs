//! Block-level encode/decode entry points.
//!
//! Dispatches to the appropriate encoder/decoder based on scalar type
//! and dimensionality.

#![allow(clippy::too_many_lines)]

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::types::{ZfpBlockError, ZfpDimensionality, ZfpScalar, ZfpScalarType};
use bytemuck;

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
// Strided block encode
// ---------------------------------------------------------------------------

/// Encode a strided 4^d block of scalars; return bits written.
pub fn encode_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
) -> usize {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
            dim1::encode_block_strided_1d_i32(bs, bytemuck::cast_slice::<T, i32>(data), strides[0])
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
            dim1::encode_block_strided_1d_i64(bs, bytemuck::cast_slice::<T, i64>(data), strides[0])
        }
        (ZfpScalarType::Float, ZfpDimensionality::D1) => {
            dim1::encode_block_strided_1d_f32(bs, bytemuck::cast_slice::<T, f32>(data), strides[0])
        }
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            dim1::encode_block_strided_1d_f64(bs, bytemuck::cast_slice::<T, f64>(data), strides[0])
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_f64(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
        ),
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_f64(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_f64(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
    }
}

// ---------------------------------------------------------------------------
// Partial strided block encode
// ---------------------------------------------------------------------------

/// Encode a partial (boundary) strided block; return bits written.
pub fn encode_partial_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
) -> usize {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => dim1::encode_partial_block_strided_1d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => dim1::encode_partial_block_strided_1d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => dim1::encode_partial_block_strided_1d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            dim1::encode_partial_block_strided_1d_f64(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                strides[0],
            )
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::encode_partial_block_strided_2d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::encode_partial_block_strided_2d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::encode_partial_block_strided_2d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            dim2::encode_partial_block_strided_2d_f64(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
            )
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::encode_partial_block_strided_3d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::encode_partial_block_strided_3d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::encode_partial_block_strided_3d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            dim3::encode_partial_block_strided_3d_f64(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
            )
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::encode_partial_block_strided_4d_i32(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::encode_partial_block_strided_4d_i64(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::encode_partial_block_strided_4d_f32(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            dim4::encode_partial_block_strided_4d_f64(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
            )
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
// Strided block decode
// ---------------------------------------------------------------------------

/// Decode a strided 4^d block of scalars; return bits read.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
pub fn decode_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
) -> usize {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_f64(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
        ),
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_f64(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
        ),
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_f64(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
        ),
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_f64(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
    }
}

// ---------------------------------------------------------------------------
// Partial strided block decode
// ---------------------------------------------------------------------------

/// Decode a partial (boundary) strided block; return bits read.
pub fn decode_partial_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
) -> usize {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => dim1::decode_partial_block_strided_1d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => dim1::decode_partial_block_strided_1d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => dim1::decode_partial_block_strided_1d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            lengths[0],
            strides[0],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            dim1::decode_partial_block_strided_1d_f64(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                strides[0],
            )
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::decode_partial_block_strided_2d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::decode_partial_block_strided_2d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::decode_partial_block_strided_2d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            lengths[0],
            lengths[1],
            strides[0],
            strides[1],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            dim2::decode_partial_block_strided_2d_f64(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
            )
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::decode_partial_block_strided_3d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::decode_partial_block_strided_3d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::decode_partial_block_strided_3d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            strides[0],
            strides[1],
            strides[2],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            dim3::decode_partial_block_strided_3d_f64(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
            )
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::decode_partial_block_strided_4d_i32(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::decode_partial_block_strided_4d_i64(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::decode_partial_block_strided_4d_f32(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths[3],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            dim4::decode_partial_block_strided_4d_f64(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
            )
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

// ---------------------------------------------------------------------------
// Reversible gather+encode helpers for compress
// ---------------------------------------------------------------------------

/// Gather a 4^d contiguous block from strided data.
///
/// `dims` is 1–4, `strides` has effective (non-zero) strides.
/// For partial blocks, `lengths` gives the count per dimension (≤ 4), and
/// elements outside the field boundary are padded with the nearest value.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
fn gather_block<T: ZfpScalar>(
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> Vec<T> {
    // Map a block index `i` (0..4) with `n` valid elements to the source data index.
    // Mirrors C's `pad_block`: position 3 always comes from position 0; positions
    // n..2 come from position n-1; positions 0..n are the real data.
    fn pad_idx(i: isize, n: isize) -> isize {
        if i < n {
            i
        } else if i == 3 {
            0
        } else {
            n - 1
        }
    }

    let block_size = 4usize.pow(u32::from(dims));
    let mut block = vec![T::default(); block_size];
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0] as isize;
            let p = data.as_ptr();
            for x in 0..4isize {
                let px = pad_idx(x, lx);
                // SAFETY: px is a valid source index within [0, lx-1].
                block[x as usize] = unsafe { *p.offset(px * sx) };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for y in 0..4isize {
                let py = pad_idx(y, ly);
                for x in 0..4isize {
                    let px = pad_idx(x, lx);
                    // SAFETY: px, py are valid source indices.
                    block[i] = unsafe { *p.offset(px * sx + py * sy) };
                    i += 1;
                }
            }
        }
        ZfpDimensionality::D3 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let lz = lengths[2] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for z in 0..4isize {
                let pz = pad_idx(z, lz);
                for y in 0..4isize {
                    let py = pad_idx(y, ly);
                    for x in 0..4isize {
                        let px = pad_idx(x, lx);
                        // SAFETY: px, py, pz are valid source indices.
                        block[i] = unsafe { *p.offset(px * sx + py * sy + pz * sz) };
                        i += 1;
                    }
                }
            }
        }
        ZfpDimensionality::D4 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let sw = strides[3];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let lz = lengths[2] as isize;
            let lw = lengths[3] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for w in 0..4isize {
                let pw = pad_idx(w, lw);
                for z in 0..4isize {
                    let pz = pad_idx(z, lz);
                    for y in 0..4isize {
                        let py = pad_idx(y, ly);
                        for x in 0..4isize {
                            let px = pad_idx(x, lx);
                            // SAFETY: px, py, pz, pw are valid source indices.
                            block[i] = unsafe { *p.offset(px * sx + py * sy + pz * sz + pw * sw) };
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    block
}

/// Scatter a 4^d contiguous block back into strided data.
///
/// Only the `lengths` elements in each dimension are written; padding elements
/// are discarded.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
fn scatter_block<T: ZfpScalar>(
    block: &[T],
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) {
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0];
            let p = data.as_mut_ptr();
            for (x, &v) in block[..lx].iter().enumerate() {
                unsafe { *p.offset(x as isize * sx) = v };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0];
            let ly = lengths[1];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for y in 0..4 {
                for x in 0..4 {
                    if x < lx && y < ly {
                        unsafe { *p.offset(x as isize * sx + y as isize * sy) = block[i] };
                    }
                    i += 1;
                }
            }
        }
        ZfpDimensionality::D3 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let lx = lengths[0];
            let ly = lengths[1];
            let lz = lengths[2];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        if x < lx && y < ly && z < lz {
                            unsafe {
                                *p.offset(x as isize * sx + y as isize * sy + z as isize * sz) =
                                    block[i];
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
        ZfpDimensionality::D4 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let sw = strides[3];
            let lx = lengths[0];
            let ly = lengths[1];
            let lz = lengths[2];
            let lw = lengths[3];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for w in 0..4 {
                for z in 0..4 {
                    for y in 0..4 {
                        for x in 0..4 {
                            if x < lx && y < ly && z < lz && w < lw {
                                unsafe {
                                    *p.offset(
                                        x as isize * sx
                                            + y as isize * sy
                                            + z as isize * sz
                                            + w as isize * sw,
                                    ) = block[i];
                                }
                            }
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Reversible encode of a (possibly partial) strided 4^d block; return bits written.
///
/// Used by `ZfpBitStream::compress` in reversible mode.
pub fn encode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> usize {
    use crate::codec::encode::reversible::{
        encode_block_reversible_1d_f32, encode_block_reversible_1d_f64,
        encode_block_reversible_1d_i32, encode_block_reversible_1d_i64,
        encode_block_reversible_2d_f32, encode_block_reversible_2d_f64,
        encode_block_reversible_2d_i32, encode_block_reversible_2d_i64,
        encode_block_reversible_3d_f32, encode_block_reversible_3d_f64,
        encode_block_reversible_3d_i32, encode_block_reversible_3d_i64,
        encode_block_reversible_4d_f32, encode_block_reversible_4d_f64,
        encode_block_reversible_4d_i32, encode_block_reversible_4d_i64,
    };

    let block = gather_block(data, dims, strides, lengths);
    match (T::scalar_type(), dims) {
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => encode_block_reversible_1d_i32(
            bs,
            as_typed_block_1d::<T, i32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => encode_block_reversible_1d_i64(
            bs,
            as_typed_block_1d::<T, i64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => encode_block_reversible_1d_f32(
            bs,
            as_typed_block_1d::<T, f32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => encode_block_reversible_1d_f64(
            bs,
            as_typed_block_1d::<T, f64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => encode_block_reversible_2d_i32(
            bs,
            as_typed_block_2d::<T, i32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => encode_block_reversible_2d_i64(
            bs,
            as_typed_block_2d::<T, i64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => encode_block_reversible_2d_f32(
            bs,
            as_typed_block_2d::<T, f32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => encode_block_reversible_2d_f64(
            bs,
            as_typed_block_2d::<T, f64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => encode_block_reversible_3d_i32(
            bs,
            as_typed_block_3d::<T, i32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => encode_block_reversible_3d_i64(
            bs,
            as_typed_block_3d::<T, i64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => encode_block_reversible_3d_f32(
            bs,
            as_typed_block_3d::<T, f32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => encode_block_reversible_3d_f64(
            bs,
            as_typed_block_3d::<T, f64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => encode_block_reversible_4d_i32(
            bs,
            as_typed_block_4d::<T, i32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => encode_block_reversible_4d_i64(
            bs,
            as_typed_block_4d::<T, i64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => encode_block_reversible_4d_f32(
            bs,
            as_typed_block_4d::<T, f32>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => encode_block_reversible_4d_f64(
            bs,
            as_typed_block_4d::<T, f64>(&block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
    }
}

/// Reversible decode of a (possibly partial) strided 4^d block; return bits read.
///
/// Used by `ZfpBitStream::decompress` in reversible mode.
pub fn decode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> usize {
    use crate::codec::decode::reversible::{
        decode_block_reversible_1d_f32, decode_block_reversible_1d_f64,
        decode_block_reversible_1d_i32, decode_block_reversible_1d_i64,
        decode_block_reversible_2d_f32, decode_block_reversible_2d_f64,
        decode_block_reversible_2d_i32, decode_block_reversible_2d_i64,
        decode_block_reversible_3d_f32, decode_block_reversible_3d_f64,
        decode_block_reversible_3d_i32, decode_block_reversible_3d_i64,
        decode_block_reversible_4d_f32, decode_block_reversible_4d_f64,
        decode_block_reversible_4d_i32, decode_block_reversible_4d_i64,
    };

    let block_size = 4usize.pow(u32::from(dims));
    let mut block = vec![T::default(); block_size];
    let bits = match (T::scalar_type(), dims) {
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => decode_block_reversible_1d_i32(
            bs,
            as_typed_block_1d_mut::<T, i32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => decode_block_reversible_1d_i64(
            bs,
            as_typed_block_1d_mut::<T, i64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => decode_block_reversible_1d_f32(
            bs,
            as_typed_block_1d_mut::<T, f32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => decode_block_reversible_1d_f64(
            bs,
            as_typed_block_1d_mut::<T, f64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => decode_block_reversible_2d_i32(
            bs,
            as_typed_block_2d_mut::<T, i32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => decode_block_reversible_2d_i64(
            bs,
            as_typed_block_2d_mut::<T, i64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => decode_block_reversible_2d_f32(
            bs,
            as_typed_block_2d_mut::<T, f32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => decode_block_reversible_2d_f64(
            bs,
            as_typed_block_2d_mut::<T, f64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => decode_block_reversible_3d_i32(
            bs,
            as_typed_block_3d_mut::<T, i32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => decode_block_reversible_3d_i64(
            bs,
            as_typed_block_3d_mut::<T, i64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => decode_block_reversible_3d_f32(
            bs,
            as_typed_block_3d_mut::<T, f32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => decode_block_reversible_3d_f64(
            bs,
            as_typed_block_3d_mut::<T, f64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => decode_block_reversible_4d_i32(
            bs,
            as_typed_block_4d_mut::<T, i32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => decode_block_reversible_4d_i64(
            bs,
            as_typed_block_4d_mut::<T, i64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => decode_block_reversible_4d_f32(
            bs,
            as_typed_block_4d_mut::<T, f32>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => decode_block_reversible_4d_f64(
            bs,
            as_typed_block_4d_mut::<T, f64>(&mut block)
                .unwrap_or_else(|_| unreachable!("iblock size matches block dimensionality")),
        ),
    };

    scatter_block(&block, data, dims, strides, lengths);
    bits
}

// ---------------------------------------------------------------------------
// Parametric strided block encode: used by ZfpBitStream::compress
// ---------------------------------------------------------------------------

/// Encode a strided 4^d block with explicit stream parameters; return bits written.
///
/// Unlike `encode_block_strided`, this uses the caller-supplied `min_bits`,
/// `max_bits`, `max_prec`, and `min_exp` instead of the lossless defaults.
pub fn encode_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => dim1::encode_block_strided_1d_i32_rate(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => dim1::encode_block_strided_1d_i64_rate(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => dim1::encode_block_strided_1d_f32_rate(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => dim1::encode_block_strided_1d_f64_rate(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_i32_rate(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_i64_rate(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_f32_rate(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => dim2::encode_block_strided_2d_f64_rate(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_i32_rate(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_i64_rate(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_f32_rate(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => dim3::encode_block_strided_3d_f64_rate(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_i32_rate(
            bs,
            bytemuck::cast_slice::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_i64_rate(
            bs,
            bytemuck::cast_slice::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_f32_rate(
            bs,
            bytemuck::cast_slice::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => dim4::encode_block_strided_4d_f64_rate(
            bs,
            bytemuck::cast_slice::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
    }
}

/// Encode a partial (boundary) strided block with explicit stream parameters; return bits written.
pub fn encode_partial_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
            dim1::encode_partial_block_strided_1d_i32_rate(
                bs,
                bytemuck::cast_slice::<T, i32>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
            dim1::encode_partial_block_strided_1d_i64_rate(
                bs,
                bytemuck::cast_slice::<T, i64>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D1) => {
            dim1::encode_partial_block_strided_1d_f32_rate(
                bs,
                bytemuck::cast_slice::<T, f32>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            dim1::encode_partial_block_strided_1d_f64_rate(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
            dim2::encode_partial_block_strided_2d_i32_rate(
                bs,
                bytemuck::cast_slice::<T, i32>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
            dim2::encode_partial_block_strided_2d_i64_rate(
                bs,
                bytemuck::cast_slice::<T, i64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D2) => {
            dim2::encode_partial_block_strided_2d_f32_rate(
                bs,
                bytemuck::cast_slice::<T, f32>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            dim2::encode_partial_block_strided_2d_f64_rate(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
            dim3::encode_partial_block_strided_3d_i32_rate(
                bs,
                bytemuck::cast_slice::<T, i32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
            dim3::encode_partial_block_strided_3d_i64_rate(
                bs,
                bytemuck::cast_slice::<T, i64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D3) => {
            dim3::encode_partial_block_strided_3d_f32_rate(
                bs,
                bytemuck::cast_slice::<T, f32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            dim3::encode_partial_block_strided_3d_f64_rate(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
            dim4::encode_partial_block_strided_4d_i32_rate(
                bs,
                bytemuck::cast_slice::<T, i32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
            dim4::encode_partial_block_strided_4d_i64_rate(
                bs,
                bytemuck::cast_slice::<T, i64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D4) => {
            dim4::encode_partial_block_strided_4d_f32_rate(
                bs,
                bytemuck::cast_slice::<T, f32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            dim4::encode_partial_block_strided_4d_f64_rate(
                bs,
                bytemuck::cast_slice::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Parametric strided block decode: used by ZfpBitStream::decompress
// ---------------------------------------------------------------------------

/// Decode a strided 4^d block with explicit stream parameters; return bits read.
///
/// Note: `min_exp` is not passed because the decode path infers it from the
/// encoded exponent bits in the bitstream.
pub fn decode_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_i32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_i64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_f32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D1) => dim1::decode_block_strided_1d_f64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_i32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_i64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_f32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D2) => dim2::decode_block_strided_2d_f64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_i32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_i64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_f32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D3) => dim3::decode_block_strided_3d_f64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_i32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_i64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, i64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
        ),
        (ZfpScalarType::Float, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_f32_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f32>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
        (ZfpScalarType::Double, ZfpDimensionality::D4) => dim4::decode_block_strided_4d_f64_rate(
            bs,
            bytemuck::cast_slice_mut::<T, f64>(data),
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        ),
    }
}

/// Decode a partial (boundary) strided block with explicit stream parameters; return bits read.
pub fn decode_partial_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};
    match (T::scalar_type(), dims) {
        // 1D
        (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
            dim1::decode_partial_block_strided_1d_i32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i32>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
            dim1::decode_partial_block_strided_1d_i64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i64>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D1) => {
            dim1::decode_partial_block_strided_1d_f32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f32>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D1) => {
            dim1::decode_partial_block_strided_1d_f64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                strides[0],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 2D
        (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
            dim2::decode_partial_block_strided_2d_i32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i32>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
            dim2::decode_partial_block_strided_2d_i64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D2) => {
            dim2::decode_partial_block_strided_2d_f32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f32>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D2) => {
            dim2::decode_partial_block_strided_2d_f64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                strides[0],
                strides[1],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 3D
        (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
            dim3::decode_partial_block_strided_3d_i32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
            dim3::decode_partial_block_strided_3d_i64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D3) => {
            dim3::decode_partial_block_strided_3d_f32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D3) => {
            dim3::decode_partial_block_strided_3d_f64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                strides[0],
                strides[1],
                strides[2],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        // 4D
        (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
            dim4::decode_partial_block_strided_4d_i32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
            dim4::decode_partial_block_strided_4d_i64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, i64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
            )
        }
        (ZfpScalarType::Float, ZfpDimensionality::D4) => {
            dim4::decode_partial_block_strided_4d_f32_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f32>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
        (ZfpScalarType::Double, ZfpDimensionality::D4) => {
            dim4::decode_partial_block_strided_4d_f64_rate(
                bs,
                bytemuck::cast_slice_mut::<T, f64>(data),
                lengths[0],
                lengths[1],
                lengths[2],
                lengths[3],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
    }
}
