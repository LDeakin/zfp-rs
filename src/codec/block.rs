//! Block-level encode/decode entry points.
//!
//! Dispatches to the appropriate encoder/decoder based on scalar type
//! and dimensionality.
//!
//! # Preconditions for the `*_strided` entry points
//!
//! [`encode_block`] and [`decode_block`] take a slice and validate its length.
//! The `*_strided` variants take a raw pointer instead, mirroring the C API:
//! `data` points at the block origin, and the gather/scatter helpers index it
//! as `*data.offset(x*sx + y*sy + z*sz + w*sw)`. Non-unit strides step beyond
//! the block's element count and negative strides step backwards from the
//! origin, so no slice could describe the memory they touch — a slice
//! reference would carry provenance over its own elements only, and indexing
//! outside it is undefined behaviour even when the allocation extends that far.
//!
//! The caller must therefore guarantee that every offset the strides generate
//! is in bounds of a single allocation, and that `data`'s provenance covers it
//! — in practice, by deriving `data` from a pointer to the whole buffer.
//! Callers that cannot should use
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress] and
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress], which
//! validate the field's index span and alignment against its buffer and derive
//! the pointer accordingly.

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::ZfpConfig;
use crate::types::{ZfpBlockError, ZfpDimensionality, ZfpScalar};
mod strided;

// Public only with `ffi`, which the C-ABI layer enables. Without it these stay
// crate-internal: they are unsafe, and the safe whole-field API covers every
// use a Rust caller has.
#[cfg(feature = "ffi")]
pub use strided::*;
#[cfg(not(feature = "ffi"))]
pub(crate) use strided::*;

// ---------------------------------------------------------------------------
// Contiguous block encode / decode
// ---------------------------------------------------------------------------

/// The strides of a contiguous 4^d block.
const CONTIGUOUS: [isize; 4] = [1, 4, 16, 64];

/// Encode a contiguous 4^d block of scalars with the given config; return
/// the number of bits written.
///
/// A reversible `config` selects lossless coding. The output matches what
/// [`compress`][crate::ZfpBitStreamMutOps::compress] writes for the same block.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` is not
/// [`dims.block_size()`][ZfpDimensionality::block_size].
pub fn encode_block<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    config: &ZfpConfig,
    data: &[T],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    if data.len() != dims.block_size() {
        return Err(ZfpBlockError);
    }
    // SAFETY: `data` holds a whole block, and contiguous strides address
    // exactly its elements.
    Ok(unsafe { encode_block_strided(bs, data.as_ptr(), dims, &CONTIGUOUS, config) })
}

/// Decode a contiguous 4^d block of scalars with the given config; return
/// the number of bits read.
///
/// `config` must match the one used to encode, rounding included.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if `data.len()` is not
/// [`dims.block_size()`][ZfpDimensionality::block_size].
pub fn decode_block<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    config: &ZfpConfig,
    data: &mut [T],
    dims: ZfpDimensionality,
) -> Result<usize, ZfpBlockError> {
    if data.len() != dims.block_size() {
        return Err(ZfpBlockError);
    }
    // SAFETY: as in `encode_block`.
    Ok(unsafe { decode_block_strided(bs, data.as_mut_ptr(), dims, &CONTIGUOUS, config) })
}

// ---------------------------------------------------------------------------
// Slice-to-array conversion helpers
// ---------------------------------------------------------------------------

/// Reinterpret a `&[T]` slice as a `[U; 4]` array via bytemuck casting for 1-D blocks.
///
/// Performs two conversions:
/// 1. `bytemuck::cast_slice` reinterprets `&[T]` → `&[U]`
/// 2. `try_into()` converts `&[U]` → `&[U; 4]`
///
/// # Errors
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

#[cfg(test)]
mod tests {
    use super::{decode_block, encode_block};
    use crate::{
        ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField, ZfpFieldMut, ZfpRounding, ZfpScalar,
        ZfpScalarType, ZfpStreamAlignment,
    };

    /// Block coding must match compressing a field that is exactly one block.
    fn check_matches_field<T: ZfpScalar>(data: &[T], dims: ZfpDimensionality, config: &ZfpConfig) {
        let n = dims.block_size();
        let d = usize::from(dims);
        let field_dims: [usize; 4] = std::array::from_fn(|axis| if axis < d { 4 } else { 0 });

        let field = ZfpField::new(&data[..n], field_dims).unwrap();
        let mut whole = ZfpBitStream::new(1 << 16);
        whole.compress(config, &field).unwrap();

        let mut block = ZfpBitStream::new(1 << 16);
        let written = encode_block(&mut block, config, &data[..n], dims).unwrap();
        assert_eq!(written as u64, block.write_pos());
        block.flush();
        assert_eq!(block.as_bytes(), whole.as_bytes(), "{config:?} {dims:?}");

        let mut from_field = vec![T::default(); n];
        whole.rewind();
        whole
            .decompress(
                config,
                &mut ZfpFieldMut::new(&mut from_field, field_dims).unwrap(),
            )
            .unwrap();
        let mut from_block = vec![T::default(); n];
        block.rewind();
        let read = decode_block(&mut block, config, &mut from_block, dims).unwrap();
        assert_eq!(read, written);
        assert_eq!(
            bytemuck::cast_slice::<T, u8>(&from_block),
            bytemuck::cast_slice::<T, u8>(&from_field)
        );
    }

    fn check_type<T: ZfpScalar>(data: &[T]) {
        for dims in [
            ZfpDimensionality::D1,
            ZfpDimensionality::D2,
            ZfpDimensionality::D3,
            ZfpDimensionality::D4,
        ] {
            for config in [
                ZfpConfig::fixed_rate(9.0, T::SCALAR_TYPE, dims, ZfpStreamAlignment::Unaligned),
                ZfpConfig::fixed_precision(19),
                ZfpConfig::fixed_accuracy(1e-3),
                ZfpConfig::reversible(),
                ZfpConfig::expert(100, 2000, 40, -30),
                ZfpConfig::fixed_accuracy(1e-3)
                    .with_rounding(ZfpRounding::First { tight_error: true }),
            ] {
                check_matches_field(data, dims, &config);
            }
        }
    }

    #[test]
    fn block_coding_matches_single_block_field_coding() {
        let f: Vec<f64> = (0..256)
            .map(|i| (f64::from(i) * 0.37).sin() * 1e3)
            .collect();
        check_type(&f);
        #[allow(clippy::cast_possible_truncation, reason = "test data")]
        check_type(&f.iter().map(|&x| x as f32).collect::<Vec<_>>());
        #[allow(clippy::cast_possible_truncation, reason = "test data")]
        check_type(&f.iter().map(|&x| x as i32).collect::<Vec<_>>());
        #[allow(clippy::cast_possible_truncation, reason = "test data")]
        check_type(&f.iter().map(|&x| (x * 1e6) as i64).collect::<Vec<_>>());
    }

    /// Each block of a truncated fixed-rate stream still reads its whole
    /// budget: reads past the end yield zeros, and skips there keep their
    /// offset. Skips were once clamped to the buffer, which moved the cursor
    /// backwards and underflowed the bit count.
    #[test]
    fn truncated_fixed_rate_blocks_read_their_budget() {
        let config = ZfpConfig::fixed_rate(
            8.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        );
        let data: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(1024);
        bs.compress(&config, &ZfpField::new(&data, [8usize, 8]).unwrap())
            .unwrap();
        // One word of four blocks' worth, so most blocks start past the end.
        let mut truncated = ZfpBitStream::from_bytes(&bs.as_bytes()[..8]);
        let mut block = [0f64; 16];
        for i in 1..=4 {
            let read =
                decode_block(&mut truncated, &config, &mut block, ZfpDimensionality::D2).unwrap();
            assert_eq!(read, 128);
            assert_eq!(truncated.read_pos(), 128 * i);
        }
    }

    #[test]
    fn block_coding_rejects_wrong_lengths() {
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(1024);
        assert!(encode_block(&mut bs, &config, &[0f32; 15], ZfpDimensionality::D2).is_err());
        assert!(decode_block(&mut bs, &config, &mut [0f32; 17], ZfpDimensionality::D2).is_err());
        assert_eq!(bs.write_pos(), 0);
    }
}
