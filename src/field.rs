#![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
//! `ZfpField` and `ZfpFieldMut`: uncompressed array descriptors.

use crate::types::{
    ZfpDimensionality, ZfpDims, ZfpFieldError, ZfpMetadataError, ZfpScalar, ZfpScalarType,
    ZfpStrides,
};
use bytemuck;
use std::fmt;

/// Decoded 52-bit ZFP field metadata.
///
/// This is the structured form of the header metadata word: scalar type plus
/// array dimensions. Strides are not represented in ZFP headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZfpFieldMetadata {
    /// Scalar type stored in the field.
    pub scalar_type: ZfpScalarType,
    /// Field dimensions as `[nx, ny, nz, nw]`; unused dimensions are zero.
    pub dims: [usize; 4],
}

impl ZfpFieldMetadata {
    /// Decode a 52-bit metadata word.
    #[must_use]
    pub fn from_bits(meta: u64) -> Option<Self> {
        let dims = decode_metadata(meta)?;
        let scalar_type = match meta & 0x3 {
            0 => ZfpScalarType::I32,
            1 => ZfpScalarType::I64,
            2 => ZfpScalarType::F32,
            _ => ZfpScalarType::F64,
        };
        Some(Self { scalar_type, dims })
    }

    /// Encode this metadata as the 52-bit header word.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpMetadataError::InvalidDims`] if the first dimension is zero
    /// or a dimension follows a zero one, or
    /// [`ZfpMetadataError::DimensionTooLarge`] if a dimension exceeds the
    /// encodable range.
    pub fn to_bits(self) -> Result<u64, ZfpMetadataError> {
        field_metadata(self.scalar_type, &self.dims)
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Whether `dims` has a nonzero first dimension and no nonzero dimension after
/// a zero one.
pub(crate) fn valid_dims(dims: &[usize; 4]) -> bool {
    let active = dims.iter().take_while(|&&n| n != 0).count();
    active > 0 && dims[active..].iter().all(|&n| n == 0)
}

/// Check that a field's dimensions are well formed and that `data` covers its
/// strided span with the scalar type's alignment.
fn validate(
    scalar_type: ZfpScalarType,
    dims: &[usize; 4],
    strides: &[isize; 4],
    data: &[u8],
) -> Result<(), ZfpFieldError> {
    if !valid_dims(dims) {
        return Err(ZfpFieldError::InvalidDims { dims: *dims });
    }
    if !logical_shape_fits(dims) {
        return Err(ZfpFieldError::ShapeTooLarge { dims: *dims });
    }
    let required = checked_size_bytes(dims, strides, scalar_type).unwrap_or(usize::MAX);
    if data.len() < required {
        return Err(ZfpFieldError::InsufficientData {
            required,
            actual: data.len(),
        });
    }
    if !scalar_type.is_aligned(data.as_ptr()) {
        return Err(ZfpFieldError::MisalignedData {
            align: scalar_type.align(),
        });
    }
    Ok(())
}

/// Borrow `byte_count` bytes at `ptr`, or nothing if `ptr` is null.
///
/// # Safety
/// If `ptr` is non-null, it must be valid for reads of `byte_count` bytes for `'a`.
unsafe fn raw_slice<'a>(ptr: *const u8, byte_count: usize) -> &'a [u8] {
    if ptr.is_null() || byte_count == 0 {
        &[]
    } else {
        // SAFETY: forwarded from the caller.
        unsafe { std::slice::from_raw_parts(ptr, byte_count) }
    }
}

/// Mutable [`raw_slice`].
///
/// # Safety
/// If `ptr` is non-null, it must be valid for reads and writes of `byte_count`
/// bytes, and not otherwise accessed, for `'a`.
unsafe fn raw_slice_mut<'a>(ptr: *mut u8, byte_count: usize) -> &'a mut [u8] {
    if ptr.is_null() || byte_count == 0 {
        &mut []
    } else {
        // SAFETY: forwarded from the caller.
        unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) }
    }
}

/// Accessors shared by [`ZfpField`] and [`ZfpFieldMut`].
macro_rules! impl_field_common {
    ($ty:ident) => {
        impl $ty<'_> {
            /// Return the scalar type.
            #[must_use]
            pub fn scalar_type(&self) -> ZfpScalarType {
                self.scalar_type
            }

            /// Return the scalar type and dimensions, as stored in a ZFP header.
            ///
            /// Encode it with [`ZfpFieldMetadata::to_bits`].
            #[must_use]
            pub fn metadata(&self) -> ZfpFieldMetadata {
                ZfpFieldMetadata {
                    scalar_type: self.scalar_type,
                    dims: self.dims,
                }
            }

            /// Replace the scalar type and dimensions, and reset the strides to
            /// a contiguous layout.
            ///
            /// # Errors
            ///
            /// Returns [`ZfpFieldError`] if the data buffer does not suit the new
            /// layout. The field is unchanged in that case.
            pub fn set_metadata(
                &mut self,
                metadata: ZfpFieldMetadata,
            ) -> Result<(), ZfpFieldError> {
                validate(metadata.scalar_type, &metadata.dims, &[0; 4], self.data)?;
                self.scalar_type = metadata.scalar_type;
                self.dims = metadata.dims;
                self.strides = [0; 4];
                Ok(())
            }

            /// Replace the strides.
            ///
            /// # Errors
            ///
            /// Returns [`ZfpFieldError`] if the data buffer does not cover the new
            /// strided span. The field is unchanged in that case.
            pub fn set_strides<S: ZfpStrides>(&mut self, strides: S) -> Result<(), ZfpFieldError> {
                let strides = strides.to_array();
                validate(self.scalar_type, &self.dims, &strides, self.data)?;
                self.strides = strides;
                Ok(())
            }

            /// Return the number of active dimensions (1–4).
            #[must_use]
            pub fn dimensionality(&self) -> ZfpDimensionality {
                dimensionality(&self.dims)
            }

            /// Return `[nx, ny, nz, nw]`; unused dimensions are zero.
            #[must_use]
            pub fn dims(&self) -> [usize; 4] {
                self.dims
            }

            /// Return the total number of scalar elements. Saturates at
            /// `usize::MAX` for an unchecked field whose count overflows.
            #[must_use]
            pub fn num_elements(&self) -> usize {
                checked_num_elements(&self.dims).unwrap_or(usize::MAX)
            }

            /// Return the number of 4^d compression blocks. Saturates at
            /// `usize::MAX` for an unchecked field whose count overflows.
            #[must_use]
            pub fn num_blocks(&self) -> usize {
                checked_num_blocks(&self.dims).unwrap_or(usize::MAX)
            }

            /// Return `true` if the data layout is contiguous (all strides are 0 or
            /// match the natural row-major layout).
            #[must_use]
            pub fn is_contiguous(&self) -> bool {
                is_contiguous(&self.dims, &self.strides)
            }

            /// Return the effective strides (filling in defaults for zero entries).
            #[must_use]
            pub fn effective_strides(&self) -> [isize; 4] {
                effective_strides(&self.dims, &self.strides)
            }

            /// Return the underlying raw data bytes.
            ///
            /// The buffer starts at the *lowest* address of the strided span: with
            /// a negative stride, the element at index `[0, 0, 0, 0]` sits further
            /// in. This is the opposite of the C `zfp_field`, whose `data` member
            /// points at element `[0, 0, 0, 0]`.
            #[must_use]
            pub fn data(&self) -> &[u8] {
                self.data
            }

            /// Number of bits per scalar value (32 or 64).
            #[must_use]
            pub fn precision(&self) -> u32 {
                self.scalar_type.precision()
            }

            /// Compute the min and max scalar index offsets spanned by the field.
            ///
            /// Returns `(imin, imax)` where `imin <= 0 <= imax`.
            /// The total number of scalars spanned (including any gaps) is `imax - imin + 1`.
            /// For an unchecked field whose span overflows, returns
            /// `(isize::MIN, isize::MAX)`.
            #[must_use]
            pub fn index_span(&self) -> (isize, isize) {
                field_index_span(&self.dims, &self.strides)
            }

            /// Number of bytes spanned by the field, including gaps from non-unit
            /// strides. Saturates at `usize::MAX` for an unchecked field whose
            /// span overflows.
            #[must_use]
            pub fn size_bytes(&self) -> usize {
                self.checked_size_bytes().unwrap_or(usize::MAX)
            }

            /// [`size_bytes`][Self::size_bytes] with overflow checking; `None` if the
            /// span does not fit in a `usize`.
            pub(crate) fn checked_size_bytes(&self) -> Option<usize> {
                checked_size_bytes(&self.dims, &self.strides, self.scalar_type)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ZfpField: immutable view (for compression and metadata queries)
// ---------------------------------------------------------------------------

/// Immutable view over uncompressed data.
///
/// Equivalent to a read-only `zfp_field` in `zfp.h`. The constructors check
/// that the data buffer covers the field, so a `ZfpField` is always valid to
/// compress.
pub struct ZfpField<'a> {
    data: &'a [u8],
    /// Scalar type of the data.
    scalar_type: ZfpScalarType,
    /// `[nx, ny, nz, nw]`; unused dimensions are 0.
    dims: [usize; 4],
    /// `[sx, sy, sz, sw]`; 0 means contiguous for that axis.
    strides: [isize; 4],
}

impl<'a> ZfpField<'a> {
    /// Create a contiguous field from a typed slice with the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpFieldError`] if a dimension is zero or `data` is shorter
    /// than the field.
    pub fn new<T: ZfpScalar, D: ZfpDims>(data: &'a [T], dims: D) -> Result<Self, ZfpFieldError> {
        Self::new_strided(data, dims, [0isize; 4])
    }

    /// Create a strided field from a typed slice.
    ///
    /// `data` must cover the whole strided span, starting at its *lowest*
    /// address: with a negative stride the element at index `[0, 0, 0, 0]`
    /// sits inside the slice rather than at its start. A stride of 0 means
    /// contiguous for that axis.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpFieldError`] if a dimension is zero or `data` does not
    /// cover the strided span.
    pub fn new_strided<T: ZfpScalar, D: ZfpDims, S: ZfpStrides>(
        data: &'a [T],
        dims: D,
        strides: S,
    ) -> Result<Self, ZfpFieldError> {
        let field = Self {
            scalar_type: T::SCALAR_TYPE,
            data: bytemuck::cast_slice(data),
            dims: dims.to_array(),
            strides: strides.to_array(),
        };
        validate(field.scalar_type, &field.dims, &field.strides, field.data)?;
        Ok(field)
    }

    /// Create a field from a raw byte pointer.
    ///
    /// `ptr` addresses the *lowest* address of the strided span, not
    /// necessarily the element at index `[0, 0, 0, 0]`; see
    /// [`data`][Self::data].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpFieldError`] if a dimension is zero, `byte_count` does not
    /// cover the strided span (a null `ptr` counts as zero bytes), or `ptr`
    /// is misaligned for `scalar_type`.
    ///
    /// # Safety
    /// If `ptr` is non-null, it must point to at least `byte_count` readable
    /// bytes that remain valid and unmodified for the lifetime `'a`.
    pub unsafe fn from_raw(
        ptr: *const u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Result<Self, ZfpFieldError> {
        // SAFETY: forwarded from the caller.
        let data = unsafe { raw_slice(ptr, byte_count) };
        validate(scalar_type, &dims, &strides, data)?;
        Ok(Self {
            data,
            scalar_type,
            dims,
            strides,
        })
    }

    /// [`from_raw`][Self::from_raw] without validation, for the C ABI.
    ///
    /// C `zfp_field`s may describe a layout with no data at all. The codec
    /// still validates the field, so compressing an invalid one fails cleanly.
    ///
    /// # Safety
    /// As for [`from_raw`][Self::from_raw].
    #[cfg(feature = "ffi")]
    #[must_use]
    pub unsafe fn from_raw_unchecked(
        ptr: *const u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Self {
        Self {
            // SAFETY: forwarded from the caller.
            data: unsafe { raw_slice(ptr, byte_count) },
            scalar_type,
            dims,
            strides,
        }
    }
}

impl_field_common!(ZfpField);

// ---------------------------------------------------------------------------
// ZfpFieldMut: mutable view (for decompression output)
// ---------------------------------------------------------------------------

/// Mutable view over uncompressed data.
///
/// Equivalent to a writable `zfp_field` in `zfp.h`. The constructors check
/// that the data buffer covers the field, so a `ZfpFieldMut` is always valid
/// to decompress into.
pub struct ZfpFieldMut<'a> {
    data: &'a mut [u8],
    scalar_type: ZfpScalarType,
    /// `[nx, ny, nz, nw]`; unused dimensions are 0.
    dims: [usize; 4],
    /// `[sx, sy, sz, sw]`; 0 means contiguous.
    strides: [isize; 4],
}

impl<'a> ZfpFieldMut<'a> {
    /// Create a contiguous field from a typed mutable slice with the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpFieldError`] if a dimension is zero or `data` is shorter
    /// than the field.
    pub fn new<T: ZfpScalar, D: ZfpDims>(
        data: &'a mut [T],
        dims: D,
    ) -> Result<Self, ZfpFieldError> {
        Self::new_strided(data, dims, [0isize; 4])
    }

    /// Create a strided field from a typed mutable slice.
    ///
    /// `data` must cover the whole strided span, starting at its *lowest*
    /// address; see [`ZfpField::new_strided`].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpFieldError`] if a dimension is zero or `data` does not
    /// cover the strided span.
    pub fn new_strided<T: ZfpScalar, D: ZfpDims, S: ZfpStrides>(
        data: &'a mut [T],
        dims: D,
        strides: S,
    ) -> Result<Self, ZfpFieldError> {
        let field = Self {
            scalar_type: T::SCALAR_TYPE,
            data: bytemuck::cast_slice_mut(data),
            dims: dims.to_array(),
            strides: strides.to_array(),
        };
        validate(field.scalar_type, &field.dims, &field.strides, field.data)?;
        Ok(field)
    }

    /// Create a mutable field from a raw byte pointer.
    ///
    /// `ptr` addresses the *lowest* address of the strided span; see
    /// [`ZfpField::from_raw`].
    ///
    /// # Errors
    ///
    /// As for [`ZfpField::from_raw`].
    ///
    /// # Safety
    /// If `ptr` is non-null, it must point to at least `byte_count` bytes that
    /// are valid for reads and writes, and not otherwise accessed, for the
    /// lifetime `'a`.
    pub unsafe fn from_raw(
        ptr: *mut u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Result<Self, ZfpFieldError> {
        // SAFETY: forwarded from the caller.
        let data = unsafe { raw_slice_mut(ptr, byte_count) };
        validate(scalar_type, &dims, &strides, data)?;
        Ok(Self {
            data,
            scalar_type,
            dims,
            strides,
        })
    }

    /// [`from_raw`][Self::from_raw] without validation, for the C ABI.
    ///
    /// # Safety
    /// As for [`from_raw`][Self::from_raw].
    #[cfg(feature = "ffi")]
    #[must_use]
    pub unsafe fn from_raw_unchecked(
        ptr: *mut u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Self {
        Self {
            // SAFETY: forwarded from the caller.
            data: unsafe { raw_slice_mut(ptr, byte_count) },
            scalar_type,
            dims,
            strides,
        }
    }

    /// Return the underlying raw data bytes (mutable view).
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

impl_field_common!(ZfpFieldMut);

impl fmt::Debug for ZfpField<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZfpField")
            .field("scalar_type", &self.scalar_type)
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl fmt::Debug for ZfpFieldMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZfpFieldMut")
            .field("scalar_type", &self.scalar_type)
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("data_len", &self.data.len())
            .finish()
    }
}

/// Compute the min and max scalar index offsets spanned by a field with these
/// dimensions and strides, as for [`ZfpField::index_span`].
#[cfg(feature = "ffi")]
#[must_use]
pub fn index_span(dims: &[usize; 4], strides: &[isize; 4]) -> (isize, isize) {
    field_index_span(dims, strides)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Return the dimensionality implied by a dims array.
///
/// Always returns a value in 1..=4 (D1 is used as the default for empty fields).
pub(crate) fn dimensionality(dims: &[usize; 4]) -> ZfpDimensionality {
    if dims[1] == 0 {
        ZfpDimensionality::D1
    } else if dims[2] == 0 {
        ZfpDimensionality::D2
    } else if dims[3] == 0 {
        ZfpDimensionality::D3
    } else {
        ZfpDimensionality::D4
    }
}

fn checked_num_elements(dims: &[usize; 4]) -> Option<usize> {
    dims.iter()
        .take(usize::from(dimensionality(dims)))
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
}

pub(crate) fn checked_num_blocks(dims: &[usize; 4]) -> Option<usize> {
    dims.iter()
        .take(usize::from(dimensionality(dims)))
        .try_fold(1usize, |count, &dim| count.checked_mul(dim.div_ceil(4)))
}

/// A logical element index must fit the signed offsets used by the codec.
pub(crate) fn logical_shape_fits(dims: &[usize; 4]) -> bool {
    checked_num_elements(dims).is_some_and(|n| isize::try_from(n).is_ok())
        && checked_num_blocks(dims).is_some()
}

/// The stride an axis takes when its own is 0: the product of the dims below it.
fn natural_stride(dims: &[usize; 4], axis: usize) -> isize {
    dims[..axis]
        .iter()
        .try_fold(1usize, |stride, &dim| stride.checked_mul(dim))
        .and_then(|stride| isize::try_from(stride).ok())
        .unwrap_or(isize::MAX)
}

fn effective_strides(dims: &[usize; 4], strides: &[isize; 4]) -> [isize; 4] {
    std::array::from_fn(|axis| {
        if strides[axis] != 0 {
            strides[axis]
        } else {
            natural_stride(dims, axis)
        }
    })
}

fn is_contiguous(dims: &[usize; 4], strides: &[isize; 4]) -> bool {
    if !logical_shape_fits(dims) {
        return false;
    }
    // All strides zero means contiguous by definition
    if strides.iter().all(|&s| s == 0) {
        return true;
    }
    // Otherwise check each active stride matches the natural layout
    let eff = effective_strides(dims, strides);
    let nat = effective_strides(dims, &[0; 4]);
    let d = dimensionality(dims);
    let active = match d {
        ZfpDimensionality::D1 => 1,
        ZfpDimensionality::D2 => 2,
        ZfpDimensionality::D3 => 3,
        ZfpDimensionality::D4 => 4,
    };
    eff[..active] == nat[..active]
}

/// Compute the min and max scalar index offsets spanned by the field.
///
/// Returns `(imin, imax)` where `imin <= 0 <= imax`.
/// The total number of scalars spanned (including any gaps) is `imax - imin + 1`.
///
/// Only the axes below [`dimensionality`] take part: a dimension declared past
/// the first zero one is inert, so it must not widen the span or shift the
/// origin away from what the block walk uses.
pub(crate) fn field_index_span(dims: &[usize; 4], strides: &[isize; 4]) -> (isize, isize) {
    checked_index_span(dims, strides).unwrap_or((isize::MIN, isize::MAX))
}

/// Overflow-checked index span, including natural strides for zero entries.
#[cfg_attr(not(feature = "ffi"), allow(dead_code))]
#[must_use]
pub fn checked_index_span(dims: &[usize; 4], strides: &[isize; 4]) -> Option<(isize, isize)> {
    let mut imin: isize = 0;
    let mut imax: isize = 0;
    for axis in 0..usize::from(dimensionality(dims)) {
        if dims[axis] == 0 {
            continue;
        }
        let stride = if strides[axis] != 0 {
            strides[axis]
        } else {
            isize::try_from(
                dims[..axis]
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))?,
            )
            .ok()?
        };
        let extent = stride.checked_mul(isize::try_from(dims[axis]).ok()?.checked_sub(1)?)?;
        imin = imin.checked_add(extent.min(0))?;
        imax = imax.checked_add(extent.max(0))?;
    }
    Some((imin, imax))
}

/// Number of bytes a field with these dimensions and strides spans, using
/// checked arithmetic throughout.
///
/// The overflow-checked twin of [`field_index_span`], and it covers the same
/// axes: only those below [`dimensionality`].
///
/// Returns `None` if the span overflows, which callers treat as "no buffer can
/// possibly be large enough".
pub(crate) fn checked_size_bytes(
    dims: &[usize; 4],
    strides: &[isize; 4],
    scalar_type: ZfpScalarType,
) -> Option<usize> {
    let (imin, imax) = checked_index_span(dims, strides)?;
    let span = imax.checked_sub(imin)?.checked_add(1)?;
    usize::try_from(span).ok()?.checked_mul(scalar_type.size())
}

/// Compute the 52-bit metadata word for a field with the given type and dims.
///
/// # Errors
///
/// Returns [`ZfpMetadataError::InvalidDims`] if the dimensions are malformed, or
/// [`ZfpMetadataError::DimensionTooLarge`] if any dimension exceeds the
/// encodable range for a 52-bit metadata word.
pub(crate) fn field_metadata(
    scalar_type: ZfpScalarType,
    dims: &[usize; 4],
) -> Result<u64, ZfpMetadataError> {
    let type_bits = match scalar_type {
        ZfpScalarType::I32 => 0u64,
        ZfpScalarType::I64 => 1u64,
        ZfpScalarType::F32 => 2u64,
        ZfpScalarType::F64 => 3u64,
    };
    if !valid_dims(dims) {
        return Err(ZfpMetadataError::InvalidDims);
    }
    let d = dimensionality(dims);
    let bit_width: u32 = match d {
        ZfpDimensionality::D1 => 48,
        ZfpDimensionality::D2 => 24,
        ZfpDimensionality::D3 => 16,
        ZfpDimensionality::D4 => 12,
    };
    let max_encoded_dimension = (1u64 << bit_width) - 1;
    let mut meta = 0u64;
    let mut shift = 0;
    for &dim in &dims[..usize::from(d)] {
        let encoded = u64::try_from(dim - 1).map_err(|_| ZfpMetadataError::DimensionTooLarge)?;
        if encoded > max_encoded_dimension {
            return Err(ZfpMetadataError::DimensionTooLarge);
        }
        meta |= encoded << shift;
        shift += bit_width;
    }
    // 2 bits for dimensionality (0 = 1D, 1 = 2D, 2 = 3D, 3 = 4D)
    meta = (meta << 2) | (d as u64 - 1);
    // 2 bits for type
    meta = (meta << 2) | type_bits;
    Ok(meta)
}

/// Decode the dims portion of a 52-bit metadata word (type is ignored).
/// Returns `None` if the metadata is out of range.
pub(crate) fn decode_metadata(meta: u64) -> Option<[usize; 4]> {
    if meta >> 52 != 0 {
        return None;
    }
    let meta = meta >> 2;
    let d = ((meta & 0x3) as usize) + 1;
    let meta = meta >> 2;

    let bit_width: u32 = match d {
        1 => 48,
        2 => 24,
        3 => 16,
        4 => 12,
        _ => return None,
    };
    let mask = (1u64 << bit_width) - 1;
    let mut encoded = meta;
    let mut dims = [0; 4];
    for dim in &mut dims[..d] {
        *dim = usize::try_from((encoded & mask) + 1).ok()?;
        encoded >>= bit_width;
    }
    Some(dims)
}

#[cfg(test)]
mod tests {
    use super::checked_size_bytes;
    #[cfg(feature = "ffi")]
    use crate::types::{ZfpCompressionError, ZfpDecompressionError};
    use crate::types::{ZfpFieldError, ZfpMetadataError, ZfpScalarType};
    use crate::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut};

    #[test]
    fn metadata_roundtrips_at_dimension_width_boundaries() {
        let max_one_dim = usize::try_from(1u64 << 48).unwrap_or(usize::MAX);
        for dims in [
            [max_one_dim, 0, 0, 0],
            [1usize << 24, 1usize << 24, 0, 0],
            [1usize << 16, 1usize << 16, 1usize << 16, 0],
            [1usize << 12; 4],
        ] {
            let metadata = super::ZfpFieldMetadata {
                scalar_type: ZfpScalarType::F64,
                dims,
            };
            let bits = metadata.to_bits().unwrap();
            assert_eq!(bits >> 52, 0);
            assert_eq!(super::ZfpFieldMetadata::from_bits(bits), Some(metadata));
        }

        for dims in [
            [(1usize << 24) + 1, 1, 0, 0],
            [(1usize << 16) + 1, 1, 1, 0],
            [(1usize << 12) + 1, 1, 1, 1],
        ] {
            let metadata = super::ZfpFieldMetadata {
                scalar_type: ZfpScalarType::I32,
                dims,
            };
            assert_eq!(metadata.to_bits(), Err(ZfpMetadataError::DimensionTooLarge));
        }
        if let Ok(too_large) = usize::try_from((1u64 << 48) + 1) {
            let metadata = super::ZfpFieldMetadata {
                scalar_type: ZfpScalarType::I32,
                dims: [too_large, 0, 0, 0],
            };
            assert_eq!(metadata.to_bits(), Err(ZfpMetadataError::DimensionTooLarge));
        }
    }

    #[test]
    fn metadata_decode_checks_dimension_after_increment() {
        for dimension in [1u64 << 32, 1u64 << 48] {
            let bits = (dimension - 1) << 4;
            let expected = usize::try_from(dimension)
                .ok()
                .map(|dim| super::ZfpFieldMetadata {
                    scalar_type: ZfpScalarType::I32,
                    dims: [dim, 0, 0, 0],
                });
            assert_eq!(super::ZfpFieldMetadata::from_bits(bits), expected);
        }
    }

    #[test]
    fn checked_size_bytes_matches_size_bytes_for_ordinary_fields() {
        let data = [0f64; 64];
        let field = ZfpField::new(&data, [4usize, 4, 4]).unwrap();
        assert_eq!(field.checked_size_bytes(), Some(field.size_bytes()));

        // Negative and permuted strides still span the same buffer.
        let strided = ZfpField::new_strided(&data, [4usize, 4, 4], [1isize, -4, 16]).unwrap();
        assert_eq!(strided.checked_size_bytes(), Some(strided.size_bytes()));
    }

    #[test]
    fn inert_axes_do_not_widen_the_span() {
        // `dimensionality` stops at the first zero dim, so the C ABI treats
        // this as a 1-D field of 5 elements: `nz` and `sz` are inert. Folding
        // them into the span would demand a 3240-byte buffer and put the
        // field's origin 400 elements above its start, while the block walk
        // reads `data[0..5]`.
        let dims = [5usize, 0, 5, 0];
        let strides = [1isize, 0, -100, 0];
        assert_eq!(super::field_index_span(&dims, &strides), (0, 4));
        assert_eq!(
            checked_size_bytes(&dims, &strides, ZfpScalarType::F64),
            Some(5 * size_of::<f64>())
        );
    }

    #[test]
    fn constructors_reject_malformed_dims() {
        let data = [1.5f64; 5];
        for dims in [[5usize, 0, 5, 0], [0, 5, 0, 0], [0; 4]] {
            assert_eq!(
                ZfpField::new(&data, dims).unwrap_err(),
                ZfpFieldError::InvalidDims { dims }
            );
        }
    }

    #[test]
    fn checked_size_bytes_reports_overflow_instead_of_wrapping() {
        // The natural stride for the second axis is nx, so the span is
        // nx * ny elements, which overflows isize here.
        assert_eq!(
            checked_size_bytes(&[usize::MAX, usize::MAX, 0, 0], &[0; 4], ZfpScalarType::F64),
            None
        );
    }

    #[test]
    fn logical_shape_checks_the_signed_offset_and_block_count_boundaries() {
        let within = [isize::MAX as usize, 1, 0, 0];
        assert!(super::logical_shape_fits(&within));
        assert_eq!(
            super::checked_num_elements(&within),
            Some(isize::MAX as usize)
        );

        let beyond = [isize::MAX as usize + 1, 1, 0, 0];
        assert!(!super::logical_shape_fits(&beyond));
        assert_eq!(super::checked_num_elements(&beyond), Some(beyond[0]));

        let block_overflow = [1usize << 18; 4];
        assert_eq!(super::checked_num_blocks(&block_overflow), None);
        assert_eq!(super::checked_num_elements(&block_overflow), None);
    }

    #[test]
    fn overlapping_field_rejects_an_overflowing_logical_block_grid() {
        let n = 1usize << 18;
        let dims = [n; 4];
        // This covers the full physical span of the reported overlapping
        // layout, while the logical element and block counts overflow usize.
        let mut data = vec![0i32; 4 * (n - 1) + 1];
        let strides = [1isize; 4];
        assert_eq!(
            checked_size_bytes(&dims, &strides, ZfpScalarType::I32),
            Some(data.len() * size_of::<i32>())
        );
        let expected = ZfpFieldError::ShapeTooLarge { dims };
        assert_eq!(
            ZfpField::new_strided(&data, dims, strides).unwrap_err(),
            expected
        );
        assert_eq!(
            ZfpFieldMut::new_strided(&mut data, dims, strides).unwrap_err(),
            expected
        );
    }

    #[test]
    fn setters_reject_unrepresentable_logical_shape_without_changing_field() {
        let data = [0i32; 1];
        let mut field = ZfpField::new(&data, [1usize]).unwrap();
        let dims = [1usize << 18; 4];
        assert_eq!(
            field.set_metadata(super::ZfpFieldMetadata {
                scalar_type: ZfpScalarType::I32,
                dims,
            }),
            Err(ZfpFieldError::ShapeTooLarge { dims })
        );
        assert_eq!(field.dims(), [1, 0, 0, 0]);
        assert_eq!(field.num_elements(), 1);
        assert_eq!(field.num_blocks(), 1);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn unchecked_field_inspectors_saturate_and_codec_rejects_shape() {
        let mut data = [0i32; 1];
        let dims = [1usize << 18; 4];
        // SAFETY: the pointer and count accurately describe `data`; the
        // descriptor itself is intentionally invalid.
        let field = unsafe {
            ZfpField::from_raw_unchecked(
                data.as_ptr().cast(),
                size_of::<i32>(),
                ZfpScalarType::I32,
                dims,
                [1; 4],
            )
        };
        assert_eq!(field.num_elements(), usize::MAX);
        assert_eq!(field.num_blocks(), usize::MAX);
        assert!(!field.is_contiguous());
        assert_eq!(field.size_bytes(), (4 * (dims[0] - 1) + 1) * 4);
        let mut stream = ZfpBitStream::new(8);
        assert_eq!(
            stream.compress(&ZfpConfig::reversible(), &field),
            Err(ZfpCompressionError::Field(ZfpFieldError::ShapeTooLarge {
                dims
            }))
        );
        // SAFETY: the pointer and count accurately describe `data`.
        let mut output = unsafe {
            ZfpFieldMut::from_raw_unchecked(
                data.as_mut_ptr().cast(),
                size_of::<i32>(),
                ZfpScalarType::I32,
                dims,
                [1; 4],
            )
        };
        assert_eq!(
            stream.decompress(&ZfpConfig::reversible(), &mut output),
            Err(ZfpDecompressionError::Field(ZfpFieldError::ShapeTooLarge {
                dims
            }))
        );

        let huge = [isize::MAX as usize + 1, 1, 0, 0];
        // SAFETY: as above.
        let field = unsafe {
            ZfpField::from_raw_unchecked(
                data.as_ptr().cast(),
                size_of::<i32>(),
                ZfpScalarType::I32,
                huge,
                [1, 0, 0, 0],
            )
        };
        assert_eq!(field.effective_strides()[1], isize::MAX);
        assert_eq!(field.index_span(), (isize::MIN, isize::MAX));
        assert_eq!(field.size_bytes(), usize::MAX);
    }

    #[test]
    fn constructors_reject_a_field_larger_than_its_buffer() {
        // 1000 declared elements over a 4-element slice: compressing this used
        // to read out of bounds from entirely safe code.
        let mut data = [0f64; 4];
        let expected = ZfpFieldError::InsufficientData {
            required: 8000,
            actual: 32,
        };
        assert_eq!(ZfpField::new(&data, [1000usize]).unwrap_err(), expected);
        assert_eq!(
            ZfpFieldMut::new(&mut data, [1000usize]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn strides_that_overrun_the_buffer_are_rejected() {
        let data = [0f64; 16];
        // A 4x4 field with sy = 100 spans 1 + 3*1 + 3*100 = 304 elements.
        let expected = ZfpFieldError::InsufficientData {
            required: 304 * 8,
            actual: 128,
        };
        assert_eq!(
            ZfpField::new_strided(&data, [4usize, 4], [1isize, 100]).unwrap_err(),
            expected
        );
        let mut field = ZfpField::new(&data, [4usize, 4]).unwrap();
        assert_eq!(field.set_strides([1isize, 100]).unwrap_err(), expected);
        assert_eq!(field.effective_strides(), [1, 4, 16, 0]);
    }

    #[test]
    fn constructors_reject_misaligned_data() {
        let mut data = [0f64; 3];
        let expected = ZfpFieldError::MisalignedData { align: 8 };
        let ptr = data.as_mut_ptr().cast::<u8>().wrapping_add(4);
        // SAFETY: the 16 bytes at `ptr` are within `data`.
        let field =
            unsafe { ZfpField::from_raw(ptr, 16, ZfpScalarType::F64, [2, 0, 0, 0], [0; 4]) };
        assert_eq!(field.unwrap_err(), expected);
        // SAFETY: as above.
        let field =
            unsafe { ZfpFieldMut::from_raw(ptr, 16, ZfpScalarType::F64, [2, 0, 0, 0], [0; 4]) };
        assert_eq!(field.unwrap_err(), expected);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn codec_rejects_an_unchecked_field_larger_than_its_buffer() {
        let mut data = [0f64; 4];
        let expected = ZfpFieldError::InsufficientData {
            required: 8000,
            actual: 32,
        };
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096);
        // SAFETY: the pointer and length describe `data`.
        let field = unsafe {
            ZfpField::from_raw_unchecked(
                data.as_ptr().cast(),
                32,
                ZfpScalarType::F64,
                [1000, 0, 0, 0],
                [0; 4],
            )
        };
        assert_eq!(
            bs.compress(&config, &field),
            Err(ZfpCompressionError::Field(expected))
        );
        // SAFETY: as above.
        let mut field = unsafe {
            ZfpFieldMut::from_raw_unchecked(
                data.as_mut_ptr().cast(),
                32,
                ZfpScalarType::F64,
                [1000, 0, 0, 0],
                [0; 4],
            )
        };
        assert_eq!(
            bs.decompress(&config, &mut field),
            Err(ZfpDecompressionError::Field(expected))
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn codec_rejects_an_unchecked_misaligned_field() {
        let mut data = [0f64; 3];
        let expected = ZfpFieldError::MisalignedData { align: 8 };
        let ptr = data.as_mut_ptr().cast::<u8>().wrapping_add(4);
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096);
        // SAFETY: the 16 bytes at `ptr` are within `data`.
        let field = unsafe {
            ZfpField::from_raw_unchecked(ptr, 16, ZfpScalarType::F64, [2, 0, 0, 0], [0; 4])
        };
        assert_eq!(
            bs.compress(&config, &field),
            Err(ZfpCompressionError::Field(expected))
        );
        // SAFETY: as above.
        let mut field = unsafe {
            ZfpFieldMut::from_raw_unchecked(ptr, 16, ZfpScalarType::F64, [2, 0, 0, 0], [0; 4])
        };
        assert_eq!(
            bs.decompress(&config, &mut field),
            Err(ZfpDecompressionError::Field(expected))
        );
    }

    #[test]
    fn exactly_sized_fields_are_accepted() {
        let data = [1.5f64; 64];
        let field = ZfpField::new(&data, [4usize, 4, 4]).unwrap();
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(
            config
                .maximum_size(ZfpScalarType::F64, [4usize, 4, 4])
                .unwrap(),
        );

        let written = bs
            .compress(&config, &field)
            .expect("exact fit must compress");
        assert!(written > 0);
    }
}
