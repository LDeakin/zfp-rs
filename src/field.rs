#![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
//! `ZfpField` and `ZfpFieldMut`: uncompressed array descriptors.

use crate::types::{
    ZfpDimensionality, ZfpDims, ZfpMetadataError, ZfpScalar, ZfpScalarType, ZfpStrides,
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
            0 => ZfpScalarType::Int32,
            1 => ZfpScalarType::Int64,
            2 => ZfpScalarType::Float,
            _ => ZfpScalarType::Double,
        };
        Some(Self { scalar_type, dims })
    }

    /// Encode this metadata as the 52-bit header word.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpMetadataError::Null`] if all dimensions are zero, or
    /// [`ZfpMetadataError::DimensionTooLarge`] if a dimension exceeds the
    /// encodable range.
    pub fn to_bits(self) -> Result<u64, ZfpMetadataError> {
        field_metadata(self.scalar_type, &self.dims)
    }
}

// ---------------------------------------------------------------------------
// ZfpField: immutable view (for compression and metadata queries)
// ---------------------------------------------------------------------------

/// Immutable view over uncompressed data.
///
/// Equivalent to a read-only `zfp_field` in `zfp.h`.
pub struct ZfpField<'a> {
    data: &'a [u8],
    /// Scalar type of the data.
    scalar_type: ZfpScalarType,
    /// `[nx, ny, nz, nw]`; unused dimensions are 0.
    dims: [usize; 4],
    /// `[sx, sy, sz, sw]`; 0 means contiguous for that axis.
    strides: [isize; 4],
}

/// Generic constructors: accept dimension/stride arrays via `ZfpDims`/`ZfpStrides`.
impl<'a> ZfpField<'a> {
    /// Compute the min and max scalar index offsets spanned by a field.
    ///
    /// Returns `(imin, imax)` where `imin <= 0 <= imax`.
    /// The total number of scalars spanned (including any gaps) is `imax - imin + 1`.
    #[must_use]
    pub fn field_index_span_static(dims: &[usize; 4], strides: &[isize; 4]) -> (isize, isize) {
        field_index_span(dims, strides)
    }

    /// Create a field from a typed slice with the given dimensions.
    ///
    /// Strides default to contiguous layout.
    pub fn new<T: ZfpScalar, D: ZfpDims>(data: &'a [T], dims: D) -> Self {
        Self {
            scalar_type: T::scalar_type(),
            data: bytemuck::cast_slice(data),
            dims: dims.to_array(),
            strides: [0; 4],
        }
    }

    /// Create a strided field from a typed slice.
    pub fn new_strided<T: ZfpScalar, D: ZfpDims, S: ZfpStrides>(
        data: &'a [T],
        dims: D,
        strides: S,
    ) -> Self {
        Self {
            scalar_type: T::scalar_type(),
            data: bytemuck::cast_slice(data),
            dims: dims.to_array(),
            strides: strides.to_array(),
        }
    }

    /// Create a field from a raw byte pointer in a single call.
    ///
    /// This constructor accepts the total byte count directly, avoiding the
    /// intermediate staged descriptor mutation used by the C API and the
    /// redundant `field_index_span` computation that would be required to
    /// compute `size_bytes()`.
    ///
    /// # Safety
    /// `ptr` must point to at least `byte_count` bytes, and the pointed-to
    /// memory must remain valid for the lifetime `'a`.
    #[must_use]
    pub unsafe fn from_raw(
        ptr: *const u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Self {
        if !ptr.is_null() && byte_count > 0 {
            // SAFETY: caller guarantees ptr is valid for byte_count bytes.
            Self {
                scalar_type,
                data: unsafe { std::slice::from_raw_parts(ptr, byte_count) },
                dims,
                strides,
            }
        } else {
            Self {
                scalar_type,
                data: &[],
                dims,
                strides,
            }
        }
    }
}

/// Non-generic impl: all operations that don't depend on the specific type.
impl ZfpField<'_> {
    /// Return the scalar type.
    #[must_use]
    pub fn scalar_type(&self) -> ZfpScalarType {
        self.scalar_type
    }

    /// Set strides from a stride array.
    pub fn set_stride<S: ZfpStrides>(&mut self, strides: S) {
        self.strides = strides.to_array();
    }

    /// Return the 52-bit compact encoding of type + dimensionality + sizes.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpMetadataError::Null`] if the field has no type set, or
    /// [`ZfpMetadataError::DimensionTooLarge`] if any dimension exceeds the
    /// encodable range for a 52-bit metadata word.
    pub fn metadata(&self) -> Result<u64, ZfpMetadataError> {
        field_metadata(self.scalar_type, &self.dims)
    }

    /// Decode a 52-bit metadata word and update this field's type and dimensions.
    /// Returns `false` if the metadata is invalid.
    pub fn set_metadata(&mut self, meta: u64) -> bool {
        if let Some(metadata) = ZfpFieldMetadata::from_bits(meta) {
            self.scalar_type = metadata.scalar_type;
            self.dims = metadata.dims;
            self.strides = [0; 4];
            true
        } else {
            false
        }
    }

    /// Return the number of active dimensions (1–4).
    ///
    /// Returns `ZfpDimensionality::D1` for empty fields (0 active dimensions).
    #[must_use]
    pub fn dimensionality(&self) -> ZfpDimensionality {
        dimensionality(&self.dims)
    }

    /// Return `[nx, ny, nz, nw]`.
    #[must_use]
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    /// Return the total number of scalar elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        num_elements(&self.dims)
    }

    /// Return the number of 4^d compression blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        num_blocks(&self.dims)
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
    /// Returns an empty slice if the field was created with empty input data
    /// or via [`from_raw`][Self::from_raw] with a null pointer or zero byte count.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Raw pointer to the byte at the lowest memory address in the field,
    /// accounting for negative strides.
    #[must_use]
    pub fn begin(&self) -> Option<*const u8> {
        if self.data.is_empty() {
            return None;
        }
        let (imin, _) = self.field_index_span();
        let elem_size = self.scalar_type.size();
        Some(unsafe { self.data.as_ptr().offset(imin * elem_size as isize) })
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
    #[must_use]
    pub fn field_index_span(&self) -> (isize, isize) {
        field_index_span(&self.dims, &self.strides)
    }

    /// Number of bytes spanned by the field, including gaps from non-unit strides.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let (imin, imax) = self.field_index_span();
        let size = self.scalar_type.size();
        (imax - imin + 1) as usize * size
    }

    /// [`size_bytes`][Self::size_bytes] with overflow checking; `None` if the
    /// span does not fit in a `usize`.
    pub(crate) fn checked_size_bytes(&self) -> Option<usize> {
        checked_size_bytes(&self.dims, &self.strides, self.scalar_type)
    }
}

// ---------------------------------------------------------------------------
// ZfpFieldMut: mutable view (for decompression output)
// ---------------------------------------------------------------------------

/// Mutable view over uncompressed data.
///
/// Equivalent to a writable `zfp_field` in `zfp.h`.
///
/// Like `ZfpField`, this is non-generic and stores raw bytes with a `ZfpScalarType`.
pub struct ZfpFieldMut<'a> {
    data: &'a mut [u8],
    scalar_type: ZfpScalarType,
    /// `[nx, ny, nz, nw]`; unused dimensions are 0.
    dims: [usize; 4],
    /// `[sx, sy, sz, sw]`; 0 means contiguous.
    strides: [isize; 4],
}

/// Generic constructors: accept dimension/stride arrays via `ZfpDims`/`ZfpStrides`.
impl<'a> ZfpFieldMut<'a> {
    /// Create a field from a typed mutable slice with the given dimensions.
    ///
    /// Strides default to contiguous layout.
    pub fn new<T: ZfpScalar, D: ZfpDims>(data: &'a mut [T], dims: D) -> Self {
        Self {
            scalar_type: T::scalar_type(),
            data: bytemuck::cast_slice_mut(data),
            dims: dims.to_array(),
            strides: [0; 4],
        }
    }

    /// Create a strided field from a typed mutable slice.
    pub fn new_strided<T: ZfpScalar, D: ZfpDims, S: ZfpStrides>(
        data: &'a mut [T],
        dims: D,
        strides: S,
    ) -> Self {
        Self {
            scalar_type: T::scalar_type(),
            data: bytemuck::cast_slice_mut(data),
            dims: dims.to_array(),
            strides: strides.to_array(),
        }
    }
}

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

// Non-generic impl for ZfpFieldMut.
impl ZfpFieldMut<'_> {
    /// Return the scalar type.
    #[must_use]
    pub fn scalar_type(&self) -> ZfpScalarType {
        self.scalar_type
    }

    /// Return the 52-bit compact encoding of type + dimensionality + sizes.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpMetadataError::Null`] if the field has no type set, or
    /// [`ZfpMetadataError::DimensionTooLarge`] if any dimension exceeds the
    /// encodable range for a 52-bit metadata word.
    pub fn metadata(&self) -> Result<u64, ZfpMetadataError> {
        field_metadata(self.scalar_type, &self.dims)
    }

    /// Decode a 52-bit metadata word and update this field's type and dimensions.
    /// Returns `false` if the metadata is invalid.
    pub fn set_metadata(&mut self, meta: u64) -> bool {
        if let Some(metadata) = ZfpFieldMetadata::from_bits(meta) {
            self.scalar_type = metadata.scalar_type;
            self.dims = metadata.dims;
            self.strides = [0; 4];
            true
        } else {
            false
        }
    }

    /// Return the number of active dimensions (1–4).
    ///
    /// Returns `ZfpDimensionality::D1` for empty fields (0 active dimensions).
    #[must_use]
    pub fn dimensionality(&self) -> ZfpDimensionality {
        dimensionality(&self.dims)
    }

    /// Return `[nx, ny, nz, nw]`.
    #[must_use]
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    /// Return the total number of scalar elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        num_elements(&self.dims)
    }

    /// Return the number of 4^d compression blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        num_blocks(&self.dims)
    }

    /// Return `true` if the data layout is contiguous.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.dims, &self.strides)
    }

    /// Return the effective strides (filling in defaults for zero entries).
    #[must_use]
    pub fn effective_strides(&self) -> [isize; 4] {
        effective_strides(&self.dims, &self.strides)
    }

    /// Return the underlying raw data bytes (read-only view).
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Return the underlying raw data bytes (mutable view).
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Raw pointer to the byte at the lowest memory address in the field,
    /// accounting for negative strides.
    ///
    /// Returns `None` if the field has no data.
    #[must_use]
    pub fn begin(&self) -> Option<*mut u8> {
        if self.data.is_empty() {
            return None;
        }
        let (imin, _) = self.field_index_span();
        let elem_size = self.scalar_type.size();
        // SAFETY: imin is within the allocation since field_index_span is derived
        // from the strides/dims used when constructing the field.
        Some(unsafe {
            self.data
                .as_ptr()
                .cast_mut()
                .offset(imin * elem_size as isize)
        })
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
    #[must_use]
    pub fn field_index_span(&self) -> (isize, isize) {
        field_index_span(&self.dims, &self.strides)
    }

    /// Number of bytes spanned by the field, including gaps from non-unit strides.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let (imin, imax) = self.field_index_span();
        let size = self.scalar_type.size();
        (imax - imin + 1) as usize * size
    }

    /// [`size_bytes`][Self::size_bytes] with overflow checking; `None` if the
    /// span does not fit in a `usize`.
    pub(crate) fn checked_size_bytes(&self) -> Option<usize> {
        checked_size_bytes(&self.dims, &self.strides, self.scalar_type)
    }

    /// Create a mutable field from a raw byte pointer in a single call.
    ///
    /// This constructor accepts the total byte count directly, avoiding the
    /// intermediate staged descriptor mutation used by the C API and the
    /// redundant `field_index_span` computation that would be required to
    /// compute `size_bytes()`.
    ///
    /// # Safety
    /// `ptr` must point to at least `byte_count` bytes, and the pointed-to
    /// memory must remain valid for the lifetime `'a`.
    pub unsafe fn from_raw(
        ptr: *mut u8,
        byte_count: usize,
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        strides: [isize; 4],
    ) -> Self {
        if !ptr.is_null() && byte_count > 0 {
            // SAFETY: caller guarantees ptr is valid for byte_count bytes.
            Self {
                scalar_type,
                data: unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) },
                dims,
                strides,
            }
        } else {
            Self {
                scalar_type,
                data: &mut [],
                dims,
                strides,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Return the dimensionality implied by a dims array.
///
/// Always returns a value in 1..=4 (D1 is used as the default for empty fields).
fn dimensionality(dims: &[usize; 4]) -> ZfpDimensionality {
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

fn num_elements(dims: &[usize; 4]) -> usize {
    dims[0].max(1) * dims[1].max(1) * dims[2].max(1) * dims[3].max(1)
}

fn num_blocks(dims: &[usize; 4]) -> usize {
    let bx = dims[0].div_ceil(4);
    let by = dims[1].div_ceil(4);
    let bz = dims[2].div_ceil(4);
    let bw = dims[3].div_ceil(4);
    match dimensionality(dims) {
        ZfpDimensionality::D1 => bx,
        ZfpDimensionality::D2 => bx * by,
        ZfpDimensionality::D3 => bx * by * bz,
        ZfpDimensionality::D4 => bx * by * bz * bw,
    }
}

fn effective_strides(dims: &[usize; 4], strides: &[isize; 4]) -> [isize; 4] {
    let sx = if strides[0] != 0 { strides[0] } else { 1 };
    let sy = if strides[1] != 0 {
        strides[1]
    } else {
        dims[0] as isize
    };
    let sz = if strides[2] != 0 {
        strides[2]
    } else {
        (dims[0] * dims[1]) as isize
    };
    let sw = if strides[3] != 0 {
        strides[3]
    } else {
        (dims[0] * dims[1] * dims[2]) as isize
    };
    [sx, sy, sz, sw]
}

fn is_contiguous(dims: &[usize; 4], strides: &[isize; 4]) -> bool {
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
fn field_index_span(dims: &[usize; 4], strides: &[isize; 4]) -> (isize, isize) {
    let sx = if strides[0] != 0 { strides[0] } else { 1 };
    let sy = if strides[1] != 0 {
        strides[1]
    } else {
        dims[0] as isize
    };
    let sz = if strides[2] != 0 {
        strides[2]
    } else {
        (dims[0] * dims[1]) as isize
    };
    let sw = if strides[3] != 0 {
        strides[3]
    } else {
        (dims[0] * dims[1] * dims[2]) as isize
    };
    let dx = if dims[0] != 0 {
        sx * (dims[0] as isize - 1)
    } else {
        0
    };
    let dy = if dims[1] != 0 {
        sy * (dims[1] as isize - 1)
    } else {
        0
    };
    let dz = if dims[2] != 0 {
        sz * (dims[2] as isize - 1)
    } else {
        0
    };
    let dw = if dims[3] != 0 {
        sw * (dims[3] as isize - 1)
    } else {
        0
    };
    let imin = dx.min(0) + dy.min(0) + dz.min(0) + dw.min(0);
    let imax = dx.max(0) + dy.max(0) + dz.max(0) + dw.max(0);
    (imin, imax)
}

/// Number of bytes a field with these dimensions and strides spans, using
/// checked arithmetic throughout.
///
/// Returns `None` if the span overflows, which callers treat as "no buffer can
/// possibly be large enough".
pub(crate) fn checked_size_bytes(
    dims: &[usize; 4],
    strides: &[isize; 4],
    scalar_type: ZfpScalarType,
) -> Option<usize> {
    let effective = [
        if strides[0] != 0 { strides[0] } else { 1 },
        if strides[1] != 0 {
            strides[1]
        } else {
            isize::try_from(dims[0]).ok()?
        },
        if strides[2] != 0 {
            strides[2]
        } else {
            isize::try_from(dims[0].checked_mul(dims[1])?).ok()?
        },
        if strides[3] != 0 {
            strides[3]
        } else {
            isize::try_from(dims[0].checked_mul(dims[1])?.checked_mul(dims[2])?).ok()?
        },
    ];

    let mut imin: isize = 0;
    let mut imax: isize = 0;
    for (stride, &dim) in effective.iter().zip(dims.iter()) {
        if dim == 0 {
            continue;
        }
        let extent = stride.checked_mul(isize::try_from(dim).ok()?.checked_sub(1)?)?;
        imin = imin.checked_add(extent.min(0))?;
        imax = imax.checked_add(extent.max(0))?;
    }

    let span = imax.checked_sub(imin)?.checked_add(1)?;
    usize::try_from(span).ok()?.checked_mul(scalar_type.size())
}

/// Compute the 52-bit metadata word for a field with the given type and dims.
///
/// # Errors
///
/// Returns [`ZfpMetadataError::Null`] if dimensionality is 0, or
/// [`ZfpMetadataError::DimensionTooLarge`] if any dimension exceeds the
/// encodable range for a 52-bit metadata word.
pub(crate) fn field_metadata(
    scalar_type: ZfpScalarType,
    dims: &[usize; 4],
) -> Result<u64, ZfpMetadataError> {
    let type_bits = match scalar_type {
        ZfpScalarType::Int32 => 0u64,
        ZfpScalarType::Int64 => 1u64,
        ZfpScalarType::Float => 2u64,
        ZfpScalarType::Double => 3u64,
    };
    if dims == &[0; 4] {
        return Err(ZfpMetadataError::Null);
    }
    let d = dimensionality(dims);
    let mut meta: u64;
    match d {
        ZfpDimensionality::D1 => {
            if (dims[0] - 1) >> 48 != 0 {
                return Err(ZfpMetadataError::DimensionTooLarge);
            }
            meta = (dims[0] - 1) as u64;
        }
        ZfpDimensionality::D2 => {
            if ((dims[0] - 1) >> 24 != 0) || ((dims[1] - 1) >> 24 != 0) {
                return Err(ZfpMetadataError::DimensionTooLarge);
            }
            meta = ((dims[1] - 1) as u64) << 24 | (dims[0] - 1) as u64;
        }
        ZfpDimensionality::D3 => {
            if ((dims[0] - 1) >> 16 != 0)
                || ((dims[1] - 1) >> 16 != 0)
                || ((dims[2] - 1) >> 16 != 0)
            {
                return Err(ZfpMetadataError::DimensionTooLarge);
            }
            meta =
                ((dims[2] - 1) as u64) << 32 | ((dims[1] - 1) as u64) << 16 | (dims[0] - 1) as u64;
        }
        ZfpDimensionality::D4 => {
            if ((dims[0] - 1) >> 12 != 0)
                || ((dims[1] - 1) >> 12 != 0)
                || ((dims[2] - 1) >> 12 != 0)
                || ((dims[3] - 1) >> 12 != 0)
            {
                return Err(ZfpMetadataError::DimensionTooLarge);
            }
            meta = ((dims[3] - 1) as u64) << 36
                | ((dims[2] - 1) as u64) << 24
                | ((dims[1] - 1) as u64) << 12
                | (dims[0] - 1) as u64;
        }
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

    let dims = match d {
        1 => {
            let nx = (meta & 0x0000_ffff_ffff_ffff) as usize + 1;
            [nx, 0, 0, 0]
        }
        2 => {
            let nx = (meta & 0x00ff_ffff) as usize + 1;
            let meta = meta >> 24;
            let ny = (meta & 0x00ff_ffff) as usize + 1;
            [nx, ny, 0, 0]
        }
        3 => {
            let nx = (meta & 0xffff) as usize + 1;
            let meta = meta >> 16;
            let ny = (meta & 0xffff) as usize + 1;
            let meta = meta >> 16;
            let nz = (meta & 0xffff) as usize + 1;
            [nx, ny, nz, 0]
        }
        4 => {
            let nx = (meta & 0xfff) as usize + 1;
            let meta = meta >> 12;
            let ny = (meta & 0xfff) as usize + 1;
            let meta = meta >> 12;
            let nz = (meta & 0xfff) as usize + 1;
            let meta = meta >> 12;
            let nw = (meta & 0xfff) as usize + 1;
            [nx, ny, nz, nw]
        }
        _ => return None,
    };
    Some(dims)
}

#[cfg(test)]
mod tests {
    use super::checked_size_bytes;
    use crate::types::ZfpScalarType;
    use crate::{
        ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut,
        types::{ZfpCompressionError, ZfpDecompressionError},
    };

    #[test]
    fn checked_size_bytes_matches_size_bytes_for_ordinary_fields() {
        let data = [0f64; 64];
        let field = ZfpField::new(&data, [4usize, 4, 4]);
        assert_eq!(field.checked_size_bytes(), Some(field.size_bytes()));

        // Negative and permuted strides still span the same buffer.
        let strided = ZfpField::new_strided(&data, [4usize, 4, 4], [1isize, -4, 16]);
        assert_eq!(strided.checked_size_bytes(), Some(strided.size_bytes()));
    }

    #[test]
    fn checked_size_bytes_reports_overflow_instead_of_wrapping() {
        // The natural stride for the second axis is nx, so the span is
        // nx * ny elements, which overflows isize here.
        assert_eq!(
            checked_size_bytes(
                &[usize::MAX, usize::MAX, 0, 0],
                &[0; 4],
                ZfpScalarType::Double
            ),
            None
        );
    }

    #[test]
    fn compress_rejects_a_field_larger_than_its_buffer() {
        // 1000 declared elements over a 4-element slice: compressing this used
        // to read out of bounds from entirely safe code.
        let data = [0f64; 4];
        let field = ZfpField::new(&data, [1000usize]);
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096);

        assert_eq!(
            bs.compress(&config, &field),
            Err(ZfpCompressionError::InvalidField {
                required: 8000,
                actual: 32,
            })
        );
    }

    #[test]
    fn decompress_rejects_a_field_larger_than_its_buffer() {
        let mut data = [0f64; 4];
        let mut field = ZfpFieldMut::new(&mut data, [1000usize]);
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096);

        assert_eq!(
            bs.decompress(&config, &mut field),
            Err(ZfpDecompressionError::InvalidField {
                required: 8000,
                actual: 32,
            })
        );
    }

    #[test]
    fn strides_that_overrun_the_buffer_are_rejected() {
        let data = [0f64; 16];
        // A 4x4 field with sy = 100 spans 1 + 3*1 + 3*100 = 304 elements.
        let field = ZfpField::new_strided(&data, [4usize, 4], [1isize, 100]);
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096);

        assert_eq!(
            bs.compress(&config, &field),
            Err(ZfpCompressionError::InvalidField {
                required: 304 * 8,
                actual: 128,
            })
        );
    }

    #[test]
    fn exactly_sized_fields_are_accepted() {
        let data = [1.5f64; 64];
        let field = ZfpField::new(&data, [4usize, 4, 4]);
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(config.maximum_size(ZfpScalarType::Double, &[4, 4, 4]));

        let written = bs
            .compress(&config, &field)
            .expect("exact fit must compress");
        assert!(written > 0);
    }
}
