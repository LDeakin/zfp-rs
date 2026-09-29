//! Core ZFP types, constants, and the sealed `ZfpScalar` trait.

use bitflags::bitflags;
use std::convert::TryFrom;
use std::fmt;

mod dims;
mod strides;

pub use dims::ZfpDims;
pub use strides::ZfpStrides;

// ---------------------------------------------------------------------------
// Bitstream word type
// ---------------------------------------------------------------------------

/// ZFP bitstream word type.
///
/// Mirrors the C reference implementation's `BIT_STREAM_WORD_TYPE`, which
/// defaults to `uint64` (`unsigned long long`). All bit-level I/O in the
/// library operates on 64-bit word units.
///
/// This is a type alias for `u64`, providing semantic documentation without
/// runtime overhead. Callers can use it to make their code self-documenting
/// when working with raw bitstream words.
///
/// For zero-copy conversion between `Vec<u8>` and `Vec<ZfpBitStreamWord>`,
/// use [`bytemuck`] when the byte buffer is 8-byte aligned. The crate
/// provides [`ZfpBitStream::from_bytes`](crate::bitstream::ZfpBitStream::from_bytes)
/// as a convenience that handles both aligned and unaligned buffers.
pub type ZfpBitStreamWord = u64;

// ---------------------------------------------------------------------------
// Metadata errors
// ---------------------------------------------------------------------------

/// Errors that can occur when encoding field metadata to a 52-bit word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpMetadataError {
    /// The first dimension is zero, or a dimension follows a zero one.
    InvalidDims,
    /// One or more dimensions exceed the maximum encodable range.
    ///
    /// The 52-bit metadata word can represent:
    /// - 1-D: up to 2⁴⁸ elements
    /// - 2-D: up to 2²⁴ × 2²⁴ elements
    /// - 3-D: up to 2¹⁶ × 2¹⁶ × 2¹⁶ elements
    /// - 4-D: up to 2¹² × 2¹² × 2¹² × 2¹² elements
    DimensionTooLarge,
}

impl fmt::Display for ZfpMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpMetadataError::InvalidDims => write!(
                f,
                "invalid field dimensions: dimensions must be nonzero and precede any zero ones"
            ),
            ZfpMetadataError::DimensionTooLarge => {
                write!(
                    f,
                    "field dimension exceeds encodable range for 52-bit metadata"
                )
            }
        }
    }
}

impl std::error::Error for ZfpMetadataError {}

// ---------------------------------------------------------------------------
// Allocation errors
// ---------------------------------------------------------------------------

/// A buffer could not be allocated.
///
/// Returned by the [`ZfpBitStream`][crate::ZfpBitStream] constructors and
/// [`into_bytes`][crate::ZfpBitStream::into_bytes] when the allocator fails,
/// or the size overflows the address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZfpAllocError {
    /// Bytes requested.
    pub bytes: usize,
}

impl fmt::Display for ZfpAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot allocate {} bytes", self.bytes)
    }
}

impl std::error::Error for ZfpAllocError {}

// ---------------------------------------------------------------------------
// Block errors
// ---------------------------------------------------------------------------

/// Errors that can occur during block-level encode/decode.
///
/// Returned by the public block codec functions in
/// [`codec::block`][crate::codec::block].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZfpBlockError;

impl fmt::Display for ZfpBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data slice length does not match expected block size")
    }
}

impl std::error::Error for ZfpBlockError {}

// ---------------------------------------------------------------------------
// Compression/decompression errors
// ---------------------------------------------------------------------------

/// Errors describing a field whose layout does not suit its data buffer.
///
/// Returned by the [`ZfpField`][crate::ZfpField] and
/// [`ZfpFieldMut`][crate::ZfpFieldMut] constructors and setters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpFieldError {
    /// The first dimension is zero, or a dimension follows a zero one.
    InvalidDims {
        /// The dimensions supplied, as `[nx, ny, nz, nw]`.
        dims: [usize; 4],
    },
    /// The logical element count or block grid cannot be represented safely.
    ShapeTooLarge {
        /// The dimensions supplied, as `[nx, ny, nz, nw]`.
        dims: [usize; 4],
    },
    /// The data buffer is smaller than the field's dimensions and strides
    /// require.
    InsufficientData {
        /// Bytes spanned by the field's dimensions and strides. `usize::MAX`
        /// if the span itself overflows `usize`.
        required: usize,
        /// Bytes actually available in the data buffer.
        actual: usize,
    },
    /// The data buffer is not aligned for its scalar type.
    ///
    /// Only reachable via `from_raw`: typed slices are always aligned.
    MisalignedData {
        /// Alignment the scalar type requires, in bytes.
        align: usize,
    },
}

impl fmt::Display for ZfpFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDims { dims } => write!(
                f,
                "invalid field dimensions {dims:?}: dimensions must be nonzero and precede any zero ones"
            ),
            Self::ShapeTooLarge { dims } => {
                write!(
                    f,
                    "field dimensions {dims:?} exceed the supported logical size"
                )
            }
            Self::InsufficientData { required, actual } => write!(
                f,
                "field spans {required} bytes but its data buffer holds only {actual}"
            ),
            Self::MisalignedData { align } => write!(
                f,
                "field data buffer is not {align}-byte aligned for its scalar type"
            ),
        }
    }
}

impl std::error::Error for ZfpFieldError {}

/// Errors that can occur during compression.
///
/// Returned by [`ZfpBitStream::compress`][crate::ZfpBitStream::compress],
/// [`ZfpBitStream::compress_with_execution`][crate::ZfpBitStream::compress_with_execution]
/// and [`ZfpBitStream::write_header`][crate::ZfpBitStream::write_header].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpCompressionError {
    /// The input field is invalid.
    ///
    /// Fields built with the safe constructors are always valid; this is
    /// reachable only from the C ABI.
    Field(ZfpFieldError),
    /// The bitstream is too small to hold the compressed output.
    ///
    /// Writes past the end of the buffer are dropped, so the stream contents
    /// are incomplete. [`ZfpConfig::maximum_size`][crate::ZfpConfig::maximum_size]
    /// gives a capacity that is always sufficient.
    BufferTooSmall {
        /// Bytes the stream needed, from its start.
        required: usize,
        /// Bytes the stream can hold.
        capacity: usize,
    },
    /// Header field metadata could not be encoded.
    Metadata(ZfpMetadataError),
    /// Header mode cannot preserve the compression configuration.
    Config(crate::config::ZfpConfigError),
}

impl From<ZfpFieldError> for ZfpCompressionError {
    fn from(e: ZfpFieldError) -> Self {
        Self::Field(e)
    }
}

impl From<ZfpMetadataError> for ZfpCompressionError {
    fn from(e: ZfpMetadataError) -> Self {
        Self::Metadata(e)
    }
}

impl From<crate::config::ZfpConfigError> for ZfpCompressionError {
    fn from(e: crate::config::ZfpConfigError) -> Self {
        Self::Config(e)
    }
}

impl fmt::Display for ZfpCompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(e) => write!(f, "invalid input field: {e}"),
            Self::BufferTooSmall { required, capacity } => write!(
                f,
                "bitstream needs {required} bytes but holds only {capacity}"
            ),
            Self::Metadata(e) => write!(f, "cannot write header: {e}"),
            Self::Config(e) => write!(f, "cannot write header: {e}"),
        }
    }
}

impl std::error::Error for ZfpCompressionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Field(e) => Some(e),
            Self::Metadata(e) => Some(e),
            Self::Config(e) => Some(e),
            Self::BufferTooSmall { .. } => None,
        }
    }
}

/// Errors that can occur during decompression.
///
/// Returned by [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress] and
/// [`ZfpBitStream::decompress_with_execution`][crate::ZfpBitStream::decompress_with_execution].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpDecompressionError {
    /// The output field is invalid.
    ///
    /// Fields built with the safe constructors are always valid; this is
    /// reachable only from the C ABI.
    Field(ZfpFieldError),
}

impl From<ZfpFieldError> for ZfpDecompressionError {
    fn from(e: ZfpFieldError) -> Self {
        Self::Field(e)
    }
}

impl fmt::Display for ZfpDecompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(e) => write!(f, "invalid output field: {e}"),
        }
    }
}

impl std::error::Error for ZfpDecompressionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Field(e) => Some(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric constants (mirror zfp.h macros)
// ---------------------------------------------------------------------------

/// Smallest `min_bits` for [`ZfpConfig::expert`][crate::ZfpConfig::expert] (`ZFP_MIN_BITS`).
pub const ZFP_MIN_BITS: u32 = 1;
/// Largest bits per block any config needs (`ZFP_MAX_BITS`).
pub const ZFP_MAX_BITS: u32 = 16658;
/// Largest `max_prec` for [`ZfpConfig::expert`][crate::ZfpConfig::expert] (`ZFP_MAX_PREC`).
pub const ZFP_MAX_PREC: u32 = 64;
/// Smallest meaningful `min_exp` for [`ZfpConfig::expert`][crate::ZfpConfig::expert]
/// (`ZFP_MIN_EXP`); a smaller value selects reversible mode.
pub const ZFP_MIN_EXP: i32 = -1074;
/// Bits in the header magic section (`ZFP_MAGIC_BITS`).
pub const ZFP_MAGIC_BITS: u32 = 32;
/// Bits in the header field-metadata section (`ZFP_META_BITS`).
pub const ZFP_META_BITS: u32 = 52;
/// Bits in a short header mode section (`ZFP_MODE_SHORT_BITS`).
pub const ZFP_MODE_SHORT_BITS: u32 = 12;
/// Bits in a long header mode section (`ZFP_MODE_LONG_BITS`).
pub const ZFP_MODE_LONG_BITS: u32 = 64;
/// Largest possible header, in bits (`ZFP_HEADER_MAX_BITS`).
pub const ZFP_HEADER_MAX_BITS: u32 = 148;

// ---------------------------------------------------------------------------
// ZfpDimensionality
// ---------------------------------------------------------------------------

/// Block dimensionality: 1-D through 4-D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ZfpDimensionality {
    D1 = 1,
    D2 = 2,
    D3 = 3,
    D4 = 4,
}

impl ZfpDimensionality {
    /// Return the number of elements in a block for this dimensionality.
    ///
    /// - `D1` → 4
    /// - `D2` → 16
    /// - `D3` → 64
    /// - `D4` → 256
    #[inline]
    #[must_use]
    pub const fn block_size(self) -> usize {
        4usize.pow(self as u32)
    }
}

impl From<ZfpDimensionality> for u32 {
    fn from(d: ZfpDimensionality) -> Self {
        d as u32
    }
}

impl From<ZfpDimensionality> for usize {
    fn from(d: ZfpDimensionality) -> Self {
        d as usize
    }
}

impl TryFrom<u32> for ZfpDimensionality {
    type Error = InvalidDimensionalityError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(ZfpDimensionality::D1),
            2 => Ok(ZfpDimensionality::D2),
            3 => Ok(ZfpDimensionality::D3),
            4 => Ok(ZfpDimensionality::D4),
            _ => Err(InvalidDimensionalityError(v)),
        }
    }
}

// ---------------------------------------------------------------------------
// InvalidDimensionalityError
// ---------------------------------------------------------------------------

/// Error returned when a `u32` cannot be converted to [`ZfpDimensionality`].
///
/// Valid dimensionalities are 1 through 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidDimensionalityError(
    /// The invalid value that was supplied.
    pub u32,
);

impl fmt::Display for InvalidDimensionalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid ZFP dimensionality: {} (expected 1–4)", self.0)
    }
}

impl std::error::Error for InvalidDimensionalityError {}

// ---------------------------------------------------------------------------
// ZfpScalarType
// ---------------------------------------------------------------------------

/// Scalar element type tag: mirrors `zfp_type` in `zfp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZfpScalarType {
    /// `i32` (`zfp_type_int32`).
    I32,
    /// `i64` (`zfp_type_int64`).
    I64,
    /// `f32` (`zfp_type_float`).
    F32,
    /// `f64` (`zfp_type_double`).
    F64,
}

impl ZfpScalarType {
    /// Number of bytes per scalar value.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            ZfpScalarType::I32 | ZfpScalarType::F32 => 4,
            ZfpScalarType::I64 | ZfpScalarType::F64 => 8,
        }
    }

    /// Alignment a buffer of this scalar type requires, in bytes.
    ///
    /// This is the target's alignment for the corresponding Rust type, which
    /// is not always [`size`][Self::size]: 64-bit scalars are 4-byte aligned
    /// on some 32-bit targets. Buffers passed to
    /// [`ZfpField::from_raw`][crate::ZfpField::from_raw] must satisfy this.
    #[must_use]
    pub const fn align(self) -> usize {
        match self {
            ZfpScalarType::I32 => align_of::<i32>(),
            ZfpScalarType::I64 => align_of::<i64>(),
            ZfpScalarType::F32 => align_of::<f32>(),
            ZfpScalarType::F64 => align_of::<f64>(),
        }
    }

    /// Whether `ptr` satisfies this scalar type's alignment.
    #[must_use]
    pub fn is_aligned(self, ptr: *const u8) -> bool {
        ptr.addr().is_multiple_of(self.align())
    }

    /// Number of bits per scalar value (32 or 64).
    #[must_use]
    pub const fn precision(self) -> u32 {
        match self {
            ZfpScalarType::I32 | ZfpScalarType::F32 => 32,
            ZfpScalarType::I64 | ZfpScalarType::F64 => 64,
        }
    }
}

impl fmt::Display for ZfpScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpScalarType::I32 => write!(f, "int32"),
            ZfpScalarType::I64 => write!(f, "int64"),
            ZfpScalarType::F32 => write!(f, "float"),
            ZfpScalarType::F64 => write!(f, "double"),
        }
    }
}

// ---------------------------------------------------------------------------
// ZfpMode
// ---------------------------------------------------------------------------

/// Compression mode: mirrors `zfp_mode` in `zfp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZfpMode {
    /// Invalid / unset.
    Null,
    /// All four expert parameters set manually.
    Expert,
    /// Fixed compressed bits per scalar.
    FixedRate,
    /// Fixed number of uncompressed bits per scalar.
    FixedPrecision,
    /// Absolute error tolerance.
    FixedAccuracy,
    /// Lossless.
    Reversible,
}

impl fmt::Display for ZfpMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpMode::Null => write!(f, "null"),
            ZfpMode::Expert => write!(f, "expert"),
            ZfpMode::FixedRate => write!(f, "fixed-rate"),
            ZfpMode::FixedPrecision => write!(f, "fixed-precision"),
            ZfpMode::FixedAccuracy => write!(f, "fixed-accuracy"),
            ZfpMode::Reversible => write!(f, "reversible"),
        }
    }
}

// ---------------------------------------------------------------------------
// ZfpHeaderMask (bitflags)
// ---------------------------------------------------------------------------

bitflags! {
    /// Header section flags: mirrors `ZFP_HEADER_*` constants in `zfp.h`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ZfpHeaderMask: u32 {
        /// 32-bit magic word.
        const MAGIC = 0x1;
        /// 52-bit field metadata.
        const META  = 0x2;
        /// 12- or 64-bit compression mode.
        const MODE  = 0x4;
        /// All of the above.
        const FULL  = 0x7;
    }
}

// ---------------------------------------------------------------------------
// ZfpScalar sealed trait
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Sealed trait for types ZFP can compress: `i32`, `i64`, `f32`, `f64`.
pub trait ZfpScalar: sealed::Sealed + Copy + Default + bytemuck::Pod + 'static {
    /// The [`ZfpScalarType`] tag for this primitive type.
    const SCALAR_TYPE: ZfpScalarType;
}

impl ZfpScalar for i32 {
    const SCALAR_TYPE: ZfpScalarType = ZfpScalarType::I32;
}

impl ZfpScalar for i64 {
    const SCALAR_TYPE: ZfpScalarType = ZfpScalarType::I64;
}

impl ZfpScalar for f32 {
    const SCALAR_TYPE: ZfpScalarType = ZfpScalarType::F32;
}

impl ZfpScalar for f64 {
    const SCALAR_TYPE: ZfpScalarType = ZfpScalarType::F64;
}
