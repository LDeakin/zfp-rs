//! `ZfpConfig`: holds compression parameters only.
//!
//! A `ZfpConfig` is an immutable, [`Copy`] struct holding four expert
//! parameters and a rounding mode. It does **not** own a bitstream: that is managed separately
//! by the caller, mirroring the C API where the bitstream is externally
//! allocated.
//!
//! Compression and decompression are performed via methods on
//! [`ZfpBitStream`][crate::ZfpBitStream]
//! that take `&ZfpConfig`.

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::types::{
    ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_BITS, ZFP_MIN_EXP, ZfpDimensionality, ZfpDims, ZfpMode,
    ZfpScalarType,
};
use std::fmt;

/// Errors when encoding a configuration for a stream header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpConfigError {
    /// Expert parameters do not describe a valid codec configuration.
    InvalidParameters,
    /// The requested fixed rate is negative or non-finite, or gives a block
    /// budget of zero bits or above [`ZFP_MAX_BITS`].
    InvalidRate,
    /// The mode word would change at least one expert parameter.
    UnrepresentableMode,
}

impl fmt::Display for ZfpConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters => write!(f, "invalid ZFP compression parameters"),
            Self::InvalidRate => write!(f, "invalid ZFP fixed rate"),
            Self::UnrepresentableMode => {
                write!(
                    f,
                    "ZFP header mode cannot preserve the compression parameters"
                )
            }
        }
    }
}

impl std::error::Error for ZfpConfigError {}

/// Number of bytes per stream word ([`crate::types::ZfpBitStreamWord`]).
pub const STREAM_WORD_BYTES: usize = size_of::<crate::types::ZfpBitStreamWord>();

/// Number of bits per stream word ([`crate::types::ZfpBitStreamWord`]).
#[expect(
    clippy::cast_possible_truncation,
    reason = "supported stream word sizes fit in u32"
)]
pub const STREAM_WORD_BITS: u32 = (STREAM_WORD_BYTES * 8) as u32;

/// Maximum value that fits in a 12-bit mode word.
const MODE_SHORT_MAX: u64 = (1u64 << 12) - 2;

// ---------------------------------------------------------------------------
// ZfpStreamAlignment
// ---------------------------------------------------------------------------

/// Controls whether fixed-rate blocks are padded to a 64-bit word boundary.
///
/// Passed as the `align` argument to [`ZfpConfig::fixed_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ZfpStreamAlignment {
    /// Blocks use exactly the computed bit count (no padding).
    #[default]
    Unaligned,
    /// Each block is padded to the next 64-bit word boundary.
    WordAligned,
}

/// Validate expert-mode parameters.
#[expect(unused_variables)]
const fn valid_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> bool {
    min_bits <= max_bits && (0 < max_prec && max_prec <= 64)
}

/// Coefficient rounding, mirroring zfp's build-time `ZFP_ROUNDING_MODE` and
/// `ZFP_WITH_TIGHT_ERROR`. Unlike C zfp, this is a per-call setting.
///
/// Not part of the stream mode word: encoder and decoder must be given the
/// same value. Only [`Never`][Self::Never] matches a stock `libzfp` build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ZfpRounding {
    /// `ZFP_ROUND_NEVER`: truncate. The zfp default.
    #[default]
    Never,
    /// `ZFP_ROUND_FIRST`: bias coefficients before encoding. Changes the bitstream.
    First {
        /// `ZFP_WITH_TIGHT_ERROR`: one fewer bit plane in fixed-accuracy and expert mode.
        tight_error: bool,
    },
    /// `ZFP_ROUND_LAST`: bias coefficients after decoding.
    ///
    /// The bias is decode-only, but `tight_error` applies to both sides, so the
    /// stream matches [`Never`][Self::Never] when it is `false`. When it is
    /// `true`, fixed-accuracy encoding matches `Never` with twice the tolerance,
    /// which uses the same precision without biasing coefficients.
    ///
    /// Reversible decoding is not biased, so it stays lossless. Upstream biases
    /// it too, which makes a `ZFP_ROUND_LAST` build of `libzfp` lossy in
    /// reversible mode.
    Last {
        /// `ZFP_WITH_TIGHT_ERROR`: one fewer bit plane in fixed-accuracy and expert mode.
        tight_error: bool,
    },
}

impl ZfpRounding {
    /// Whether `ZFP_WITH_TIGHT_ERROR` is in effect.
    #[must_use]
    pub fn tight_error(self) -> bool {
        match self {
            Self::Never => false,
            Self::First { tight_error } | Self::Last { tight_error } => tight_error,
        }
    }
}

/// Compression expert parameters.
///
/// Construct a configured instance using the mode constructors:
/// - [`ZfpConfig::fixed_rate`]
/// - [`ZfpConfig::fixed_precision`]
/// - [`ZfpConfig::fixed_accuracy`]
/// - [`ZfpConfig::reversible`]
/// - [`ZfpConfig::expert`]
///
/// Every constructor either validates its parameters or maps them into range,
/// so a `ZfpConfig` always describes a valid codec configuration.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct ZfpConfig {
    /// Minimum bits per block.
    min_bits: u32,
    /// Maximum bits per block.
    max_bits: u32,
    /// Maximum precision (bits per scalar).
    max_prec: u32,
    /// Minimum exponent.
    min_exp: i32,
    /// Coefficient rounding.
    rounding: ZfpRounding,
}

// ---------------------------------------------------------------------------
// Free functions: compute results from (min_bits, max_bits, max_prec, min_exp)
// ---------------------------------------------------------------------------

/// Compute the compression mode from expert parameters.
fn mode_of(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> ZfpMode {
    if min_bits > max_bits || !(0 < max_prec && max_prec <= 64) {
        return ZfpMode::Null;
    }

    // Default expert-mode values
    if min_bits == ZFP_MIN_BITS
        && max_bits == ZFP_MAX_BITS
        && max_prec == ZFP_MAX_PREC
        && min_exp == ZFP_MIN_EXP
    {
        return ZfpMode::Expert;
    }

    // Fixed rate: minbits == maxbits, maxprec >= ZFP_MAX_PREC, minexp == ZFP_MIN_EXP
    if min_bits == max_bits
        && (1..=ZFP_MAX_BITS).contains(&max_bits)
        && max_prec >= ZFP_MAX_PREC
        && min_exp == ZFP_MIN_EXP
    {
        return ZfpMode::FixedRate;
    }

    // Fixed precision: minbits <= ZFP_MIN_BITS, maxbits >= ZFP_MAX_BITS, maxprec in [1..], minexp == ZFP_MIN_EXP
    if min_bits <= ZFP_MIN_BITS
        && max_bits >= ZFP_MAX_BITS
        && max_prec >= 1
        && min_exp == ZFP_MIN_EXP
    {
        return ZfpMode::FixedPrecision;
    }

    // Fixed accuracy: minbits <= ZFP_MIN_BITS, maxbits >= ZFP_MAX_BITS, maxprec >= ZFP_MAX_PREC, minexp >= ZFP_MIN_EXP
    if min_bits <= ZFP_MIN_BITS
        && max_bits >= ZFP_MAX_BITS
        && max_prec >= ZFP_MAX_PREC
        && min_exp >= ZFP_MIN_EXP
    {
        return ZfpMode::FixedAccuracy;
    }

    // Reversible: minbits <= ZFP_MIN_BITS, maxbits >= ZFP_MAX_BITS, maxprec >= ZFP_MAX_PREC, minexp < ZFP_MIN_EXP
    if min_bits <= ZFP_MIN_BITS
        && max_bits >= ZFP_MAX_BITS
        && max_prec >= ZFP_MAX_PREC
        && min_exp < ZFP_MIN_EXP
    {
        return ZfpMode::Reversible;
    }

    ZfpMode::Expert
}

/// Compute the compact mode encoding from expert parameters.
#[allow(clippy::cast_sign_loss)] // i32→u64 for mode encoding
#[expect(
    clippy::arithmetic_side_effects,
    reason = "`mode_of` gives these modes only for `max_bits`, `max_prec` >= 1 and `min_exp` >= `ZFP_MIN_EXP`, and the guards bound them above"
)]
fn mode_bits_of(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u64 {
    match mode_of(min_bits, max_bits, max_prec, min_exp) {
        ZfpMode::FixedRate if max_bits <= 2048 => u64::from(max_bits - 1),
        ZfpMode::FixedPrecision if max_prec <= 128 => u64::from(max_prec - 1) + 2048,
        ZfpMode::Reversible => 2048 + 128,
        ZfpMode::FixedAccuracy if min_exp <= 843 => {
            (min_exp - ZFP_MIN_EXP) as u64 + (2048 + 128 + 1)
        }
        ZfpMode::Null | ZfpMode::Expert => {
            encode_expert_mode(min_bits, max_bits, max_prec, min_exp)
        }
        _ => {
            // Fallback for modes whose guard conditions are not met
            // (e.g. FixedRate with max_bits > 2048, FixedPrecision with
            // max_prec > 128, FixedAccuracy with min_exp > 843). All fall
            // through to the long-form 64-bit expert encoding.
            encode_expert_mode(min_bits, max_bits, max_prec, min_exp)
        }
    }
}

/// [`ZfpConfig::mode`] of the expert parameters, for the C ABI.
#[cfg(feature = "ffi")]
#[must_use]
pub fn compression_mode_from_params(
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> ZfpMode {
    mode_of(min_bits, max_bits, max_prec, min_exp)
}

/// [`ZfpConfig::rate`] of the expert parameters, or `0.0` as in C.
#[cfg(feature = "ffi")]
#[must_use]
pub fn rate_from_params(
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
    dims: ZfpDimensionality,
) -> f64 {
    ZfpConfig::raw(min_bits, max_bits, max_prec, min_exp)
        .rate(dims)
        .unwrap_or(0.0)
}

/// [`ZfpConfig::precision`] of the expert parameters, or `0` as in C.
#[cfg(feature = "ffi")]
#[must_use]
pub fn precision_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u32 {
    ZfpConfig::raw(min_bits, max_bits, max_prec, min_exp)
        .precision()
        .unwrap_or(0)
}

/// [`ZfpConfig::accuracy`] of the expert parameters, or `0.0` as in C.
#[cfg(feature = "ffi")]
#[must_use]
pub fn accuracy_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> f64 {
    ZfpConfig::raw(min_bits, max_bits, max_prec, min_exp)
        .accuracy()
        .unwrap_or(0.0)
}

/// [`ZfpConfig::mode_bits`] of the expert parameters, for the C ABI.
#[cfg(feature = "ffi")]
#[must_use]
pub fn mode_bits_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u64 {
    mode_bits_of(min_bits, max_bits, max_prec, min_exp)
}

/// Encode expert-mode parameters into the 64-bit long-form mode word.
#[allow(clippy::cast_sign_loss)] // i32→u64 for mode encoding
#[expect(
    clippy::arithmetic_side_effects,
    reason = "each field is clamped to at least 1 before it is decremented"
)]
fn encode_expert_mode(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u64 {
    let min_bits = u64::from(min_bits.clamp(1, 0x8000) - 1);
    let max_bits = u64::from(max_bits.clamp(1, 0x8000) - 1);
    let max_prec = u64::from(max_prec.clamp(1, 0x0080) - 1);
    // Saturating: `ZfpConfig::expert` accepts any `i32`.
    let min_exp = min_exp.saturating_add(16495).clamp(0, 0x7fff) as u64;
    let mut mode = 0u64;
    mode = (mode << 15) | min_exp;
    mode = (mode << 7) | max_prec;
    mode = (mode << 15) | max_bits;
    mode = (mode << 15) | min_bits;
    mode = (mode << 12) | 0xfffu64;
    mode
}

// ---------------------------------------------------------------------------
// ZfpConfig: parameter configuration
// ---------------------------------------------------------------------------

impl ZfpConfig {
    /// Decode a mode value back into a `ZfpConfig`.
    ///
    /// This is the inverse of [`mode_bits`][Self::mode_bits].
    /// Returns `None` if the mode is invalid (e.g., precision out of range).
    ///
    /// Rounding is not encoded in the mode word; the result uses
    /// [`ZfpRounding::Never`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // encoded bounded by prior branch conditions
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "each field is masked or bounded by its branch, well inside `u32` and `i32`"
    )]
    pub fn from_mode_bits(encoded: u64) -> Option<Self> {
        let (min_bits, max_bits, max_prec, min_exp) = if encoded <= MODE_SHORT_MAX {
            if encoded < 2048 {
                // fixed rate
                let b = encoded as u32 + 1;
                (b, b, ZFP_MAX_PREC, ZFP_MIN_EXP)
            } else if encoded < (2048 + 128) {
                // fixed precision
                let p = encoded as u32 + 1 - 2048;
                (ZFP_MIN_BITS, ZFP_MAX_BITS, p, ZFP_MIN_EXP)
            } else if encoded == (2048 + 128) {
                // reversible
                (ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP - 1)
            } else {
                // fixed accuracy
                let e = encoded as i32 + ZFP_MIN_EXP - (2048 + 128 + 1);
                (ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, e)
            }
        } else {
            // 64-bit encoding
            let m = encoded >> 12;
            let min_bits = (m & 0x7fff) as u32 + 1;
            let m = m >> 15;
            let max_bits = (m & 0x7fff) as u32 + 1;
            let m = m >> 15;
            let max_prec = (m & 0x007f) as u32 + 1;
            let m = m >> 7;
            let min_exp = (m & 0x7fff) as i32 - 16495;
            (min_bits, max_bits, max_prec, min_exp)
        };

        Self::expert(min_bits, max_bits, max_prec, min_exp).ok()
    }

    /// Fixed-rate mode.
    ///
    /// Configures fixed-rate compression with the given rate (compressed bits
    /// per scalar) for the given data type and dimensionality. The bits per
    /// block are rounded to the nearest integer, and raised to the minimum
    /// the type needs; [`rate`][Self::rate] returns the resulting rate.
    ///
    /// Pass [`ZfpStreamAlignment::WordAligned`] to pad each block to the next
    /// 64-bit word boundary; pass [`ZfpStreamAlignment::Unaligned`] for exact bit packing.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpConfigError::InvalidRate`] if `rate` is negative or
    /// non-finite, rounds to zero bits for an integer type, or produces a block
    /// budget above [`ZFP_MAX_BITS`], including after word alignment. A zero
    /// rate is raised to the minimum for float types, like any rate too small
    /// for the block header.
    pub fn fixed_rate(
        rate: f64,
        ty: ZfpScalarType,
        dims: ZfpDimensionality,
        align: ZfpStreamAlignment,
    ) -> Result<Self, ZfpConfigError> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(ZfpConfigError::InvalidRate);
        }
        let n = dims.block_values();
        let rounded = (f64::from(n) * rate + 0.5).floor();
        if !rounded.is_finite() || rounded > f64::from(ZFP_MAX_BITS) {
            return Err(ZfpConfigError::InvalidRate);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // The checks above bound rounded to [0, ZFP_MAX_BITS].
        let mut bits = rounded as u32;

        match ty {
            ZfpScalarType::F32 if bits < 1 + 8 => {
                bits = 1 + 8;
            }
            ZfpScalarType::F64 if bits < 1 + 11 => {
                bits = 1 + 11;
            }
            ZfpScalarType::F32 | ZfpScalarType::F64 | ZfpScalarType::I32 | ZfpScalarType::I64 => {}
        }

        if align == ZfpStreamAlignment::WordAligned {
            bits = bits
                .checked_next_multiple_of(STREAM_WORD_BITS)
                .ok_or(ZfpConfigError::InvalidRate)?;
        }

        if bits == 0 || bits > ZFP_MAX_BITS {
            return Err(ZfpConfigError::InvalidRate);
        }

        Ok(Self {
            min_bits: bits,
            max_bits: bits,
            max_prec: ZFP_MAX_PREC,
            min_exp: ZFP_MIN_EXP,
            rounding: ZfpRounding::Never,
        })
    }

    /// Fixed-precision mode.
    ///
    /// Configures the stream for fixed-precision compression with the given
    /// number of uncompressed bits per scalar. Zero selects the full precision
    /// of 64 bits; values above 64 are clamped to 64.
    #[must_use]
    pub fn fixed_precision(precision: u32) -> Self {
        Self {
            min_bits: ZFP_MIN_BITS,
            max_bits: ZFP_MAX_BITS,
            max_prec: if precision > 0 {
                precision.min(ZFP_MAX_PREC)
            } else {
                ZFP_MAX_PREC
            },
            min_exp: ZFP_MIN_EXP,
            rounding: ZfpRounding::Never,
        }
    }

    /// Fixed-accuracy mode.
    ///
    /// Configures the stream for fixed-accuracy compression with the given
    /// absolute error tolerance. Zero, negative values, and NaN select the
    /// default minimum exponent. Positive infinity follows `frexp` semantics
    /// and selects exponent `-1`.
    #[must_use]
    pub fn fixed_accuracy(tolerance: f64) -> Self {
        let emin = if tolerance > 0.0 {
            // At most 1025, for infinity, so this cannot overflow.
            let (_, e) = libm::frexp(tolerance);
            e.saturating_sub(1)
        } else {
            ZFP_MIN_EXP
        };
        Self {
            min_bits: ZFP_MIN_BITS,
            max_bits: ZFP_MAX_BITS,
            max_prec: ZFP_MAX_PREC,
            min_exp: emin,
            rounding: ZfpRounding::Never,
        }
    }

    /// Reversible (lossless) mode.
    #[must_use]
    pub fn reversible() -> Self {
        Self {
            min_bits: ZFP_MIN_BITS,
            max_bits: ZFP_MAX_BITS,
            max_prec: ZFP_MAX_PREC,
            min_exp: ZFP_MIN_EXP - 1,
            rounding: ZfpRounding::Never,
        }
    }

    /// Expert mode with explicit parameters.
    ///
    /// This validates the codec parameters, not their header representation.
    /// Use [`checked_mode_bits`][Self::checked_mode_bits] to check whether a
    /// header can preserve them;
    /// [`write_header`][crate::ZfpBitStreamMutOps::write_header] performs that
    /// check when writing a mode.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpConfigError::InvalidParameters`] if `min_bits > max_bits`
    /// or `max_prec` is not in `1..=64`, the failure case of C
    /// `zfp_stream_set_params`.
    pub const fn expert(
        min_bits: u32,
        max_bits: u32,
        max_prec: u32,
        min_exp: i32,
    ) -> Result<Self, ZfpConfigError> {
        if valid_params(min_bits, max_bits, max_prec, min_exp) {
            Ok(Self::raw(min_bits, max_bits, max_prec, min_exp))
        } else {
            Err(ZfpConfigError::InvalidParameters)
        }
    }

    /// Expert parameters from a C `zfp_stream`, without validation.
    ///
    /// C lets a `zfp_stream` hold any parameters, and compresses with them.
    /// So does this config: the codec never panics, but an invalid combination
    /// gives a config whose [`mode`][Self::mode] is [`ZfpMode::Null`], and
    /// output that is only as meaningful as C's.
    #[cfg(feature = "ffi")]
    #[must_use]
    pub const fn from_raw_params(
        min_bits: u32,
        max_bits: u32,
        max_prec: u32,
        min_exp: i32,
    ) -> Self {
        Self::raw(min_bits, max_bits, max_prec, min_exp)
    }

    /// Expert parameters, unvalidated, with no rounding.
    const fn raw(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> Self {
        Self {
            min_bits,
            max_bits,
            max_prec,
            min_exp,
            rounding: ZfpRounding::Never,
        }
    }

    /// Block-codec defaults when the caller passes no stream config: no rate
    /// constraint and full precision, down to the type's smallest exponent.
    #[cfg(feature = "internals")]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "at most 64 bits for each of 256 values"
    )]
    pub(crate) const fn block_default(ty: ZfpScalarType, dims: ZfpDimensionality) -> Self {
        let values = dims.block_values();
        // max_bits exceeds every value at full precision, plus the float exponent.
        let (max_bits, max_prec, min_exp) = match ty {
            ZfpScalarType::I32 => (32 * values + 1, 32, ZFP_MIN_EXP),
            ZfpScalarType::I64 => (64 * values + 1, 64, ZFP_MIN_EXP),
            ZfpScalarType::F32 => ((8 + 1) + 32 * values, 32, -149),
            ZfpScalarType::F64 => ((11 + 1) + 64 * values, 64, -1074),
        };
        Self::raw(0, max_bits, max_prec, min_exp)
    }

    /// Default expert-mode config with all parameters at their maximum range.
    ///
    /// Equivalent to `ZfpConfig::expert(ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP)`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_bits: ZFP_MIN_BITS,
            max_bits: ZFP_MAX_BITS,
            max_prec: ZFP_MAX_PREC,
            min_exp: ZFP_MIN_EXP,
            rounding: ZfpRounding::Never,
        }
    }

    // --- Field accessors ---

    /// Minimum bits per block.
    #[must_use]
    pub fn min_bits(&self) -> u32 {
        self.min_bits
    }

    /// Maximum bits per block.
    #[must_use]
    pub fn max_bits(&self) -> u32 {
        self.max_bits
    }

    /// Maximum precision (bits per scalar).
    #[must_use]
    pub fn max_prec(&self) -> u32 {
        self.max_prec
    }

    /// Minimum exponent.
    #[must_use]
    pub fn min_exp(&self) -> i32 {
        self.min_exp
    }

    /// Coefficient rounding. Defaults to [`ZfpRounding::Never`].
    #[must_use]
    pub fn rounding(&self) -> ZfpRounding {
        self.rounding
    }

    /// Set the coefficient rounding.
    ///
    /// Rounding is not encoded in the stream, so decompression must use the
    /// same value the stream was compressed with.
    #[must_use]
    pub fn with_rounding(mut self, rounding: ZfpRounding) -> Self {
        self.rounding = rounding;
        self
    }

    // --- Inspectors ---

    /// Return the compression mode these parameters select.
    #[must_use]
    pub fn mode(&self) -> ZfpMode {
        mode_of(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Whether these parameters select the reversible coder, as C's
    /// `REVERSIBLE` does. Unlike [`mode`][Self::mode], this ignores the other
    /// parameters.
    #[inline]
    pub(crate) fn is_reversible(&self) -> bool {
        self.min_exp < ZFP_MIN_EXP
    }

    /// Return the rate (compressed bits per scalar) for the given
    /// dimensionality, or [`None`] if this is not a fixed-rate config.
    #[must_use]
    pub fn rate(&self, dims: ZfpDimensionality) -> Option<f64> {
        (self.mode() == ZfpMode::FixedRate)
            .then(|| f64::from(self.max_bits) / f64::from(dims.block_values()))
    }

    /// Return the precision (uncompressed bits per scalar), or [`None`] if
    /// this is not a fixed-precision config.
    #[must_use]
    pub fn precision(&self) -> Option<u32> {
        (self.mode() == ZfpMode::FixedPrecision).then_some(self.max_prec)
    }

    /// Return the absolute error tolerance, or [`None`] if this is not a
    /// fixed-accuracy config.
    #[must_use]
    pub fn accuracy(&self) -> Option<f64> {
        (self.mode() == ZfpMode::FixedAccuracy).then(|| libm::ldexp(1.0, self.min_exp))
    }

    /// Return the C-compatible compact 12- or 64-bit mode encoding.
    ///
    /// This low-level encoding clamps expert parameters to the wire format's
    /// range. Use [`checked_mode_bits`][Self::checked_mode_bits] before writing
    /// a header whose decoded configuration must preserve the parameters.
    #[must_use]
    pub fn mode_bits(&self) -> u64 {
        mode_bits_of(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Encode a mode only if reading it preserves every expert parameter.
    ///
    /// Rounding is not included in the mode word; the decoder must receive it
    /// separately. This check is conservative for parameters that could happen
    /// to produce the same output for a particular field.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpConfigError::InvalidParameters`] for codec-invalid
    /// parameters, or [`ZfpConfigError::UnrepresentableMode`] if the mode word
    /// changes any of the four expert parameters.
    pub fn checked_mode_bits(&self) -> Result<u64, ZfpConfigError> {
        if !valid_params(self.min_bits, self.max_bits, self.max_prec, self.min_exp) {
            return Err(ZfpConfigError::InvalidParameters);
        }
        let mode = self.mode_bits();
        let decoded = Self::from_mode_bits(mode).ok_or(ZfpConfigError::UnrepresentableMode)?;
        if (self.min_bits, self.max_bits, self.max_prec, self.min_exp)
            != (
                decoded.min_bits,
                decoded.max_bits,
                decoded.max_prec,
                decoded.min_exp,
            )
        {
            return Err(ZfpConfigError::UnrepresentableMode);
        }
        Ok(mode)
    }

    /// Return the maximum compressed size in bytes, header included, for a
    /// field with the given type and dimensions.
    ///
    /// `dims` takes the same forms as [`ZfpField::new`][crate::ZfpField::new],
    /// including the zero-padded `[usize; 4]` from
    /// [`ZfpField::dims`][crate::ZfpField::dims]. A stream this large always
    /// holds the output of [`write_header`][crate::ZfpBitStreamMutOps::write_header]
    /// with [`ZfpHeaderMask::FULL`][crate::ZfpHeaderMask::FULL] followed by
    /// [`compress`][crate::ZfpBitStreamMutOps::compress].
    ///
    /// Returns [`None`] if the dimensions are malformed (as for
    /// [`ZfpFieldError::InvalidDims`][crate::ZfpFieldError::InvalidDims]) or
    /// the size overflows `usize`.
    #[must_use]
    pub fn maximum_size(&self, ty: ZfpScalarType, dims: impl ZfpDims) -> Option<usize> {
        let dims = dims.to_array();
        if !crate::field::valid_dims(&dims) {
            return None;
        }
        let dimensionality = crate::field::dimensionality(&dims);
        let maxbits = self.block_bits(ty, dimensionality);

        let blocks = dims
            .iter()
            .take(usize::from(dimensionality))
            .try_fold(1usize, |acc, &n| acc.checked_mul(n.div_ceil(4)))?;
        // Maximum header size in bits (mirrors ZFP_HEADER_MAX_BITS / zfp_stream_maximum_size in zfp.c).
        let header_max: u64 = 148;
        let total_bits = (blocks as u64)
            .checked_mul(u64::from(maxbits))
            .and_then(|bits| bits.checked_add(header_max))
            .and_then(|bits| bits.checked_next_multiple_of(u64::from(STREAM_WORD_BITS)))?;
        usize::try_from(total_bits / 8).ok()
    }

    /// The most bits a block of this type and dimensionality can take.
    ///
    /// As C's `zfp_stream_maximum_size`, this is the most a block can need,
    /// capped at `max_bits` and raised to `min_bits`, except that it is never
    /// less than the headers. A block writes those whatever `max_bits` is, so
    /// C under-reports when `max_bits` is smaller.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "at most 19 header bits and 64 bits for each of 256 values"
    )]
    pub(crate) fn block_bits(&self, ty: ZfpScalarType, dims: ZfpDimensionality) -> u32 {
        let values = dims.block_values();
        let type_prec = match ty {
            ZfpScalarType::I32 | ZfpScalarType::F32 => 32u32,
            ZfpScalarType::I64 | ZfpScalarType::F64 => 64u32,
        };
        let header = if self.is_reversible() {
            // Precision bits, after a zero-block bit, a path bit and the
            // exponent for floats (mirrors zfp_stream_maximum_size in zfp.c).
            match ty {
                ZfpScalarType::I32 => 5,
                ZfpScalarType::I64 => 6,
                ZfpScalarType::F32 => 1 + 1 + 8 + 5,
                ZfpScalarType::F64 => 1 + 1 + 11 + 6,
            }
        } else {
            // A zero-block bit and the exponent for floats.
            match ty {
                ZfpScalarType::F32 => 1 + 8,
                ZfpScalarType::F64 => 1 + 11,
                ZfpScalarType::I32 | ZfpScalarType::I64 => 0,
            }
        };
        let most = header + values - 1 + values * self.max_prec.min(type_prec);
        most.min(self.max_bits).max(header).max(self.min_bits)
    }
}

impl Default for ZfpConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::STREAM_WORD_BITS;
    use crate::config::ZfpStreamAlignment;
    use crate::types::{
        ZFP_MAX_PREC, ZFP_MIN_BITS, ZFP_MIN_EXP, ZfpDimensionality, ZfpMode, ZfpScalarType,
    };
    use crate::{ZFP_MAX_BITS, ZfpConfig, ZfpConfigError};

    #[test]
    fn rounding_defaults_to_never_and_round_trips() {
        use crate::ZfpRounding;
        let config = ZfpConfig::fixed_accuracy(1e-6);
        assert_eq!(config.rounding(), ZfpRounding::Never);
        assert!(!config.rounding().tight_error());

        let rounded = config.with_rounding(ZfpRounding::First { tight_error: true });
        assert_eq!(rounded.rounding(), ZfpRounding::First { tight_error: true });
        assert!(rounded.rounding().tight_error());
        // Rounding is orthogonal to the expert parameters.
        assert_eq!(rounded.mode_bits(), config.mode_bits());
        assert_eq!(rounded.min_exp(), config.min_exp());
        // ...and is not recovered from the mode word.
        assert_eq!(
            ZfpConfig::from_mode_bits(rounded.mode_bits())
                .unwrap()
                .rounding(),
            ZfpRounding::Never
        );
    }

    #[test]
    fn new_is_expert_mode() {
        let config = ZfpConfig::new();
        assert_eq!(config.mode(), ZfpMode::Expert);
        assert_eq!(
            config,
            ZfpConfig::expert(
                ZFP_MIN_BITS,
                crate::types::ZFP_MAX_BITS,
                ZFP_MAX_PREC,
                ZFP_MIN_EXP
            )
            .unwrap()
        );
    }

    #[test]
    fn fixed_rate_creates_fixed_rate_stream() {
        for zfp_type in [
            ZfpScalarType::I32,
            ZfpScalarType::I64,
            ZfpScalarType::F32,
            ZfpScalarType::F64,
        ] {
            for dims in [
                ZfpDimensionality::D1,
                ZfpDimensionality::D2,
                ZfpDimensionality::D3,
                ZfpDimensionality::D4,
            ] {
                let config =
                    ZfpConfig::fixed_rate(8.0, zfp_type, dims, ZfpStreamAlignment::Unaligned)
                        .unwrap();
                assert_eq!(
                    config.mode(),
                    ZfpMode::FixedRate,
                    "type={zfp_type:?} dims={dims:?}"
                );
            }
        }
    }

    #[test]
    fn fixed_rate_with_align_rounds_to_word_boundary() {
        let config = ZfpConfig::fixed_rate(
            8.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::WordAligned,
        )
        .unwrap();
        assert_eq!(config.max_bits(), 512);

        let config = ZfpConfig::fixed_rate(
            5.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::WordAligned,
        )
        .unwrap();
        assert_eq!(config.max_bits(), 320);
    }

    #[test]
    fn fixed_rate_min_bits_enforcement() {
        let config = ZfpConfig::fixed_rate(
            0.1,
            ZfpScalarType::F32,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        assert_eq!(config.min_bits(), 9);
        assert_eq!(config.max_bits(), 9);

        let config = ZfpConfig::fixed_rate(
            0.1,
            ZfpScalarType::F64,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        assert_eq!(config.min_bits(), 12);
        assert_eq!(config.max_bits(), 12);
    }

    #[test]
    fn fixed_rate_rejects_invalid_and_unrepresentable_rates() {
        use ZfpDimensionality::D1;
        use ZfpScalarType::{F64, I32};
        for rate in [
            f64::NEG_INFINITY,
            -1.0,
            -f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::from(u32::MAX) / 4.0,
            f64::MAX,
        ] {
            for align in [
                ZfpStreamAlignment::Unaligned,
                ZfpStreamAlignment::WordAligned,
            ] {
                assert_eq!(
                    ZfpConfig::fixed_rate(rate, F64, D1, align),
                    Err(ZfpConfigError::InvalidRate),
                    "rate={rate:?}, align={align:?}"
                );
            }
        }
        assert_eq!(
            ZfpConfig::fixed_rate(0.01, I32, D1, ZfpStreamAlignment::Unaligned),
            Err(ZfpConfigError::InvalidRate)
        );
        assert_eq!(
            ZfpConfig::fixed_rate(
                f64::from(ZFP_MAX_BITS) / 4.0,
                I32,
                D1,
                ZfpStreamAlignment::Unaligned
            )
            .unwrap()
            .max_bits(),
            ZFP_MAX_BITS
        );
        assert_eq!(
            ZfpConfig::fixed_rate(
                f64::from(ZFP_MAX_BITS) / 4.0,
                I32,
                D1,
                ZfpStreamAlignment::WordAligned
            ),
            Err(ZfpConfigError::InvalidRate)
        );
        let aligned_max = ZFP_MAX_BITS / STREAM_WORD_BITS * STREAM_WORD_BITS;
        assert_eq!(
            ZfpConfig::fixed_rate(
                f64::from(aligned_max) / 4.0,
                I32,
                D1,
                ZfpStreamAlignment::WordAligned
            )
            .unwrap()
            .max_bits(),
            aligned_max
        );
        assert_eq!(
            ZfpConfig::fixed_rate(
                f64::from(aligned_max + 1) / 4.0,
                I32,
                D1,
                ZfpStreamAlignment::WordAligned
            ),
            Err(ZfpConfigError::InvalidRate)
        );
    }

    /// A zero rate is too small for any block, like any rate below the header:
    /// float types are raised to the header, and integer types have no budget.
    #[test]
    fn fixed_rate_raises_a_zero_rate_to_the_float_header() {
        use ZfpDimensionality::D1;
        use ZfpScalarType::{F32, F64, I32};
        use ZfpStreamAlignment::{Unaligned, WordAligned};
        for rate in [-0.0, 0.0, 0.01] {
            let bits = |ty, align| ZfpConfig::fixed_rate(rate, ty, D1, align).map(|c| c.max_bits());
            assert_eq!(bits(F32, Unaligned), Ok(1 + 8), "rate={rate:?}");
            assert_eq!(bits(F64, Unaligned), Ok(1 + 11), "rate={rate:?}");
            assert_eq!(
                bits(F64, WordAligned),
                Ok(STREAM_WORD_BITS),
                "rate={rate:?}"
            );
            assert_eq!(
                bits(I32, Unaligned),
                Err(ZfpConfigError::InvalidRate),
                "rate={rate:?}"
            );
        }
        assert_eq!(
            ZfpConfig::fixed_rate(0.0, F64, D1, Unaligned)
                .unwrap()
                .max_bits(),
            1 + 11
        );
    }

    #[test]
    fn legacy_precision_and_tolerance_edge_values_are_stable() {
        assert_eq!(ZfpConfig::fixed_precision(0).max_prec(), ZFP_MAX_PREC);
        assert_eq!(
            ZfpConfig::fixed_precision(u32::MAX).max_prec(),
            ZFP_MAX_PREC
        );
        for tolerance in [-1.0, -0.0, 0.0, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(ZfpConfig::fixed_accuracy(tolerance).min_exp(), ZFP_MIN_EXP);
        }
        assert_eq!(ZfpConfig::fixed_accuracy(f64::INFINITY).min_exp(), -1);
    }

    #[test]
    fn fixed_precision_creates_fixed_precision_stream() {
        for prec in 1..ZFP_MAX_PREC {
            let config = ZfpConfig::fixed_precision(prec);
            assert_eq!(config.mode(), ZfpMode::FixedPrecision, "prec={prec}");
            assert_eq!(config.precision(), Some(prec));
        }
    }

    #[test]
    fn fixed_accuracy_creates_fixed_accuracy_stream() {
        for acc_exp in -20..0 {
            let tol = libm::ldexp(1.0, acc_exp);
            let config = ZfpConfig::fixed_accuracy(tol);
            assert_eq!(config.mode(), ZfpMode::FixedAccuracy, "acc_exp={acc_exp}");
            assert_eq!(config.accuracy().map(f64::to_bits), Some(tol.to_bits()));
        }
    }

    #[test]
    fn reversible_creates_reversible_stream() {
        let config = ZfpConfig::reversible();
        assert_eq!(config.mode(), ZfpMode::Reversible);
    }

    #[test]
    fn expert_sets_custom_params() {
        let config = ZfpConfig::expert(10, 100, 50, -500).unwrap();
        assert_eq!(config.mode(), ZfpMode::Expert);
    }

    #[test]
    fn copy_clone() {
        let config = ZfpConfig::fixed_rate(
            8.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let cloned = config;
        assert_eq!(config, cloned);
        assert!(std::ptr::eq(&raw const config, &raw const config)); // Copy, not moved
    }
}
