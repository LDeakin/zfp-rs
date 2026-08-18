//! `ZfpConfig`: holds compression parameters only.
//!
//! A `ZfpConfig` is an immutable, [`Copy`] struct holding four expert
//! parameters. It does **not** own a bitstream: that is managed separately
//! by the caller, mirroring the C API where the bitstream is externally
//! allocated.
//!
//! Compression and decompression are performed via methods on
//! [`ZfpBitStream`][crate::ZfpBitStream]
//! that take `&ZfpConfig`.

use crate::types::{
    ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_BITS, ZFP_MIN_EXP, ZfpDimensionality, ZfpMode,
    ZfpScalarType,
};

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
    None,
    /// Each block is padded to the next 64-bit word boundary.
    WordAligned,
}

/// Validate expert-mode parameters.
#[expect(unused_variables)]
fn valid_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> bool {
    min_bits <= max_bits && (0 < max_prec && max_prec <= 64)
}

/// Compression expert parameters.
///
/// Construct a configured instance using the mode constructors:
/// - [`ZfpConfig::fixed_rate`]
/// - [`ZfpConfig::fixed_precision`]
/// - [`ZfpConfig::fixed_accuracy`]
/// - [`ZfpConfig::reversible`]
/// - [`ZfpConfig::expert`]
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
}

// ---------------------------------------------------------------------------
// Free functions: compute results from (min_bits, max_bits, max_prec, min_exp)
// ---------------------------------------------------------------------------

/// Compute the compression mode from expert parameters without a `ZfpConfig`.
///
/// This is the parameter-less version of [`ZfpConfig::compression_mode`].
#[must_use]
pub fn compression_mode_from_params(
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> ZfpMode {
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

/// Compute the effective rate for a dimensionality from expert parameters.
#[must_use]
pub fn rate_from_params(
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
    dims: ZfpDimensionality,
) -> f64 {
    if compression_mode_from_params(min_bits, max_bits, max_prec, min_exp) == ZfpMode::FixedRate {
        f64::from(max_bits) / f64::from(1u32 << (2 * u32::from(dims)))
    } else {
        0.0
    }
}

/// Compute the effective precision from expert parameters.
#[must_use]
pub fn precision_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u32 {
    if compression_mode_from_params(min_bits, max_bits, max_prec, min_exp)
        == ZfpMode::FixedPrecision
    {
        max_prec
    } else {
        0
    }
}

/// Compute the effective accuracy tolerance from expert parameters.
#[must_use]
pub fn accuracy_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> f64 {
    if compression_mode_from_params(min_bits, max_bits, max_prec, min_exp) == ZfpMode::FixedAccuracy
    {
        libm::ldexp(1.0, min_exp)
    } else {
        0.0
    }
}

/// Compute the compact mode encoding from expert parameters.
#[must_use]
#[allow(clippy::cast_sign_loss)] // i32→u64 for mode encoding
pub fn mode_bits_from_params(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> u64 {
    match compression_mode_from_params(min_bits, max_bits, max_prec, min_exp) {
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

/// Encode expert-mode parameters into the 64-bit long-form mode word.
#[allow(clippy::cast_sign_loss)] // i32→u64 for mode encoding
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
    /// This is the inverse of [`mode_bits_from_params`].
    /// Returns `None` if the mode is invalid (e.g., precision out of range).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // encoded bounded by prior branch conditions
    pub fn from_mode(encoded: u64) -> Option<Self> {
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

        if !valid_params(min_bits, max_bits, max_prec, min_exp) {
            return None;
        }
        Some(Self {
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        })
    }

    /// Fixed-rate mode.
    ///
    /// Configures the stream for fixed-rate compression with the given
    /// rate (bits per scalar), data type, and dimensionality.
    ///
    /// Returns the computed rate, rounded to the nearest integer bit count.
    ///
    /// Pass [`ZfpStreamAlignment::WordAligned`] to pad each block to the next
    /// 64-bit word boundary; pass [`ZfpStreamAlignment::None`] for exact bit packing.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // n≥1, rate>0, result fits in u32 for valid params
    #[allow(clippy::cast_sign_loss)] // f64→u32 for bit count
    pub fn fixed_rate(
        rate: f64,
        ty: ZfpScalarType,
        dims: ZfpDimensionality,
        align: ZfpStreamAlignment,
    ) -> Self {
        let n = 1u32 << (2 * u32::from(dims));
        let mut bits = (f64::from(n) * rate + 0.5).floor() as u32;

        match ty {
            ZfpScalarType::Float if bits < 1 + 8 => {
                bits = 1 + 8;
            }
            ZfpScalarType::Double if bits < 1 + 11 => {
                bits = 1 + 11;
            }
            ZfpScalarType::Float
            | ZfpScalarType::Double
            | ZfpScalarType::Int32
            | ZfpScalarType::Int64 => {}
        }

        if align == ZfpStreamAlignment::WordAligned {
            bits = bits.next_multiple_of(STREAM_WORD_BITS);
        }

        Self {
            min_bits: bits,
            max_bits: bits,
            max_prec: ZFP_MAX_PREC,
            min_exp: ZFP_MIN_EXP,
        }
    }

    /// Fixed-precision mode.
    ///
    /// Configures the stream for fixed-precision compression with the given
    /// number of uncompressed bits per scalar.
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
        }
    }

    /// Fixed-accuracy mode.
    ///
    /// Configures the stream for fixed-accuracy compression with the given
    /// absolute error tolerance.
    #[must_use]
    pub fn fixed_accuracy(tolerance: f64) -> Self {
        let emin = if tolerance > 0.0 {
            let (_, e) = libm::frexp(tolerance);
            e - 1
        } else {
            ZFP_MIN_EXP
        };
        Self {
            min_bits: ZFP_MIN_BITS,
            max_bits: ZFP_MAX_BITS,
            max_prec: ZFP_MAX_PREC,
            min_exp: emin,
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
        }
    }

    /// Expert mode with explicit parameters.
    #[must_use]
    pub fn expert(min_bits: u32, max_bits: u32, max_prec: u32, min_exp: i32) -> Self {
        Self {
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        }
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

    // --- Inspectors ---

    /// Return the current compression mode.
    #[must_use]
    pub fn compression_mode(&self) -> ZfpMode {
        compression_mode_from_params(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Return the effective rate for the given dimensionality.
    #[must_use]
    pub fn rate(&self, dims: ZfpDimensionality) -> f64 {
        rate_from_params(
            self.min_bits,
            self.max_bits,
            self.max_prec,
            self.min_exp,
            dims,
        )
    }

    /// Return the current precision.
    #[must_use]
    pub fn precision(&self) -> u32 {
        precision_from_params(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Return the current accuracy tolerance.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        accuracy_from_params(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Return the compact 12- or 64-bit mode encoding.
    #[must_use]
    pub fn mode_bits(&self) -> u64 {
        mode_bits_from_params(self.min_bits, self.max_bits, self.max_prec, self.min_exp)
    }

    /// Return the maximum compressed size in bytes for a field with the given type and dims.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // zfp only supports up to 4 dimensions
    pub fn maximum_size(&self, ty: ZfpScalarType, dims: &[usize]) -> usize {
        let d = dims.len() as u32;
        if d == 0 || d > 4 {
            return 0;
        }
        let reversible = self.min_exp < ZFP_MIN_EXP;
        let values = 1u32 << (2 * d);
        let type_prec = match ty {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 32u32,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 64u32,
        };
        // Extra bits for reversible mode: 1 sign + 1 exponent + M mantissa bits + E exponent bits,
        // where M and E depend on float/double precision (mirrors zfp_reversible_size in zfp.c).
        let mut extra_bits: u32 = if reversible {
            match ty {
                ZfpScalarType::Int32 => 5, // 1 sign + 1 exponent + 5 mantissa + 1 exponent (min: 5 bits)
                ZfpScalarType::Int64 => 6, // 1 sign + 1 exponent + 6 mantissa + 1 exponent (min: 6 bits)
                ZfpScalarType::Float => 1 + 1 + 8 + 5, // sign + exp + 8 mantissa + 5 exponent
                ZfpScalarType::Double => 1 + 1 + 11 + 6, // sign + exp + 11 mantissa + 6 exponent
            }
        } else {
            match ty {
                // 1 sign bit + float exponent width (mirrors zfp_stream_maximum_size in zfp.c).
                ZfpScalarType::Float => 1 + 8,
                // 1 sign bit + double exponent width.
                ZfpScalarType::Double => 1 + 11,
                ZfpScalarType::Int32 | ZfpScalarType::Int64 => 0,
            }
        };
        extra_bits += values - 1 + values * self.max_prec.min(type_prec);
        let maxbits = extra_bits.min(self.max_bits).max(self.min_bits);

        let Some(blocks) = dims
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n.div_ceil(4)))
        else {
            return 0;
        };
        // Maximum header size in bits (mirrors ZFP_HEADER_MAX_BITS / zfp_stream_maximum_size in zfp.c).
        let header_max: u64 = 148;
        let Some(total_bits) = (blocks as u64)
            .checked_mul(u64::from(maxbits))
            .and_then(|bits| bits.checked_add(header_max))
            .and_then(|bits| bits.checked_next_multiple_of(u64::from(STREAM_WORD_BITS)))
        else {
            return 0;
        };
        let bytes = total_bits / 8;
        if bytes as usize as u64 != bytes {
            return 0;
        }
        bytes as usize
    }
}

impl Default for ZfpConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::ZfpConfig;
    use crate::config::ZfpStreamAlignment;
    use crate::types::{
        ZFP_MAX_PREC, ZFP_MIN_BITS, ZFP_MIN_EXP, ZfpDimensionality, ZfpMode, ZfpScalarType,
    };

    #[test]
    fn new_is_expert_mode() {
        let config = ZfpConfig::new();
        assert_eq!(config.compression_mode(), ZfpMode::Expert);
        assert_eq!(
            config,
            ZfpConfig::expert(
                ZFP_MIN_BITS,
                crate::types::ZFP_MAX_BITS,
                ZFP_MAX_PREC,
                ZFP_MIN_EXP
            )
        );
    }

    #[test]
    fn fixed_rate_creates_fixed_rate_stream() {
        for zfp_type in [
            ZfpScalarType::Int32,
            ZfpScalarType::Int64,
            ZfpScalarType::Float,
            ZfpScalarType::Double,
        ] {
            for dims in [
                ZfpDimensionality::D1,
                ZfpDimensionality::D2,
                ZfpDimensionality::D3,
                ZfpDimensionality::D4,
            ] {
                let config = ZfpConfig::fixed_rate(8.0, zfp_type, dims, ZfpStreamAlignment::None);
                assert_eq!(
                    config.compression_mode(),
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
            ZfpScalarType::Double,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::WordAligned,
        );
        assert_eq!(config.max_bits(), 512);

        let config = ZfpConfig::fixed_rate(
            5.0,
            ZfpScalarType::Double,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::WordAligned,
        );
        assert_eq!(config.max_bits(), 320);
    }

    #[test]
    fn fixed_rate_min_bits_enforcement() {
        let config = ZfpConfig::fixed_rate(
            0.1,
            ZfpScalarType::Float,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::None,
        );
        assert_eq!(config.min_bits(), 9);
        assert_eq!(config.max_bits(), 9);

        let config = ZfpConfig::fixed_rate(
            0.1,
            ZfpScalarType::Double,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::None,
        );
        assert_eq!(config.min_bits(), 12);
        assert_eq!(config.max_bits(), 12);
    }

    #[test]
    fn fixed_precision_creates_fixed_precision_stream() {
        for prec in 1..ZFP_MAX_PREC {
            let config = ZfpConfig::fixed_precision(prec);
            assert_eq!(
                config.compression_mode(),
                ZfpMode::FixedPrecision,
                "prec={prec}"
            );
            assert_eq!(config.precision(), prec);
        }
    }

    #[test]
    fn fixed_accuracy_creates_fixed_accuracy_stream() {
        for acc_exp in -20..0 {
            let tol = libm::ldexp(1.0, acc_exp);
            let config = ZfpConfig::fixed_accuracy(tol);
            assert_eq!(
                config.compression_mode(),
                ZfpMode::FixedAccuracy,
                "acc_exp={acc_exp}"
            );
            assert_eq!(config.accuracy().to_bits(), tol.to_bits());
        }
    }

    #[test]
    fn reversible_creates_reversible_stream() {
        let config = ZfpConfig::reversible();
        assert_eq!(config.compression_mode(), ZfpMode::Reversible);
    }

    #[test]
    fn expert_sets_custom_params() {
        let config = ZfpConfig::expert(10, 100, 50, -500);
        assert_eq!(config.compression_mode(), ZfpMode::Expert);
    }

    #[test]
    fn copy_clone() {
        let config = ZfpConfig::fixed_rate(
            8.0,
            ZfpScalarType::Double,
            ZfpDimensionality::D3,
            ZfpStreamAlignment::None,
        );
        let cloned = config;
        assert_eq!(config, cloned);
        assert!(std::ptr::eq(&raw const config, &raw const config)); // Copy, not moved
    }
}
