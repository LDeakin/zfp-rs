//! Bounded, structured input model shared by the fuzz target bodies.
//!
//! Every type here maps raw fuzzer bytes onto a *valid* configuration. Clamps
//! mirror the proptest strategies in `tests/proptest/compress_compat.rs` so the
//! two testing approaches cover the same space, with one deliberate exception:
//! float payloads here are unfiltered (see [`crate::scalar::decode_scalars`]).
//!
//! Both a byte-oriented constructor (`from_bytes`/`from_byte`) and an
//! [`Arbitrary`] impl are provided for each type. The byte-oriented form gives
//! `decompress_stream` and `header_decode` a fixed-width framing whose layout
//! is stable across `arbitrary` releases, so committed seeds and regressions do
//! not silently rot on a dependency bump.

use arbitrary::{Arbitrary, Result, Unstructured};
use zfp_rs::{
    ZfpConfig, ZfpExecution, ZfpStreamAlignment,
    types::{ZfpDimensionality, ZfpScalarType},
};

use crate::limits::{MAX_ELEMENTS, MAX_SIDE};

// ---------------------------------------------------------------------------
// Scalar type
// ---------------------------------------------------------------------------

/// Which of the four ZFP scalar types a target should instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    I32,
    I64,
    F32,
    F64,
}

impl ScalarKind {
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        match b % 4 {
            0 => Self::I32,
            1 => Self::I64,
            2 => Self::F32,
            _ => Self::F64,
        }
    }

    #[must_use]
    pub fn scalar_type(self) -> ZfpScalarType {
        match self {
            Self::I32 => ZfpScalarType::Int32,
            Self::I64 => ZfpScalarType::Int64,
            Self::F32 => ZfpScalarType::Float,
            Self::F64 => ZfpScalarType::Double,
        }
    }
}

impl<'a> Arbitrary<'a> for ScalarKind {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Self::from_byte(u8::arbitrary(u)?))
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// A field shape bounded to [`MAX_ELEMENTS`] scalars.
///
/// `zfp-rs` derives dimensionality from trailing zeros in a `[usize; 4]`
/// (`dimensionality()` in `src/field.rs`), so a single `[usize; 4]` call site
/// covers all four ranks and no per-rank macro is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    rank: usize,
    dims: [usize; 4],
}

impl Shape {
    /// Build a shape from a rank byte and four side bytes.
    ///
    /// Sides are clamped per-rank via [`MAX_SIDE`], then trimmed from the last
    /// axis inward until the element count fits [`MAX_ELEMENTS`]. Every axis
    /// stays at least 1, so the result is always a usable field.
    #[must_use]
    pub fn from_bytes(rank: u8, sides: [u8; 4]) -> Self {
        let rank = 1 + usize::from(rank % 4);
        let max_side = MAX_SIDE[rank - 1];

        let mut dims = [0usize; 4];
        for axis in 0..rank {
            dims[axis] = 1 + usize::from(sides[axis]) % max_side;
        }

        // Trim from the outermost axis inward until the field fits the budget.
        // Trimming inward keeps the innermost axis (the one the 1-D transform
        // runs along) as long as possible, which is the more interesting shape.
        for axis in (0..rank).rev() {
            while dims[..rank].iter().product::<usize>() > MAX_ELEMENTS && dims[axis] > 1 {
                dims[axis] -= 1;
            }
        }

        Self { rank, dims }
    }

    #[must_use]
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[must_use]
    pub fn elements(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }

    #[must_use]
    pub fn dimensionality(&self) -> ZfpDimensionality {
        dimensionality_of(self.rank)
    }
}

/// Rank (1-4) as a [`ZfpDimensionality`].
///
/// # Panics
/// Panics outside 1-4. Every caller derives the rank as `1 + b % 4`.
#[must_use]
pub fn dimensionality_of(rank: usize) -> ZfpDimensionality {
    let rank = u32::try_from(rank).expect("rank is 1..=4");
    ZfpDimensionality::try_from(rank).expect("rank is 1..=4")
}

impl<'a> Arbitrary<'a> for Shape {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let rank = u8::arbitrary(u)?;
        let sides = <[u8; 4]>::arbitrary(u)?;
        Ok(Self::from_bytes(rank, sides))
    }
}

// ---------------------------------------------------------------------------
// Compression mode
// ---------------------------------------------------------------------------

/// A compression mode plus its parameter, before binding to a scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeSpec {
    /// Fixed rate, expressed as bits per block (`1..=2048`).
    FixedRate {
        bits_per_block: u32,
        align: ZfpStreamAlignment,
    },
    /// Fixed precision (`1..=64`).
    FixedPrecision { precision: u32 },
    /// Fixed accuracy, expressed as the minimum exponent (`-1075..=843`).
    FixedAccuracy { min_exp: i32 },
    /// Lossless.
    Reversible,
    /// All four expert parameters, normalised so `min_bits <= max_bits`.
    Expert {
        min_bits: u32,
        max_bits: u32,
        max_prec: u32,
        min_exp: i32,
    },
}

/// Upper bound on expert `max_bits`, mirroring `ZFP_MAX_BITS`.
const MAX_BITS: u32 = 16658;

/// Lower bound on expert `max_bits`.
///
/// A block always emits its exponent header (1 + 11 bits for `f64`) before it
/// can honour a bit budget, so a `max_bits` below that is not a rate the
/// encoder can meet — it overshoots, and `maximum_size`, which trusts
/// `max_bits`, then under-reports the buffer the encoder needs. The C reference
/// has the same behaviour (it wraps the budget subtraction and overshoots
/// further), and `ZfpConfig::fixed_rate` clamps to `1 + 11` for exactly this
/// reason. Only the unvalidated `ZfpConfig::expert` can reach it, so the
/// generator declines to. Low-rate configurations are still covered through
/// `FixedRate`, which does the clamping itself.
const MIN_EXPERT_BITS: u32 = 64;

impl ModeSpec {
    /// Build a mode from a family selector, a 24-bit parameter and a flag byte.
    #[must_use]
    pub fn from_bytes(family: u8, param: [u8; 3], flags: u8) -> Self {
        let raw = u32::from(param[0]) | (u32::from(param[1]) << 8) | (u32::from(param[2]) << 16);
        match family % 5 {
            0 => Self::FixedRate {
                bits_per_block: 1 + raw % 2048,
                align: if flags & 1 == 0 {
                    ZfpStreamAlignment::None
                } else {
                    ZfpStreamAlignment::WordAligned
                },
            },
            1 => Self::FixedPrecision {
                precision: 1 + raw % 64,
            },
            2 => Self::FixedAccuracy {
                // -1075..=843, matching `mode_strategy()` in the proptests plus
                // one below ZFP_MIN_EXP, which selects reversible.
                min_exp: -1075 + i32::try_from(raw % 1919).unwrap_or(0),
            },
            3 => Self::Reversible,
            _ => {
                let min_bits = MIN_EXPERT_BITS + raw % (MAX_BITS - MIN_EXPERT_BITS);
                let max_bits = min_bits + (u32::from(flags) % (MAX_BITS - min_bits + 1));
                Self::Expert {
                    min_bits,
                    max_bits,
                    max_prec: 1 + (raw >> 8) % 64,
                    min_exp: -1074 + i32::try_from((raw >> 16) % 1918).unwrap_or(0),
                }
            }
        }
    }

    /// Bind this mode to a scalar type and dimensionality.
    ///
    /// Returns `None` for parameter combinations `zfp-rs` cannot represent.
    #[must_use]
    pub fn to_config(self, ty: ZfpScalarType, dims: ZfpDimensionality) -> Option<ZfpConfig> {
        let config = match self {
            Self::FixedRate {
                bits_per_block,
                align,
            } => {
                // `fixed_rate` takes bits per *scalar*; convert from the
                // per-block budget, exactly as `apply_mode_rust` does in
                // `tests/proptest/compress_compat.rs`.
                let block = f64::from(1u32 << (2 * u32::from(dims)));
                ZfpConfig::fixed_rate(f64::from(bits_per_block) / block, ty, dims, align)
            }
            Self::FixedPrecision { precision } => ZfpConfig::fixed_precision(precision),
            Self::FixedAccuracy { min_exp } => ZfpConfig::fixed_accuracy(libm::ldexp(1.0, min_exp)),
            Self::Reversible => ZfpConfig::reversible(),
            Self::Expert {
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            } => ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp),
        };
        Some(config)
    }
}

impl<'a> Arbitrary<'a> for ModeSpec {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let family = u8::arbitrary(u)?;
        let param = <[u8; 3]>::arbitrary(u)?;
        let flags = u8::arbitrary(u)?;
        Ok(Self::from_bytes(family, param, flags))
    }
}

// ---------------------------------------------------------------------------
// Execution policy
// ---------------------------------------------------------------------------

/// Serial or (when the `rayon` feature is on) parallel execution.
///
/// `ZfpExecution::Rayon` exists unconditionally in `zfp-rs` but silently falls
/// back to serial when the feature is off (`src/bitstream/owned.rs`). Gating
/// the variant here stops the fuzzer burning half its inputs on a duplicate
/// code path in non-rayon builds, mirroring `execution_strategy()` in the
/// proptests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecSpec {
    Serial,
    #[cfg(feature = "rayon")]
    Rayon {
        threads: u32,
        chunk_size: u32,
    },
}

impl ExecSpec {
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        #[cfg(feature = "rayon")]
        {
            if b.is_multiple_of(2) {
                Self::Serial
            } else {
                Self::Rayon {
                    threads: u32::from(b >> 1) % 5,
                    chunk_size: u32::from(b >> 3) % 65,
                }
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            let _ = b;
            Self::Serial
        }
    }

    #[must_use]
    pub fn to_execution(self) -> ZfpExecution {
        match self {
            Self::Serial => ZfpExecution::Serial,
            #[cfg(feature = "rayon")]
            Self::Rayon {
                threads,
                chunk_size,
            } => ZfpExecution::Rayon {
                threads,
                chunk_size,
            },
        }
    }
}

impl<'a> Arbitrary<'a> for ExecSpec {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(Self::from_byte(u8::arbitrary(u)?))
    }
}
