//! Compression modes and execution policies shared by the whole-field
//! compatibility tests, applied alike to zfp-rs and `zfp-sys`.
#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use zfp_rs::{ZfpConfig, ZfpDimensionality, ZfpExecution, types::ZfpScalarType};

// ---------------------------------------------------------------------------
// Execution strategy
// ---------------------------------------------------------------------------

pub(crate) fn execution_strategy() -> impl Strategy<Value = ZfpExecution> {
    #[cfg(feature = "rayon")]
    {
        prop_oneof![
            Just(ZfpExecution::Serial),
            (0u32..=4u32, 0u32..=64u32).prop_map(|(threads, chunk_size)| {
                ZfpExecution::Rayon {
                    threads,
                    chunk_size,
                }
            }),
        ]
    }
    #[cfg(not(feature = "rayon"))]
    {
        Just(ZfpExecution::Serial)
    }
}

// ---------------------------------------------------------------------------
// Compression mode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) enum Mode {
    FixedRate(u32),      // bits per block (1..=2048)
    FixedPrecision(u32), // precision (1..=64)
    FixedAccuracy(i32),  // min_exp (-1074..=843)
    Reversible,
    Expert {
        min_bits: u32,
        max_bits: u32,
        max_prec: u32,
        min_exp: i32,
    },
}

impl Mode {
    /// Whether an all-zero float block is padded to `min_bits`.
    ///
    /// C's reversible encoder writes such a block as one bit, although its
    /// decoder skips to `min_bits`, so C cannot decode its own stream. zfp-rs
    /// pads, and so differs from C there.
    pub(crate) fn pads_zero_blocks(&self) -> bool {
        matches!(*self, Mode::Expert { min_bits, min_exp, .. } if min_bits > 1 && min_exp < -1074)
    }
}

pub(crate) fn mode_strategy() -> impl Strategy<Value = Mode> {
    prop_oneof![
        (1u32..=2048u32).prop_map(Mode::FixedRate),
        (1u32..=64u32).prop_map(Mode::FixedPrecision),
        (-1074i32..=843i32).prop_map(Mode::FixedAccuracy),
        Just(Mode::Reversible),
        expert_strategy(),
    ]
}

/// The four preset modes, without expert parameters.
pub(crate) fn preset_mode_strategy() -> impl Strategy<Value = Mode> {
    prop_oneof![
        (1u32..=2048u32).prop_map(Mode::FixedRate),
        (1u32..=64u32).prop_map(Mode::FixedPrecision),
        (-1074i32..=843i32).prop_map(Mode::FixedAccuracy),
        Just(Mode::Reversible),
    ]
}

/// Expert parameters, reversible or not, on which C and zfp-rs agree.
///
/// `max_bits` is at least 19, the longest block header (reversible `f64`).
/// Below a header, C's unsigned budget wraps around and leaves the block
/// unbounded, where zfp-rs gives it no budget.
pub(crate) fn expert_strategy() -> impl Strategy<Value = Mode> {
    (
        prop_oneof![19u32..=600u32, 19u32..=16658u32],
        1u32..=64u32,
        prop_oneof![Just(-1075i32), -1074i32..=843i32],
    )
        .prop_flat_map(|(max_bits, max_prec, min_exp)| {
            (1u32..=max_bits).prop_map(move |min_bits| Mode::Expert {
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            })
        })
}

pub(crate) fn apply_mode_rust(
    config: &mut ZfpConfig,
    mode: &Mode,
    ty: ZfpScalarType,
    dims: ZfpDimensionality,
) {
    match *mode {
        Mode::FixedRate(bits) => {
            *config = ZfpConfig::fixed_rate(
                f64::from(bits) / f64::from(1u32 << (2 * u32::from(dims))),
                ty,
                dims,
                zfp_rs::ZfpStreamAlignment::Unaligned,
            )
            .unwrap();
        }
        Mode::FixedPrecision(p) => {
            *config = ZfpConfig::fixed_precision(p);
        }
        Mode::FixedAccuracy(e) => {
            *config = ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e));
        }
        Mode::Reversible => {
            *config = ZfpConfig::reversible();
        }
        Mode::Expert {
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        } => {
            *config = ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp)
                .expect("the strategy generates valid parameters");
        }
    }
}

/// # Safety
/// `zfp` must be an open stream.
pub(crate) unsafe fn apply_mode_c(
    zfp: *mut zfp_sys::zfp_stream,
    mode: &Mode,
    c_type: zfp_sys::zfp_type,
    dims: u32,
) {
    match *mode {
        Mode::FixedRate(bits) => {
            zfp_sys::zfp_stream_set_rate(
                zfp,
                f64::from(bits) / f64::from(1u32 << (2 * dims)),
                c_type,
                dims,
                0,
            );
        }
        Mode::FixedPrecision(p) => {
            zfp_sys::zfp_stream_set_precision(zfp, p);
        }
        Mode::FixedAccuracy(e) => {
            zfp_sys::zfp_stream_set_accuracy(zfp, libm::ldexp(1.0, e));
        }
        Mode::Reversible => {
            zfp_sys::zfp_stream_set_reversible(zfp);
        }
        Mode::Expert {
            min_bits,
            max_bits,
            max_prec,
            min_exp,
        } => {
            assert_ne!(
                zfp_sys::zfp_stream_set_params(zfp, min_bits, max_bits, max_prec, min_exp),
                0
            );
        }
    }
}
