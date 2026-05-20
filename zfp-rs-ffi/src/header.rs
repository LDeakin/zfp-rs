//! Header and config API: C-level wrappers for config constructors and data symbols.
//!
//! Implements:
//! - Config constructors: `zfp_config_none`, `zfp_config_rate`,
//!   `zfp_config_precision`, `zfp_config_accuracy`, `zfp_config_reversible`,
//!   `zfp_config_expert`
//! - Data symbols: `stream_word_bits`, `zfp_codec_version`, `zfp_library_version`,
//!   `zfp_version_string`
//! - Constants (already in abi.rs)

use crate::abi::{
    zfp_bool, zfp_config, zfp_config__bindgen_ty_1, zfp_config__bindgen_ty_1__bindgen_ty_1,
    zfp_false, zfp_mode_zfp_mode_expert, zfp_mode_zfp_mode_fixed_accuracy,
    zfp_mode_zfp_mode_fixed_precision, zfp_mode_zfp_mode_fixed_rate, zfp_mode_zfp_mode_null,
    zfp_mode_zfp_mode_reversible,
};
use zfp_rs::STREAM_WORD_BITS;

// ===========================================================================
// Data symbols
// ===========================================================================

/// ZFP codec version (5).
pub const ZFP_CODEC_VERSION: u32 = 5;

/// ZFP library version components (major=1, minor=0, patch=1, tweak=0).
pub const ZFP_LIBRARY_MAJOR: u32 = 1;
pub const ZFP_LIBRARY_MINOR: u32 = 0;
pub const ZFP_LIBRARY_PATCH: u32 = 1;
pub const ZFP_LIBRARY_TWEAK: u32 = 0;

/// Version string.
pub const ZFP_VERSION_STRING: &str = "1.0.1";

/// C `stream_word_bits` symbol.
///
/// In the C API this is a global variable. We expose it as a constant.
#[unsafe(no_mangle)]
pub static stream_word_bits: usize = STREAM_WORD_BITS as usize;

/// C `zfp_codec_version` symbol.
#[unsafe(no_mangle)]
pub static zfp_codec_version: u32 = ZFP_CODEC_VERSION;

/// C `zfp_library_version` symbol (encoded as major*10000 + minor*100 + patch).
#[unsafe(no_mangle)]
pub static zfp_library_version: u32 =
    ZFP_LIBRARY_MAJOR * 10000 + ZFP_LIBRARY_MINOR * 100 + ZFP_LIBRARY_PATCH;

/// C `zfp_version_string` symbol.
pub static ZFP_VERSION_STRING_BYTES: [u8; 6] = *b"1.0.1\0";

// ===========================================================================
// Config constructors
// ===========================================================================

/// Create a `zfp_config` with null mode.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_none() -> zfp_config {
    zfp_config {
        mode: zfp_mode_zfp_mode_null,
        arg: unsafe { std::mem::zeroed() },
    }
}

/// Create a `zfp_config` for fixed-rate mode.
///
/// # Arguments
/// * `rate` - the rate parameter
/// * `align` - whether to align the rate to word boundaries
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_rate(rate: f64, align: zfp_bool) -> zfp_config {
    // Negative rate means "don't align"
    let rate = if align == zfp_false && rate >= 0.0 {
        -rate
    } else {
        rate
    };

    zfp_config {
        mode: zfp_mode_zfp_mode_fixed_rate,
        arg: unsafe { zfp_config__bindgen_ty_1 { rate } },
    }
}

/// Create a `zfp_config` for fixed-precision mode.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_precision(precision: u32) -> zfp_config {
    zfp_config {
        mode: zfp_mode_zfp_mode_fixed_precision,
        arg: unsafe { zfp_config__bindgen_ty_1 { precision } },
    }
}

/// Create a `zfp_config` for fixed-accuracy mode.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_accuracy(tolerance: f64) -> zfp_config {
    zfp_config {
        mode: zfp_mode_zfp_mode_fixed_accuracy,
        arg: unsafe { zfp_config__bindgen_ty_1 { tolerance } },
    }
}

/// Create a `zfp_config` for reversible (lossless) mode.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_reversible() -> zfp_config {
    zfp_config {
        mode: zfp_mode_zfp_mode_reversible,
        arg: unsafe { std::mem::zeroed() },
    }
}

/// Create a `zfp_config` for expert mode.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn zfp_config_expert(
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
) -> zfp_config {
    zfp_config {
        mode: zfp_mode_zfp_mode_expert,
        arg: unsafe {
            zfp_config__bindgen_ty_1 {
                expert: zfp_config__bindgen_ty_1__bindgen_ty_1 {
                    minbits,
                    maxbits,
                    maxprec,
                    minexp,
                },
            }
        },
    }
}
