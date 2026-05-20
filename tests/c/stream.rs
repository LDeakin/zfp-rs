#![allow(clippy::cast_precision_loss)] // usize→f64 for rate computation
#![allow(clippy::float_cmp)] // exact float comparison in tests
//! Port of `zfp/tests/src/misc/testZfpParameters.c`.

use zfp_rs::types::{ZFP_MAX_PREC, ZFP_MIN_EXP};
use zfp_rs::{ZfpConfig, ZfpDimensionality, ZfpMode, ZfpScalarType, ZfpStreamAlignment};

// expert mode compression parameters (from testZfpParameters.c)
const MIN_BITS: u32 = 11;
const MAX_BITS: u32 = 1001;
const MAX_PREC: u32 = 52;
const MIN_EXP: i32 = -1000;
const MAX_EXP: i32 = 1023;

fn setup() -> ZfpConfig {
    ZfpConfig::new()
}

// ---------------------------------------------------------------------------
// compression_mode() tests
// ---------------------------------------------------------------------------

#[test]
fn given_opened_zfp_stream_when_zfp_stream_compression_mode_expect_returns_expert_enum() {
    let config = setup();
    // default values imply expert mode
    assert_eq!(config.compression_mode(), ZfpMode::Expert);
}

#[cfg(feature = "ffi")]
#[test]
fn given_zfp_stream_set_with_invalid_params_when_zfp_stream_compression_mode_expect_returns_null_enum()
 {
    // With immutable ZfpConfig, we can't force invalid config.
    // The expert constructor accepts any values, but compression_mode() will
    // return Null for truly invalid configs.
    let setup = setup();
    // Invalid: min_bits > max_bits
    let config = ZfpConfig::expert(
        setup.max_bits() + 1,
        setup.max_bits(),
        setup.max_prec(),
        setup.min_exp(),
    );
    // Since min_bits > max_bits, compression_mode returns Null
    assert_eq!(config.compression_mode(), ZfpMode::Null);
}

#[test]
fn given_zfp_stream_set_with_fixed_rate_when_zfp_stream_compression_mode_expect_returns_fixed_rate_enum()
 {
    for zfp_type in [
        ZfpScalarType::Int32,
        ZfpScalarType::Int64,
        ZfpScalarType::Float,
        ZfpScalarType::Double,
    ] {
        let max_rate = match zfp_type {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 32,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 64,
        };
        for dims in [
            ZfpDimensionality::D1,
            ZfpDimensionality::D2,
            ZfpDimensionality::D3,
            ZfpDimensionality::D4,
        ] {
            for rate in 1..=max_rate {
                for align in [ZfpStreamAlignment::None, ZfpStreamAlignment::WordAligned] {
                    let config = ZfpConfig::fixed_rate(f64::from(rate), zfp_type, dims, align);
                    let mode = config.compression_mode();
                    assert_eq!(
                        mode,
                        ZfpMode::FixedRate,
                        "type={zfp_type:?} rate={rate} align={align:?} dims={dims:?} → {mode:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn given_zfp_stream_set_with_fixed_precision_when_zfp_stream_compression_mode_expect_returns_fixed_precision_enum()
 {
    // ZFP_MAX_PREC is treated as expert mode; test 1..ZFP_MAX_PREC-1
    for prec in 1..ZFP_MAX_PREC {
        let config = ZfpConfig::fixed_precision(prec);
        let mode = config.compression_mode();
        assert_eq!(mode, ZfpMode::FixedPrecision, "prec={prec} → {mode:?}");
    }
}

#[test]
fn given_zfp_stream_set_with_fixed_accuracy_when_zfp_stream_compression_mode_expect_returns_fixed_accuracy_enum()
 {
    // loop accExp from MAX_EXP down to ZFP_MIN_EXP+1 (using ZFP_MIN_EXP implies expert mode)
    let mut acc_exp = MAX_EXP;
    while acc_exp > ZFP_MIN_EXP {
        let tol = libm::ldexp(1.0, acc_exp);
        if tol == 0.0 {
            acc_exp -= 1;
            continue;
        }
        let config = ZfpConfig::fixed_accuracy(tol);
        let mode = config.compression_mode();
        assert_eq!(mode, ZfpMode::FixedAccuracy, "acc_exp={acc_exp} → {mode:?}");
        acc_exp -= 1;
    }
}

#[test]
fn given_zfp_stream_set_with_reversible_when_zfp_stream_compression_mode_expect_returns_reversible_enum()
 {
    let config = ZfpConfig::reversible();
    let mode = config.compression_mode();
    assert_eq!(mode, ZfpMode::Reversible, "→ {mode:?}");
}

// ---------------------------------------------------------------------------
// set_mode() tests
// ---------------------------------------------------------------------------

#[test]
fn given_zfp_stream_in_expert_mode_when_set_mode_with_expert_mode_bits_expect_params_unchanged() {
    let config = setup();

    let mode_bits = config.mode_bits();

    // set a non-expert mode, then restore via from_mode
    let non_expert = ZfpConfig::fixed_precision(ZFP_MAX_PREC - 2);
    assert_ne!(non_expert.compression_mode(), ZfpMode::Expert);

    let restored = ZfpConfig::from_mode(mode_bits).unwrap();
    assert_eq!(restored.compression_mode(), ZfpMode::Expert);
    assert_eq!(restored, config);
}

#[test]
fn given_zfp_stream_set_fixed_rate_when_set_mode_with_those_bits_expect_fixed_rate_mode_set() {
    for zfp_type in [
        ZfpScalarType::Int32,
        ZfpScalarType::Int64,
        ZfpScalarType::Float,
        ZfpScalarType::Double,
    ] {
        let max_rate = match zfp_type {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 32,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 64,
        };
        for dims in [
            ZfpDimensionality::D1,
            ZfpDimensionality::D2,
            ZfpDimensionality::D3,
            ZfpDimensionality::D4,
        ] {
            for rate in 1..=max_rate {
                for align in [ZfpStreamAlignment::None, ZfpStreamAlignment::WordAligned] {
                    let config = ZfpConfig::fixed_rate(f64::from(rate), zfp_type, dims, align);
                    assert_eq!(config.compression_mode(), ZfpMode::FixedRate);

                    let mode_bits = config.mode_bits();

                    let restored = ZfpConfig::from_mode(mode_bits).unwrap();
                    assert_eq!(
                        restored.compression_mode(),
                        ZfpMode::FixedRate,
                        "type={zfp_type:?} rate={rate} align={align:?} dims={dims:?} → {:?}",
                        restored.compression_mode()
                    );
                    assert_eq!(restored, config);
                }
            }
        }
    }
}

#[test]
fn given_zfp_stream_set_fixed_precision_when_set_mode_with_those_bits_expect_fixed_precision_mode_set()
 {
    for prec in 1..ZFP_MAX_PREC {
        let config = ZfpConfig::fixed_precision(prec);
        assert_eq!(config.compression_mode(), ZfpMode::FixedPrecision);

        let mode_bits = config.mode_bits();

        let restored = ZfpConfig::from_mode(mode_bits).unwrap();
        assert_eq!(
            restored.compression_mode(),
            ZfpMode::FixedPrecision,
            "prec={prec} → {:?}",
            restored.compression_mode()
        );
        assert_eq!(restored, config);
    }
}

#[test]
fn given_zfp_stream_set_fixed_accuracy_when_set_mode_with_those_bits_expect_fixed_accuracy_mode_set()
 {
    let mut acc_exp = MAX_EXP;
    while acc_exp > ZFP_MIN_EXP {
        let tol = libm::ldexp(1.0, acc_exp);
        if tol == 0.0 {
            acc_exp -= 1;
            continue;
        }
        let config = ZfpConfig::fixed_accuracy(tol);
        assert_eq!(config.compression_mode(), ZfpMode::FixedAccuracy);

        let mode_bits = config.mode_bits();

        let restored = ZfpConfig::from_mode(mode_bits).unwrap();
        assert_eq!(
            restored.compression_mode(),
            ZfpMode::FixedAccuracy,
            "acc_exp={acc_exp} → {:?}",
            restored.compression_mode()
        );
        assert_eq!(restored, config);
        acc_exp -= 1;
    }
}

#[test]
fn given_zfp_stream_set_reversible_when_set_mode_with_those_bits_expect_reversible_mode_set() {
    let config = ZfpConfig::reversible();
    assert_eq!(config.compression_mode(), ZfpMode::Reversible);

    let mode_bits = config.mode_bits();

    let restored = ZfpConfig::from_mode(mode_bits).unwrap();
    assert_eq!(
        restored.compression_mode(),
        ZfpMode::Reversible,
        "→ {:?}",
        restored.compression_mode()
    );
    assert_eq!(restored, config);
}

#[test]
fn given_zfp_stream_with_expert_params_when_set_mode_with_those_bits_expect_expert_params_set() {
    let config = ZfpConfig::expert(MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP);

    let mode_bits = config.mode_bits();

    let restored = ZfpConfig::from_mode(mode_bits).unwrap();
    assert_eq!(
        restored.compression_mode(),
        ZfpMode::Expert,
        "→ {:?}",
        restored.compression_mode()
    );
    assert_eq!(restored, config);
}

// ---------------------------------------------------------------------------
// set_params / accessors / maximum_size tests
// ---------------------------------------------------------------------------

#[test]
fn given_zfp_stream_when_set_params_with_valid_params_expect_returns_true() {
    // Valid params create a stream without error
    let config = ZfpConfig::expert(MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP);
    assert_eq!(config.compression_mode(), ZfpMode::Expert);
}

#[test]
fn given_zfp_stream_when_set_params_with_invalid_params_expect_returns_false() {
    // Invalid params (min_bits > max_bits) create a stream with Null mode
    let config = ZfpConfig::expert(MAX_BITS + 1, MAX_BITS, MAX_PREC, MIN_EXP);
    assert_eq!(config.compression_mode(), ZfpMode::Null);
}

#[test]
fn given_zfp_stream_when_zfp_stream_rate_expect_rate_returned() {
    for zfp_type in [
        ZfpScalarType::Int32,
        ZfpScalarType::Int64,
        ZfpScalarType::Float,
        ZfpScalarType::Double,
    ] {
        let scalar_bits = match zfp_type {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 32usize,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 64,
        };
        for dims in [
            ZfpDimensionality::D1,
            ZfpDimensionality::D2,
            ZfpDimensionality::D3,
            ZfpDimensionality::D4,
        ] {
            for i in 1usize..=4 {
                let rate = scalar_bits as f64 * i as f64 / 4.0;
                for align in [ZfpStreamAlignment::None, ZfpStreamAlignment::WordAligned] {
                    let config = ZfpConfig::fixed_rate(rate, zfp_type, dims, align);
                    let actual = config.rate(dims);
                    // When align=WordAligned, fixed_rate rounds up to the next word boundary (64 bits),
                    // so the actual rate may differ from the requested rate.
                    let expected =
                        f64::from(config.max_bits()) / f64::from(1u32 << (2 * u32::from(dims)));
                    assert_eq!(
                        actual, expected,
                        "type={zfp_type:?} rate={rate} align={align:?} dims={dims:?}: got {actual}, want {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn given_zfp_stream_when_zfp_stream_precision_expect_precision_returned() {
    for prec in 1..ZFP_MAX_PREC {
        let config = ZfpConfig::fixed_precision(prec);
        let actual = config.precision();
        assert_eq!(actual, prec, "prec={prec}: got {actual}");
    }
}

#[test]
fn given_zfp_stream_when_zfp_stream_accuracy_expect_accuracy_returned() {
    let mut acc_exp = MAX_EXP;
    while acc_exp > ZFP_MIN_EXP {
        let tol = libm::ldexp(1.0, acc_exp);
        if tol == 0.0 {
            acc_exp -= 1;
            continue;
        }
        let config = ZfpConfig::fixed_accuracy(tol);
        let actual = config.accuracy();
        assert_eq!(actual, tol, "acc_exp={acc_exp}: got {actual}, want {tol}");
        acc_exp -= 1;
    }
}

#[test]
fn given_zfp_stream_when_maximum_size_expect_nonzero_size_returned() {
    // use a non-trivial field shape to confirm a real calculation
    let size = ZfpConfig::new().maximum_size(ZfpScalarType::Double, &[33, 401]);
    assert!(size > 0, "maximum_size returned 0");
}
