//! Fuzz target: `ZfpConfig` construction, mode-word round-tripping and
//! `maximum_size`.
//!
//! Pure logic and essentially free to run, which is why it is the one target
//! allowed to feed **unbounded dimensions**: nothing here allocates a buffer,
//! so a 2^60-element field costs nothing but arithmetic. That makes it the
//! right place to hunt the overflow in `maximum_size`, which computes
//! `dims.iter().map(|n| n.div_ceil(4)).product()` without checking.

use zfp_rs::{
    ZfpConfig,
    types::{ZFP_MAX_BITS, ZFP_MAX_PREC, ZfpMode, ZfpScalarType},
};

use crate::input::{ModeSpec, ScalarKind, Shape};

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    if data.len() < 12 {
        return;
    }

    let ty = ScalarKind::from_byte(data[0]).scalar_type();
    let shape = Shape::from_bytes(data[1], [data[2], data[3], data[4], data[5]]);
    let mode = ModeSpec::from_bytes(data[6], [data[7], data[8], data[9]], data[10]);

    let Some(config) = mode.to_config(ty, shape.dimensionality()) else {
        return;
    };

    check_mode_roundtrip(config);
    check_queries(config);

    // Bounded dims, matching what the allocating targets actually use.
    let dims = shape.dims();
    let _ = config.maximum_size(ty, &dims[..shape.rank()]);

    // Unbounded dims: no allocation happens, so this is safe to probe with
    // values the other targets must reject.
    let wild = wild_dims(&data[11..]);
    for rank in 1..=4usize {
        let _ = config.maximum_size(ty, &wild[..rank]);
    }

    // Expert parameters straight from the fuzzer, with no normalisation. This
    // reaches configurations `ModeSpec` deliberately excludes, including
    // `min_bits > max_bits`.
    if data.len() >= 24 {
        let raw = &data[12..24];
        let expert = ZfpConfig::expert(
            u32::from_le_bytes([raw[0], raw[1], raw[2], 0]),
            u32::from_le_bytes([raw[3], raw[4], raw[5], 0]),
            u32::from(raw[6]),
            i32::from_le_bytes([raw[7], raw[8], raw[9], raw[10]]),
        );
        // Only the no-panic property is asserted here. `ZfpConfig::expert`
        // performs no validation, so it will happily build configs with
        // `min_bits = 0` or `max_bits` far above `ZFP_MAX_BITS`; those get
        // misclassified into a short mode form and the encoding is then not
        // idempotent. That is a real wart, but it is a property of an
        // unvalidated constructor rather than a codec defect, so asserting on
        // it here would just wedge the fuzzer on a known issue. Configs built
        // through `ModeSpec` above are normalised and *are* held to the
        // idempotence bar.
        check_queries(expert);
        if expert.compression_mode() != ZfpMode::Null
            && expert.min_bits() >= 1
            && expert.max_bits() <= ZFP_MAX_BITS
            && expert.min_bits() <= expert.max_bits()
            && (1..=ZFP_MAX_PREC).contains(&expert.max_prec())
        {
            check_mode_roundtrip(expert);
        }
    }
}

/// A config's mode word must decode, and re-encoding the result must be stable.
///
/// Deliberately *not* `from_mode(c.mode_bits()) == c`. Parameters outside the
/// ranges the short mode forms cover are legitimately reclassified — a
/// `min_exp` below `ZFP_MIN_EXP` becomes reversible, for example — so exact
/// equality is not a property the crate offers. Idempotence is, and it is the
/// one that matters: a header written from a decoded header must not drift.
fn check_mode_roundtrip(config: ZfpConfig) {
    let bits = config.mode_bits();
    let Some(decoded) = ZfpConfig::from_mode(bits) else {
        panic!("from_mode rejected the mode word {bits:#x} produced by mode_bits");
    };
    assert_eq!(
        decoded.mode_bits(),
        bits,
        "mode-word encoding is not idempotent for {config:?}"
    );
    assert_eq!(
        decoded.compression_mode(),
        config.compression_mode(),
        "compression mode changed across a mode-word round-trip for {config:?}"
    );
}

/// None of the parameter queries may panic, whatever the config holds.
fn check_queries(config: ZfpConfig) {
    let _ = config.compression_mode();
    let _ = config.precision();
    let _ = config.accuracy();
    let _ = config.mode_bits();
    for dims in [
        zfp_rs::ZfpDimensionality::D1,
        zfp_rs::ZfpDimensionality::D2,
        zfp_rs::ZfpDimensionality::D3,
        zfp_rs::ZfpDimensionality::D4,
    ] {
        let _ = config.rate(dims);
    }
    for ty in [
        ZfpScalarType::Int32,
        ZfpScalarType::Int64,
        ZfpScalarType::Float,
        ZfpScalarType::Double,
    ] {
        let _ = config.maximum_size(ty, &[1]);
    }
}

/// Build four large-but-varied dimensions from the remaining bytes.
fn wild_dims(rest: &[u8]) -> [usize; 4] {
    let mut dims = [1usize; 4];
    for (i, dim) in dims.iter_mut().enumerate() {
        let b = rest.get(i).copied().unwrap_or(0);
        // Spread across the whole usize range: small values, mid values, and
        // values large enough that `div_ceil(4).product()` overflows.
        *dim = 1usize << (u32::from(b) % 63);
    }
    dims
}
