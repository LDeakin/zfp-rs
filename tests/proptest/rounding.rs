//! Property-based tests for `ZfpRounding`.
//!
//! No C counterpart: `zfp-sys` only exposes the coupled `ZFP_ROUND_FIRST` +
//! `ZFP_WITH_TIGHT_ERROR` build. These assert the relationships the modes are
//! defined by instead. Byte-equality with C is covered by
//! `zfp-round-tests/tests/c_rounding.rs` (encode) and `zfp-rs-ffi`'s `ffi_compat`
//! with `round-tight-error` (encode and decode), and the other settings by
//! `zfp-rs-ffi/tests/c_rounding_builds.rs`.

use proptest::prelude::*;
use zfp_rs::codec::block::decode_block;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField, ZfpFieldMut, ZfpHeaderMask, ZfpRounding,
};

fn rounding_strategy() -> impl Strategy<Value = ZfpRounding> {
    prop_oneof![
        Just(ZfpRounding::Never),
        any::<bool>().prop_map(|tight_error| ZfpRounding::First { tight_error }),
        any::<bool>().prop_map(|tight_error| ZfpRounding::Last { tight_error }),
    ]
}

fn normal_f64s() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        any::<f64>().prop_filter("normal or zero", |f| !f.is_subnormal()),
        64,
    )
}

/// Values a reversible block codes in few bit planes. `ZFP_ROUND_LAST`'s bias
/// is a no-op at full precision, which arbitrary bit patterns need.
fn small_integer_f64s() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec((-1000i32..1000).prop_map(f64::from), 64)
}

/// Compress then decompress a 4x4x4 `f64` block with the given config.
fn round_trip(config: &ZfpConfig, data: &[f64]) -> (Vec<u8>, Vec<f64>) {
    let mut bs = ZfpBitStream::new(4096).unwrap();
    bs.compress(config, &ZfpField::new(data, [4usize, 4, 4]).unwrap())
        .expect("compress");
    bs.flush();
    let bytes = bs.as_bytes().to_vec();

    let mut out = vec![0f64; data.len()];
    bs.rewind();
    bs.decompress(
        config,
        &mut ZfpFieldMut::new(&mut out, [4usize, 4, 4]).unwrap(),
    )
    .expect("decompress");
    (bytes, out)
}

proptest! {
    /// `ZFP_ROUND_LAST`'s bias is decode-only, so its stream is byte-identical
    /// to an unbiased encoder using the same precision.
    #[test]
    fn round_last_matches_the_stream_of_its_precision_peer(
        data in normal_f64s(),
        e in -1074i32..=843i32,
        tight_error in any::<bool>(),
    ) {
        let base = ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e));
        // `Never` does not bias coefficients. Doubling its tolerance drops one
        // bit plane, matching `Last`'s precision when tight error is enabled.
        let reference = if tight_error {
            ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e + 1))
        } else {
            base
        };
        let (want, _) = round_trip(&reference, &data);
        let (got, _) = round_trip(&base.with_rounding(ZfpRounding::Last { tight_error }), &data);
        prop_assert_eq!(got, want);
    }

    /// Every variant round-trips within the requested fixed-accuracy tolerance.
    ///
    /// Values are bounded: random f64 bit patterns are almost all astronomically
    /// large, where an absolute error bound says nothing useful.
    #[test]
    fn fixed_accuracy_holds_for_every_rounding(
        data in prop::collection::vec(-1.0e6f64..1.0e6f64, 64),
        e in -20i32..=10i32,
        rounding in rounding_strategy(),
    ) {
        let tolerance = libm::ldexp(1.0, e);
        let config = ZfpConfig::fixed_accuracy(tolerance).with_rounding(rounding);
        let (_, out) = round_trip(&config, &data);
        for (i, (&want, &got)) in data.iter().zip(out.iter()).enumerate() {
            prop_assert!(
                (want - got).abs() <= tolerance,
                "index {}: |{} - {}| > {}", i, want, got, tolerance
            );
        }
    }

    /// `ZFP_ROUND_FIRST` and `ZFP_WITH_TIGHT_ERROR` both change the encoded bits,
    /// so neither can silently no-op.
    #[test]
    fn round_first_and_tight_error_change_the_bitstream(
        data in prop::collection::vec(-1.0e6f64..1.0e6f64, 64),
        e in -20i32..=10i32,
    ) {
        // A flat block codes to a single bit plane, where the bias is invisible.
        let max_abs = data.iter().fold(0f64, |m, v| m.max(v.abs()));
        prop_assume!(max_abs > 1.0);

        let base = ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e));
        let (never, _) = round_trip(&base, &data);
        let (first, _) = round_trip(
            &base.with_rounding(ZfpRounding::First { tight_error: false }),
            &data,
        );
        let (tight, _) = round_trip(
            &base.with_rounding(ZfpRounding::First { tight_error: true }),
            &data,
        );
        prop_assert_ne!(&first, &never);
        prop_assert_ne!(&tight, &first);
    }

    /// Reversible mode, stream and values, is unaffected by rounding.
    ///
    /// Upstream's `revdecode.c` shares `decode_ints` with the lossy path, so
    /// `inv_round` biases reversible coefficients too, and a `ZFP_ROUND_LAST`
    /// build is lossy in reversible mode. This crate does not reproduce that.
    #[test]
    fn reversible_is_lossless_under_every_rounding(
        data in prop_oneof![normal_f64s(), small_integer_f64s()],
        rounding in rounding_strategy(),
    ) {
        let config = ZfpConfig::reversible().with_rounding(rounding);
        let (bytes, out) = round_trip(&config, &data);

        // Reversible encode never rounds: `revencode.c` calls `encode_ints`
        // directly, bypassing the `fwd_round` in `encode_block`.
        let (want_bytes, _) = round_trip(&ZfpConfig::reversible(), &data);
        prop_assert_eq!(&bytes, &want_bytes);

        let mut block = vec![0f64; data.len()];
        decode_block(
            &mut ZfpBitStream::from_bytes(&bytes).unwrap(),
            &config,
            &mut block,
            ZfpDimensionality::D3,
        )
        .expect("decode block");
        prop_assert_eq!(
            block.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        for (&want, &got) in data.iter().zip(out.iter()) {
            prop_assert!(want.to_bits() == got.to_bits(), "{} != {}", want, got);
        }
    }
}

/// The header cannot carry rounding, so a reader must reapply it.
#[test]
fn header_config_needs_the_encoders_rounding() {
    let data: Vec<f64> = (0..64).map(|i| f64::from(i).sin() * 1.0e3).collect();
    let rounding = ZfpRounding::First { tight_error: true };
    let config = ZfpConfig::fixed_accuracy(1.0e-3).with_rounding(rounding);
    let field = ZfpField::new(&data, [4usize, 4, 4]).unwrap();

    let mut bs = ZfpBitStream::new(4096).unwrap();
    bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)
        .expect("write header");
    bs.compress(&config, &field).expect("compress");
    bs.flush();
    let (_, want) = round_trip(&config, &data);

    bs.rewind();
    let header = bs.read_header(ZfpHeaderMask::FULL).expect("header");
    let read = header.config.expect("mode");
    assert_eq!(read.rounding(), ZfpRounding::Never);
    let mut out = vec![0f64; data.len()];
    bs.decompress(
        &read.with_rounding(rounding),
        &mut ZfpFieldMut::new(&mut out, [4usize, 4, 4]).unwrap(),
    )
    .expect("decompress");
    assert_eq!(out, want);
}
