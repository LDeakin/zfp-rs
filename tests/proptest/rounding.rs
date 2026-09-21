//! Property-based tests for `ZfpRounding`.
//!
//! No C counterpart: `zfp-sys` only exposes the coupled `ZFP_ROUND_FIRST` +
//! `ZFP_WITH_TIGHT_ERROR` build. These assert the relationships the modes are
//! defined by instead. Byte-equality with C is covered by `tests/c_rounding.rs`.

use proptest::prelude::*;
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpRounding};

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

/// Compress then decompress a 4x4x4 `f64` block with the given config.
fn round_trip(config: &ZfpConfig, data: &[f64]) -> (Vec<u8>, Vec<f64>) {
    let mut bs = ZfpBitStream::new(4096);
    bs.compress(config, &ZfpField::new(data, [4usize, 4, 4]))
        .expect("compress");
    bs.flush();
    let bytes = bs.as_bytes().to_vec();

    let mut out = vec![0f64; data.len()];
    bs.rewind();
    bs.decompress(config, &mut ZfpFieldMut::new(&mut out, [4usize, 4, 4]))
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

    /// Reversible mode is unaffected by `Never` and `First`, and its stream is
    /// unaffected by all three.
    ///
    /// `Last` is the exception on decode: upstream's `revdecode.c` shares
    /// `decode_ints` with the lossy path, so `inv_round` biases reversible
    /// coefficients too. That makes `Last` + reversible lossy in C, and this
    /// crate matches it rather than silently diverging.
    #[test]
    fn reversible_is_lossless_except_under_round_last(
        data in normal_f64s(),
        rounding in rounding_strategy(),
    ) {
        let config = ZfpConfig::reversible().with_rounding(rounding);
        let (bytes, out) = round_trip(&config, &data);

        // Reversible encode never rounds: `revencode.c` calls `encode_ints`
        // directly, bypassing the `fwd_round` in `encode_block`.
        let (want_bytes, _) = round_trip(&ZfpConfig::reversible(), &data);
        prop_assert_eq!(bytes, want_bytes);

        if matches!(rounding, ZfpRounding::Last { .. }) {
            return Ok(());
        }
        for (&want, &got) in data.iter().zip(out.iter()) {
            prop_assert!(want.to_bits() == got.to_bits(), "{} != {}", want, got);
        }
    }
}
