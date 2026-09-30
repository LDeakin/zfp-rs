//! The examples in `docs/differences-from-c.md`, checked against `zfp-sys`.
//!
//! Unless stated otherwise, each compresses the document's 16 `f64` values,
//! `(0.37 * i).sin() * 100.0` for `i` in `0..16`, as a 1-D field.

use crate::c_1d::{
    Mode, Scalar, bits, c_compress, c_decompress, rs_compress, rs_decompress, rs_try_decompress,
    with_c_stream,
};
use proptest::prelude::*;
use zfp_rs::{ZfpDecompressionError, ZfpScalar};

/// The document's 16 values.
fn values() -> Vec<f64> {
    (0..16u8)
        .map(|i| (0.37 * f64::from(i)).sin() * 100.0)
        .collect()
}

fn c_maximum_size<T: Scalar>(mode: Mode, n: usize) -> usize {
    let mut buf = [0u8; 8];
    with_c_stream::<T, _>(
        mode,
        &mut buf,
        std::ptr::null_mut(),
        n,
        |zfp, field| unsafe { zfp_sys::zfp_stream_maximum_size(zfp, field) },
    )
}

/// The largest error in `data` once C decodes `bytes`, to two significant
/// figures.
fn c_error<T: Scalar>(mode: Mode, bytes: &[u8], data: &[T]) -> String {
    let out = c_decompress::<T>(mode, bytes, data.len());
    let err = data
        .iter()
        .zip(&out)
        .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
        .fold(0.0, f64::max);
    format!("{err:.1e}")
}

/// "`max_bits` below the block header": stream sizes, in bytes.
#[test]
fn max_bits_below_the_header() {
    let data = values();
    for (mode, c_len, rs_len) in [
        (Mode::Expert(1, 11, 64, -1074), 136, 8),
        (Mode::Expert(12, 12, 64, -1074), 8, 8),
        (Mode::Expert(1, 18, 64, -1075), 120, 16),
        (Mode::Expert(19, 19, 64, -1075), 16, 16),
    ] {
        let c = c_compress(mode, &data);
        let rs = rs_compress(mode, &data);
        assert_eq!((c.len(), rs.len()), (c_len, rs_len), "{mode:?}");
        if c_len == rs_len {
            assert_eq!(c, rs, "{mode:?}");
        }
    }
}

/// An expert mode whose `max_bits` is at least `header`, and 16 values.
fn from_the_header<T: Scalar>(
    header: u32,
    reversible: bool,
    values: impl Strategy<Value = Vec<T>>,
) -> impl Strategy<Value = (Mode, Vec<T>)> {
    let min_exp = if reversible {
        Just(-1075).boxed()
    } else {
        (-1074i32..=843).boxed()
    };
    (header.max(1)..=40, 1u32..=64, min_exp, values).prop_flat_map(
        |(max_bits, max_prec, min_exp, data)| {
            (1..=max_bits).prop_map(move |min_bits| {
                (
                    Mode::Expert(min_bits, max_bits, max_prec, min_exp),
                    data.clone(),
                )
            })
        },
    )
}

fn ints<T: Scalar + prop::arbitrary::Arbitrary>() -> impl Strategy<Value = Vec<T>> {
    prop::collection::vec(any::<T>(), 16)
}

/// Floats, none of them zero, so no reversible block is all zero, and none
/// too small for C's scale. The integers take the reversible coder's longest
/// header, which casts them to integers, where arbitrary floats do not cast
/// exactly.
fn floats<T: Scalar + From<i16>>(
    any_float: impl Strategy<Value = T>,
) -> impl Strategy<Value = Vec<T>> {
    let integer = (-1000i16..=1000)
        .prop_filter("nonzero", |&i| i != 0)
        .prop_map(T::from);
    prop_oneof![
        prop::collection::vec(any_float, 16),
        prop::collection::vec(integer, 16),
    ]
}

fn comparable_f32() -> impl Strategy<Value = f32> {
    let min = libm::ldexpf(1.0, -98);
    any::<f32>().prop_filter("NaN or at least 2^-98", move |f| {
        f.is_nan() || f.abs() >= min
    })
}

fn comparable_f64() -> impl Strategy<Value = f64> {
    let min = libm::ldexp(1.0, -962);
    any::<f64>().prop_filter("NaN or at least 2^-962", move |f| {
        f.is_nan() || f.abs() >= min
    })
}

macro_rules! from_the_header_tests {
    ($($name:ident: $scalar:ty, $header:expr, $reversible:expr, $values:expr;)+) => {
        proptest! {
            $(
                /// "`max_bits` below the block header": at or above the
                /// largest header, C's bytes.
                #[test]
                fn $name((mode, data) in from_the_header::<$scalar>($header, $reversible, $values)) {
                    prop_assert_eq!(rs_compress(mode, &data), c_compress(mode, &data), "{:?}", mode);
                }
            )+
        }
    };
}

from_the_header_tests! {
    i32_matches_c_from_the_header: i32, 0, false, ints();
    i64_matches_c_from_the_header: i64, 0, false, ints();
    f32_matches_c_from_the_header: f32, 9, false, floats(comparable_f32());
    f64_matches_c_from_the_header: f64, 12, false, floats(comparable_f64());
    i32_reversible_matches_c_from_the_header: i32, 5, true, ints();
    i64_reversible_matches_c_from_the_header: i64, 6, true, ints();
    f32_reversible_matches_c_from_the_header: f32, 15, true, floats(comparable_f32());
    f64_reversible_matches_c_from_the_header: f64, 19, true, floats(comparable_f64());
}

/// "`maximum_size` under-reports".
#[test]
fn maximum_size_under_reports() {
    let data = values();
    let mode = Mode::Expert(1, 11, 64, -1074);
    assert_eq!(c_maximum_size::<f64>(mode, 16), 24);
    assert_eq!(c_compress(mode, &data).len(), 136);
    let config = mode.config::<f64>();
    assert_eq!(config.maximum_size(f64::SCALAR_TYPE, [16usize]), Some(32));
    assert_eq!(rs_compress(mode, &data).len(), 8);
}

/// "Reversible all-zero blocks are not padded", with the first four values
/// zero.
#[test]
fn reversible_zero_blocks_are_padded() {
    let mut data = values();
    data[..4].fill(0.0);

    // C cannot decode its own stream for any `min_bits` above 1: its decoder
    // reads past the end of it, as far as `min_bits` past the first block.
    // zfp-rs reports a stream it runs off the end of, and decodes the stream
    // as C does when given the zeros C reads there. Its own stream
    // round-trips.
    for min_bits in (2..=128).chain([1000, 16658]) {
        let mode = Mode::Expert(min_bits, 16658, 64, -1075);
        let c = c_compress(mode, &data);
        let c_out = c_decompress::<f64>(mode, &c, 16);
        assert_ne!(bits(&c_out), bits(&data), "{mode:?}");
        match rs_try_decompress::<f64>(mode, &c, 16) {
            Ok(out) => assert_eq!(bits(&out), bits(&c_out), "{mode:?}"),
            Err(ZfpDecompressionError::Truncated { .. }) => {}
            Err(e) => panic!("{mode:?}: {e}"),
        }
        let mut padded = c.clone();
        padded.resize(c.len() + (1 << 16), 0);
        assert_eq!(bits(&rs_decompress::<f64>(mode, &padded, 16)), bits(&c_out));
        // For these, C's decoder reads past the end of its own stream by more
        // than the padding of the last word.
        if min_bits >= 1000 {
            assert!(
                matches!(
                    rs_try_decompress::<f64>(mode, &c, 16),
                    Err(ZfpDecompressionError::Truncated { .. })
                ),
                "{mode:?}"
            );
        }
        let rs = rs_compress(mode, &data);
        assert_eq!(bits(&rs_decompress::<f64>(mode, &rs, 16)), bits(&data));
        if min_bits == 100 {
            assert_eq!((c.len(), rs.len()), (96, 104));
        }
    }

    let mode = Mode::Expert(1, 16658, 64, -1075);
    let c = c_compress(mode, &data);
    assert_eq!(c.len(), 96);
    assert_eq!(bits(&c_decompress::<f64>(mode, &c, 16)), bits(&data));
    assert_eq!(rs_compress(mode, &data), c);
}

/// "Blocks of tiny magnitude lose their values": stream sizes, and the
/// largest error once C decodes each stream.
#[test]
fn tiny_blocks_lose_their_values() {
    fn row<T: Scalar>(mode: Mode, data: &[T]) -> [(usize, String); 2] {
        [c_compress(mode, data), rs_compress(mode, data)]
            .map(|bytes| (bytes.len(), c_error(mode, &bytes, data)))
    }
    let tiny32: Vec<f32> = (0..16u8).map(|i| (f32::from(i) - 7.5) * 1.0e-31).collect();
    let tiny64: Vec<f64> = values().iter().map(|v| v * 1.0e-300).collect();
    for (got, want) in [
        (
            row(Mode::Precision(16), &tiny32),
            [(40, "2.3e-30"), (32, "3.1e-35")],
        ),
        (
            row(Mode::Rate(8.0), &tiny32),
            [(16, "2.3e-30"), (16, "6.1e-33")],
        ),
        (
            row(Mode::Precision(16), &tiny64),
            [(40, "4.8e-298"), (40, "1.1e-302")],
        ),
    ] {
        assert_eq!(got, want.map(|(len, err)| (len, err.to_string())));
    }
}
