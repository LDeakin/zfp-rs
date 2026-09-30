//! The examples in `docs/differences-from-c.md`, checked against `zfp-sys`.
//!
//! Unless stated otherwise, each compresses the document's 16 `f64` values,
//! `(0.37 * i).sin() * 100.0` for `i` in `0..16`, as a 1-D field.

use proptest::prelude::*;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDecompressionError, ZfpDimensionality, ZfpField, ZfpFieldMut,
    ZfpScalar, ZfpStreamAlignment,
};

trait Scalar: ZfpScalar + Copy + Default + std::fmt::Debug {
    const C_TYPE: zfp_sys::zfp_type;
    fn to_bits_u64(self) -> u64;
    fn to_f64(self) -> f64;
}

impl Scalar for i32 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int32;
    fn to_bits_u64(self) -> u64 {
        u64::from(self.cast_unsigned())
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Scalar for i64 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int64;
    fn to_bits_u64(self) -> u64 {
        self.cast_unsigned()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Scalar for f32 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Scalar for f64 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// The document's 16 values.
fn values() -> Vec<f64> {
    (0..16u8)
        .map(|i| (0.37 * f64::from(i)).sin() * 100.0)
        .collect()
}

fn bits<T: Scalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits_u64()).collect()
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// `min_bits`, `max_bits`, `max_prec`, `min_exp`.
    Expert(u32, u32, u32, i32),
    Rate(f64),
    Precision(u32),
}

impl Mode {
    fn config<T: Scalar>(self) -> ZfpConfig {
        match self {
            Mode::Expert(min_bits, max_bits, max_prec, min_exp) => {
                ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp).unwrap()
            }
            Mode::Rate(rate) => ZfpConfig::fixed_rate(
                rate,
                T::SCALAR_TYPE,
                ZfpDimensionality::D1,
                ZfpStreamAlignment::Unaligned,
            )
            .unwrap(),
            Mode::Precision(p) => ZfpConfig::fixed_precision(p),
        }
    }

    /// # Safety
    /// `zfp` must be an open stream.
    unsafe fn set_c<T: Scalar>(self, zfp: *mut zfp_sys::zfp_stream) {
        unsafe {
            match self {
                Mode::Expert(min_bits, max_bits, max_prec, min_exp) => {
                    assert_ne!(
                        zfp_sys::zfp_stream_set_params(zfp, min_bits, max_bits, max_prec, min_exp),
                        0
                    );
                }
                Mode::Rate(rate) => {
                    zfp_sys::zfp_stream_set_rate(zfp, rate, T::C_TYPE, 1, 0);
                }
                Mode::Precision(p) => {
                    zfp_sys::zfp_stream_set_precision(zfp, p);
                }
            }
        }
    }
}

fn rs_compress<T: Scalar>(mode: Mode, data: &[T]) -> Vec<u8> {
    let field = ZfpField::new(data, [data.len()]).unwrap();
    let mut bs = ZfpBitStream::new(1 << 16).unwrap();
    bs.compress(&mode.config::<T>(), &field).unwrap();
    bs.as_bytes().to_vec()
}

fn rs_try_decompress<T: Scalar>(
    mode: Mode,
    bytes: &[u8],
    n: usize,
) -> Result<Vec<T>, ZfpDecompressionError> {
    let mut out = vec![T::default(); n];
    let mut field = ZfpFieldMut::new(&mut out, [n]).unwrap();
    let mut bs = ZfpBitStream::from_bytes(bytes).unwrap();
    bs.decompress(&mode.config::<T>(), &mut field)?;
    Ok(out)
}

fn rs_decompress<T: Scalar>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
    rs_try_decompress(mode, bytes, n).unwrap()
}

/// Run `f` on a C stream in `mode` over `buf`, with a 1-D field of `n` values
/// at `data`.
fn with_c_stream<T: Scalar, R>(
    mode: Mode,
    buf: &mut [u8],
    data: *mut T,
    n: usize,
    f: impl FnOnce(*mut zfp_sys::zfp_stream, *mut zfp_sys::zfp_field) -> R,
) -> R {
    unsafe {
        let bs = zfp_sys::stream_open(buf.as_mut_ptr().cast(), buf.len());
        assert!(!bs.is_null());
        let zfp = zfp_sys::zfp_stream_open(bs);
        assert!(!zfp.is_null());
        mode.set_c::<T>(zfp);
        let field = zfp_sys::zfp_field_1d(data.cast(), T::C_TYPE, n);
        assert!(!field.is_null());
        let r = f(zfp, field);
        zfp_sys::zfp_field_free(field);
        zfp_sys::zfp_stream_close(zfp);
        zfp_sys::stream_close(bs);
        r
    }
}

fn c_compress<T: Scalar>(mode: Mode, data: &[T]) -> Vec<u8> {
    let mut buf = vec![0u8; 1 << 16];
    let size = with_c_stream(
        mode,
        &mut buf,
        data.as_ptr().cast_mut(),
        data.len(),
        |zfp, field| unsafe { zfp_sys::zfp_compress(zfp, field) },
    );
    assert!(size > 0);
    buf.truncate(size);
    buf
}

fn c_decompress<T: Scalar>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
    // Zero padding, as C reads past the end of a stream that does not decode.
    let mut buf = bytes.to_vec();
    buf.resize(bytes.len() + (1 << 16), 0);
    let mut out = vec![T::default(); n];
    let size = with_c_stream(mode, &mut buf, out.as_mut_ptr(), n, |zfp, field| unsafe {
        zfp_sys::zfp_decompress(zfp, field)
    });
    assert!(size > 0);
    out
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
