//! Fixed-accuracy blocks spanning a wide dynamic range, cross-checked against
//! `zfp-sys`.
//!
//! The tolerance is not a bound for such blocks: the largest value sets the
//! block exponent, and 64 bits of precision need not reach the smallest.
//! C behaves the same (`zfp/docs/source/faq.rst`, Q17), so these tests pin
//! zfp-rs to C's bytes and values, and record where the tolerance holds.

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut};

const C_DOUBLE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;

/// Run `f` on a fixed-accuracy C stream over `buf`.
fn with_c_stream<R>(
    buf: &mut [u8],
    tolerance: f64,
    f: impl FnOnce(*mut zfp_sys::zfp_stream) -> R,
) -> R {
    unsafe {
        let bs = zfp_sys::stream_open(buf.as_mut_ptr().cast(), buf.len());
        assert!(!bs.is_null());
        let zfp = zfp_sys::zfp_stream_open(bs);
        assert!(!zfp.is_null());
        zfp_sys::zfp_stream_set_accuracy(zfp, tolerance);
        let r = f(zfp);
        zfp_sys::zfp_stream_close(zfp);
        zfp_sys::stream_close(bs);
        r
    }
}

fn c_compress(tolerance: f64, data: &[f64; 4]) -> Vec<u8> {
    let mut buf = vec![0u8; 1024];
    let size = with_c_stream(&mut buf, tolerance, |zfp| unsafe {
        let field = zfp_sys::zfp_field_1d(data.as_ptr().cast_mut().cast(), C_DOUBLE, 4);
        assert!(!field.is_null());
        let size = zfp_sys::zfp_compress(zfp, field);
        zfp_sys::zfp_field_free(field);
        assert!(size > 0);
        size
    });
    buf.truncate(size);
    buf
}

fn c_decompress(tolerance: f64, bytes: &[u8]) -> [f64; 4] {
    // A spare word, as the decoder reads a word at a time.
    let mut buf = bytes.to_vec();
    buf.resize(bytes.len() + 8, 0);
    let mut out = [0.0; 4];
    with_c_stream(&mut buf, tolerance, |zfp| unsafe {
        let field = zfp_sys::zfp_field_1d(out.as_mut_ptr().cast(), C_DOUBLE, 4);
        assert!(!field.is_null());
        assert!(zfp_sys::zfp_decompress(zfp, field) > 0);
        zfp_sys::zfp_field_free(field);
    });
    out
}

/// Compress `data` with both libraries, check that the bytes and both decoders
/// agree, and return the decoded values.
fn roundtrip_matches_c(tolerance: f64, data: [f64; 4]) -> [f64; 4] {
    let config = ZfpConfig::fixed_accuracy(tolerance);
    let field = ZfpField::new(&data, [4usize]).unwrap();
    let mut bs = ZfpBitStream::new(1024);
    bs.compress(&config, &field).unwrap();
    let bytes = bs.as_bytes().to_vec();
    assert_eq!(
        bytes,
        c_compress(tolerance, &data),
        "compressed bytes differ"
    );

    let mut out = [0.0; 4];
    let mut field = ZfpFieldMut::new(&mut out, [4usize]).unwrap();
    ZfpBitStream::from_bytes(&bytes)
        .decompress(&config, &mut field)
        .unwrap();
    let c_out = c_decompress(tolerance, &bytes);
    assert_eq!(
        out.map(f64::to_bits),
        c_out.map(f64::to_bits),
        "decoders differ"
    );
    out
}

fn max_error(data: &[f64; 4], out: &[f64; 4]) -> f64 {
    data.iter()
        .zip(out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

const WIDE: [f64; 4] = [7.9e-24, 1e30, -1e30, 1e30];

#[test]
fn wide_range_block_meets_a_coarse_tolerance() {
    let out = roundtrip_matches_c(6.2e26, WIDE);
    assert!(max_error(&WIDE, &out) <= 6.2e26, "{out:?}");
}

/// Precision runs out above the small value, which decodes to zero: an error
/// of 7.9e-24 against a tolerance of 1e-24.
#[test]
fn wide_range_block_exceeds_a_fine_tolerance_as_in_c() {
    let out = roundtrip_matches_c(1e-24, WIDE);
    assert_eq!(out[0].to_bits(), 0.0f64.to_bits(), "{out:?}");
    assert!(max_error(&WIDE, &out) > 1e-24, "{out:?}");
}

#[test]
fn largest_magnitudes_reconstruct_exactly() {
    let data = [f64::MAX, f64::MAX, -f64::MAX, f64::MAX];
    let out = roundtrip_matches_c(1.0, data);
    assert_eq!(out.map(f64::to_bits), data.map(f64::to_bits));
}
