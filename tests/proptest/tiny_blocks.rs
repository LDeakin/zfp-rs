//! Blocks too small for C's encoder, cross-checked against `zfp-sys`.
//!
//! C's `fwd_cast` scales a block by `2^(30 - emax)` (`2^(62 - emax)` for
//! `f64`), computed in the block's own type. Below 2^-98 (2^-962) the scale
//! overflows, and every value casts to the minimum integer. zfp-rs scales such
//! blocks exactly. The stream format is unchanged, so C decodes zfp-rs's
//! streams, and reversible mode, where C falls back to coding the raw bits,
//! still matches C byte for byte.
#![expect(unsafe_op_in_unsafe_fn)]

use zfp_rs::types::ZfpScalarType;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField, ZfpFieldMut, ZfpScalar,
    ZfpStreamAlignment,
};

trait Float: ZfpScalar + Copy + Default + PartialEq + std::fmt::Debug {
    const TYPE: ZfpScalarType;
    const C_TYPE: zfp_sys::zfp_type;
    fn to_bits_u64(self) -> u64;
    fn from_f64(v: f64) -> Self;
    /// `self * 2^e`.
    fn scale(self, e: i32) -> Self;
}

impl Float for f32 {
    const TYPE: ZfpScalarType = ZfpScalarType::F32;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn scale(self, e: i32) -> Self {
        libm::ldexpf(self, e)
    }
}

impl Float for f64 {
    const TYPE: ZfpScalarType = ZfpScalarType::F64;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn scale(self, e: i32) -> Self {
        libm::ldexp(self, e)
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Rate(f64),
    Precision(u32),
    Reversible,
}

const MODES: [Mode; 3] = [Mode::Rate(16.0), Mode::Precision(20), Mode::Reversible];

fn rs_config<T: Float>(mode: Mode) -> ZfpConfig {
    match mode {
        Mode::Rate(rate) => ZfpConfig::fixed_rate(
            rate,
            T::TYPE,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::Unaligned,
        ),
        Mode::Precision(p) => ZfpConfig::fixed_precision(p),
        Mode::Reversible => ZfpConfig::reversible(),
    }
}

unsafe fn set_c_mode<T: Float>(zfp: *mut zfp_sys::zfp_stream, mode: Mode) {
    match mode {
        Mode::Rate(rate) => {
            zfp_sys::zfp_stream_set_rate(zfp, rate, T::C_TYPE, 1, 0);
        }
        Mode::Precision(p) => {
            zfp_sys::zfp_stream_set_precision(zfp, p);
        }
        Mode::Reversible => zfp_sys::zfp_stream_set_reversible(zfp),
    }
}

fn rs_compress<T: Float>(mode: Mode, data: &[T]) -> Vec<u8> {
    let field = ZfpField::new(data, [data.len()]).unwrap();
    let mut bs = ZfpBitStream::new(1 << 16);
    bs.compress(&rs_config::<T>(mode), &field).unwrap();
    bs.as_bytes().to_vec()
}

fn rs_decompress<T: Float>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let mut field = ZfpFieldMut::new(&mut out, [n]).unwrap();
    let mut bs = ZfpBitStream::from_bytes(bytes);
    bs.decompress(&rs_config::<T>(mode), &mut field).unwrap();
    out
}

/// Run `f` on a C stream over `buf`.
fn with_c_stream<R>(buf: &mut [u8], f: impl FnOnce(*mut zfp_sys::zfp_stream) -> R) -> R {
    unsafe {
        let bs = zfp_sys::stream_open(buf.as_mut_ptr().cast(), buf.len());
        assert!(!bs.is_null());
        let zfp = zfp_sys::zfp_stream_open(bs);
        assert!(!zfp.is_null());
        let r = f(zfp);
        zfp_sys::zfp_stream_close(zfp);
        zfp_sys::stream_close(bs);
        r
    }
}

fn c_compress<T: Float>(mode: Mode, data: &[T]) -> Vec<u8> {
    let mut buf = vec![0u8; 1 << 16];
    let size = with_c_stream(&mut buf, |zfp| unsafe {
        set_c_mode::<T>(zfp, mode);
        let field = zfp_sys::zfp_field_1d(data.as_ptr().cast_mut().cast(), T::C_TYPE, data.len());
        assert!(!field.is_null());
        let size = zfp_sys::zfp_compress(zfp, field);
        zfp_sys::zfp_field_free(field);
        assert!(size > 0);
        size
    });
    buf.truncate(size);
    buf
}

fn c_decompress<T: Float>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
    // A spare word, as the decoder reads a word at a time.
    let mut buf = bytes.to_vec();
    buf.resize(bytes.len() + 8, 0);
    let mut out = vec![T::default(); n];
    with_c_stream(&mut buf, |zfp| unsafe {
        set_c_mode::<T>(zfp, mode);
        let field = zfp_sys::zfp_field_1d(out.as_mut_ptr().cast(), T::C_TYPE, n);
        assert!(!field.is_null());
        assert!(zfp_sys::zfp_decompress(zfp, field) > 0);
        zfp_sys::zfp_field_free(field);
    });
    out
}

fn bits<T: Float>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits_u64()).collect()
}

/// A tiny block compresses as the same block scaled up by `2^up` does, and C
/// decodes zfp-rs's stream to the scaled block's values scaled back down.
fn check_scale_invariance<T: Float>(tiny: &[T], up: i32) {
    let scaled: Vec<T> = tiny.iter().map(|v| v.scale(up)).collect();
    for mode in MODES {
        let bytes = rs_compress(mode, tiny);
        let c_out = c_decompress::<T>(mode, &bytes, tiny.len());
        let rs_out = rs_decompress::<T>(mode, &bytes, tiny.len());
        assert_eq!(bits(&c_out), bits(&rs_out), "{mode:?}: decoders differ");

        let want = rs_decompress::<T>(mode, &rs_compress(mode, &scaled), tiny.len());
        let want: Vec<T> = want.iter().map(|v| v.scale(-up)).collect();
        assert_eq!(bits(&c_out), bits(&want), "{mode:?}: {tiny:?}");
    }
}

/// Values that C, in fixed-precision mode, decodes as `-1.58e-30, -1.58e-30,
/// ...`.
fn tiny_f32() -> Vec<f32> {
    (0..16u8).map(|i| (f32::from(i) - 7.5) * 1.0e-31).collect()
}

#[test]
fn c_decodes_zfp_rs_tiny_f32_blocks() {
    check_scale_invariance(&tiny_f32(), 64);
    // Subnormals among normal values.
    let mixed: Vec<f32> = (0..16u8)
        .map(|i| {
            if i % 3 == 0 {
                f32::from_bits(u32::from(i) * 977)
            } else {
                f32::from(i) * 1.0e-36
            }
        })
        .collect();
    check_scale_invariance(&mixed, 64);
}

#[test]
fn c_decodes_zfp_rs_tiny_f64_blocks() {
    let tiny: Vec<f64> = (0..16u8).map(|i| (f64::from(i) - 7.5) * 1.0e-300).collect();
    check_scale_invariance(&tiny, 64);
}

/// C's own stream of a tiny block does not hold its values.
#[test]
fn c_loses_tiny_blocks() {
    let tiny = tiny_f32();
    let max = tiny.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mode = Mode::Precision(20);
    let c_bytes = c_compress(mode, &tiny);
    assert_ne!(c_bytes, rs_compress(mode, &tiny));
    let c_out = c_decompress::<f32>(mode, &c_bytes, tiny.len());
    let err = tiny
        .iter()
        .zip(&c_out)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(err > max, "{c_out:?}");
}

/// Below 2^-120 (2^-1013), C's decoder's own scale `2^(emax - 30)` underflows,
/// so both libraries decode such a block to zeros.
#[test]
fn underflowing_blocks_decode_to_zeros() {
    let tiny: Vec<f32> = (1..=16u8).map(|i| f32::from(i) * 1.0e-38).collect();
    let tiny64: Vec<f64> = (1..=16u8).map(|i| f64::from(i) * 1.0e-309).collect();
    for mode in [Mode::Rate(16.0), Mode::Precision(20)] {
        let bytes = rs_compress(mode, &tiny);
        let c_out = c_decompress::<f32>(mode, &bytes, tiny.len());
        assert_eq!(
            bits(&c_out),
            bits(&rs_decompress::<f32>(mode, &bytes, tiny.len()))
        );
        assert!(c_out.iter().all(|&v| v == 0.0), "{c_out:?}");

        let bytes = rs_compress(mode, &tiny64);
        let c_out = c_decompress::<f64>(mode, &bytes, tiny64.len());
        assert_eq!(
            bits(&c_out),
            bits(&rs_decompress::<f64>(mode, &bytes, tiny64.len()))
        );
        assert!(c_out.iter().all(|&v| v == 0.0), "{c_out:?}");
    }
}

/// A block whose largest magnitude is `2^e`.
fn block_of<T: Float>(e: i32) -> Vec<T> {
    [1.0, -0.5, 0.25, -0.125]
        .map(|m| T::from_f64(m).scale(e))
        .to_vec()
}

/// C's encoder loses a block exactly when its largest magnitude is below
/// `2^encode`, and C's decoder, when it is below `2^decode`.
fn check_thresholds<T: Float>(encode: i32, decode: i32) {
    for mode in [Mode::Rate(16.0), Mode::Precision(20)] {
        let at = block_of::<T>(encode);
        assert_eq!(rs_compress(mode, &at), c_compress(mode, &at), "{mode:?}");
        let below = block_of::<T>(encode - 1);
        assert_ne!(
            rs_compress(mode, &below),
            c_compress(mode, &below),
            "{mode:?}"
        );

        let at = block_of::<T>(decode);
        let c_out = c_decompress::<T>(mode, &rs_compress(mode, &at), 4);
        assert!(c_out.iter().any(|&v| v != T::default()), "{mode:?}");
        let below = block_of::<T>(decode - 1);
        let c_out = c_decompress::<T>(mode, &rs_compress(mode, &below), 4);
        assert!(c_out.iter().all(|&v| v == T::default()), "{mode:?}");
    }
    check_scale_invariance(&block_of::<T>(decode), 64);
}

#[test]
fn tiny_block_thresholds() {
    check_thresholds::<f32>(-98, -120);
    check_thresholds::<f64>(-962, -1013);
}

/// C reinterprets a tiny reversible block, as its cast overflows, and so does
/// zfp-rs.
#[test]
fn reversible_tiny_blocks_match_c() {
    let tiny = tiny_f32();
    let subnormal: Vec<f32> = (0..16u32).map(|i| f32::from_bits(i * 12_345)).collect();
    let tiny64: Vec<f64> = (0..16u8).map(|i| (f64::from(i) - 7.5) * 1.0e-300).collect();
    let mode = Mode::Reversible;
    for data in [&tiny, &subnormal] {
        let bytes = rs_compress(mode, data);
        assert_eq!(bytes, c_compress(mode, data));
        assert_eq!(
            bits(&rs_decompress::<f32>(mode, &bytes, data.len())),
            bits(data)
        );
    }
    let bytes = rs_compress(mode, &tiny64);
    assert_eq!(bytes, c_compress(mode, &tiny64));
    assert_eq!(
        bits(&rs_decompress::<f64>(mode, &bytes, tiny64.len())),
        bits(&tiny64)
    );
}
