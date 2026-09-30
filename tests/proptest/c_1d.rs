//! Coding 1-D fields with zfp-rs and with `zfp-sys`, for the tests that
//! compare the two on hand-picked values.

use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDecompressionError, ZfpDimensionality, ZfpField, ZfpFieldMut,
    ZfpScalar, ZfpStreamAlignment,
};

pub(crate) trait Scalar: ZfpScalar + Copy + Default + std::fmt::Debug {
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

pub(crate) fn bits<T: Scalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits_u64()).collect()
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode {
    /// `min_bits`, `max_bits`, `max_prec`, `min_exp`.
    Expert(u32, u32, u32, i32),
    Rate(f64),
    Precision(u32),
    Reversible,
}

impl Mode {
    pub(crate) fn config<T: Scalar>(self) -> ZfpConfig {
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
            Mode::Reversible => ZfpConfig::reversible(),
        }
    }

    /// # Safety
    /// `zfp` must be an open stream.
    pub(crate) unsafe fn set_c<T: Scalar>(self, zfp: *mut zfp_sys::zfp_stream) {
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
                Mode::Reversible => zfp_sys::zfp_stream_set_reversible(zfp),
            }
        }
    }
}

pub(crate) fn rs_compress<T: Scalar>(mode: Mode, data: &[T]) -> Vec<u8> {
    let field = ZfpField::new(data, [data.len()]).unwrap();
    let mut bs = ZfpBitStream::new(1 << 16).unwrap();
    bs.compress(&mode.config::<T>(), &field).unwrap();
    bs.as_bytes().to_vec()
}

pub(crate) fn rs_try_decompress<T: Scalar>(
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

pub(crate) fn rs_decompress<T: Scalar>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
    rs_try_decompress(mode, bytes, n).unwrap()
}

/// Run `f` on a C stream in `mode` over `buf`, with a 1-D field of `n` values
/// at `data`.
pub(crate) fn with_c_stream<T: Scalar, R>(
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

pub(crate) fn c_compress<T: Scalar>(mode: Mode, data: &[T]) -> Vec<u8> {
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

pub(crate) fn c_decompress<T: Scalar>(mode: Mode, bytes: &[u8], n: usize) -> Vec<T> {
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
