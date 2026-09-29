//! Cross-checks against libzfp built with each rounding setting that
//! `zfp-sys` cannot build: `ZFP_ROUND_FIRST` without `ZFP_WITH_TIGHT_ERROR`,
//! and `ZFP_ROUND_LAST` with and without it. `zfp-round-tests` covers
//! `ZFP_ROUND_FIRST` with tight error, which `zfp-sys` does build.
//!
//! As in `ffi_compat`, CMake builds each library as a shared object, which is
//! loaded at run time so that its symbols do not clash with this crate's.

use libloading::Library;
use proptest::prelude::*;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpRounding, ZfpScalar};
use zfp_rs_ffi::{
    zfp_type, zfp_type_zfp_type_double, zfp_type_zfp_type_float, zfp_type_zfp_type_int32,
    zfp_type_zfp_type_int64,
};

/// The C builds, by the rounding they are built with.
const BUILDS: [ZfpRounding; 3] = [
    ZfpRounding::First { tight_error: false },
    ZfpRounding::Last { tight_error: false },
    ZfpRounding::Last { tight_error: true },
];

struct Api {
    _lib: Library,
    stream_open: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    stream_close: unsafe extern "C" fn(*mut c_void),
    zfp_stream_open: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    zfp_stream_close: unsafe extern "C" fn(*mut c_void),
    zfp_stream_set_params: unsafe extern "C" fn(*mut c_void, u32, u32, u32, i32) -> i32,
    zfp_field_1d: unsafe extern "C" fn(*mut c_void, zfp_type, usize) -> *mut c_void,
    zfp_field_2d: unsafe extern "C" fn(*mut c_void, zfp_type, usize, usize) -> *mut c_void,
    zfp_field_3d: unsafe extern "C" fn(*mut c_void, zfp_type, usize, usize, usize) -> *mut c_void,
    zfp_field_4d:
        unsafe extern "C" fn(*mut c_void, zfp_type, usize, usize, usize, usize) -> *mut c_void,
    zfp_field_free: unsafe extern "C" fn(*mut c_void),
    zfp_compress: unsafe extern "C" fn(*mut c_void, *const c_void) -> usize,
    zfp_decompress: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize,
}

/// The C library built with `rounding`.
fn api(rounding: ZfpRounding) -> &'static Api {
    static APIS: [OnceLock<Api>; BUILDS.len()] = [const { OnceLock::new() }; BUILDS.len()];
    let build = BUILDS
        .iter()
        .position(|&b| b == rounding)
        .expect("a C build");
    APIS[build].get_or_init(|| unsafe {
        let path = build_shared_zfp(rounding);
        let lib = Library::new(&path)
            .unwrap_or_else(|err| panic!("failed to load {}: {err}", path.display()));
        Api {
            stream_open: sym(&lib, b"stream_open\0"),
            stream_close: sym(&lib, b"stream_close\0"),
            zfp_stream_open: sym(&lib, b"zfp_stream_open\0"),
            zfp_stream_close: sym(&lib, b"zfp_stream_close\0"),
            zfp_stream_set_params: sym(&lib, b"zfp_stream_set_params\0"),
            zfp_field_1d: sym(&lib, b"zfp_field_1d\0"),
            zfp_field_2d: sym(&lib, b"zfp_field_2d\0"),
            zfp_field_3d: sym(&lib, b"zfp_field_3d\0"),
            zfp_field_4d: sym(&lib, b"zfp_field_4d\0"),
            zfp_field_free: sym(&lib, b"zfp_field_free\0"),
            zfp_compress: sym(&lib, b"zfp_compress\0"),
            zfp_decompress: sym(&lib, b"zfp_decompress\0"),
            _lib: lib,
        }
    })
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> T {
    unsafe { *lib.get::<T>(name).expect("symbol") }
}

fn build_shared_zfp(rounding: ZfpRounding) -> PathBuf {
    let (mode, tight_error) = match rounding {
        ZfpRounding::Never => unreachable!("zfp-sys builds the default"),
        ZfpRounding::First { tight_error } => ("FIRST", tight_error),
        ZfpRounding::Last { tight_error } => ("LAST", tight_error),
    };
    let mut defines = vec![format!("-DZFP_ROUNDING_MODE=ZFP_ROUND_{mode}")];
    // The directory names match `ffi_compat`'s.
    let mut suffix = format!("-{}", mode.to_lowercase());
    if tight_error {
        defines.push("-DZFP_WITH_TIGHT_ERROR=ON".to_string());
        suffix.push_str("-tight");
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let build_dir = workspace
        .join("target")
        .join(format!("zfp-rs-ffi-c-ref-shared{suffix}"));
    run(Command::new("cmake")
        .arg("-S")
        .arg(workspace.join("zfp"))
        .arg("-B")
        .arg(&build_dir)
        .arg("-DBUILD_SHARED_LIBS=ON")
        .arg("-DBUILD_TESTING=OFF")
        .arg("-DBUILD_UTILITIES=OFF")
        .arg("-DBUILD_EXAMPLES=OFF")
        .arg("-DZFP_WITH_OPENMP=OFF")
        .args(&defines));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("zfp"));
    if cfg!(target_os = "macos") {
        build_dir.join("lib").join("libzfp.dylib")
    } else if cfg!(target_os = "windows") {
        build_dir.join("bin").join("zfp.dll")
    } else {
        build_dir.join("lib").join("libzfp.so")
    }
}

fn run(command: &mut Command) {
    let output = command.output().expect("cmake runs");
    assert!(
        output.status.success(),
        "{command:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

trait Scalar: ZfpScalar + Copy + Default + std::fmt::Debug {
    const TYPE: zfp_type;
    fn to_bits_u64(self) -> u64;
}

impl Scalar for i32 {
    const TYPE: zfp_type = zfp_type_zfp_type_int32;
    fn to_bits_u64(self) -> u64 {
        u64::from(self.cast_unsigned())
    }
}

impl Scalar for i64 {
    const TYPE: zfp_type = zfp_type_zfp_type_int64;
    fn to_bits_u64(self) -> u64 {
        self.cast_unsigned()
    }
}

impl Scalar for f32 {
    const TYPE: zfp_type = zfp_type_zfp_type_float;
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Scalar for f64 {
    const TYPE: zfp_type = zfp_type_zfp_type_double;
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}

fn bits<T: Scalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits_u64()).collect()
}

/// `min_bits`, `max_bits`, `max_prec`, `min_exp`.
type Params = (u32, u32, u32, i32);

fn config((min_bits, max_bits, max_prec, min_exp): Params, rounding: ZfpRounding) -> ZfpConfig {
    ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp)
        .unwrap()
        .with_rounding(rounding)
}

/// Run `f` on a stream of the C build for `rounding`, with `params`, over
/// `buf`, and a field of `dims` at `data`.
fn with_c_stream<T: Scalar, R>(
    rounding: ZfpRounding,
    params: Params,
    buf: &mut [u8],
    data: *mut T,
    dims: &[usize],
    f: impl FnOnce(&Api, *mut c_void, *mut c_void) -> R,
) -> R {
    let api = api(rounding);
    let (min_bits, max_bits, max_prec, min_exp) = params;
    let data = data.cast();
    unsafe {
        let bs = (api.stream_open)(buf.as_mut_ptr().cast(), buf.len());
        assert!(!bs.is_null());
        let zfp = (api.zfp_stream_open)(bs);
        assert_ne!(
            (api.zfp_stream_set_params)(zfp, min_bits, max_bits, max_prec, min_exp),
            0
        );
        let field = match *dims {
            [nx] => (api.zfp_field_1d)(data, T::TYPE, nx),
            [nx, ny] => (api.zfp_field_2d)(data, T::TYPE, nx, ny),
            [nx, ny, nz] => (api.zfp_field_3d)(data, T::TYPE, nx, ny, nz),
            [nx, ny, nz, nw] => (api.zfp_field_4d)(data, T::TYPE, nx, ny, nz, nw),
            _ => unreachable!("1-4 dimensions"),
        };
        assert!(!field.is_null());
        let r = f(api, zfp, field);
        (api.zfp_field_free)(field);
        (api.zfp_stream_close)(zfp);
        (api.stream_close)(bs);
        r
    }
}

const CAPACITY: usize = 1 << 20;

fn c_compress<T: Scalar>(
    rounding: ZfpRounding,
    params: Params,
    data: &[T],
    dims: &[usize],
) -> Vec<u8> {
    let mut buf = vec![0u8; CAPACITY];
    let data = data.as_ptr().cast_mut();
    let size = with_c_stream(
        rounding,
        params,
        &mut buf,
        data,
        dims,
        |api, zfp, field| unsafe { (api.zfp_compress)(zfp, field) },
    );
    assert!(size > 0);
    buf.truncate(size);
    buf
}

fn c_decompress<T: Scalar>(
    rounding: ZfpRounding,
    params: Params,
    bytes: &[u8],
    dims: &[usize],
) -> Vec<T> {
    // A spare word, as the decoder reads a word at a time.
    let mut buf = bytes.to_vec();
    buf.resize(bytes.len() + 8, 0);
    let mut out = vec![T::default(); dims.iter().product()];
    let data = out.as_mut_ptr();
    let size = with_c_stream(
        rounding,
        params,
        &mut buf,
        data,
        dims,
        |api, zfp, field| unsafe { (api.zfp_decompress)(zfp, field) },
    );
    assert!(size > 0);
    out
}

/// `dims` with unused axes zero, as zfp-rs takes them.
fn rs_dims(dims: &[usize]) -> [usize; 4] {
    std::array::from_fn(|axis| dims.get(axis).copied().unwrap_or(0))
}

fn rs_compress<T: Scalar>(config: &ZfpConfig, data: &[T], dims: &[usize]) -> Vec<u8> {
    let mut bs = ZfpBitStream::new(CAPACITY).unwrap();
    bs.compress(config, &ZfpField::new(data, rs_dims(dims)).unwrap())
        .unwrap();
    bs.as_bytes().to_vec()
}

fn rs_decompress<T: Scalar>(config: &ZfpConfig, bytes: &[u8], dims: &[usize]) -> Vec<T> {
    let mut out = vec![T::default(); dims.iter().product()];
    ZfpBitStream::from_bytes(bytes)
        .unwrap()
        .decompress(
            config,
            &mut ZfpFieldMut::new(&mut out, rs_dims(dims)).unwrap(),
        )
        .unwrap();
    out
}

/// C biases reversible coefficients when it rounds last, which makes its
/// reversible mode lossy. zfp-rs does not, and writes the same streams.
#[test]
fn c_reversible_decoding_is_lossy_when_rounding_last() {
    let params = (1, 16658, 64, -1075);
    let data: Vec<i32> = (0..16).map(|i| 8 * i).collect();
    for rounding in BUILDS {
        let config = config(params, rounding);
        let bytes = rs_compress(&config, &data, &[16]);
        assert_eq!(c_compress(rounding, params, &data, &[16]), bytes);
        let c_out = c_decompress::<i32>(rounding, params, &bytes, &[16]);
        if matches!(rounding, ZfpRounding::Last { .. }) {
            assert_eq!(c_out[..6], [1, 10, 20, 32, 33, 42], "{rounding:?}");
        } else {
            assert_eq!(c_out, data, "{rounding:?}");
        }
        assert_eq!(rs_decompress::<i32>(&config, &bytes, &[16]), data);
    }
}

/// Lossy expert parameters. `max_bits` is at least 19, the longest block
/// header, below which C's budget wraps around.
fn params() -> impl Strategy<Value = Params> {
    (
        prop_oneof![19u32..=600, 19u32..=16658],
        1u32..=64,
        -1074i32..=843,
    )
        .prop_flat_map(|(max_bits, max_prec, min_exp)| {
            (1..=max_bits).prop_map(move |min_bits| (min_bits, max_bits, max_prec, min_exp))
        })
}

/// A field of 1-4 dimensions, each 1-9 long, and its values.
fn field<T: Scalar>(
    value: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (Vec<usize>, Vec<T>)> {
    prop::collection::vec(1usize..=9, 1..=4).prop_flat_map(move |dims| {
        let n = dims.iter().product::<usize>();
        (Just(dims), prop::collection::vec(value.clone(), n))
    })
}

// C's scale overflows for a block whose largest magnitude is below 2^-98
// (`f32`) or 2^-962 (`f64`), where zfp-rs deliberately differs.
fn comparable_f32() -> impl Strategy<Value = f32> + Clone {
    let min = f32::from_bits(29 << 23);
    any::<f32>().prop_filter("zero, NaN or at least 2^-98", move |f| {
        *f == 0.0 || f.is_nan() || f.abs() >= min
    })
}

fn comparable_f64() -> impl Strategy<Value = f64> + Clone {
    let min = f64::from_bits(61 << 52);
    any::<f64>().prop_filter("zero, NaN or at least 2^-962", move |f| {
        *f == 0.0 || f.is_nan() || f.abs() >= min
    })
}

/// zfp-rs, with each C build's rounding, encodes and decodes as that build
/// does.
fn check_each_build<T: Scalar>(
    params: Params,
    dims: &[usize],
    data: &[T],
) -> Result<(), TestCaseError> {
    for rounding in BUILDS {
        let config = config(params, rounding);
        let bytes = rs_compress(&config, data, dims);
        prop_assert_eq!(
            &c_compress(rounding, params, data, dims),
            &bytes,
            "{:?}",
            rounding
        );
        prop_assert_eq!(
            bits(&c_decompress::<T>(rounding, params, &bytes, dims)),
            bits(&rs_decompress::<T>(&config, &bytes, dims)),
            "{:?}",
            rounding
        );
    }
    Ok(())
}

proptest! {
    #[test]
    fn i32_matches_each_rounding_build(params in params(), (dims, data) in field(any::<i32>())) {
        check_each_build(params, &dims, &data)?;
    }

    #[test]
    fn i64_matches_each_rounding_build(params in params(), (dims, data) in field(any::<i64>())) {
        check_each_build(params, &dims, &data)?;
    }

    #[test]
    fn f32_matches_each_rounding_build(params in params(), (dims, data) in field(comparable_f32())) {
        check_each_build(params, &dims, &data)?;
    }

    #[test]
    fn f64_matches_each_rounding_build(params in params(), (dims, data) in field(comparable_f64())) {
        check_each_build(params, &dims, &data)?;
    }
}
