//! The upstream C library (`zfp-sys`).
//!
//! This must never link `zfp-rs-ffi`, which defines the same `zfp_*` and `stream_*`
//! symbols: the linker would bind every `zfp_sys` call to the Rust definitions,
//! and these benchmarks would silently measure `zfp-rs` instead of C.
//! `assert_c_reference_is_linked` guards against that.
//! The remaining variants live in `api_compare`.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]

#[macro_use]
mod common;

use common::{BenchScalar, Case, ModeKind, OMP_THREADS, STREAM_PAD_BYTES, dims4, rust_config};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::ffi::c_void;
use std::hint::black_box;
use std::time::Duration;

trait CScalar: BenchScalar {
    const C_TYPE: zfp_sys::zfp_type;
}

impl CScalar for i32 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int32;
}

impl CScalar for i64 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int64;
}

impl CScalar for f32 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;
}

impl CScalar for f64 {
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;
}

/// Assert that the `zfp_sys` functions resolve into the upstream `libzfp`.
///
/// Fails if another definition of the `zfp_*` symbols is linked into this binary.
/// Relies on a position-independent executable (the default), where a function's
/// address is its resolved definition and not a PLT stub in the executable.
#[cfg(unix)]
fn assert_c_reference_is_linked() {
    let function = zfp_sys::zfp_compress as *const c_void;
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: `info` is a valid `Dl_info` to write to.
    let found = unsafe { libc::dladdr(function, info.as_mut_ptr()) };
    assert!(found != 0, "dladdr could not locate zfp_sys::zfp_compress");
    // SAFETY: `dladdr` succeeded, so `info` is initialised.
    let info = unsafe { info.assume_init() };
    assert!(
        !info.dli_fname.is_null(),
        "dladdr returned no object name for zfp_sys::zfp_compress"
    );
    // SAFETY: a non-null `dli_fname` is a NUL-terminated string that outlives this call.
    let object = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy();
    let file_name = object.rsplit('/').next().unwrap_or_default();
    assert!(
        file_name.starts_with("libzfp"),
        "zfp_sys::zfp_compress resolves into `{object}`, not the upstream libzfp: \
         another definition of the zfp C symbols (zfp-rs-ffi?) is linked into this binary"
    );
}

#[cfg(not(unix))]
fn assert_c_reference_is_linked() {}

unsafe fn apply_mode_c<T: CScalar>(zfp: *mut zfp_sys::zfp_stream, case: Case) {
    match case.mode {
        ModeKind::FixedRate => unsafe {
            zfp_sys::zfp_stream_set_rate(zfp, common::RATE, T::C_TYPE, case.dims, 0);
        },
        ModeKind::FixedPrecision => unsafe {
            zfp_sys::zfp_stream_set_precision(zfp, common::PRECISION);
        },
        ModeKind::FixedAccuracy => unsafe {
            zfp_sys::zfp_stream_set_accuracy(zfp, common::ACCURACY);
        },
        ModeKind::Reversible => unsafe {
            zfp_sys::zfp_stream_set_reversible(zfp);
        },
    }
}

unsafe fn c_field<T: CScalar>(data: *mut c_void, dims: &[usize]) -> *mut zfp_sys::zfp_field {
    match dims {
        [nx] => unsafe { zfp_sys::zfp_field_1d(data, T::C_TYPE, *nx) },
        [nx, ny] => unsafe { zfp_sys::zfp_field_2d(data, T::C_TYPE, *nx, *ny) },
        [nx, ny, nz] => unsafe { zfp_sys::zfp_field_3d(data, T::C_TYPE, *nx, *ny, *nz) },
        [nx, ny, nz, nw] => unsafe { zfp_sys::zfp_field_4d(data, T::C_TYPE, *nx, *ny, *nz, *nw) },
        _ => std::ptr::null_mut(),
    }
}

struct CStream {
    zfp: *mut zfp_sys::zfp_stream,
    bs: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CStream {
    fn new(capacity: usize) -> Self {
        let mut buf = vec![0u8; capacity + STREAM_PAD_BYTES];
        let bs = unsafe { zfp_sys::stream_open(buf.as_mut_ptr().cast::<c_void>(), buf.len()) };
        assert!(!bs.is_null());
        let zfp = unsafe { zfp_sys::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self { zfp, bs, buf }
    }

    fn with_bytes(bytes: &[u8]) -> Self {
        let mut stream = Self::new(bytes.len());
        stream.buf[..bytes.len()].copy_from_slice(bytes);
        unsafe {
            zfp_sys::stream_close(stream.bs);
            stream.bs =
                zfp_sys::stream_open(stream.buf.as_mut_ptr().cast::<c_void>(), stream.buf.len());
            zfp_sys::zfp_stream_set_bit_stream(stream.zfp, stream.bs);
        }
        stream
    }

    fn rewind(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_rewind(self.zfp);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let size = unsafe { zfp_sys::stream_size(self.bs) };
        self.buf[..size].to_vec()
    }
}

impl Drop for CStream {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_close(self.zfp);
            zfp_sys::stream_close(self.bs);
        }
    }
}

struct CField(*mut zfp_sys::zfp_field);

impl Drop for CField {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_field_free(self.0);
        }
    }
}

fn compressed_c<T: CScalar>(data: &[T], dims: &[usize], case: Case) -> Vec<u8> {
    let capacity = rust_config::<T>(case)
        .maximum_size(T::RUST_TYPE, dims4(dims))
        .expect("maximum size");
    let stream = CStream::new(capacity);
    unsafe {
        apply_mode_c::<T>(stream.zfp, case);
        let field = CField(c_field::<T>(
            data.as_ptr().cast_mut().cast::<c_void>(),
            dims,
        ));
        assert!(!field.0.is_null());
        assert!(zfp_sys::zfp_compress(stream.zfp, field.0) > 0);
    }
    stream.bytes()
}

fn bench_case<T: CScalar>(criterion: &mut Criterion, case: Case) {
    let (data, dims) = T::generate(case.dims);
    let elements = data.len() as u64;
    let capacity = rust_config::<T>(case)
        .maximum_size(T::RUST_TYPE, dims4(&dims))
        .expect("maximum size");
    let case_label = case.label();

    let mut group = criterion.benchmark_group("api_compare");
    group.throughput(Throughput::Elements(elements));

    {
        let mut stream = CStream::new(capacity);
        unsafe {
            apply_mode_c::<T>(stream.zfp, case);
        }
        let field = unsafe {
            CField(c_field::<T>(
                data.as_ptr().cast_mut().cast::<c_void>(),
                &dims,
            ))
        };
        assert!(!field.0.is_null());
        group.bench_function(
            BenchmarkId::new("compress", format!("{case_label}/zfp-sys")),
            |b| {
                b.iter(|| {
                    stream.rewind();
                    let bytes = unsafe { zfp_sys::zfp_compress(stream.zfp, field.0) };
                    black_box(bytes);
                });
            },
        );
    }

    // --- Parallel compress benchmarks ---
    for &threads in OMP_THREADS {
        let mut stream = CStream::new(capacity);
        unsafe {
            apply_mode_c::<T>(stream.zfp, case);
            zfp_sys::zfp_stream_set_omp_threads(stream.zfp, threads);
        }
        let field = unsafe {
            CField(c_field::<T>(
                data.as_ptr().cast_mut().cast::<c_void>(),
                &dims,
            ))
        };
        assert!(!field.0.is_null());
        group.bench_function(
            BenchmarkId::new("compress", format!("{case_label}/zfp-sys-omp{threads}")),
            |b| {
                b.iter(|| {
                    stream.rewind();
                    let bytes = unsafe { zfp_sys::zfp_compress(stream.zfp, field.0) };
                    black_box(bytes);
                });
            },
        );
    }

    let c_bytes = compressed_c::<T>(&data, &dims, case);

    {
        let mut output = vec![T::default(); data.len()];
        let mut stream = CStream::with_bytes(&c_bytes);
        unsafe {
            apply_mode_c::<T>(stream.zfp, case);
        }
        let field = unsafe { CField(c_field::<T>(output.as_mut_ptr().cast::<c_void>(), &dims)) };
        assert!(!field.0.is_null());
        group.bench_function(
            BenchmarkId::new("decompress", format!("{case_label}/zfp-sys")),
            |b| {
                b.iter(|| {
                    stream.rewind();
                    let bytes = unsafe { zfp_sys::zfp_decompress(stream.zfp, field.0) };
                    black_box(bytes);
                });
            },
        );
    }

    // NOTE: zfp-sys-omp decompress is not benchmarked because the zfp C library
    // (v1.0.1) does not yet implement OpenMP decompression. The decompress function
    // table for OMP contains only NULL entries, causing zfp_decompress to return 0
    // immediately without processing any data. See zfp/src/zfp.c:
    //
    //     /* OpenMP; not yet supported */
    //     {{{ NULL }}}}

    group.finish();
}

fn api_compare_c(criterion: &mut Criterion) {
    assert_c_reference_is_linked();
    run_cases!(criterion, bench_case);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(100));
    targets = api_compare_c
}
criterion_main!(benches);
