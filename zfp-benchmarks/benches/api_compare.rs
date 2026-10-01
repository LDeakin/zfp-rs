//! `zfp-rs` and `zfp-rs-ffi`.
//!
//! The upstream C library (`zfp-sys`) is benchmarked by `api_compare_c`.
//! It cannot share a binary with `zfp-rs-ffi`; see `common`.

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
use zfp_rs::{ZfpBitStream, ZfpExecution, ZfpField, ZfpFieldMut};
use zfp_rs_ffi as ffi;

trait FfiScalar: BenchScalar {
    const FFI_TYPE: ffi::zfp_type;
}

impl FfiScalar for i32 {
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_int32;
}

impl FfiScalar for i64 {
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_int64;
}

impl FfiScalar for f32 {
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_float;
}

impl FfiScalar for f64 {
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_double;
}

unsafe fn apply_mode_ffi<T: FfiScalar>(zfp: *mut ffi::zfp_stream, case: Case) {
    match case.mode {
        ModeKind::FixedRate => unsafe {
            ffi::zfp_stream_set_rate(zfp, common::RATE, T::FFI_TYPE, case.dims, 0);
        },
        ModeKind::FixedPrecision => unsafe {
            ffi::zfp_stream_set_precision(zfp, common::PRECISION);
        },
        ModeKind::FixedAccuracy => unsafe {
            ffi::zfp_stream_set_accuracy(zfp, common::ACCURACY);
        },
        ModeKind::Reversible => unsafe {
            ffi::zfp_stream_set_reversible(zfp);
        },
    }
}

fn rust_field<'a, T: BenchScalar>(data: &'a [T], dims: &[usize]) -> ZfpField<'a> {
    ZfpField::new(data, dims4(dims)).unwrap()
}

fn rust_field_mut<'a, T: BenchScalar>(data: &'a mut [T], dims: &[usize]) -> ZfpFieldMut<'a> {
    ZfpFieldMut::new(data, dims4(dims)).unwrap()
}

fn ffi_field<T: FfiScalar>(data: *mut c_void, dims: &[usize]) -> *mut ffi::zfp_field {
    match dims {
        [nx] => unsafe { ffi::zfp_field_1d(data, T::FFI_TYPE, *nx) },
        [nx, ny] => unsafe { ffi::zfp_field_2d(data, T::FFI_TYPE, *nx, *ny) },
        [nx, ny, nz] => unsafe { ffi::zfp_field_3d(data, T::FFI_TYPE, *nx, *ny, *nz) },
        [nx, ny, nz, nw] => unsafe { ffi::zfp_field_4d(data, T::FFI_TYPE, *nx, *ny, *nz, *nw) },
        _ => std::ptr::null_mut(),
    }
}

struct FfiStream {
    zfp: *mut ffi::zfp_stream,
    bs: *mut c_void,
    _buf: Option<Vec<u8>>,
}

impl FfiStream {
    fn new(capacity: usize) -> Self {
        let bs = unsafe { ffi::stream_open(std::ptr::null_mut(), capacity + STREAM_PAD_BYTES) };
        assert!(!bs.is_null());
        let zfp = unsafe { ffi::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self {
            zfp,
            bs,
            _buf: None,
        }
    }

    fn with_bytes(bytes: &[u8]) -> Self {
        let mut padded = bytes.to_vec();
        padded.extend_from_slice(&[0; STREAM_PAD_BYTES]);
        let bs = unsafe { ffi::stream_open(padded.as_mut_ptr().cast::<c_void>(), padded.len()) };
        assert!(!bs.is_null());
        let zfp = unsafe { ffi::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self {
            zfp,
            bs,
            _buf: Some(padded),
        }
    }

    fn rewind(&mut self) {
        unsafe {
            ffi::zfp_stream_rewind(self.zfp);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        unsafe {
            let size = ffi::stream_size(self.bs);
            let ptr = ffi::stream_data(self.bs);
            std::slice::from_raw_parts(ptr.cast::<u8>(), size).to_vec()
        }
    }
}

impl Drop for FfiStream {
    fn drop(&mut self) {
        unsafe {
            ffi::zfp_stream_close(self.zfp);
            ffi::stream_close(self.bs);
        }
    }
}

struct FfiField(*mut ffi::zfp_field);

impl Drop for FfiField {
    fn drop(&mut self) {
        unsafe {
            ffi::zfp_field_free(self.0);
        }
    }
}

fn compressed_rust<T: BenchScalar>(data: &[T], dims: &[usize], case: Case) -> Vec<u8> {
    let config = rust_config::<T>(case);
    let field = rust_field(data, dims);
    let mut bs = ZfpBitStream::new(
        config
            .maximum_size(T::RUST_TYPE, dims4(dims))
            .expect("maximum size"),
    )
    .expect("the stream allocates");
    let bytes = bs
        .compress(&config, &field)
        .expect("rust compression failed");
    bs.as_bytes()[..bytes].to_vec()
}

fn compressed_ffi<T: FfiScalar>(data: &[T], dims: &[usize], case: Case) -> Vec<u8> {
    let capacity = rust_config::<T>(case)
        .maximum_size(T::RUST_TYPE, dims4(dims))
        .expect("maximum size");
    let stream = FfiStream::new(capacity);
    unsafe {
        apply_mode_ffi::<T>(stream.zfp, case);
        let field = FfiField(ffi_field::<T>(
            data.as_ptr().cast_mut().cast::<c_void>(),
            dims,
        ));
        assert!(!field.0.is_null());
        assert!(ffi::zfp_compress(stream.zfp, field.0) > 0);
    }
    stream.bytes()
}

fn bench_case<T: FfiScalar>(criterion: &mut Criterion, case: Case) {
    let (data, dims) = T::generate(case.dims);
    let elements = data.len() as u64;
    let capacity = rust_config::<T>(case)
        .maximum_size(T::RUST_TYPE, dims4(&dims))
        .expect("maximum size");
    let case_label = case.label();

    let mut group = criterion.benchmark_group("api_compare");
    group.throughput(Throughput::Elements(elements));

    {
        let config = rust_config::<T>(case);
        let field = rust_field(&data, &dims);
        let mut bs = ZfpBitStream::new(capacity).expect("the stream allocates");
        group.bench_function(
            BenchmarkId::new("compress", format!("{case_label}/zfp-rs")),
            |b| {
                b.iter(|| {
                    bs.rewind();
                    let bytes = bs
                        .compress(black_box(&config), black_box(&field))
                        .expect("rust compression failed");
                    black_box(bytes);
                });
            },
        );
    }

    {
        let mut stream = FfiStream::new(capacity);
        unsafe {
            apply_mode_ffi::<T>(stream.zfp, case);
        }
        let field = FfiField(ffi_field::<T>(
            data.as_ptr().cast_mut().cast::<c_void>(),
            &dims,
        ));
        assert!(!field.0.is_null());
        group.bench_function(
            BenchmarkId::new("compress", format!("{case_label}/zfp-rs-ffi")),
            |b| {
                b.iter(|| {
                    stream.rewind();
                    let bytes = unsafe { ffi::zfp_compress(stream.zfp, field.0) };
                    black_box(bytes);
                });
            },
        );
    }

    // --- Parallel compress benchmarks ---
    for &threads in OMP_THREADS {
        {
            let config = rust_config::<T>(case);
            let field = rust_field(&data, &dims);
            let mut bs = ZfpBitStream::new(capacity).expect("the stream allocates");
            group.bench_function(
                BenchmarkId::new("compress", format!("{case_label}/zfp-rs-rayon{threads}")),
                |b| {
                    b.iter(|| {
                        bs.rewind();
                        let execution = ZfpExecution::Rayon {
                            threads,
                            chunk_size: 0,
                        };
                        let bytes = bs
                            .compress_with_execution(
                                black_box(&config),
                                black_box(&field),
                                black_box(execution),
                            )
                            .expect("rust compression failed");
                        black_box(bytes);
                    });
                },
            );
        }

        {
            let mut stream = FfiStream::new(capacity);
            unsafe {
                apply_mode_ffi::<T>(stream.zfp, case);
                ffi::zfp_stream_set_omp_threads(stream.zfp, threads);
            }
            let field = FfiField(ffi_field::<T>(
                data.as_ptr().cast_mut().cast::<c_void>(),
                &dims,
            ));
            assert!(!field.0.is_null());
            group.bench_function(
                BenchmarkId::new("compress", format!("{case_label}/zfp-rs-ffi-omp{threads}")),
                |b| {
                    b.iter(|| {
                        stream.rewind();
                        let bytes = unsafe { ffi::zfp_compress(stream.zfp, field.0) };
                        black_box(bytes);
                    });
                },
            );
        }
    }

    let rust_bytes = compressed_rust::<T>(&data, &dims, case);
    let ffi_bytes = compressed_ffi::<T>(&data, &dims, case);

    {
        let config = rust_config::<T>(case);
        let mut output = vec![T::default(); data.len()];
        let mut bs = ZfpBitStream::from_bytes(&rust_bytes).expect("the stream allocates");
        group.bench_function(
            BenchmarkId::new("decompress", format!("{case_label}/zfp-rs")),
            |b| {
                b.iter(|| {
                    bs.rewind();
                    let mut field = rust_field_mut(&mut output, &dims);
                    let bytes = bs
                        .decompress(black_box(&config), black_box(&mut field))
                        .expect("rust decompression failed");
                    black_box(bytes);
                });
            },
        );
    }

    {
        let mut output = vec![T::default(); data.len()];
        let mut stream = FfiStream::with_bytes(&ffi_bytes);
        unsafe {
            apply_mode_ffi::<T>(stream.zfp, case);
        }
        let field = FfiField(ffi_field::<T>(output.as_mut_ptr().cast::<c_void>(), &dims));
        assert!(!field.0.is_null());
        group.bench_function(
            BenchmarkId::new("decompress", format!("{case_label}/zfp-rs-ffi")),
            |b| {
                b.iter(|| {
                    stream.rewind();
                    let bytes = unsafe { ffi::zfp_decompress(stream.zfp, field.0) };
                    black_box(bytes);
                });
            },
        );
    }

    // --- Parallel decompress benchmarks ---
    //
    // NOTE: Parallel decompression is supported only for fixed-rate encoding.
    // The other modes (fixed-precision, fixed-accuracy, reversible) do not
    // produce bitstreams with a known layout, so the chunks cannot be split
    // across threads in advance.
    //
    // The upstream C library does not implement OpenMP decompression, so there
    // is no `zfp-sys-omp` counterpart; see `api_compare_c`.

    if case.mode == ModeKind::FixedRate {
        for &threads in OMP_THREADS {
            {
                let config = rust_config::<T>(case);
                let mut output = vec![T::default(); data.len()];
                let mut bs = ZfpBitStream::from_bytes(&rust_bytes).expect("the stream allocates");
                group.bench_function(
                    BenchmarkId::new("decompress", format!("{case_label}/zfp-rs-rayon{threads}")),
                    |b| {
                        b.iter(|| {
                            bs.rewind();
                            let mut field = rust_field_mut(&mut output, &dims);
                            let execution = ZfpExecution::Rayon {
                                threads,
                                chunk_size: 0,
                            };
                            let bytes = bs
                                .decompress_with_execution(
                                    black_box(&config),
                                    black_box(&mut field),
                                    black_box(execution),
                                )
                                .expect("rust decompression failed");
                            black_box(bytes);
                        });
                    },
                );
            }

            {
                let mut output = vec![T::default(); data.len()];
                let mut stream = FfiStream::with_bytes(&ffi_bytes);
                unsafe {
                    apply_mode_ffi::<T>(stream.zfp, case);
                    ffi::zfp_stream_set_omp_threads(stream.zfp, threads);
                }
                let field = FfiField(ffi_field::<T>(output.as_mut_ptr().cast::<c_void>(), &dims));
                assert!(!field.0.is_null());
                group.bench_function(
                    BenchmarkId::new(
                        "decompress",
                        format!("{case_label}/zfp-rs-ffi-omp{threads}"),
                    ),
                    |b| {
                        b.iter(|| {
                            stream.rewind();
                            let bytes = unsafe { ffi::zfp_decompress(stream.zfp, field.0) };
                            black_box(bytes);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn api_compare(criterion: &mut Criterion) {
    run_cases!(criterion, bench_case);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(100));
    targets = api_compare
}
criterion_main!(benches);
