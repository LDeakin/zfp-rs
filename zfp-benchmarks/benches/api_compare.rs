#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::ffi::c_void;
use std::hint::black_box;
use std::time::Duration;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpExecution, ZfpField, ZfpFieldMut, ZfpScalar,
    ZfpScalarType, ZfpStreamAlignment,
};
use zfp_rs_ffi as ffi;

const RATE: f64 = 8.0;
const PRECISION: u32 = 16;
const ACCURACY: f64 = 0.003_906_25;
const STREAM_PAD_BYTES: usize = 8;

#[derive(Clone, Copy)]
enum ScalarKind {
    I32,
    I64,
    F32,
    F64,
}

impl ScalarKind {
    const fn label(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ModeKind {
    FixedRate,
    FixedPrecision,
    FixedAccuracy,
    Reversible,
}

impl ModeKind {
    const fn label(self) -> &'static str {
        match self {
            Self::FixedRate => "fixed_rate",
            Self::FixedPrecision => "fixed_precision",
            Self::FixedAccuracy => "fixed_accuracy",
            Self::Reversible => "reversible",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    scalar: ScalarKind,
    dims: u32,
    mode: ModeKind,
}

impl Case {
    fn label(self) -> String {
        format!(
            "{}_d{}_{}",
            self.scalar.label(),
            self.dims,
            self.mode.label()
        )
    }

    fn dimensionality(self) -> ZfpDimensionality {
        ZfpDimensionality::try_from(self.dims).expect("benchmark dimensions are valid")
    }
}

const SCALARS: &[ScalarKind] = &[
    ScalarKind::I32,
    ScalarKind::I64,
    ScalarKind::F32,
    ScalarKind::F64,
];
const DIMS: &[u32] = &[2, 3, 4];
const MODES: &[ModeKind] = &[
    ModeKind::FixedRate,
    ModeKind::FixedPrecision,
    ModeKind::FixedAccuracy,
    ModeKind::Reversible,
];
const OMP_THREADS: &[u32] = &[2, 3];

trait BenchScalar: ZfpScalar + bytemuck::Pod + Default + Copy + 'static {
    const RUST_TYPE: ZfpScalarType;
    const C_TYPE: zfp_sys::zfp_type;
    const FFI_TYPE: ffi::zfp_type;

    fn sample(index: usize) -> Self;

    fn generate(dims: u32) -> (Vec<Self>, Vec<usize>) {
        let shape = shape_for_dims(dims);
        let elements = shape.iter().product();
        let data = (0..elements).map(Self::sample).collect();
        (data, shape)
    }
}

fn shape_for_dims(dims: u32) -> Vec<usize> {
    match dims {
        2 => vec![1_024, 1_024],
        3 => vec![128, 128, 64],
        4 => vec![64, 64, 16, 16],
        _ => panic!("unsupported dimensionality"),
    }
}

impl BenchScalar for i32 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::Int32;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int32;
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_int32;

    fn sample(index: usize) -> Self {
        let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((mixed & 0x000f_ffff) as i32) - 524_288
    }
}

impl BenchScalar for i64 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::Int64;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_int64;
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_int64;

    fn sample(index: usize) -> Self {
        let mixed = (index as i64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (mixed & 0x0000_ffff_ffff_ffff) - 140_737_488_355_328
    }
}

impl BenchScalar for f32 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::Float;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_float;
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_float;

    fn sample(index: usize) -> Self {
        let x = index as f64;
        ((x * 0.001).sin() * 32.0 + (x * 0.000_037).cos() * 4.0) as f32
    }
}

impl BenchScalar for f64 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::Double;
    const C_TYPE: zfp_sys::zfp_type = zfp_sys::zfp_type_zfp_type_double;
    const FFI_TYPE: ffi::zfp_type = ffi::zfp_type_zfp_type_double;

    fn sample(index: usize) -> Self {
        let x = index as f64;
        (x * 0.001).sin() * 32.0 + (x * 0.000_037).cos() * 4.0
    }
}

fn rust_config<T: BenchScalar>(case: Case) -> ZfpConfig {
    match case.mode {
        ModeKind::FixedRate => ZfpConfig::fixed_rate(
            RATE,
            T::RUST_TYPE,
            case.dimensionality(),
            ZfpStreamAlignment::None,
        ),
        ModeKind::FixedPrecision => ZfpConfig::fixed_precision(PRECISION),
        ModeKind::FixedAccuracy => ZfpConfig::fixed_accuracy(ACCURACY),
        ModeKind::Reversible => ZfpConfig::reversible(),
    }
}

unsafe fn apply_mode_c<T: BenchScalar>(zfp: *mut zfp_sys::zfp_stream, case: Case) {
    match case.mode {
        ModeKind::FixedRate => unsafe {
            zfp_sys::zfp_stream_set_rate(zfp, RATE, T::C_TYPE, case.dims, 0);
        },
        ModeKind::FixedPrecision => unsafe {
            zfp_sys::zfp_stream_set_precision(zfp, PRECISION);
        },
        ModeKind::FixedAccuracy => unsafe {
            zfp_sys::zfp_stream_set_accuracy(zfp, ACCURACY);
        },
        ModeKind::Reversible => unsafe {
            zfp_sys::zfp_stream_set_reversible(zfp);
        },
    }
}

unsafe fn apply_mode_ffi<T: BenchScalar>(zfp: *mut ffi::zfp_stream, case: Case) {
    match case.mode {
        ModeKind::FixedRate => unsafe {
            ffi::zfp_stream_set_rate(zfp, RATE, T::FFI_TYPE, case.dims, 0);
        },
        ModeKind::FixedPrecision => unsafe {
            ffi::zfp_stream_set_precision(zfp, PRECISION);
        },
        ModeKind::FixedAccuracy => unsafe {
            ffi::zfp_stream_set_accuracy(zfp, ACCURACY);
        },
        ModeKind::Reversible => unsafe {
            ffi::zfp_stream_set_reversible(zfp);
        },
    }
}

fn rust_field<'a, T: BenchScalar>(data: &'a [T], dims: &[usize]) -> ZfpField<'a> {
    match dims {
        [nx] => ZfpField::new(data, [*nx]),
        [nx, ny] => ZfpField::new(data, [*nx, *ny]),
        [nx, ny, nz] => ZfpField::new(data, [*nx, *ny, *nz]),
        [nx, ny, nz, nw] => ZfpField::new(data, [*nx, *ny, *nz, *nw]),
        _ => panic!("unsupported dimensionality"),
    }
}

fn rust_field_mut<'a, T: BenchScalar>(data: &'a mut [T], dims: &[usize]) -> ZfpFieldMut<'a> {
    match dims {
        [nx] => ZfpFieldMut::new(data, [*nx]),
        [nx, ny] => ZfpFieldMut::new(data, [*nx, *ny]),
        [nx, ny, nz] => ZfpFieldMut::new(data, [*nx, *ny, *nz]),
        [nx, ny, nz, nw] => ZfpFieldMut::new(data, [*nx, *ny, *nz, *nw]),
        _ => panic!("unsupported dimensionality"),
    }
}

unsafe fn c_field<T: BenchScalar>(data: *mut c_void, dims: &[usize]) -> *mut zfp_sys::zfp_field {
    match dims {
        [nx] => unsafe { zfp_sys::zfp_field_1d(data, T::C_TYPE, *nx) },
        [nx, ny] => unsafe { zfp_sys::zfp_field_2d(data, T::C_TYPE, *nx, *ny) },
        [nx, ny, nz] => unsafe { zfp_sys::zfp_field_3d(data, T::C_TYPE, *nx, *ny, *nz) },
        [nx, ny, nz, nw] => unsafe { zfp_sys::zfp_field_4d(data, T::C_TYPE, *nx, *ny, *nz, *nw) },
        _ => std::ptr::null_mut(),
    }
}

fn ffi_field<T: BenchScalar>(data: *mut c_void, dims: &[usize]) -> *mut ffi::zfp_field {
    match dims {
        [nx] => unsafe { ffi::zfp_field_1d(data, T::FFI_TYPE, *nx) },
        [nx, ny] => unsafe { ffi::zfp_field_2d(data, T::FFI_TYPE, *nx, *ny) },
        [nx, ny, nz] => unsafe { ffi::zfp_field_3d(data, T::FFI_TYPE, *nx, *ny, *nz) },
        [nx, ny, nz, nw] => unsafe { ffi::zfp_field_4d(data, T::FFI_TYPE, *nx, *ny, *nz, *nw) },
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

struct CField(*mut zfp_sys::zfp_field);

impl Drop for CField {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_field_free(self.0);
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
    let mut bs = ZfpBitStream::new(config.maximum_size(T::RUST_TYPE, dims));
    let bytes = bs
        .compress(&config, &field)
        .expect("rust compression failed");
    bs.as_bytes()[..bytes].to_vec()
}

fn compressed_c<T: BenchScalar>(data: &[T], dims: &[usize], case: Case) -> Vec<u8> {
    let capacity = rust_config::<T>(case).maximum_size(T::RUST_TYPE, dims);
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

fn compressed_ffi<T: BenchScalar>(data: &[T], dims: &[usize], case: Case) -> Vec<u8> {
    let capacity = rust_config::<T>(case).maximum_size(T::RUST_TYPE, dims);
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

fn bench_case<T: BenchScalar>(criterion: &mut Criterion, case: Case) {
    let (data, dims) = T::generate(case.dims);
    let elements = data.len() as u64;
    let capacity = rust_config::<T>(case).maximum_size(T::RUST_TYPE, &dims);
    let case_label = case.label();

    let mut group = criterion.benchmark_group("api_compare");
    group.throughput(Throughput::Elements(elements));

    {
        let config = rust_config::<T>(case);
        let field = rust_field(&data, &dims);
        let mut bs = ZfpBitStream::new(capacity);
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
            let mut bs = ZfpBitStream::new(capacity);
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
    let c_bytes = compressed_c::<T>(&data, &dims, case);
    let ffi_bytes = compressed_ffi::<T>(&data, &dims, case);

    {
        let config = rust_config::<T>(case);
        let mut output = vec![T::default(); data.len()];
        let mut bs = ZfpBitStream::from_bytes(&rust_bytes);
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

    if case.mode == ModeKind::FixedRate {
        for &threads in OMP_THREADS {
            {
                let config = rust_config::<T>(case);
                let mut output = vec![T::default(); data.len()];
                let mut bs = ZfpBitStream::from_bytes(&rust_bytes);
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

            // NOTE: zfp-sys-omp decompress is not benchmarked because the zfp C library
            // (v1.0.1) does not yet implement OpenMP decompression. The decompress function
            // table for OMP contains only NULL entries, causing zfp_decompress to return 0
            // immediately without processing any data. See zfp/src/zfp.c:
            //
            //     /* OpenMP; not yet supported */
            //     {{{ NULL }}}}

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
    for &scalar in SCALARS {
        for &dims in DIMS {
            for &mode in MODES {
                let case = Case { scalar, dims, mode };
                match scalar {
                    ScalarKind::I32 => bench_case::<i32>(criterion, case),
                    ScalarKind::I64 => bench_case::<i64>(criterion, case),
                    ScalarKind::F32 => bench_case::<f32>(criterion, case),
                    ScalarKind::F64 => bench_case::<f64>(criterion, case),
                }
            }
        }
    }
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
