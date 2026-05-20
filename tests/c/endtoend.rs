#![allow(clippy::cast_possible_wrap)] // usize↔isize for stride computation
#![allow(clippy::cast_possible_truncation)] // usize→u32/u64/i32 for test constants
#![allow(clippy::cast_sign_loss)] // isize→usize for index computation
#![allow(clippy::cast_precision_loss)] // usize→f64 for rate computation
//! Port of `zfp/tests/src/endtoend/testZfpSerial{1-4}{d,f,i,l}.c` (16 files).
//!
//! Each C file includes `serialExecBase.c` and `testcases/serial.c`.
//! Data generation is delegated to the `zfp-test-utils` workspace crate,
//! which compiles the upstream C smooth-random-number utilities so that the
//! generated arrays are byte-identical to those produced by the reference
//! implementation.
#![allow(dead_code)] // Ported from upstream; some variants/functions are intentionally unexercised.

use std::sync::OnceLock;

use zfp_rs::ZfpStreamAlignment;
use zfp_rs::bitstream::ZfpBitStream;
use zfp_rs::config::ZfpConfig;
use zfp_rs::field::{ZfpField, ZfpFieldMut};
use zfp_rs::types::{ZfpDimensionality, ZfpScalar, ZfpScalarType as LibZfpScalarType};

use super::checksums::{
    Subject, TestType, ZfpMode as CsumMode, ZfpScalarType as CsumType, compute_key,
    compute_key_original_input, get_checksum, hash_array32, hash_array64, hash_bitstream,
    hash_strided_array32, hash_strided_array64,
};

use zfp_test_utils::{
    gen_smooth_rand_doubles, gen_smooth_rand_floats, gen_smooth_rand_ints32, gen_smooth_rand_ints64,
};

// ---------------------------------------------------------------------------
// Stride configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrideConfig {
    AsIs,
    Reversed,
    Interleaved,
    Permuted,
}

/// Holds the stride-rearranged backing data together with the logical start
/// offset and the per-axis strides.
struct StridedData<T> {
    data: Vec<T>,
    /// Index within `data` of the logical element at position [0,0,0,0].
    start_offset: usize,
    strides: [isize; 4],
    config: StrideConfig,
}

impl<T: Copy + Default> StridedData<T> {
    fn build(src: &[T], n: [usize; 4], dims: u32, config: StrideConfig) -> Self {
        let nx = n[0];
        let ny = if dims >= 2 { n[1] } else { 1 };
        let nz = if dims >= 3 { n[2] } else { 1 };
        let nw = if dims >= 4 { n[3] } else { 1 };
        let total = nx * ny * nz * nw;

        match config {
            StrideConfig::AsIs => StridedData {
                data: src.to_vec(),
                start_offset: 0,
                strides: [0; 4],
                config,
            },

            StrideConfig::Reversed => {
                // reversed: last element first, stride = -1 per axis
                let data: Vec<T> = src.iter().rev().copied().collect();
                let sx: isize = -1;
                let sy: isize = -(nx as isize);
                let sz: isize = -(nx as isize) * ny as isize;
                let sw: isize = -(nx as isize) * ny as isize * nz as isize;
                // The C test adjusts the pointer to the last element; we encode
                // that by setting start_offset = total - 1.
                StridedData {
                    data,
                    start_offset: total - 1,
                    strides: [sx, sy, sz, sw],
                    config,
                }
            }

            StrideConfig::Interleaved => {
                // every even slot holds data; odd slots are default
                let mut data = vec![T::default(); total * 2];
                for (i, &v) in src.iter().enumerate() {
                    data[2 * i] = v;
                }
                let sx: isize = 2;
                let sy: isize = 2 * nx as isize;
                let sz: isize = 2 * nx as isize * ny as isize;
                let sw: isize = 2 * nx as isize * ny as isize * nz as isize;
                StridedData {
                    data,
                    start_offset: 0,
                    strides: [sx, sy, sz, sw],
                    config,
                }
            }

            StrideConfig::Permuted => {
                assert!(dims >= 2, "permuted stride requires dims >= 2");
                let mut data = vec![T::default(); total];
                permute_square_array(src, &mut data, nx, dims);
                let strides = permuted_strides(n, dims);
                StridedData {
                    data,
                    start_offset: 0,
                    strides,
                    config,
                }
            }
        }
    }
}

/// Permute a square array: reverse index order (ijkl → lkji).
/// Matches `permuteSquareArray` in `stridedOperations.c`.
fn permute_square_array<T: Copy + Default>(src: &[T], dst: &mut [T], side: usize, dims: u32) {
    match dims {
        2 => {
            for j in 0..side {
                for i in 0..side {
                    dst[j * side + i] = src[i * side + j];
                }
            }
        }
        3 => {
            for k in 0..side {
                for j in 0..side {
                    for i in 0..side {
                        let out = k * side * side + j * side + i;
                        let inp = i * side * side + j * side + k;
                        dst[out] = src[inp];
                    }
                }
            }
        }
        4 => {
            for l in 0..side {
                for k in 0..side {
                    for j in 0..side {
                        for i in 0..side {
                            let out = l * side * side * side + k * side * side + j * side + i;
                            let inp = i * side * side * side + j * side * side + k * side + l;
                            dst[out] = src[inp];
                        }
                    }
                }
            }
        }
        _ => panic!("permute_square_array: unsupported dims {dims}"),
    }
}

/// Compute strides for the permuted layout: [nx^(d-1), ..., nx, 1].
/// Matches `getPermutedStrides` in `stridedOperations.c`.
fn permuted_strides(n: [usize; 4], dims: u32) -> [isize; 4] {
    match dims {
        2 => [n[0] as isize, 1, 0, 0],
        3 => [(n[0] * n[1]) as isize, n[0] as isize, 1, 0],
        4 => [
            (n[0] * n[1] * n[2]) as isize,
            (n[0] * n[1]) as isize,
            n[0] as isize,
            1,
        ],
        _ => panic!("permuted_strides: unsupported dims {dims}"),
    }
}

// ---------------------------------------------------------------------------
// Compression parameter helpers  (from zfpCompressionParams.c)
// ---------------------------------------------------------------------------

fn fixed_precision_param(p: usize) -> u32 {
    1u32 << (p + 3)
}
fn fixed_rate_param(p: usize) -> usize {
    1usize << (p + 3)
}
fn fixed_accuracy_param(p: usize) -> f64 {
    libm::ldexp(1.0, -((1usize << p) as i32))
}

// ---------------------------------------------------------------------------
// ZFP field construction helpers
// ---------------------------------------------------------------------------

fn make_field<T: ZfpScalar>(
    data: &[T],
    dims: u32,
    n: [usize; 4],
    strides: [isize; 4],
) -> ZfpField<'_> {
    let zero = [0isize; 4];
    match dims {
        1 => {
            if strides == zero {
                ZfpField::new(data, [n[0]])
            } else {
                ZfpField::new_strided(data, [n[0]], [strides[0]])
            }
        }
        2 => {
            if strides == zero {
                ZfpField::new(data, [n[0], n[1]])
            } else {
                ZfpField::new_strided(data, [n[0], n[1]], [strides[0], strides[1]])
            }
        }
        3 => {
            if strides == zero {
                ZfpField::new(data, [n[0], n[1], n[2]])
            } else {
                ZfpField::new_strided(
                    data,
                    [n[0], n[1], n[2]],
                    [strides[0], strides[1], strides[2]],
                )
            }
        }
        4 => {
            if strides == zero {
                ZfpField::new(data, [n[0], n[1], n[2], n[3]])
            } else {
                ZfpField::new_strided(
                    data,
                    [n[0], n[1], n[2], n[3]],
                    [strides[0], strides[1], strides[2], strides[3]],
                )
            }
        }
        _ => panic!("make_field: invalid dims {dims}"),
    }
}

fn make_field_mut<T: ZfpScalar>(
    data: &mut [T],
    dims: u32,
    n: [usize; 4],
    strides: [isize; 4],
) -> ZfpFieldMut<'_> {
    let zero = [0isize; 4];
    match dims {
        1 => {
            if strides == zero {
                ZfpFieldMut::new(data, [n[0]])
            } else {
                ZfpFieldMut::new_strided(data, [n[0]], [strides[0]])
            }
        }
        2 => {
            if strides == zero {
                ZfpFieldMut::new(data, [n[0], n[1]])
            } else {
                ZfpFieldMut::new_strided(data, [n[0], n[1]], [strides[0], strides[1]])
            }
        }
        3 => {
            if strides == zero {
                ZfpFieldMut::new(data, [n[0], n[1], n[2]])
            } else {
                ZfpFieldMut::new_strided(
                    data,
                    [n[0], n[1], n[2]],
                    [strides[0], strides[1], strides[2]],
                )
            }
        }
        4 => {
            if strides == zero {
                ZfpFieldMut::new(data, [n[0], n[1], n[2], n[3]])
            } else {
                ZfpFieldMut::new_strided(
                    data,
                    [n[0], n[1], n[2], n[3]],
                    [strides[0], strides[1], strides[2], strides[3]],
                )
            }
        }
        _ => panic!("make_field_mut: invalid dims {dims}"),
    }
}

// ---------------------------------------------------------------------------
// Build n array (square arrays)
// ---------------------------------------------------------------------------

fn build_n(side_len: usize, dims: u32) -> [usize; 4] {
    let mut n = [0usize; 4];
    for slot in n.iter_mut().take(dims as usize) {
        *slot = side_len;
    }
    n
}

// ---------------------------------------------------------------------------
// Decompressed-array hash (mirrors isDecompressedArrayChecksumsMatch)
// ---------------------------------------------------------------------------

fn hash_decomp_array<T: ZfpScalar>(
    data: &[T],
    data_offset: usize,
    n: [usize; 4],
    strides: [isize; 4],
    stride_config: StrideConfig,
) -> u64 {
    match std::mem::size_of::<T>() {
        4 => {
            // SAFETY: T is a 4-byte ZfpScalar (i32 or f32); reinterpreting as u32 is valid.
            let u32_data =
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u32>(), data.len()) };
            match stride_config {
                StrideConfig::AsIs => u64::from(hash_array32(u32_data, 1)),
                StrideConfig::Interleaved => u64::from(hash_array32(u32_data, 2)),
                StrideConfig::Reversed | StrideConfig::Permuted => u64::from(hash_strided_array32(
                    &u32_data[data_offset..],
                    n,
                    [strides[0], strides[1], strides[2], strides[3]],
                )),
            }
        }
        8 => {
            // SAFETY: T is an 8-byte ZfpScalar (i64 or f64); reinterpreting as u64 is valid.
            let u64_data =
                unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u64>(), data.len()) };
            match stride_config {
                StrideConfig::AsIs => hash_array64(u64_data, 1),
                StrideConfig::Interleaved => hash_array64(u64_data, 2),
                StrideConfig::Reversed | StrideConfig::Permuted => hash_strided_array64(
                    &u64_data[data_offset..],
                    n,
                    [strides[0], strides[1], strides[2], strides[3]],
                ),
            }
        }
        _ => unreachable!("unexpected ZfpScalar size"),
    }
}

// ---------------------------------------------------------------------------
// Compression mode enum
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum CompressionMode {
    FixedPrecision,
    FixedRate,
    FixedAccuracy,
    Reversible,
}

fn to_csum_mode(mode: CompressionMode) -> CsumMode {
    match mode {
        CompressionMode::FixedPrecision => CsumMode::FixedPrecision,
        CompressionMode::FixedRate => CsumMode::FixedRate,
        CompressionMode::FixedAccuracy => CsumMode::FixedAccuracy,
        CompressionMode::Reversible => CsumMode::Reversible,
    }
}

// ---------------------------------------------------------------------------
// Core compress/decompress test loop
// ---------------------------------------------------------------------------

fn run_compress_decompress<T: ZfpScalar>(
    dims: u32,
    checksum_type: CsumType,
    n: [usize; 4],
    strided: &StridedData<T>,
    mode: CompressionMode,
    num_params: usize,
) {
    let total_strided_len = strided.data.len();

    for param_num in 0..num_params {
        let config = match mode {
            CompressionMode::FixedPrecision => {
                ZfpConfig::fixed_precision(fixed_precision_param(param_num))
            }
            CompressionMode::FixedRate => ZfpConfig::fixed_rate(
                fixed_rate_param(param_num) as f64,
                T::scalar_type(),
                ZfpDimensionality::try_from(dims).unwrap(),
                ZfpStreamAlignment::None,
            ),
            CompressionMode::FixedAccuracy => {
                ZfpConfig::fixed_accuracy(fixed_accuracy_param(param_num))
            }
            CompressionMode::Reversible => ZfpConfig::reversible(),
        };

        let buf_size = config.maximum_size(T::scalar_type(), &n[..dims as usize]);
        let mut bs = ZfpBitStream::new(buf_size);

        // Compression field: pass the full backing array. For reversed layouts,
        // start_offset marks the logical element [0,0,0,0] within `data`, but the
        // stream's imin calculation handles negative strides automatically — it
        // computes the minimum-address element as data[-imin] from data[0].
        let field = make_field(&strided.data, dims, n, strided.strides);

        let compressed_bytes = bs.compress(&config, &field).unwrap();
        assert!(
            compressed_bytes > 0,
            "compression returned 0 bytes (mode={:?}, param={param_num})",
            to_csum_mode(mode) as u32
        );

        // Checksum compressed bitstream
        {
            let bs_hash = hash_bitstream(&bs.as_bytes()[..compressed_bytes]);
            let (k1, k2) = compute_key(
                TestType::Array,
                Subject::CompressedBitstream,
                n,
                to_csum_mode(mode),
                param_num as u64,
            );
            let expected = get_checksum(dims, checksum_type, k1, k2).unwrap_or_else(|| {
                panic!(
                    "no checksum: compressed bitstream dims={dims} type={:?} mode={:?} param={param_num}",
                    checksum_type as u32,
                    to_csum_mode(mode) as u32
                )
            });
            assert_eq!(
                bs_hash,
                expected,
                "compressed bitstream checksum mismatch (mode={:?}, param={param_num})",
                to_csum_mode(mode) as u32
            );
        }

        // Decompress into a fresh buffer — pass the full backing array so that
        // the stream's imin calculation can handle negative strides correctly.
        let mut decomp_data = vec![T::default(); total_strided_len];
        bs.rewind();
        {
            let mut decomp_field = make_field_mut(&mut decomp_data, dims, n, strided.strides);
            let decomp_bytes = bs.decompress(&config, &mut decomp_field).unwrap();
            assert_eq!(
                decomp_bytes,
                compressed_bytes,
                "decompress byte count mismatch (mode={:?}, param={param_num})",
                to_csum_mode(mode) as u32
            );
        }

        // Checksum decompressed array
        {
            let arr_hash = hash_decomp_array(
                &decomp_data,
                strided.start_offset,
                n,
                strided.strides,
                strided.config,
            );
            let (k1, k2) = compute_key(
                TestType::Array,
                Subject::DecompressedArray,
                n,
                to_csum_mode(mode),
                param_num as u64,
            );
            let expected = get_checksum(dims, checksum_type, k1, k2).unwrap_or_else(|| {
                panic!(
                    "no checksum: decompressed array dims={dims} type={:?} mode={:?} param={param_num}",
                    checksum_type as u32,
                    to_csum_mode(mode) as u32
                )
            });
            assert_eq!(
                arr_hash,
                expected,
                "decompressed array checksum mismatch (mode={:?}, param={param_num})",
                to_csum_mode(mode) as u32
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible round-trip test
// ---------------------------------------------------------------------------

fn run_reversible<T: ZfpScalar + PartialEq + std::fmt::Debug>(
    dims: u32,
    checksum_type: CsumType,
    n: [usize; 4],
    strided: &StridedData<T>,
) {
    let total_strided_len = strided.data.len();

    let config = ZfpConfig::reversible();
    let buf_size = config.maximum_size(T::scalar_type(), &n[..dims as usize]);
    let mut bs = ZfpBitStream::new(buf_size);

    let field = make_field(&strided.data, dims, n, strided.strides);
    let compressed_bytes = bs.compress(&config, &field).unwrap();
    assert!(
        compressed_bytes > 0,
        "reversible compression returned 0 bytes"
    );

    // Checksum compressed bitstream
    {
        let bs_hash = hash_bitstream(&bs.as_bytes()[..compressed_bytes]);
        let (k1, k2) = compute_key(
            TestType::Array,
            Subject::CompressedBitstream,
            n,
            CsumMode::Reversible,
            0,
        );
        let expected = get_checksum(dims, checksum_type, k1, k2)
            .expect("no checksum entry for reversible compressed bitstream");
        assert_eq!(
            bs_hash, expected,
            "reversible compressed bitstream checksum mismatch"
        );
    }

    // Decompress
    let mut decomp_data = vec![T::default(); total_strided_len];
    bs.rewind();
    {
        let mut decomp_field = make_field_mut(&mut decomp_data, dims, n, strided.strides);
        let decomp_bytes = bs.decompress(&config, &mut decomp_field).unwrap();
        assert_eq!(
            decomp_bytes, compressed_bytes,
            "reversible decompress byte count mismatch"
        );
    }

    // Verify bit-exact round-trip
    if strided.config == StrideConfig::AsIs {
        let logical_len = n[..dims as usize].iter().product();
        assert_eq!(
            &strided.data[..logical_len],
            &decomp_data[..logical_len],
            "reversible round-trip failed (AsIs)"
        );
    } else {
        check_strided_equal(
            &strided.data,
            &decomp_data,
            strided.start_offset,
            n,
            strided.strides,
            dims,
        );
    }
}

fn check_strided_equal<T: PartialEq + std::fmt::Debug>(
    a: &[T],
    b: &[T],
    start_offset: usize,
    n: [usize; 4],
    strides: [isize; 4],
    dims: u32,
) {
    let nx = n[0];
    let ny = if dims >= 2 { n[1] } else { 1 };
    let nz = if dims >= 3 { n[2] } else { 1 };
    let nw = if dims >= 4 { n[3] } else { 1 };

    let base = start_offset as isize;
    let mut off_l: isize = 0;
    for _l in 0..nw {
        let mut off_k = off_l;
        for _k in 0..nz {
            let mut off_j = off_k;
            for _j in 0..ny {
                let mut off_i = off_j;
                for _i in 0..nx {
                    let idx = (base + off_i) as usize;
                    assert_eq!(
                        a[idx], b[idx],
                        "reversible round-trip mismatch at strided index {idx}"
                    );
                    off_i += strides[0];
                }
                off_j += strides[1];
            }
            off_k += strides[2];
        }
        off_l += strides[3];
    }
}

// ---------------------------------------------------------------------------
// Fixed-rate rounding test (SetRateWithWriteRandomAccess)
// ---------------------------------------------------------------------------

fn test_set_rate_align<T: ZfpScalar>(dims: u32) {
    // C uses ZFP_RATE_PARAM_BITS = 19 for all types (from universalConsts.h)
    let rate = 19.0f64;
    let zfp_dims = ZfpDimensionality::try_from(dims).unwrap();
    let params_no_align =
        ZfpConfig::fixed_rate(rate, T::scalar_type(), zfp_dims, ZfpStreamAlignment::None);
    let params_align = ZfpConfig::fixed_rate(
        rate,
        T::scalar_type(),
        zfp_dims,
        ZfpStreamAlignment::WordAligned,
    );
    let rate_without_align = params_no_align.rate(zfp_dims);
    let rate_with_align = params_align.rate(zfp_dims);
    assert!(
        rate_with_align >= rate_without_align,
        "rateWithAlign ({rate_with_align}) >= rateWithoutAlign ({rate_without_align}) failed"
    );
    let bits_per_block = (rate_with_align * f64::from(4u32.pow(dims)) + 0.5).floor() as u64;
    assert_eq!(
        bits_per_block % 64,
        0,
        "bits_per_block ({bits_per_block}) must be a multiple of stream_word_bits (64)"
    );
}

// ---------------------------------------------------------------------------
// Fixed-rate bitrate validation test
// ---------------------------------------------------------------------------

fn test_bitrate<T: ZfpScalar>(dims: u32, n: [usize; 4], src: &[T]) {
    for param_num in 0..3 {
        let rate_param = fixed_rate_param(param_num);
        let config = ZfpConfig::fixed_rate(
            rate_param as f64,
            T::scalar_type(),
            ZfpDimensionality::try_from(dims).unwrap(),
            ZfpStreamAlignment::None,
        );
        let buf_size = config.maximum_size(T::scalar_type(), &n[..dims as usize]);
        let mut bs = ZfpBitStream::new(buf_size);
        let field = make_field(src, dims, n, [0; 4]);
        let compressed_bytes = bs.compress(&config, &field).unwrap();
        assert!(
            compressed_bytes > 0,
            "bitrate test: compression failed (param={param_num})"
        );

        let compressed_bits = compressed_bytes * 8;
        let padded: [usize; 4] = [
            (n[0] + 3) & !3,
            if dims >= 2 { (n[1] + 3) & !3 } else { 1 },
            if dims >= 3 { (n[2] + 3) & !3 } else { 1 },
            if dims >= 4 { (n[3] + 3) & !3 } else { 1 },
        ];
        let padded_len: usize = padded[..dims as usize].iter().product();
        let expected_bits = rate_param * padded_len;
        let expected_bits = (expected_bits + 63) & !63;

        assert_eq!(
            compressed_bits, expected_bits,
            "bitrate mismatch (param={param_num}): got {compressed_bits} bits, expected {expected_bits}"
        );
    }
}

// ---------------------------------------------------------------------------
// Fixed-accuracy value validation test
// ---------------------------------------------------------------------------

#[allow(clippy::similar_names)]
fn test_accuracy<T: ZfpScalar>(dims: u32, n: [usize; 4], strided: &StridedData<T>) {
    let nx = n[0];
    let ny = if dims >= 2 { n[1] } else { 1 };
    let nz = if dims >= 3 { n[2] } else { 1 };
    let nw = if dims >= 4 { n[3] } else { 1 };
    let total_strided_len = strided.data.len();

    for param_num in 0..3 {
        let tolerance = fixed_accuracy_param(param_num);
        let config = ZfpConfig::fixed_accuracy(tolerance);
        let buf_size = config.maximum_size(T::scalar_type(), &n[..dims as usize]);
        let mut bs = ZfpBitStream::new(buf_size);

        let field = make_field(&strided.data, dims, n, strided.strides);
        let compressed_bytes = bs.compress(&config, &field).unwrap();
        assert!(
            compressed_bytes > 0,
            "accuracy test: compression failed (param={param_num})"
        );

        let mut decomp_data = vec![T::default(); total_strided_len];
        bs.rewind();
        {
            let mut decomp_field = make_field_mut(&mut decomp_data, dims, n, strided.strides);
            let decomp_bytes = bs.decompress(&config, &mut decomp_field).unwrap();
            assert_eq!(
                decomp_bytes, compressed_bytes,
                "accuracy test: decompress byte count mismatch (param={param_num})"
            );
        }

        // Check each element is within tolerance with stride-aware traversal
        let base = strided.start_offset as isize;
        let strides = strided.strides;
        let mut off_l: isize = 0;
        for _l in 0..nw {
            let mut off_k = off_l;
            for _k in 0..nz {
                let mut off_j = off_k;
                for _j in 0..ny {
                    let mut off_i = off_j;
                    for _i in 0..nx {
                        let idx = (base + off_i) as usize;
                        check_within_tolerance(
                            strided.data[idx],
                            decomp_data[idx],
                            tolerance,
                            param_num,
                        );
                        off_i += strides[0];
                    }
                    off_j += strides[1];
                }
                off_k += strides[2];
            }
            off_l += strides[3];
        }
    }
}

fn check_within_tolerance<T: ZfpScalar>(orig: T, dec: T, tolerance: f64, param_num: usize) {
    match std::mem::size_of::<T>() {
        4 => {
            // SAFETY: T is f32; the bits are valid f32.
            let o = unsafe { *(&raw const orig).cast::<f32>() };
            let d = unsafe { *(&raw const dec).cast::<f32>() };
            let diff = f64::from((o - d).abs());
            assert!(
                diff <= tolerance,
                "fixed-accuracy error {diff} > tolerance {tolerance} (param={param_num})"
            );
        }
        8 => {
            // SAFETY: T is f64; the bits are valid f64.
            let o = unsafe { *(&raw const orig).cast::<f64>() };
            let d = unsafe { *(&raw const dec).cast::<f64>() };
            let diff = (o - d).abs();
            assert!(
                diff <= tolerance,
                "fixed-accuracy error {diff} > tolerance {tolerance} (param={param_num})"
            );
        }
        _ => panic!("check_within_tolerance: unexpected type size"),
    }
}

// ---------------------------------------------------------------------------
// Data checksum test
// ---------------------------------------------------------------------------

fn test_data_checksum<T: ZfpScalar>(dims: u32, checksum_type: CsumType, src: &[T], n: [usize; 4]) {
    let checksum: u64 = match std::mem::size_of::<T>() {
        4 => {
            // SAFETY: T is a 4-byte ZfpScalar; reinterpreting as u32 is valid.
            let u32_data =
                unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u32>(), src.len()) };
            u64::from(hash_array32(u32_data, 1))
        }
        8 => {
            // SAFETY: T is an 8-byte ZfpScalar; reinterpreting as u64 is valid.
            let u64_data =
                unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u64>(), src.len()) };
            hash_array64(u64_data, 1)
        }
        _ => unreachable!(),
    };

    let (k1, k2) = compute_key_original_input(TestType::Array, n);
    let expected = get_checksum(dims, checksum_type, k1, k2)
        .expect("no checksum entry for original input data");
    assert_eq!(checksum, expected, "data checksum mismatch");
}

// ---------------------------------------------------------------------------
// Typed data generation — routes to the correct C generator per type T
// ---------------------------------------------------------------------------

fn generate_typed<T: ZfpScalar>(dims: u32) -> (Vec<T>, usize, usize) {
    match T::scalar_type() {
        LibZfpScalarType::Int32 => {
            let (v, s, t) = gen_smooth_rand_ints32(dims);
            reinterpret_vec(v, s, t)
        }
        LibZfpScalarType::Int64 => {
            let (v, s, t) = gen_smooth_rand_ints64(dims);
            reinterpret_vec(v, s, t)
        }
        LibZfpScalarType::Float => {
            let (v, s, t) = gen_smooth_rand_floats(dims);
            reinterpret_vec(v, s, t)
        }
        LibZfpScalarType::Double => {
            let (v, s, t) = gen_smooth_rand_doubles(dims);
            reinterpret_vec(v, s, t)
        }
    }
}

/// Reinterpret-cast `Vec<U>` as `Vec<T>` where `size_of::<U>() == size_of::<T>()`.
///
/// # Safety
/// Called only with (U, T) pairs that are the same ZFP scalar type (i32/i32, f32/f32, etc.)
/// because `generate_typed` matches on `T::scalar_type()` before dispatching here.
fn reinterpret_vec<U, T>(v: Vec<U>, side: usize, total: usize) -> (Vec<T>, usize, usize) {
    assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
    assert_eq!(std::mem::align_of::<U>(), std::mem::align_of::<T>());
    let mut v = std::mem::ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr().cast::<T>();
    let len = v.len();
    let cap = v.capacity();
    // SAFETY: T and U have the same size and alignment (asserted above), and this
    // is called only when T is the concrete type matching U (e.g., both are f32).
    let out = unsafe { Vec::from_raw_parts(ptr, len, cap) };
    (out, side, total)
}

// ---------------------------------------------------------------------------
// Macro: generates 11 test functions for one (dims, scalar type) combination
// ---------------------------------------------------------------------------

macro_rules! endtoend_tests {
    ($mod_name:ident, $dims:literal, $type_variant:ident, $rust_type:ty, $is_float:literal) => {
        mod $mod_name {
            use super::*;

            const DIMS: u32 = $dims;
            const CSUM_TYPE: CsumType = CsumType::$type_variant;

            fn get_data() -> &'static (Vec<$rust_type>, usize, usize) {
                static DATA: OnceLock<(Vec<$rust_type>, usize, usize)> = OnceLock::new();
                DATA.get_or_init(|| generate_typed::<$rust_type>(DIMS))
            }

            fn make_n() -> [usize; 4] {
                let (_, side_len, _) = get_data();
                build_n(*side_len, DIMS)
            }

            // ----------------------------------------------------------------
            // 1. Data generation checksum
            // ----------------------------------------------------------------

            #[test]
            fn when_seeded_random_smooth_data_generated_expect_checksum_matches() {
                let (data, side_len, total_len) = get_data();
                let n = build_n(*side_len, DIMS);
                test_data_checksum::<$rust_type>(DIMS, CSUM_TYPE, &data[..*total_len], n);
            }

            // ----------------------------------------------------------------
            // 2. Reversed array — fixed-precision (1 param)
            // ----------------------------------------------------------------

            #[test]
            fn given_reversed_array_when_zfp_compress_decompress_fixed_precision_expect_bitstream_and_array_checksums_match(
            ) {
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::Reversed);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedPrecision,
                    1,
                );
            }

            // ----------------------------------------------------------------
            // 3. Interleaved array — fixed-precision (1 param)
            // ----------------------------------------------------------------

            #[test]
            fn given_interleaved_array_when_zfp_compress_decompress_fixed_precision_expect_bitstream_and_array_checksums_match(
            ) {
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::Interleaved);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedPrecision,
                    1,
                );
            }

            // ----------------------------------------------------------------
            // 4. Permuted array — fixed-precision (1 param, dims >= 2 only)
            // ----------------------------------------------------------------

            #[test]
            fn given_permuted_array_when_zfp_compress_decompress_fixed_precision_expect_bitstream_and_array_checksums_match(
            ) {
                if DIMS < 2 {
                    return;
                }
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::Permuted);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedPrecision,
                    1,
                );
            }

            // ----------------------------------------------------------------
            // 5. Fixed-precision (3 params)
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_decompress_fixed_precision_expect_bitstream_and_array_checksums_match(
            ) {
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::AsIs);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedPrecision,
                    3,
                );
            }

            // ----------------------------------------------------------------
            // 6. Fixed-rate: rate rounding
            // ----------------------------------------------------------------

            #[test]
            fn given_zfp_stream_when_set_rate_with_write_random_access_expect_rate_rounded_up_properly(
            ) {
                test_set_rate_align::<$rust_type>(DIMS);
            }

            // ----------------------------------------------------------------
            // 7. Fixed-rate: compress/decompress (3 params)
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_decompress_fixed_rate_expect_bitstream_and_array_checksums_match(
            ) {
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::AsIs);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedRate,
                    3,
                );
            }

            // ----------------------------------------------------------------
            // 8. Fixed-rate: bitrate validation (3 params)
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_fixed_rate_expect_compressed_bitrate_comparable_to_chosen_rate(
            ) {
                let (data, side_len, total_len) = get_data();
                let n = build_n(*side_len, DIMS);
                test_bitrate::<$rust_type>(DIMS, n, &data[..*total_len]);
            }

            // ----------------------------------------------------------------
            // 9. Fixed-accuracy: compress/decompress (3 config, floats only)
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_decompress_fixed_accuracy_expect_bitstream_and_array_checksums_match(
            ) {
                if !$is_float {
                    return;
                }
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::AsIs);
                run_compress_decompress::<$rust_type>(
                    DIMS,
                    CSUM_TYPE,
                    n,
                    &strided,
                    CompressionMode::FixedAccuracy,
                    3,
                );
            }

            // ----------------------------------------------------------------
            // 10. Fixed-accuracy: value validation (3 config, floats only)
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_fixed_accuracy_expect_compressed_values_within_accuracy(
            ) {
                if !$is_float {
                    return;
                }
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::AsIs);
                test_accuracy::<$rust_type>(DIMS, n, &strided);
            }

            // ----------------------------------------------------------------
            // 11. Reversible
            // ----------------------------------------------------------------

            #[test]
            fn given_array_when_zfp_compress_decompress_reversible_expect_bitstream_and_array_checksums_match(
            ) {
                let (data, side_len, _) = get_data();
                let n = build_n(*side_len, DIMS);
                let strided =
                    StridedData::<$rust_type>::build(data, n, DIMS, StrideConfig::AsIs);
                run_reversible::<$rust_type>(DIMS, CSUM_TYPE, n, &strided);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 16 test suite instantiations
// ---------------------------------------------------------------------------

endtoend_tests!(dim1_double, 1, Double, f64, true);
endtoend_tests!(dim1_float, 1, Float, f32, true);
endtoend_tests!(dim1_int32, 1, Int32, i32, false);
endtoend_tests!(dim1_int64, 1, Int64, i64, false);

endtoend_tests!(dim2_double, 2, Double, f64, true);
endtoend_tests!(dim2_float, 2, Float, f32, true);
endtoend_tests!(dim2_int32, 2, Int32, i32, false);
endtoend_tests!(dim2_int64, 2, Int64, i64, false);

endtoend_tests!(dim3_double, 3, Double, f64, true);
endtoend_tests!(dim3_float, 3, Float, f32, true);
endtoend_tests!(dim3_int32, 3, Int32, i32, false);
endtoend_tests!(dim3_int64, 3, Int64, i64, false);

endtoend_tests!(dim4_double, 4, Double, f64, true);
endtoend_tests!(dim4_float, 4, Float, f32, true);
endtoend_tests!(dim4_int32, 4, Int32, i32, false);
endtoend_tests!(dim4_int64, 4, Int64, i64, false);
