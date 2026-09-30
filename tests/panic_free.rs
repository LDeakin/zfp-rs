//! Every public entry point with unclamped arguments: none may panic.
//!
//! The test profile has overflow checks and debug assertions on, so arithmetic
//! overflow counts as a panic here. Each test asserts only that the calls
//! return; what they return is checked elsewhere.
//!
//! Field shapes, block counts and `min_bits` are bounded to keep the run
//! short. Those bounds are about time only: a larger field does more of the
//! same work, and cannot reach a panic a smaller one does not.

use proptest::prelude::*;
use zfp_rs::codec::block::{decode_block, encode_block};
use zfp_rs::{
    ZfpBitStream, ZfpBitStreamRef, ZfpBitStreamRefMut, ZfpConfig, ZfpDimensionality, ZfpExecution,
    ZfpField, ZfpFieldMetadata, ZfpFieldMut, ZfpHeaderMask, ZfpRounding, ZfpScalar, ZfpScalarType,
    ZfpStreamAlignment,
};

/// The most blocks a field may have for the codec to run on it.
const MAX_BLOCKS: usize = 1024;

/// The largest `min_bits` a config may pad each block to.
const MAX_MIN_BITS: u32 = 1 << 14;

fn scalar_type() -> impl Strategy<Value = ZfpScalarType> {
    prop_oneof![
        Just(ZfpScalarType::I32),
        Just(ZfpScalarType::I64),
        Just(ZfpScalarType::F32),
        Just(ZfpScalarType::F64),
    ]
}

fn dimensionality() -> impl Strategy<Value = ZfpDimensionality> {
    prop_oneof![
        Just(ZfpDimensionality::D1),
        Just(ZfpDimensionality::D2),
        Just(ZfpDimensionality::D3),
        Just(ZfpDimensionality::D4),
    ]
}

fn alignment() -> impl Strategy<Value = ZfpStreamAlignment> {
    prop_oneof![
        Just(ZfpStreamAlignment::Unaligned),
        Just(ZfpStreamAlignment::WordAligned),
    ]
}

fn rounding() -> impl Strategy<Value = ZfpRounding> {
    prop_oneof![
        Just(ZfpRounding::Never),
        any::<bool>().prop_map(|tight_error| ZfpRounding::First { tight_error }),
        any::<bool>().prop_map(|tight_error| ZfpRounding::Last { tight_error }),
    ]
}

fn execution() -> impl Strategy<Value = ZfpExecution> {
    prop_oneof![
        Just(ZfpExecution::Serial),
        (0..4u32, any::<u32>()).prop_map(|(threads, chunk_size)| ZfpExecution::Rayon {
            threads,
            chunk_size
        }),
        (0..4u32, 0..3u32).prop_map(|(threads, chunk_size)| ZfpExecution::Rayon {
            threads,
            chunk_size
        }),
    ]
}

/// A `u32` that is often small, sometimes at a boundary, and sometimes any.
fn wide_u32() -> impl Strategy<Value = u32> {
    prop_oneof![0..=80u32, Just(u32::MAX), any::<u32>()]
}

fn wide_i32() -> impl Strategy<Value = i32> {
    prop_oneof![
        -1100..=1100i32,
        Just(i32::MIN),
        Just(i32::MAX),
        any::<i32>()
    ]
}

fn wide_u64() -> impl Strategy<Value = u64> {
    prop_oneof![
        0..=2048u64,
        Just(u64::MAX),
        (u64::MAX - 256)..=u64::MAX,
        any::<u64>()
    ]
}

/// A config from every constructor. Arguments a constructor rejects fall back
/// to the default config, as the rejection itself is checked separately.
fn config() -> impl Strategy<Value = ZfpConfig> {
    let config = prop_oneof![
        (
            prop_oneof![0.0..70.0, any::<f64>()],
            scalar_type(),
            dimensionality(),
            alignment()
        )
            .prop_map(|(rate, ty, dims, align)| {
                ZfpConfig::fixed_rate(rate, ty, dims, align).unwrap_or_default()
            }),
        wide_u32().prop_map(ZfpConfig::fixed_precision),
        any::<f64>().prop_map(ZfpConfig::fixed_accuracy),
        Just(ZfpConfig::reversible()),
        (0..=MAX_MIN_BITS, wide_u32(), wide_u32(), wide_i32()).prop_map(
            |(min_bits, max_bits, max_prec, min_exp)| {
                ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp).unwrap_or_default()
            }
        ),
        any::<u64>().prop_map(|mode| ZfpConfig::from_mode_bits(mode).unwrap_or_default()),
    ];
    (config, rounding()).prop_map(|(config, rounding)| config.with_rounding(rounding))
}

fn dim() -> impl Strategy<Value = usize> {
    prop_oneof![0..=9usize, Just(usize::MAX), any::<usize>()]
}

fn stride() -> impl Strategy<Value = isize> {
    prop_oneof![
        Just(0isize),
        -20..=20isize,
        Just(isize::MIN),
        Just(isize::MAX),
        any::<isize>()
    ]
}

/// Dimensions, mostly of a small valid field of rank 1 to 4, and otherwise
/// anything.
fn shape() -> impl Strategy<Value = [usize; 4]> {
    let valid = (1..=4usize, [1..=7usize, 1..=7usize, 1..=7usize, 1..=7usize]).prop_map(
        |(rank, mut dims)| {
            for d in dims.iter_mut().skip(rank) {
                *d = 0;
            }
            dims
        },
    );
    prop_oneof![4 => valid, 1 => [dim(), dim(), dim(), dim()]]
}

/// Strides, mostly contiguous or small, and otherwise anything.
fn strides() -> impl Strategy<Value = [isize; 4]> {
    prop_oneof![
        3 => Just([0isize; 4]),
        2 => [-9..=9isize, -60..=60isize, -400..=400isize, -3000..=3000isize],
        1 => [stride(), stride(), stride(), stride()],
    ]
}

/// Leave `bs` somewhere arbitrary: after a partial write, or seeked anywhere.
fn position(bs: &mut ZfpBitStream, how: u8, bits: u32, offset: u64) {
    match how % 4 {
        0 => {}
        1 => {
            bs.write_bits(u64::MAX, bits);
        }
        2 => bs.seek_write(offset),
        _ => bs.seek_read(offset),
    }
}

#[allow(clippy::too_many_arguments)]
fn roundtrip<T: ZfpScalar>(
    data: &[T],
    dims: [usize; 4],
    strides: [isize; 4],
    config: &ZfpConfig,
    capacity: usize,
    start: (u8, u32, u64),
    mask: ZfpHeaderMask,
    execution: ZfpExecution,
) {
    let Ok(field) = ZfpField::new_strided(data, dims, strides) else {
        return;
    };
    let _ = (
        field.num_elements(),
        field.is_contiguous(),
        field.effective_strides(),
        field.index_span(),
        field.size_bytes(),
    );
    let _ = config.maximum_size(T::SCALAR_TYPE, field.dims());
    if field.num_blocks() > MAX_BLOCKS {
        return;
    }

    let mut bs = ZfpBitStream::new(capacity).unwrap();
    position(&mut bs, start.0, start.1, start.2);
    let _ = bs.write_header(config, &field.metadata(), mask);
    let _ = bs.compress_with_execution(config, &field, execution);
    let _ = (
        bs.as_bytes(),
        bs.write_pos(),
        bs.read_pos(),
        bs.overflowed(),
    );

    let mut out = vec![T::default(); data.len()];
    for (how, rest) in [(0, bs.write_pos()), (start.0, start.2)] {
        let Ok(mut out_field) = ZfpFieldMut::new_strided(&mut out, dims, strides) else {
            return;
        };
        bs.rewind();
        position(&mut bs, how, start.1, rest);
        let _ = bs.read_header(mask);
        let _ = bs.decompress_with_execution(config, &mut out_field, execution);
    }
}

macro_rules! roundtrip_tests {
    ($($name:ident: $t:ty),* $(,)?) => {$(
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(128))]
            #[test]
            fn $name(
                data in prop::collection::vec(any::<$t>(), 0..=2500),
                dims in shape(),
                strides in strides(),
                config in config(),
                capacity in 0..=4096usize,
                start in (any::<u8>(), wide_u32(), wide_u64()),
                mask in (0..8u32).prop_map(ZfpHeaderMask::from_bits_truncate),
                execution in execution(),
            ) {
                roundtrip(&data, dims, strides, &config, capacity, start, mask, execution);
            }
        }
    )*};
}

roundtrip_tests! {
    roundtrip_i32_is_total: i32,
    roundtrip_i64_is_total: i64,
    roundtrip_f32_is_total: f32,
    roundtrip_f64_is_total: f64,
}

proptest! {
    #[test]
    fn config_constructors_and_queries_are_total(
        rate in any::<f64>(),
        ty in scalar_type(),
        dims in dimensionality(),
        align in alignment(),
        precision in any::<u32>(),
        tolerance in any::<f64>(),
        (min_bits, max_bits, max_prec, min_exp) in (any::<u32>(), any::<u32>(), wide_u32(), any::<i32>()),
        mode in any::<u64>(),
        field_dims in [dim(), dim(), dim(), dim()],
    ) {
        let configs = [
            ZfpConfig::fixed_rate(rate, ty, dims, align).ok(),
            Some(ZfpConfig::fixed_precision(precision)),
            Some(ZfpConfig::fixed_accuracy(tolerance)),
            Some(ZfpConfig::reversible()),
            ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp).ok(),
            ZfpConfig::from_mode_bits(mode),
        ];
        for config in configs.into_iter().flatten() {
            let _ = (
                config.mode(),
                config.rate(dims),
                config.precision(),
                config.accuracy(),
                config.mode_bits(),
                config.checked_mode_bits(),
                config.maximum_size(ty, field_dims),
            );
        }
    }

    #[test]
    fn metadata_is_total(ty in scalar_type(), dims in [dim(), dim(), dim(), dim()], bits in any::<u64>()) {
        let _ = ZfpFieldMetadata { scalar_type: ty, dims }.to_bits();
        let _ = ZfpFieldMetadata::from_bits(bits);
    }

    #[test]
    fn field_setters_are_total(
        len in 0..=300usize,
        dims in [dim(), dim(), dim(), dim()],
        strides in [stride(), stride(), stride(), stride()],
        ty in scalar_type(),
        new_dims in [dim(), dim(), dim(), dim()],
        new_strides in [stride(), stride(), stride(), stride()],
    ) {
        let data = vec![0.5f64; len];
        let mut field = ZfpField::new(&data, [len.max(1)]).unwrap_or_else(|_| ZfpField::new(&[0.0f64; 1], 1usize).unwrap());
        let _ = field.set_strides(strides);
        let _ = field.set_metadata(ZfpFieldMetadata { scalar_type: ty, dims });
        let _ = field.set_strides(new_strides);
        let _ = field.set_metadata(ZfpFieldMetadata { scalar_type: ty, dims: new_dims });
        let _ = (field.num_elements(), field.num_blocks(), field.index_span(), field.size_bytes());
    }

    /// Arbitrary sequences of cursor operations with unclamped operands, on
    /// owned and borrowed streams of a few words.
    #[test]
    fn bitstream_operations_are_total(
        words in 0..=4usize,
        ops in prop::collection::vec((0..16u8, any::<u64>(), wide_u32(), wide_u64()), 0..64),
    ) {
        let mut owned = ZfpBitStream::new(words * 8).unwrap();
        let mut backing = vec![0u64; words];
        let mut source = ZfpBitStream::from_words(vec![0x0123_4567_89ab_cdef; words + 1]);
        for &(op, value, n, offset) in &ops {
            apply(&mut owned, &mut source, op, value, n, offset);
        }
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut backing);
        for &(op, value, n, offset) in &ops {
            apply(&mut borrowed, &mut source, op, value, n, offset);
        }
        let _ = owned.into_bytes().unwrap();
        let _ = ZfpBitStream::from_bytes(borrowed.backing_bytes()).unwrap();
    }

    #[test]
    fn header_reading_is_total(words in prop::collection::vec(any::<u64>(), 0..4), mask in 0..8u32, offset in wide_u64()) {
        let mut bs = ZfpBitStreamRef::from_words(&words);
        bs.seek_read(offset);
        let _ = bs.read_header(ZfpHeaderMask::from_bits_truncate(mask));
    }

    /// Block coding with a slice of any length, from any cursor position.
    #[test]
    fn block_coding_is_total(
        len in 0..=300usize,
        dims in dimensionality(),
        config in config(),
        capacity in 0..=2048usize,
        start in (any::<u8>(), wide_u32(), wide_u64()),
    ) {
        let data: Vec<f64> = (0..len).map(|i| (i as f64).sin() * 1e3).collect();
        let mut bs = ZfpBitStream::new(capacity).unwrap();
        position(&mut bs, start.0, start.1, start.2);
        let _ = encode_block(&mut bs, &config, &data, dims);
        bs.rewind();
        position(&mut bs, start.0, start.1, start.2);
        let mut out = vec![0f64; len];
        let _ = decode_block(&mut bs, &config, &mut out, dims);
    }
}

fn apply<S: zfp_rs::ZfpBitStreamMutOps>(
    bs: &mut S,
    source: &mut ZfpBitStream,
    op: u8,
    value: u64,
    n: u32,
    offset: u64,
) {
    match op {
        0 => {
            bs.write_bits(value, n);
        }
        1 => {
            bs.read_bits(n);
        }
        2 => bs.write_bit(value & 1 != 0),
        3 => {
            bs.read_bit();
        }
        4 => bs.write_word(value),
        5 => {
            bs.read_word();
        }
        6 => bs.seek_write(offset),
        7 => bs.seek_read(offset),
        8 => bs.skip(offset),
        9 => bs.pad(offset),
        10 => {
            bs.align();
        }
        11 => {
            bs.flush();
        }
        12 => bs.rewind(),
        13 => {
            source.seek_read(value);
            bs.copy_from(source, offset);
        }
        14 => {
            let _ = (
                bs.read_pos(),
                bs.write_pos(),
                bs.capacity(),
                bs.overflowed(),
            );
        }
        _ => {
            let _ = (bs.as_bytes(), bs.as_words(), bs.backing_words());
        }
    }
}

#[cfg(feature = "ffi")]
mod ffi {
    use super::{MAX_BLOCKS, MAX_MIN_BITS, dimensionality, shape, strides, wide_i32, wide_u32};
    use proptest::prelude::*;
    use zfp_rs::codec::block::{decode_partial_block_strided, encode_partial_block_strided};
    use zfp_rs::codec::promote;
    use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpScalarType};

    fn raw_config() -> impl Strategy<Value = ZfpConfig> {
        (0..=MAX_MIN_BITS, wide_u32(), wide_u32(), wide_i32()).prop_map(
            |(min_bits, max_bits, max_prec, min_exp)| {
                ZfpConfig::from_raw_params(min_bits, max_bits, max_prec, min_exp)
            },
        )
    }

    proptest! {
        /// Unvalidated C parameters and unchecked fields, as the C ABI passes.
        #[test]
        fn raw_parameters_and_unchecked_fields_are_total(
            data in prop::collection::vec(any::<f32>(), 0..=2500),
            dims in shape(),
            strides in strides(),
            config in raw_config(),
            capacity in 0..=4096usize,
        ) {
            let _ = (
                config.mode(),
                config.mode_bits(),
                config.checked_mode_bits(),
                config.maximum_size(ZfpScalarType::F32, dims),
            );
            let bytes = std::mem::size_of_val(data.as_slice());
            // SAFETY: `data` is valid for `bytes` bytes, and outlives `field`.
            let field = unsafe {
                ZfpField::from_raw_unchecked(data.as_ptr().cast(), bytes, ZfpScalarType::F32, dims, strides)
            };
            let _ = (field.num_elements(), field.num_blocks(), field.index_span(), field.size_bytes());
            if field.num_blocks() > MAX_BLOCKS {
                return Ok(());
            }
            let mut bs = ZfpBitStream::new(capacity).unwrap();
            let _ = bs.compress(&config, &field);
            bs.rewind();
            let mut out = data.clone();
            // SAFETY: `out` is valid for `bytes` bytes, and outlives `out_field`.
            let mut out_field = unsafe {
                ZfpFieldMut::from_raw_unchecked(out.as_mut_ptr().cast(), bytes, ZfpScalarType::F32, dims, strides)
            };
            let _ = bs.decompress(&config, &mut out_field);
        }

        /// Partial-block lengths of any size are rejected or coded.
        #[test]
        fn strided_block_lengths_are_total(
            (first, rest) in (any::<usize>(), [0..=5usize, 0..=5usize, 0..=5usize]),
            short in 0..=5usize,
            dims in dimensionality(),
            config in raw_config(),
        ) {
            // The first length is sometimes huge, but mostly near the limits.
            let first = if first % 2 == 0 { first } else { first % 7 };
            let lengths = [first, rest[0], rest[1], rest[2]];
            let lengths = if short < 3 { [short; 4] } else { lengths };
            let data = [1.5f64; 256];
            let mut out = [0f64; 256];
            let strides = [1, 4, 16, 64];
            let mut bs = ZfpBitStream::new(8192).unwrap();
            let reversible = ZfpConfig::reversible();
            // SAFETY: with contiguous strides, a block of lengths up to 4
            // lies within the 256 values; longer lengths are rejected first.
            unsafe {
                let _ = encode_partial_block_strided(&mut bs, data.as_ptr(), dims, lengths, &strides, &config);
                let _ = encode_partial_block_strided(&mut bs, data.as_ptr(), dims, lengths, &strides, &reversible);
                bs.rewind();
                let _ = decode_partial_block_strided(&mut bs, out.as_mut_ptr(), dims, lengths, &strides, &config);
                let _ = decode_partial_block_strided(&mut bs, out.as_mut_ptr(), dims, lengths, &strides, &reversible);
            }
        }

        #[test]
        fn promotion_is_total(dst_len in 0..=300usize, src_len in 0..=300usize, dims in dimensionality()) {
            let mut wide = vec![0i32; dst_len];
            let _ = promote::promote_u16_to_i32(&mut wide, &vec![7u16; src_len], dims);
            let mut narrow = vec![0u8; dst_len];
            let _ = promote::demote_i32_to_u8(&mut narrow, &vec![-7i32; src_len], dims);
        }
    }
}
