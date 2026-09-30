//! Decompressing a stream that ends before the field does is an error.
//!
//! Reads past the end of a buffer yield zeros, and loading a word there, where
//! C reads out of the buffer, is flagged. These tests check that a complete
//! stream never trips that, whatever its mode, shape or type, and that a
//! stream missing words that decoding reads always does.

use zfp_rs::{
    ZfpBitStream, ZfpBitStreamRef, ZfpConfig, ZfpDecompressionError, ZfpDimensionality,
    ZfpExecution, ZfpField, ZfpFieldMut, ZfpScalar, ZfpScalarType, ZfpStreamAlignment,
};

/// Values that vary in sign and magnitude, so blocks have many bit planes.
trait Sample: ZfpScalar + Copy + Default + std::fmt::Debug + PartialEq {
    const TYPE: ZfpScalarType;
    fn sample(i: usize) -> Self;
}

fn noise(i: usize) -> u64 {
    (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40
}

impl Sample for i32 {
    const TYPE: ZfpScalarType = ZfpScalarType::I32;
    fn sample(i: usize) -> Self {
        noise(i) as i32 - (1 << 23)
    }
}
impl Sample for i64 {
    const TYPE: ZfpScalarType = ZfpScalarType::I64;
    fn sample(i: usize) -> Self {
        (noise(i) as i64 - (1 << 23)) << 20
    }
}
impl Sample for f32 {
    const TYPE: ZfpScalarType = ZfpScalarType::F32;
    fn sample(i: usize) -> Self {
        (noise(i) as f32 - 8.0e6) * 1.0e-3
    }
}
impl Sample for f64 {
    const TYPE: ZfpScalarType = ZfpScalarType::F64;
    fn sample(i: usize) -> Self {
        (noise(i) as f64 - 8.0e6) * 1.0e-3
    }
}

/// Shapes with whole blocks, partial blocks and a single value, in each rank.
const SHAPES: [[usize; 4]; 12] = [
    [1, 0, 0, 0],
    [4, 0, 0, 0],
    [5, 0, 0, 0],
    [17, 0, 0, 0],
    [1, 1, 0, 0],
    [4, 4, 0, 0],
    [5, 7, 0, 0],
    [8, 8, 0, 0],
    [4, 4, 4, 0],
    [5, 3, 6, 0],
    [4, 4, 4, 4],
    [5, 2, 3, 4],
];

fn rank(shape: &[usize; 4]) -> ZfpDimensionality {
    match shape.iter().filter(|&&n| n > 0).count() {
        1 => ZfpDimensionality::D1,
        2 => ZfpDimensionality::D2,
        3 => ZfpDimensionality::D3,
        _ => ZfpDimensionality::D4,
    }
}

/// A config with a name, and whether every block has the same size.
struct Case {
    name: String,
    config: ZfpConfig,
    fixed_size: bool,
}

/// One config per mode, and several rates: the whole-word rates end the
/// stream exactly on a word boundary.
fn cases<T: Sample>(dims: ZfpDimensionality) -> Vec<Case> {
    let mut cases = Vec::new();
    for rate in [1.0, 4.0, 8.0, 16.0, 23.0] {
        for align in [
            ZfpStreamAlignment::Unaligned,
            ZfpStreamAlignment::WordAligned,
        ] {
            cases.push(Case {
                name: format!("fixed_rate({rate}, {align:?})"),
                config: ZfpConfig::fixed_rate(rate, T::TYPE, dims, align).unwrap(),
                fixed_size: true,
            });
        }
    }
    let variable = [
        ("fixed_precision(1)", ZfpConfig::fixed_precision(1)),
        ("fixed_precision(9)", ZfpConfig::fixed_precision(9)),
        ("fixed_precision(64)", ZfpConfig::fixed_precision(64)),
        ("fixed_accuracy(1e-3)", ZfpConfig::fixed_accuracy(1.0e-3)),
        ("fixed_accuracy(8)", ZfpConfig::fixed_accuracy(8.0)),
        ("reversible", ZfpConfig::reversible()),
        (
            "expert(100, 200, 24, -20)",
            ZfpConfig::expert(100, 200, 24, -20).unwrap(),
        ),
    ];
    for (name, config) in variable {
        cases.push(Case {
            name: name.to_string(),
            config,
            fixed_size: false,
        });
    }
    cases
}

/// The executions to decode with: serial, and Rayon where it is built.
fn executions() -> Vec<ZfpExecution> {
    let mut executions = vec![ZfpExecution::Serial];
    if cfg!(feature = "rayon") {
        executions.push(ZfpExecution::Rayon {
            threads: 2,
            chunk_size: 1,
        });
    }
    executions
}

/// What decoding a stream returned, and the values it decoded.
type Decoded<T> = (Result<usize, ZfpDecompressionError>, Vec<T>);

/// Decode `bs` as the field of `shape`, without rewinding first.
fn decode<T: Sample>(
    bs: &mut impl zfp_rs::ZfpBitStreamOps,
    case: &Case,
    shape: [usize; 4],
    execution: ZfpExecution,
) -> Decoded<T> {
    let mut out = vec![T::default(); shape.iter().filter(|&&n| n > 0).product()];
    let mut field = ZfpFieldMut::new(&mut out, shape).unwrap();
    let result = bs.decompress_with_execution(&case.config, &mut field, execution);
    (result, out)
}

fn for_each_stream<T: Sample>(check: impl Fn(&Case, [usize; 4], &[u64], usize, &ZfpBitStream)) {
    for shape in SHAPES {
        let n = shape.iter().filter(|&&n| n > 0).product();
        let data: Vec<T> = (0..n).map(T::sample).collect();
        let field = ZfpField::new(&data, shape).unwrap();
        for case in cases::<T>(rank(&shape)) {
            let capacity = case.config.maximum_size(T::TYPE, shape).unwrap();
            let mut bs = ZfpBitStream::new(capacity).unwrap();
            let size = bs.compress(&case.config, &field).unwrap();
            let words = bs.as_words().to_vec();
            assert_eq!(size, words.len() * 8, "{} {shape:?}", case.name);
            check(&case, shape, &words, size, &bs);
        }
    }
}

/// A stream is never reported truncated when it is whole: read from an
/// exactly-sized buffer, owned or borrowed, however it ends in its last word.
fn whole_streams_are_not_truncated<T: Sample>() {
    for_each_stream::<T>(|case, shape, words, size, _| {
        let context = format!("{:?} {} {shape:?}", T::TYPE, case.name);
        for execution in executions() {
            let mut borrowed = ZfpBitStreamRef::from_words(words);
            assert_eq!(
                decode::<T>(&mut borrowed, case, shape, execution).0,
                Ok(size),
                "borrowed, {execution:?}: {context}"
            );
            let mut owned = ZfpBitStream::from_words(words.to_vec());
            assert_eq!(
                decode::<T>(&mut owned, case, shape, execution).0,
                Ok(size),
                "owned, {execution:?}: {context}"
            );
        }
    });
}

/// A stream missing its last word, or half its words, is reported, with as
/// many bytes as the buffer holds, unless decoding loads none of the missing
/// words. That happens only for fixed-rate padding skipped to a word boundary,
/// and the stream then decodes as the whole stream does, to the same size.
/// Filling the missing words with ones shows that decoding never read them. A
/// reported fixed-rate stream reports the size it needs.
fn missing_words_are_reported<T: Sample>() {
    for_each_stream::<T>(|case, shape, words, size, _| {
        let context = format!("{:?} {} {shape:?}", T::TYPE, case.name);
        let serial = ZfpExecution::Serial;
        let whole = decode::<T>(&mut ZfpBitStreamRef::from_words(words), case, shape, serial);
        for kept in [words.len() - 1, words.len() / 2] {
            let context = format!("{kept} of {} words, {context}", words.len());
            let mut ones = words.to_vec();
            ones[kept..].fill(u64::MAX);
            let ones = decode::<T>(&mut ZfpBitStreamRef::from_words(&ones), case, shape, serial);
            for execution in executions() {
                let mut bs = ZfpBitStreamRef::from_words(&words[..kept]);
                let decoded = decode::<T>(&mut bs, case, shape, execution);
                match decoded.0 {
                    Err(ZfpDecompressionError::Truncated { required, capacity }) => {
                        assert_eq!(capacity, kept * 8, "{execution:?}: {context}");
                        assert!(required > capacity, "{execution:?}: {context}");
                        if case.fixed_size {
                            assert_eq!(required, size, "{execution:?}: {context}");
                        }
                    }
                    Ok(_) if case.fixed_size => {
                        assert_eq!(decoded, whole, "{execution:?}: {context}");
                        assert_eq!(ones, whole, "{execution:?}: {context}");
                    }
                    _ => panic!("{execution:?}: {context}: {:?}", decoded.0),
                }
            }
        }
    });
}

/// A fixed-rate stream whose last block is all zeros is missing only padding
/// if cut at a word boundary after that block's first bit. C reads nothing
/// past the cut there, so the stream decodes as a whole one does, to the size
/// C returns, which exceeds the buffer. One word less, and the block itself is
/// missing.
#[test]
fn missing_padding_is_not_truncated() {
    let config = ZfpConfig::fixed_rate(
        64.0,
        ZfpScalarType::F64,
        ZfpDimensionality::D1,
        ZfpStreamAlignment::Unaligned,
    )
    .unwrap();
    let data: [f64; 8] = std::array::from_fn(|i| if i < 4 { f64::sample(i) } else { 0.0 });
    let field = ZfpField::new(&data, [8usize, 0, 0, 0]).unwrap();
    let mut bs = ZfpBitStream::new(config.maximum_size(ZfpScalarType::F64, [8]).unwrap()).unwrap();
    assert_eq!(bs.compress(&config, &field), Ok(64));
    let words = bs.as_words();
    let case = Case {
        name: String::new(),
        config,
        fixed_size: true,
    };
    let shape = [8, 0, 0, 0];
    for execution in executions() {
        // Block 1 starts at word 4; only its first bit is read before the skip.
        let mut padding = ZfpBitStreamRef::from_words(&words[..5]);
        let decoded = decode::<f64>(&mut padding, &case, shape, execution);
        assert_eq!(decoded, (Ok(64), data.to_vec()), "{execution:?}");
        let mut block = ZfpBitStreamRef::from_words(&words[..4]);
        assert_eq!(
            decode::<f64>(&mut block, &case, shape, execution).0,
            Err(ZfpDecompressionError::Truncated {
                required: 64,
                capacity: 32
            }),
            "{execution:?}"
        );
    }
}

/// The count is of whole words. A buffer cut inside its last word excludes that
/// word, as a borrowed stream does, and so decodes as the stream missing that
/// word does, or includes it zero-padded, as `ZfpBitStream::from_bytes` does,
/// and so is whole.
fn a_cut_inside_the_last_word_is_seen_only_without_padding<T: Sample>() {
    for_each_stream::<T>(|case, shape, words, size, bs| {
        let context = format!("{:?} {} {shape:?}", T::TYPE, case.name);
        let cut = &bs.as_bytes()[..size - 1];
        let serial = ZfpExecution::Serial;

        let mut borrowed = ZfpBitStreamRef::from_bytes(cut).unwrap();
        let mut shorter = ZfpBitStreamRef::from_words(&words[..words.len() - 1]);
        assert_eq!(
            decode::<T>(&mut borrowed, case, shape, serial),
            decode::<T>(&mut shorter, case, shape, serial),
            "{context}"
        );

        let mut padded = ZfpBitStream::from_bytes(cut).unwrap();
        let result = decode::<T>(&mut padded, case, shape, serial).0;
        assert_eq!(result, Ok(size), "{context}");
    });
}

/// A cursor already past the end has nothing left to read.
fn a_cursor_past_the_end_is_reported<T: Sample>() {
    for_each_stream::<T>(|case, shape, words, size, _| {
        let mut bs = ZfpBitStreamRef::from_words(words);
        bs.seek_read(size as u64 * 8 + 1);
        let result = decode::<T>(&mut bs, case, shape, ZfpExecution::Serial).0;
        assert!(
            matches!(result, Err(ZfpDecompressionError::Truncated { .. })),
            "{:?} {} {shape:?}: {result:?}",
            T::TYPE,
            case.name
        );
    });
}

macro_rules! for_each_type {
    ($($test:ident => $generic:ident),* $(,)?) => {$(
        mod $test {
            use super::*;

            #[test]
            fn i32() {
                $generic::<i32>();
            }
            #[test]
            fn i64() {
                $generic::<i64>();
            }
            #[test]
            fn f32() {
                $generic::<f32>();
            }
            #[test]
            fn f64() {
                $generic::<f64>();
            }
        }
    )*};
}

for_each_type! {
    whole_streams_are_not_truncated => whole_streams_are_not_truncated,
    missing_words_are_reported => missing_words_are_reported,
    a_cut_inside_the_last_word_is_seen_only_without_padding => a_cut_inside_the_last_word_is_seen_only_without_padding,
    a_cursor_past_the_end_is_reported => a_cursor_past_the_end_is_reported,
}
