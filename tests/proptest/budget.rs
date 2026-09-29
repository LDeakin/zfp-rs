//! Property-based tests for expert configurations with tight limits.
//!
//! No C counterpart for small `max_bits`: when it is below a float block's
//! exponent header, C's budget wraps around, and zfp-rs writes just the header
//! instead. These assert the properties stream sizes depend on: the output
//! fits `maximum_size`, Rayon compression matches serial compression, and
//! decompression reads back exactly what compression wrote.

use proptest::prelude::*;
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpScalar};

const ZFP_MAX_BITS: u32 = 16658;

/// Any expert config with a small `max_bits`, lossy or reversible.
fn tight_config() -> impl Strategy<Value = ZfpConfig> {
    (
        1u32..=40,
        1u32..=64,
        prop_oneof![Just(-1075), -1074i32..=843],
    )
        .prop_flat_map(|(max_bits, max_prec, min_exp)| {
            (1u32..=max_bits)
                .prop_map(move |min_bits| ZfpConfig::expert(min_bits, max_bits, max_prec, min_exp))
        })
}

/// A reversible config whose `max_bits` and `max_prec` never bind, but whose
/// `min_bits` pads every block.
fn padded_reversible_config() -> impl Strategy<Value = ZfpConfig> {
    (1u32..=ZFP_MAX_BITS).prop_map(|min_bits| ZfpConfig::expert(min_bits, ZFP_MAX_BITS, 64, -1075))
}

/// Field dimensions of 1–4 dimensions, each 1–9 long.
fn dims() -> impl Strategy<Value = [usize; 4]> {
    prop::collection::vec(1usize..=9, 1..=4)
        .prop_map(|v| std::array::from_fn(|axis| v.get(axis).copied().unwrap_or(0)))
}

/// A 1-D field of `blocks` blocks, some of them all zero.
fn blocks_of<T: ZfpScalar + std::fmt::Debug>(
    value: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = Vec<T>> {
    prop::collection::vec(
        prop_oneof![Just(vec![T::default(); 4]), prop::collection::vec(value, 4),],
        1..=12,
    )
    .prop_map(|blocks| blocks.concat())
}

/// Compress into a stream of `maximum_size` bytes, serially and with Rayon,
/// then decompress, returning the decompressed values.
fn check<T: ZfpScalar>(
    config: &ZfpConfig,
    data: &[T],
    dims: [usize; 4],
) -> Result<Vec<T>, TestCaseError> {
    let field = ZfpField::new(data, dims).unwrap();
    let capacity = config.maximum_size(T::SCALAR_TYPE, dims).unwrap();
    let mut serial = ZfpBitStream::new(capacity);
    let size = serial.compress(config, &field);
    prop_assert!(size.is_ok(), "{size:?}, maximum_size {capacity}");

    #[cfg(feature = "rayon")]
    for (threads, chunk_size) in [(0, 0), (3, 0), (3, 1), (2, 3)] {
        let mut parallel = ZfpBitStream::new(capacity);
        parallel
            .compress_with_execution(
                config,
                &field,
                zfp_rs::ZfpExecution::Rayon {
                    threads,
                    chunk_size,
                },
            )
            .unwrap();
        prop_assert_eq!(parallel.as_bytes(), serial.as_bytes());
    }

    let mut out = vec![T::default(); data.len()];
    serial.rewind();
    let read = serial
        .decompress(config, &mut ZfpFieldMut::new(&mut out, dims).unwrap())
        .unwrap();
    prop_assert_eq!(read, serial.as_bytes().len());
    Ok(out)
}

macro_rules! budget_tests {
    ($fits:ident, $round_trips:ident, $scalar:ty) => {
        proptest! {
            #[test]
            fn $fits(
                config in tight_config(),
                (dims, data) in dims().prop_flat_map(|dims| {
                    let n = dims.iter().filter(|&&n| n > 0).product::<usize>();
                    (Just(dims), prop::collection::vec(any::<$scalar>(), n))
                }),
            ) {
                check(&config, &data, dims)?;
            }

            /// Padding to `min_bits`, all-zero float blocks included, is
            /// skipped on decode, so the next block starts in the right place.
            #[test]
            fn $round_trips(
                config in padded_reversible_config(),
                data in blocks_of(any::<$scalar>()),
            ) {
                let out = check(&config, &data, [data.len(), 0, 0, 0])?;
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&out),
                    bytemuck::cast_slice::<$scalar, u8>(&data)
                );
            }
        }
    };
}

budget_tests!(
    i32_fits_maximum_size,
    i32_reversible_min_bits_round_trips,
    i32
);
budget_tests!(
    i64_fits_maximum_size,
    i64_reversible_min_bits_round_trips,
    i64
);
budget_tests!(
    f32_fits_maximum_size,
    f32_reversible_min_bits_round_trips,
    f32
);
budget_tests!(
    f64_fits_maximum_size,
    f64_reversible_min_bits_round_trips,
    f64
);
