//! Property-based tests for expert configurations with tight limits.
//!
//! These assert the properties stream sizes depend on: the output fits
//! `maximum_size`, Rayon compression matches serial compression, and
//! decompression reads back exactly what compression wrote.

use proptest::prelude::*;
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpScalar};

const ZFP_MAX_BITS: u32 = 16658;

/// A reversible config whose `max_bits` and `max_prec` never bind, but whose
/// `min_bits` pads every block.
fn padded_reversible_config() -> impl Strategy<Value = ZfpConfig> {
    (1u32..=ZFP_MAX_BITS).prop_map(|min_bits| ZfpConfig::expert(min_bits, ZFP_MAX_BITS, 64, -1075))
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
    ($round_trips:ident, $scalar:ty) => {
        proptest! {
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

budget_tests!(i32_reversible_min_bits_round_trips, i32);
budget_tests!(i64_reversible_min_bits_round_trips, i64);
budget_tests!(f32_reversible_min_bits_round_trips, f32);
budget_tests!(f64_reversible_min_bits_round_trips, f64);
