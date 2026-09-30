//! Bit-level I/O: equivalent to the C `bitstream` type.
//!
//! Bits are written/read LSB-first in 64-bit word units, matching the reference
//! C implementation exactly (default `BIT_STREAM_WORD_TYPE` = `uint64`).
//!
//! `ZfpBitStream` is the central I/O type. It owns the compressed byte buffer
//! and provides methods for compression, decompression, and header I/O.
//! This mirrors the C API where the `bitstream` struct is the active I/O handle.

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

pub(crate) mod borrowed;
mod core;
mod ops;
mod owned;

pub use borrowed::{ZfpBitStreamRef, ZfpBitStreamRefMut};
pub use ops::{ZfpBitStreamMutOps, ZfpBitStreamOps};
pub use owned::ZfpBitStream;

#[cfg(feature = "rayon")]
pub(crate) use core::vec_with_capacity;
pub(crate) use core::{BitReader, BitWriter, overread, reset_overread};

#[cfg(test)]
pub(crate) use core::WSIZE;

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        reason = "test constants are chosen to fit the target integer types"
    )]

    use super::*;

    const WORD1: u64 = u64::MAX;
    const WORD2: u64 = 0x5555_5555_5555_5555;
    const STREAM_WORD_CAPACITY: usize = 3;

    fn setup() -> ZfpBitStream {
        ZfpBitStream::new(STREAM_WORD_CAPACITY * 8).unwrap()
    }

    #[test]
    fn when_alignment_expect_matching_stream_word_bits() {
        assert_eq!(WSIZE, 64);
    }

    #[test]
    fn when_bitstream_opened_expect_proper_length_and_boundaries() {
        let num_words: usize = 4;
        let s = ZfpBitStream::new(num_words * 8).unwrap();
        assert_eq!(s.capacity(), num_words * 8);
        assert_eq!(s.state.word_pos, 0);
    }

    #[test]
    fn given_borrowed_read_only_words_when_read_bits_expect_shared_cursor_logic() {
        let words = [0x0123_4567_89ab_cdefu64];
        let mut s = ZfpBitStreamRef::from_words(&words);

        assert_eq!(s.read_bits(8), 0xef);
        assert_eq!(s.read_bits(8), 0xcd);
        assert_eq!(s.read_pos(), 16);
        assert_eq!(s.capacity(), 8);
    }

    #[test]
    fn given_borrowed_mut_words_when_write_bits_expect_caller_buffer_updated() {
        let mut words = [0u64; 2];
        {
            let mut s = ZfpBitStreamRefMut::from_words(&mut words);
            s.write_bits(0x0123_4567_89ab_cdef, WSIZE);
            assert_eq!(s.as_bytes().len(), 8);
        }

        assert_eq!(words[0], 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn owned_conversions_round_trip_committed_data() {
        // A trailing partial word is zero-padded rather than dropped.
        let s = ZfpBitStream::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(s.capacity(), 16);
        assert_eq!(s.backing_bytes()[8..], [9, 0, 0, 0, 0, 0, 0, 0]);

        // `into_words` and `into_bytes` both return exactly what was written.
        let mut s = ZfpBitStream::new(64).unwrap();
        s.write_bits(0xabc, 12);
        assert!(s.as_words().is_empty());
        assert_eq!(s.into_words(), vec![0xabc]);

        let mut s = ZfpBitStream::from_words(vec![0; 8]);
        s.write_word(WORD2);
        s.write_bit(true);
        assert_eq!(s.as_words(), [WORD2]);
        let bytes = s.into_bytes().unwrap();
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes[8], 1);
    }

    #[test]
    fn given_empty_data_when_field_new_expect_insufficient_data_error() {
        use crate::field::ZfpField;
        use crate::types::ZfpFieldError;

        assert_eq!(
            ZfpField::new(&[] as &[f32], [4]).unwrap_err(),
            ZfpFieldError::InsufficientData {
                required: 16,
                actual: 0
            }
        );
    }

    #[test]
    fn given_undersized_stream_when_compress_expect_buffer_too_small_not_panic() {
        use crate::config::ZfpConfig;
        use crate::field::ZfpField;
        use crate::types::ZfpCompressionError;

        let data: Vec<f64> = (0..256).map(|i| f64::from(i).sin()).collect();
        let field = ZfpField::new(&data, [16usize, 16]).unwrap();
        let config = ZfpConfig::reversible();

        let mut big = ZfpBitStream::new(1 << 16).unwrap();
        let size = big.compress(&config, &field).unwrap();
        assert!(!big.overflowed());

        let mut small = ZfpBitStream::new(size - 8).unwrap();
        assert_eq!(
            small.compress(&config, &field),
            Err(ZfpCompressionError::BufferTooSmall {
                required: size,
                capacity: size - 8,
            })
        );
        assert!(small.overflowed());
        // The committed bytes are the prefix that fit.
        assert_eq!(small.as_bytes(), &big.as_bytes()[..size - 8]);
        small.rewind();
        assert!(!small.overflowed());

        let mut words = vec![0u64; 2];
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut words);
        assert_eq!(
            borrowed.compress(&config, &field),
            Err(ZfpCompressionError::BufferTooSmall {
                required: size,
                capacity: 16,
            })
        );
    }

    #[test]
    fn given_undersized_stream_when_write_header_expect_buffer_too_small_and_nothing_written() {
        use crate::config::ZfpConfig;
        use crate::field::ZfpField;
        use crate::types::{ZfpCompressionError, ZfpHeaderMask};

        let data = [0.0f32; 16];
        let field = ZfpField::new(&data, [4usize, 4]).unwrap();
        let config = ZfpConfig::fixed_precision(8);

        // 32 + 52 + 12 = 96 bits do not fit in one word.
        let mut bs = ZfpBitStream::new(8).unwrap();
        assert_eq!(
            bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL),
            Err(ZfpCompressionError::BufferTooSmall {
                required: 16,
                capacity: 8,
            })
        );
        assert_eq!(bs.write_pos(), 0);
        assert_eq!(
            bs.write_header(&config, &field.metadata(), ZfpHeaderMask::MAGIC),
            Ok(32)
        );
    }

    /// Past bit `u64::MAX`, `write_pos` wraps back to a small offset, which
    /// passed the capacity check: `write_header` returned `Ok` and dropped
    /// every bit.
    #[test]
    fn given_cursor_past_bit_u64_max_when_write_header_expect_buffer_too_small() {
        use crate::config::ZfpConfig;
        use crate::field::ZfpFieldMetadata;
        use crate::types::{ZfpCompressionError, ZfpHeaderMask, ZfpScalarType};

        let metadata = ZfpFieldMetadata {
            scalar_type: ZfpScalarType::F64,
            dims: [16, 0, 0, 0],
        };
        let mut bs = ZfpBitStream::new(64).unwrap();
        bs.seek_write(u64::MAX - 63);
        bs.pad(128);
        assert_eq!(bs.write_pos(), 64);
        // Two words past the cursor's word 2^58 + 1.
        let required = usize::try_from(((1u128 << 58) + 3) * 8).unwrap_or(usize::MAX);
        assert_eq!(
            bs.write_header(
                &ZfpConfig::fixed_precision(16),
                &metadata,
                ZfpHeaderMask::FULL
            ),
            Err(ZfpCompressionError::BufferTooSmall {
                required,
                capacity: 64,
            })
        );
        assert_eq!(bs.write_pos(), 64);
    }

    #[test]
    fn given_null_pointer_when_field_from_raw_expect_insufficient_data_error() {
        use crate::field::ZfpFieldMut;
        use crate::types::{ZfpFieldError, ZfpScalarType};

        // SAFETY: a null pointer is treated as an empty buffer.
        let result = unsafe {
            ZfpFieldMut::from_raw(
                std::ptr::null_mut(),
                0,
                ZfpScalarType::F32,
                [4, 0, 0, 0],
                [0; 4],
            )
        };
        assert_eq!(
            result.unwrap_err(),
            ZfpFieldError::InsufficientData {
                required: 16,
                actual: 0
            }
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn given_fixed_rate_stream_when_rayon_decompress_expect_serial_cursor_and_size() {
        use crate::config::{ZfpConfig, ZfpStreamAlignment};
        use crate::execution::ZfpExecution;
        use crate::field::{ZfpField, ZfpFieldMut};
        use crate::types::{ZfpDimensionality, ZfpScalarType};

        let data: Vec<f64> = (0..300).map(|i| f64::from(i).cos()).collect();
        let field = ZfpField::new(&data, [300usize]).unwrap();
        let config = ZfpConfig::fixed_rate(
            7.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let mut bs =
            ZfpBitStream::new(config.maximum_size(ZfpScalarType::F64, 300usize).unwrap()).unwrap();
        let written = bs.compress(&config, &field).unwrap();

        let mut decode = |execution| {
            let mut out = vec![0f64; 300];
            let mut out_field = ZfpFieldMut::new(&mut out, [300usize]).unwrap();
            bs.rewind();
            let read = bs
                .decompress_with_execution(&config, &mut out_field, execution)
                .unwrap();
            (read, bs.read_pos(), out)
        };
        let serial = decode(ZfpExecution::Serial);
        let parallel = decode(ZfpExecution::Rayon {
            threads: 3,
            chunk_size: 7,
        });
        assert_eq!(serial.0, written);
        assert_eq!(parallel, serial);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn given_borrowed_mut_words_when_rayon_compress_expect_owned_output_match() {
        use crate::config::{ZfpConfig, ZfpStreamAlignment};
        use crate::execution::ZfpExecution;
        use crate::field::ZfpField;
        use crate::types::{ZfpDimensionality, ZfpScalarType};

        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..17).map(|i| i as f32 * 0.25).collect();
        let field = ZfpField::new(&data, [data.len()]).unwrap();
        let config = ZfpConfig::fixed_rate(
            5.0,
            ZfpScalarType::F32,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let execution = ZfpExecution::Rayon {
            threads: 2,
            chunk_size: 2,
        };

        let mut owned = ZfpBitStream::new(1024).unwrap();
        let owned_size = owned
            .compress_with_execution(&config, &field, execution)
            .unwrap();

        let mut borrowed_words = vec![0u64; 128];
        let borrowed_size = {
            let mut borrowed = ZfpBitStreamRefMut::from_words(&mut borrowed_words);
            borrowed
                .compress_with_execution(&config, &field, execution)
                .unwrap()
        };

        assert_eq!(borrowed_size, owned_size);
        assert_eq!(
            bytemuck::cast_slice::<u64, u8>(&borrowed_words)[..borrowed_size],
            owned.as_bytes()[..owned_size]
        );
    }

    #[test]
    fn given_rewound_bitstream_when_write_word_expect_word_written_at_stream_begin() {
        let mut s = setup();
        let prev_size = s.as_bytes().len();
        s.write_word(WORD1);
        assert_eq!(s.as_bytes().len(), prev_size + 8);
        assert_eq!(s.word_at(0), WORD1);
    }

    #[test]
    fn when_write_two_words_expect_words_written_to_stream_consecutively() {
        let mut s = setup();
        s.write_word(WORD1);
        s.write_word(WORD2);
        assert_eq!(s.as_bytes().len(), 16);
        assert_eq!(s.word_at(0), WORD1);
        assert_eq!(s.word_at(1), WORD2);
    }

    #[test]
    fn given_bitstream_with_one_written_word_rewound_when_write_word_expect_newer_word_overwrites()
    {
        let mut s = setup();
        s.write_word(WORD1);
        s.rewind();
        s.write_word(WORD2);
        assert_eq!(s.word_at(0), WORD2);
    }

    #[test]
    fn when_read_word_expect_word_returned() {
        let mut s = setup();
        s.write_word(WORD1);
        s.rewind();
        assert_eq!(s.read_word(), WORD1);
    }

    #[test]
    fn when_read_two_words_expect_return_consecutive_words_in_order() {
        let mut s = setup();
        s.write_word(WORD1);
        s.write_word(WORD2);
        s.rewind();
        assert_eq!(s.read_word(), WORD1);
        assert_eq!(s.read_word(), WORD2);
    }

    #[test]
    fn given_started_buffer_when_stream_pad_expect_padded_word_written() {
        let existing_bit_count: u32 = 12;
        let existing_buffer: u64 = 0xfff;

        let mut s = setup();
        s.write_bits(existing_buffer, existing_bit_count);
        let prev_size = s.as_bytes().len();

        s.pad(u64::from(WSIZE - existing_bit_count));

        assert_eq!(s.as_bytes().len(), prev_size + 8);
        s.rewind();
        assert_eq!(s.read_word(), existing_buffer);
    }

    #[test]
    fn given_started_buffer_when_stream_pad_overflows_buffer_expect_proper_words_written() {
        let num_words: usize = 2;
        let existing_bit_count: u32 = 12;
        let existing_buffer: u64 = 0xfff;
        let pad_amount = num_words as u32 * WSIZE - existing_bit_count;

        let mut s = setup();
        s.write_word(0);
        s.write_word(WORD1);
        s.rewind();
        s.write_bits(existing_buffer, existing_bit_count);
        let prev_size = s.as_bytes().len();

        s.pad(u64::from(pad_amount));

        assert_eq!(s.as_bytes().len(), prev_size + num_words * 8);
        s.rewind();
        assert_eq!(s.read_word(), existing_buffer);
    }

    #[test]
    fn when_write_bit_expect_bit_written_to_buffer_from_lsb() {
        let place: u32 = 3;
        let mut s = setup();
        s.write_bits(0, place);
        s.write_bit(true);
        assert_eq!(s.buffer_bits(), place + 1);
        assert_eq!(s.buffer_value(), 1u64 << place);
    }

    #[test]
    fn given_bitstream_buffer_one_bit_from_full_when_write_bit_expect_bit_written_to_buffer_written_to_stream_and_buffer_reset()
     {
        let place = WSIZE - 1;
        let mut s = setup();
        s.write_bits(0, place);
        s.write_bit(true);
        assert_eq!(s.as_bytes().len(), 8);
        assert_eq!(s.word_at(0), 1u64 << place);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn given_bitstream_with_bit_in_buffer_when_read_bit_expect_one_bit_read_from_lsb() {
        let mut s = setup();
        s.write_bit(true);
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();
        let bit = s.read_bit();
        assert!(bit);
        assert_eq!(s.buffer_bits(), prev_bits - 1);
        assert_eq!(s.buffer_value(), prev_buffer >> 1);
    }

    #[test]
    fn given_bitstream_with_empty_buffer_when_read_bit_expect_load_next_word_to_buffer() {
        let mut s = setup();
        s.write_word(0);
        s.write_word(WORD1);
        s.rewind();
        s.read_word();
        // ptr is now at word 1, buffer=0, bits=0
        assert_eq!(s.buffer_value(), 0);
        let bit = s.read_bit();
        assert!(bit);
        assert_eq!(s.buffer_bits(), WSIZE - 1);
        assert_eq!(s.buffer_value(), WORD1 >> 1);
    }

    #[test]
    fn when_write_zero_bits_expect_nop() {
        let mut s = setup();
        let remaining = s.write_bits(WORD1, 0);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
        assert_eq!(remaining, WORD1);
    }

    #[test]
    fn when_write_bits_expect_bits_written_to_buffer_from_lsb() {
        let n: u32 = 3;
        let mask: u64 = 0x7;
        let mut s = setup();
        let remaining = s.write_bits(WORD1, n);
        assert_eq!(s.buffer_bits(), n);
        assert_eq!(s.buffer_value(), WORD1 & mask);
        assert_eq!(remaining, WORD1 >> n);
    }

    #[test]
    fn when_write_bits_fills_buffer_exactly_expect_word_written_to_stream() {
        let existing_bit_count: u32 = 5;
        let n = WSIZE - existing_bit_count;
        let completing_word: u64 = WORD2 & 0x07ff_ffff_ffff_ffff;

        let mut s = setup();
        s.write_bits(WORD1, existing_bit_count);
        let remaining = s.write_bits(completing_word, n);

        s.rewind();
        let read_word = s.read_word();
        assert_eq!(read_word, 0x1f + 0xaaaa_aaaa_aaaa_aaa0);
        assert_eq!(remaining, 0);
    }

    #[test]
    fn when_write_bits_overflows_buffer_expect_overflow_written_to_new_buffer() {
        let existing_bit_count: u32 = 5;
        let num_bits_to_write = WSIZE - 1;
        let overflow_bit_count = num_bits_to_write - (WSIZE - existing_bit_count);
        let word_to_write: u64 = WORD2 + 0x8000_0000_0000_0000;
        let overflowed_bits = word_to_write >> (WSIZE - existing_bit_count);
        let expected_buffer_result = overflowed_bits & 0xf;

        let mut s = setup();
        s.write_bits(WORD1, existing_bit_count);
        let remaining = s.write_bits(word_to_write, num_bits_to_write);

        assert_eq!(s.buffer_bits(), overflow_bit_count);
        assert_eq!(s.buffer_value(), expected_buffer_result);
        assert_eq!(remaining, word_to_write >> num_bits_to_write);
    }

    #[test]
    fn when_read_zero_bits_expect_nop() {
        // The C test injects s->buffer = WORD1; s->bits = wsize directly.
        // We replicate via seek_read to a non-zero offset so that buffer has a known
        // value with bits > 0, then verify read_bits(0) leaves state unchanged.
        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        // seek_read(wsize - 5 = 59): bits = wsize - 59 = 5, buffer = WORD1 >> 59.
        // Use seek_read(0) ... no. Use partial read instead.
        s.rewind();
        s.read_bits(5); // bits = wsize - 5 = 59, buffer = WORD1 >> 5

        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();
        let prev_pos = s.state.word_pos;

        let read_bits = s.read_bits(0);

        assert_eq!(s.buffer_bits(), prev_bits);
        assert_eq!(read_bits, 0);
        assert_eq!(s.buffer_value(), prev_buffer);
        assert_eq!(s.state.word_pos, prev_pos);
    }

    #[test]
    fn when_read_bits_expect_bits_read_in_order_lsb() {
        let bits_to_read: u32 = 2;
        let mask: u64 = 0x3;
        let mut s = setup();
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        let read_bits = s.read_bits(bits_to_read);
        assert_eq!(s.buffer_bits(), WSIZE - bits_to_read);
        assert_eq!(read_bits, WORD2 & mask);
        assert_eq!(s.buffer_value(), WORD2 >> bits_to_read);
    }

    #[test]
    fn given_bitstream_buffer_empty_with_next_word_available_when_read_bits_wsize_expect_entire_next_word_returned()
     {
        let mut s = setup();
        s.write_bits(0, WSIZE);
        s.write_bits(WORD1, WSIZE);
        s.rewind();
        s.read_word(); // advance ptr past first word; bits=0, buffer=0
        assert_eq!(s.buffer_value(), 0);
        let read_bits = s.read_bits(WSIZE);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(read_bits, WORD1);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn when_read_bits_spreads_across_two_words_expect_bits_combined_from_both_words() {
        let partial_word_bit_count: u32 = 16;
        let read_bit_count = WSIZE - 3; // 61
        let num_overflowed_bits = read_bit_count - partial_word_bit_count; // 45
        let expected_buffer_bit_count = WSIZE - num_overflowed_bits; // 19

        let partial_word1: u64 = WORD1 & 0xffff; // 0xffff

        // The C test sets s->buffer = stream_read_word(s); s->bits = 16 directly.
        // Replicate via seek_read(WSIZE - partial_word_bit_count = 48):
        //   buffer = words[0] >> 48, bits = 16.
        // For buffer == 0xffff, we need words[0] == 0xffff << 48 == 0xffff_0000_0000_0000.
        // Achieved by: write 48 zero bits, then write partial_word1 in 16 bits (filling word 0).
        let mut s = setup();
        s.write_bits(0, WSIZE - partial_word_bit_count); // bits 0-47 of word0 = 0
        s.write_bits(partial_word1, partial_word_bit_count); // bits 48-63 of word0 = 0xffff
        s.write_bits(WORD2, WSIZE); // word1 = WORD2

        // seek_read(48): reads words[0] → buffer = words[0] >> 48 = 0xffff, bits = 16.
        s.seek_read(u64::from(WSIZE - partial_word_bit_count));
        assert_eq!(s.buffer_bits(), partial_word_bit_count);
        assert_eq!(s.buffer_value(), partial_word1);

        let read_bits = s.read_bits(read_bit_count);

        assert_eq!(s.buffer_bits(), expected_buffer_bit_count);
        // Low partial_word_bit_count bits from word0, then num_overflowed_bits from word1.
        let expected_value = partial_word1
            | ((WORD2 & ((1u64 << num_overflowed_bits) - 1)) << partial_word_bit_count);
        assert_eq!(read_bits, expected_value);
        let expected_buffer = WORD2 >> num_overflowed_bits;
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_write_pos_expect_returns_written_bit_count() {
        let write_bit_count1 = WSIZE;
        let write_bit_count2: u32 = 6;

        let mut s = setup();
        s.write_bits(WORD1, write_bit_count1);
        s.write_bits(WORD1, write_bit_count2);

        assert_eq!(
            s.write_pos(),
            u64::from(write_bit_count1 + write_bit_count2)
        );
    }

    #[test]
    fn when_read_pos_expect_returns_read_bit_count() {
        let read_bit_count1 = WSIZE - 6;
        let read_bit_count2 = WSIZE;

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD1, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count1);
        s.read_bits(read_bit_count2);

        assert_eq!(s.read_pos(), u64::from(read_bit_count1 + read_bit_count2));
    }

    #[test]
    fn when_seek_write_to_multiple_of_wsize_expect_ptr_aligned_buffer_empty() {
        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.seek_write(u64::from(WSIZE));
        assert_eq!(s.state.word_pos, 1);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn when_seek_write_to_non_multiple_of_wsize_expect_masked_word_loaded_to_buffer() {
        let bit_offset = u64::from(WSIZE) + 5;
        let mask: u64 = 0x1f;

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.seek_write(bit_offset);

        assert_eq!(s.state.word_pos, 1);
        assert_eq!(s.buffer_bits(), (bit_offset % u64::from(WSIZE)) as u32);
        assert_eq!(s.buffer_value(), WORD2 & mask);
    }

    #[test]
    fn when_seek_read_to_multiple_of_wsize_expect_ptr_aligned_buffer_empty() {
        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.seek_read(u64::from(WSIZE));
        assert_eq!(s.state.word_pos, 1);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn when_seek_read_to_non_multiple_of_wsize_expect_masked_word_loaded_to_buffer() {
        let bit_offset = u64::from(WSIZE) + 5;
        let expected_bits = WSIZE - (bit_offset % u64::from(WSIZE)) as u32;
        let expected_buffer = WORD2 >> (bit_offset % u64::from(WSIZE));

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.seek_read(bit_offset);

        assert_eq!(s.state.word_pos, 2);
        assert_eq!(s.buffer_bits(), expected_bits);
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_skip_zero_bits_expect_nop() {
        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        s.read_bits(2);

        let prev_pos = s.state.word_pos;
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();

        s.skip(0);

        assert_eq!(s.state.word_pos, prev_pos);
        assert_eq!(s.buffer_bits(), prev_bits);
        assert_eq!(s.buffer_value(), prev_buffer);
    }

    #[test]
    fn when_skip_within_buffer_expect_masked_buffer() {
        let read_bit_count: u32 = 3;
        let skip_count: u64 = 5;
        let total_offset = u64::from(read_bit_count) + skip_count;
        let expected_bits = WSIZE - (total_offset % u64::from(WSIZE)) as u32;
        let expected_buffer = WORD1 >> (total_offset % u64::from(WSIZE));

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);
        let prev_pos = s.state.word_pos;

        s.skip(skip_count);

        assert_eq!(s.state.word_pos, prev_pos);
        assert_eq!(s.buffer_bits(), expected_bits);
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_skip_past_buffer_end_expect_new_masked_word_in_buffer() {
        let read_bit_count: u32 = 3;
        let skip_count = u64::from(WSIZE) + 5;
        let total_offset = u64::from(read_bit_count) + skip_count;
        let expected_bits = WSIZE - (total_offset % u64::from(WSIZE)) as u32;
        let expected_buffer = WORD2 >> (total_offset % u64::from(WSIZE));

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);

        s.skip(skip_count);

        assert_eq!(s.state.word_pos, 2);
        assert_eq!(s.buffer_bits(), expected_bits);
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_seek_read_past_end_expect_offset_kept_and_zeros_read() {
        let mut s = setup();
        let offset = 5 * u64::from(WSIZE) + 3;
        s.seek_read(offset);
        assert_eq!(s.read_pos(), offset);
        assert_eq!(s.read_bits(10), 0);
        assert_eq!(s.read_pos(), offset + 10);
        s.skip(100);
        assert_eq!(s.read_pos(), offset + 110);
    }

    /// Loading a word past the end, where C reads out of the buffer, is
    /// flagged. A seek to a word boundary there loads nothing, as in C.
    #[test]
    fn when_read_loads_word_past_end_expect_overread() {
        let end = STREAM_WORD_CAPACITY as u64 * u64::from(WSIZE);
        let mut s = setup();
        s.seek_read(end);
        s.skip(u64::from(WSIZE));
        assert!(!overread(&s));
        s.seek_read(end + 3);
        assert!(overread(&s));
        s.seek_read(0);
        assert!(overread(&s), "the flag outlives the next seek");

        let mut s = setup();
        s.seek_read(end - 1);
        assert_eq!(s.read_bits(1), 0);
        assert!(!overread(&s));
        assert!(!s.read_bit());
        assert!(overread(&s));

        let mut s = setup();
        s.seek_read(end + 3);
        reset_overread(&mut s);
        assert!(overread(&s), "the cursor holds bits of a missing word");
        s.seek_read(end);
        reset_overread(&mut s);
        assert!(!overread(&s));
    }

    /// The block decoders' reader peeks past the end of the buffer without
    /// flagging it, and flags consuming a bit there.
    #[test]
    fn when_bit_reader_consumes_past_end_expect_overread_but_not_when_peeking() {
        let end = STREAM_WORD_CAPACITY as u64 * u64::from(WSIZE);
        let mut s = setup();
        s.seek_read(end - 4);
        {
            let mut reader = BitReader::new(&mut s);
            assert_eq!(reader.peek(), 0);
            reader.consume(4);
        }
        assert!(!overread(&s));
        {
            let mut reader = BitReader::new(&mut s);
            reader.consume(1);
        }
        assert!(overread(&s));
    }

    #[test]
    fn when_seek_write_past_end_expect_offset_kept_and_writes_dropped() {
        let mut s = setup();
        let offset = 5 * u64::from(WSIZE) + 3;
        s.seek_write(offset);
        assert_eq!(s.write_pos(), offset);
        s.write_bits(0x3ff, 10);
        s.flush();
        assert!(s.overflowed());
        assert_eq!(s.write_pos(), 6 * u64::from(WSIZE));
        assert!(s.backing_words().iter().all(|&w| w == 0));
        assert_eq!(s.as_bytes().len(), s.capacity());
    }

    #[test]
    fn when_align_expect_buffer_empty_bits_zero() {
        let read_bit_count: u32 = 3;

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);
        let prev_pos = s.state.word_pos;

        s.align();

        assert_eq!(s.state.word_pos, prev_pos);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn given_empty_buffer_when_flush_expect_nop() {
        let mut s = setup();
        let prev_pos = s.state.word_pos;
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();

        let pad_count = s.flush();

        assert_eq!(s.state.word_pos, prev_pos);
        assert_eq!(s.buffer_bits(), prev_bits);
        assert_eq!(s.buffer_value(), prev_buffer);
        assert_eq!(pad_count, 0);
    }

    #[test]
    fn when_flush_expect_padded_word_written_to_stream() {
        let prev_buffer_bit_count: u32 = 8;

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD1, WSIZE);
        s.rewind();
        s.write_bits(WORD2, prev_buffer_bit_count);
        let prev_pos = s.state.word_pos;

        let pad_count = s.flush();

        assert_eq!(s.state.word_pos, prev_pos + 1);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
        assert_eq!(pad_count, WSIZE - prev_buffer_bit_count);
    }

    #[test]
    fn when_capacity_cannot_be_allocated_expect_alloc_error() {
        use crate::types::ZfpAllocError;

        // More than the address space: the layout itself overflows.
        assert_eq!(
            ZfpBitStream::new(usize::MAX).unwrap_err(),
            ZfpAllocError { bytes: usize::MAX }
        );
        // A valid layout that no allocator can satisfy.
        let huge = isize::MAX.cast_unsigned() & !7;
        assert_eq!(
            ZfpBitStream::new(huge).unwrap_err(),
            ZfpAllocError { bytes: huge }
        );
        assert_eq!(ZfpBitStream::new(0).unwrap().capacity(), 0);
        assert_eq!(ZfpBitStream::new(9).unwrap().capacity(), 16);
    }

    /// Every piece of cursor state, and the buffer, for comparing streams.
    fn snapshot(s: &ZfpBitStream) -> (usize, u64, u32, bool, Vec<u64>) {
        (
            s.state.word_pos,
            s.state.buffer,
            s.state.bits,
            s.state.overflowed,
            s.words.clone(),
        )
    }

    #[test]
    fn when_bit_count_above_64_expect_64_bits() {
        let mut words = setup();
        words.write_word(WORD2);
        words.write_word(WORD1);
        let bytes = words.into_bytes().unwrap();

        for n in [65, 200, u32::MAX] {
            let mut clamped = ZfpBitStream::from_bytes(&bytes).unwrap();
            let mut exact = ZfpBitStream::from_bytes(&bytes).unwrap();
            clamped.read_bits(3);
            exact.read_bits(3);
            assert_eq!(clamped.read_bits(n), exact.read_bits(64), "n={n}");
            assert_eq!(snapshot(&clamped), snapshot(&exact), "n={n}");

            let mut clamped = setup();
            let mut exact = setup();
            clamped.write_bits(1, 1);
            exact.write_bits(1, 1);
            assert_eq!(clamped.write_bits(WORD2, n), 0, "n={n}");
            exact.write_bits(WORD2, 64);
            assert_eq!(snapshot(&clamped), snapshot(&exact), "n={n}");
        }
    }

    #[test]
    fn when_put_bit_above_one_expect_wrapping_addition_as_in_c() {
        use crate::bitstream::core::BitStreamStorageMut;

        let mut s = setup();
        s.write_bits(0, 32);
        s.put_bit(u32::MAX);
        assert_eq!(s.buffer_value(), 0xffff_ffff_0000_0000);
        s.put_bit(1);
        assert_eq!(s.buffer_value(), 0x0000_0001_0000_0000);
        assert_eq!(s.buffer_bits(), 34);
        s.read_bits(64);
    }

    #[test]
    fn when_pad_past_end_expect_same_state_as_writing_word_by_word() {
        for start in [0, 5, 64 + 63, 2 * 64, 3 * 64 + 1, 7 * 64 + 9] {
            for n in [0, 1, 63, 64, 65, 3 * 64, 6 * 64 + 5, 11 * 64] {
                let mut bulk = setup();
                let mut stepwise = setup();
                for s in [&mut bulk, &mut stepwise] {
                    s.seek_write(start);
                    s.write_bits(0x2a5, 10);
                }
                bulk.pad(n);
                let mut remaining = n;
                while remaining > 0 {
                    let step = remaining.min(64);
                    stepwise.write_bits(0, step as u32);
                    remaining -= step;
                }
                assert_eq!(snapshot(&bulk), snapshot(&stepwise), "start={start} n={n}");
            }
        }
    }

    #[test]
    fn when_pad_huge_expect_prompt_return_and_overflow() {
        let mut s = setup();
        s.write_bits(0x3, 2);
        s.pad(u64::MAX);
        assert!(s.overflowed());
        assert_eq!(s.word_at(0), 0x3);
        assert!(s.backing_words()[1..].iter().all(|&w| w == 0));
    }

    #[test]
    fn when_copy_past_end_expect_same_state_as_copying_word_by_word() {
        for src_start in [0, 3, 64, 2 * 64 + 17, 5 * 64] {
            for dst_start in [0, 7, 3 * 64, 4 * 64 + 1] {
                for n in [0, 5, 64, 65, 4 * 64 + 3, 9 * 64, 13 * 64 + 60] {
                    let mut src_bulk = setup();
                    src_bulk.write_word(WORD2);
                    src_bulk.write_word(WORD1);
                    src_bulk.write_word(0x0123_4567_89ab_cdef);
                    let bytes = src_bulk.into_bytes().unwrap();

                    let mut src_bulk = ZfpBitStream::from_bytes(&bytes).unwrap();
                    let mut src_step = ZfpBitStream::from_bytes(&bytes).unwrap();
                    let mut dst_bulk = setup();
                    let mut dst_step = setup();
                    for s in [&mut src_bulk, &mut src_step] {
                        s.seek_read(src_start);
                    }
                    for s in [&mut dst_bulk, &mut dst_step] {
                        s.seek_write(dst_start);
                    }

                    dst_bulk.copy_from(&mut src_bulk, n);
                    let mut remaining = n;
                    while remaining > 0 {
                        let step = remaining.min(64) as u32;
                        let w = src_step.read_bits(step);
                        dst_step.write_bits(w, step);
                        remaining -= u64::from(step);
                    }

                    let at = format!("src={src_start} dst={dst_start} n={n}");
                    assert_eq!(snapshot(&src_bulk), snapshot(&src_step), "{at}");
                    assert_eq!(snapshot(&dst_bulk), snapshot(&dst_step), "{at}");
                }
            }
        }
    }

    #[test]
    fn when_copy_huge_expect_prompt_return_and_overflow() {
        let mut src = setup();
        src.write_word(WORD1);
        src.rewind();
        let mut dst = setup();
        dst.copy_from(&mut src, u64::MAX);
        assert!(dst.overflowed());
        assert_eq!(dst.word_at(0), WORD1);
        assert!(dst.backing_words()[1..].iter().all(|&w| w == 0));
    }

    #[test]
    fn when_stream_copy_expect_bits_copied_to_dest_bitstream() {
        let src_offset: u64 = u64::from(WSIZE) - 6;
        let dst_offset: u64 = 5;
        let copy_bits: usize = WSIZE as usize + 4;

        let num_word2_bits_written_to_word = dst_offset + (u64::from(WSIZE) - src_offset);
        let expected_written_word =
            ((WORD1 >> src_offset) << dst_offset) + (WORD2 << num_word2_bits_written_to_word);
        let expected_bits = ((dst_offset as usize + copy_bits) % WSIZE as usize) as u32;
        let expected_buffer =
            (WORD2 >> num_word2_bits_written_to_word) & ((1u64 << expected_bits) - 1);

        let mut src = setup();
        src.write_word(WORD1);
        src.write_word(WORD2);
        src.flush();
        src.seek_read(src_offset);

        let mut dst = ZfpBitStream::new(STREAM_WORD_CAPACITY * 8).unwrap();
        dst.seek_write(dst_offset);

        dst.copy_from(&mut src, copy_bits as u64);

        assert_eq!(dst.state.word_pos, 1);
        assert_eq!(dst.buffer_bits(), expected_bits);
        assert_eq!(dst.word_at(0), expected_written_word);
        assert_eq!(dst.buffer_value(), expected_buffer);
    }
}
