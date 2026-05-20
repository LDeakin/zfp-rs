//! Bit-level I/O: equivalent to the C `bitstream` type.
//!
//! Bits are written/read LSB-first in 64-bit word units, matching the reference
//! C implementation exactly (default `BIT_STREAM_WORD_TYPE` = `uint64`).
//!
//! `ZfpBitStream` is the central I/O type. It owns the compressed byte buffer
//! and provides methods for compression, decompression, and header I/O.
//! This mirrors the C API where the `bitstream` struct is the active I/O handle.

pub(crate) mod borrowed;
mod core;
mod ops;
mod owned;

pub use borrowed::{ZfpBitStreamRef, ZfpBitStreamRefMut};
pub use ops::{ZfpBitStreamMutOps, ZfpBitStreamOps};
pub use owned::ZfpBitStream;

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
        ZfpBitStream::new(STREAM_WORD_CAPACITY * 8)
    }

    #[test]
    fn when_alignment_expect_matching_stream_word_bits() {
        assert_eq!(WSIZE, 64);
    }

    #[test]
    fn when_bitstream_opened_expect_proper_length_and_boundaries() {
        let num_words: usize = 4;
        let s = ZfpBitStream::new(num_words * 8);
        assert_eq!(s.capacity(), num_words * 8);
        assert_eq!(s.word_pos(), 0);
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
            let mut s = ZfpBitStreamRefMut::from_words_mut(&mut words);
            s.write_bits(0x0123_4567_89ab_cdef, WSIZE);
            assert_eq!(s.size(), 8);
        }

        assert_eq!(words[0], 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn given_compression_field_without_data_when_compress_expect_no_data_error() {
        use crate::config::ZfpConfig;
        use crate::field::ZfpField;
        use crate::types::ZfpCompressionError;

        let field = ZfpField::new(&[] as &[f32], [4]);

        let mut bs = ZfpBitStream::new(1024);
        let err = bs
            .compress(&ZfpConfig::fixed_precision(8), &field)
            .unwrap_err();

        assert_eq!(err, ZfpCompressionError::NoData);
    }

    #[test]
    fn given_decompression_field_without_data_when_decompress_expect_no_data_error() {
        use crate::config::ZfpConfig;
        use crate::field::ZfpFieldMut;
        use crate::types::{ZfpDecompressionError, ZfpScalarType};

        // SAFETY: a null pointer with a zero byte count is accepted by
        // `from_raw` and produces a typed field with no output storage.
        let mut field = unsafe {
            ZfpFieldMut::from_raw(
                std::ptr::null_mut(),
                0,
                ZfpScalarType::Float,
                [4, 0, 0, 0],
                [0; 4],
            )
        };

        let mut bs = ZfpBitStream::new(1024);
        let err = bs
            .decompress(&ZfpConfig::fixed_precision(8), &mut field)
            .unwrap_err();

        assert_eq!(err, ZfpDecompressionError::NoData);
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
        let field = ZfpField::new(&data, [data.len()]);
        let config = ZfpConfig::fixed_rate(
            5.0,
            ZfpScalarType::Float,
            ZfpDimensionality::D1,
            ZfpStreamAlignment::None,
        );
        let execution = ZfpExecution::Rayon {
            threads: 2,
            chunk_size: 2,
        };

        let mut owned = ZfpBitStream::new(1024);
        let owned_size = owned
            .compress_with_execution(&config, &field, execution)
            .unwrap();

        let mut borrowed_words = vec![0u64; 128];
        let borrowed_size = {
            let mut borrowed = ZfpBitStreamRefMut::from_words_mut(&mut borrowed_words);
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
        let prev_size = s.size();
        s.write_word(WORD1);
        assert_eq!(s.size(), prev_size + 8);
        assert_eq!(s.word_at(0), WORD1);
    }

    #[test]
    fn when_write_two_words_expect_words_written_to_stream_consecutively() {
        let mut s = setup();
        s.write_word(WORD1);
        s.write_word(WORD2);
        assert_eq!(s.size(), 16);
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
        let prev_size = s.size();

        s.pad((WSIZE - existing_bit_count) as usize);

        assert_eq!(s.size(), prev_size + 8);
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
        let prev_size = s.size();

        s.pad(pad_amount as usize);

        assert_eq!(s.size(), prev_size + num_words * 8);
        s.rewind();
        assert_eq!(s.read_word(), existing_buffer);
    }

    #[test]
    fn when_write_bit_expect_bit_written_to_buffer_from_lsb() {
        let place: u32 = 3;
        let mut s = setup();
        s.write_bits(0, place);
        s.write_bit(1);
        assert_eq!(s.buffer_bits(), place + 1);
        assert_eq!(s.buffer_value(), 1u64 << place);
    }

    #[test]
    fn given_bitstream_buffer_one_bit_from_full_when_write_bit_expect_bit_written_to_buffer_written_to_stream_and_buffer_reset()
     {
        let place = WSIZE - 1;
        let mut s = setup();
        s.write_bits(0, place);
        s.write_bit(1);
        assert_eq!(s.size(), 8);
        assert_eq!(s.word_at(0), 1u64 << place);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn given_bitstream_with_bit_in_buffer_when_read_bit_expect_one_bit_read_from_lsb() {
        let mut s = setup();
        s.write_bit(1);
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();
        let bit = s.read_bit();
        assert_eq!(bit, 1);
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
        assert_eq!(bit, 1);
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
        let prev_pos = s.word_pos();

        let read_bits = s.read_bits(0);

        assert_eq!(s.buffer_bits(), prev_bits);
        assert_eq!(read_bits, 0);
        assert_eq!(s.buffer_value(), prev_buffer);
        assert_eq!(s.word_pos(), prev_pos);
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
        assert_eq!(s.word_pos(), 1);
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

        assert_eq!(s.word_pos(), 1);
        assert_eq!(s.buffer_bits(), (bit_offset % u64::from(WSIZE)) as u32);
        assert_eq!(s.buffer_value(), WORD2 & mask);
    }

    #[test]
    fn when_seek_read_to_multiple_of_wsize_expect_ptr_aligned_buffer_empty() {
        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.seek_read(u64::from(WSIZE));
        assert_eq!(s.word_pos(), 1);
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

        assert_eq!(s.word_pos(), 2);
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

        let prev_pos = s.word_pos();
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();

        s.skip(0);

        assert_eq!(s.word_pos(), prev_pos);
        assert_eq!(s.buffer_bits(), prev_bits);
        assert_eq!(s.buffer_value(), prev_buffer);
    }

    #[test]
    fn when_skip_within_buffer_expect_masked_buffer() {
        let read_bit_count: u32 = 3;
        let skip_count: usize = 5;
        let total_offset = u64::from(read_bit_count) + skip_count as u64;
        let expected_bits = WSIZE - (total_offset % u64::from(WSIZE)) as u32;
        let expected_buffer = WORD1 >> (total_offset % u64::from(WSIZE));

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);
        let prev_pos = s.word_pos();

        s.skip(skip_count);

        assert_eq!(s.word_pos(), prev_pos);
        assert_eq!(s.buffer_bits(), expected_bits);
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_skip_past_buffer_end_expect_new_masked_word_in_buffer() {
        let read_bit_count: u32 = 3;
        let skip_count: usize = WSIZE as usize + 5;
        let total_offset = u64::from(read_bit_count) + skip_count as u64;
        let expected_bits = WSIZE - (total_offset % u64::from(WSIZE)) as u32;
        let expected_buffer = WORD2 >> (total_offset % u64::from(WSIZE));

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);

        s.skip(skip_count);

        assert_eq!(s.word_pos(), 2);
        assert_eq!(s.buffer_bits(), expected_bits);
        assert_eq!(s.buffer_value(), expected_buffer);
    }

    #[test]
    fn when_align_expect_buffer_empty_bits_zero() {
        let read_bit_count: u32 = 3;

        let mut s = setup();
        s.write_bits(WORD1, WSIZE);
        s.write_bits(WORD2, WSIZE);
        s.rewind();
        s.read_bits(read_bit_count);
        let prev_pos = s.word_pos();

        s.align();

        assert_eq!(s.word_pos(), prev_pos);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
    }

    #[test]
    fn given_empty_buffer_when_flush_expect_nop() {
        let mut s = setup();
        let prev_pos = s.word_pos();
        let prev_bits = s.buffer_bits();
        let prev_buffer = s.buffer_value();

        let pad_count = s.flush();

        assert_eq!(s.word_pos(), prev_pos);
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
        let prev_pos = s.word_pos();

        let pad_count = s.flush();

        assert_eq!(s.word_pos(), prev_pos + 1);
        assert_eq!(s.buffer_bits(), 0);
        assert_eq!(s.buffer_value(), 0);
        assert_eq!(pad_count, (WSIZE - prev_buffer_bit_count) as usize);
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

        let mut dst = ZfpBitStream::new(STREAM_WORD_CAPACITY * 8);
        dst.seek_write(dst_offset);

        dst.copy_from(&mut src, copy_bits);

        assert_eq!(dst.word_pos(), 1);
        assert_eq!(dst.buffer_bits(), expected_bits);
        assert_eq!(dst.word_at(0), expected_written_word);
        assert_eq!(dst.buffer_value(), expected_buffer);
    }
}
