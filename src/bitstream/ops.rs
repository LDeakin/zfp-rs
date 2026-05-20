use crate::bitstream::core::{
    BitStreamStorage, BitStreamStorageMut, WSIZE, align_impl, as_committed_bytes, backing_bytes,
    flush_impl, pad_impl, read_bit_impl, read_bits_impl, read_pos_impl, read_word_impl,
    rewind_impl, seek_read_impl, seek_write_impl, skip_impl, write_bit_impl, write_bits_impl,
    write_pos_impl, write_word_impl,
};
use crate::config::STREAM_WORD_BYTES;
use crate::types::ZfpBitStreamWord;

/// Common read/cursor/inspection operations for ZFP bitstreams.
pub trait ZfpBitStreamOps {
    /// Read one full 64-bit word.
    fn read_word(&mut self) -> u64;
    /// Read `n` bits (0 <= n <= 64) from the stream, LSB first.
    fn read_bits(&mut self, n: u32) -> u64;
    /// Read a single bit (0 or 1).
    fn read_bit(&mut self) -> u32;
    /// Rewind the stream to the beginning (bit position 0).
    fn rewind(&mut self);
    /// Position the stream for reading at `offset` bits from the beginning.
    fn seek_read(&mut self, offset: u64);
    /// Return the current read bit offset (`stream_rtell`).
    fn read_pos(&self) -> u64;
    /// Return the current write bit offset (`stream_wtell`).
    fn write_pos(&self) -> u64;
    /// Return the backing word buffer (for parallel decompression access).
    fn words(&self) -> &[ZfpBitStreamWord];
    /// Skip `n` bits forward in the read cursor.
    fn skip(&mut self, n: usize);
    /// Discard buffered read bits and align to the next word boundary.
    fn align(&mut self) -> u32;
    /// Total number of bits written so far, matching `stream_wtell`.
    fn bits_written(&self) -> usize;
    /// Index of the next word to be read/written.
    fn word_pos(&self) -> usize;
    /// Byte capacity of the stream (`stream_capacity`).
    fn capacity(&self) -> usize;
    /// Committed byte size (`size` = `word_pos * word_bytes`).
    fn size(&self) -> usize;
    /// Return the committed bytes as a byte slice.
    fn as_bytes(&self) -> &[u8];
    /// Return the complete backing buffer as a byte slice.
    fn backing_bytes(&self) -> &[u8];
    /// The backing buffer's start pointer.
    #[cfg(feature = "ffi")]
    fn data_ptr(&self) -> *mut std::os::raw::c_void;
}

/// Mutating operations for writable ZFP bitstreams.
pub trait ZfpBitStreamMutOps: ZfpBitStreamOps {
    /// Write one full 64-bit word; returns the word previously at that position.
    fn write_word(&mut self, word: u64) -> u64;
    /// Write the low `n` bits of `value`; return the overflow (bits above `n`).
    fn write_bits(&mut self, value: u64, n: u32) -> u64;
    /// Write a single bit (must be 0 or 1); returns the bit written.
    fn write_bit(&mut self, bit: u32) -> u32;
    /// Position the stream for writing at `offset` bits from the beginning.
    fn seek_write(&mut self, offset: u64);
    /// Append `n` zero-bits to the write stream (`stream_pad`).
    fn pad(&mut self, n: usize);
    /// Flush the write buffer to the next word boundary; return padding bits written.
    fn flush(&mut self) -> usize;
    /// Copy `n` bits from `src` into `self` (`stream_copy`).
    fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: usize);
}

impl<T: BitStreamStorage + ?Sized> ZfpBitStreamOps for T {
    fn read_word(&mut self) -> u64 {
        read_word_impl(self)
    }

    fn read_bits(&mut self, n: u32) -> u64 {
        read_bits_impl(self, n)
    }

    fn read_bit(&mut self) -> u32 {
        read_bit_impl(self)
    }

    fn rewind(&mut self) {
        rewind_impl(self);
    }

    fn seek_read(&mut self, offset: u64) {
        seek_read_impl(self, offset);
    }

    fn read_pos(&self) -> u64 {
        read_pos_impl(self)
    }

    fn write_pos(&self) -> u64 {
        write_pos_impl(self)
    }

    fn words(&self) -> &[ZfpBitStreamWord] {
        <Self as BitStreamStorage>::words(self)
    }

    fn skip(&mut self, n: usize) {
        skip_impl(self, n);
    }

    fn align(&mut self) -> u32 {
        align_impl(self)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "bitstream capacities are addressable as usize on supported targets"
    )]
    fn bits_written(&self) -> usize {
        self.write_pos() as usize
    }

    fn word_pos(&self) -> usize {
        self.state().word_pos
    }

    fn capacity(&self) -> usize {
        self.words().len() * STREAM_WORD_BYTES
    }

    fn size(&self) -> usize {
        self.state().word_pos * STREAM_WORD_BYTES
    }

    fn as_bytes(&self) -> &[u8] {
        as_committed_bytes(self)
    }

    fn backing_bytes(&self) -> &[u8] {
        backing_bytes(self)
    }

    #[cfg(feature = "ffi")]
    fn data_ptr(&self) -> *mut std::os::raw::c_void {
        self.words()
            .as_ptr()
            .cast_mut()
            .cast::<std::os::raw::c_void>()
    }
}

impl<T: BitStreamStorageMut + ?Sized> ZfpBitStreamMutOps for T {
    fn write_word(&mut self, word: u64) -> u64 {
        write_word_impl(self, word)
    }

    fn write_bits(&mut self, value: u64, n: u32) -> u64 {
        write_bits_impl(self, value, n)
    }

    fn write_bit(&mut self, bit: u32) -> u32 {
        write_bit_impl(self, bit)
    }

    fn seek_write(&mut self, offset: u64) {
        seek_write_impl(self, offset);
    }

    fn pad(&mut self, n: usize) {
        pad_impl(self, n);
    }

    fn flush(&mut self) -> usize {
        flush_impl(self)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: usize) {
        let mut remaining = n;
        while remaining > WSIZE as usize {
            let w = src.read_bits(WSIZE);
            write_bits_impl(self, w, WSIZE);
            remaining -= WSIZE as usize;
        }
        if remaining > 0 {
            let w = src.read_bits(remaining as u32);
            write_bits_impl(self, w, remaining as u32);
        }
    }
}
