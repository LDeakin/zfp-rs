use crate::bitstream::core::{
    BitStreamState, BitStreamStorage, BitStreamStorageMut, bytes_to_words,
};
use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::{STREAM_WORD_BYTES, ZfpConfig};
use crate::field::{ZfpField, ZfpFieldMut};
use crate::types::{ZfpBitStreamWord, ZfpHeaderMask};

/// Owns a byte buffer and tracks a read/write bit cursor.
///
/// Internally stores data as a `Vec<u64>` (words) plus a bit-count and
/// in-progress word buffer, matching the C `bitstream` layout exactly.
pub struct ZfpBitStream {
    pub(crate) words: Vec<ZfpBitStreamWord>,
    pub(crate) state: BitStreamState,
}

impl BitStreamStorage for ZfpBitStream {
    fn words(&self) -> &[ZfpBitStreamWord] {
        &self.words
    }

    fn state(&self) -> &BitStreamState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BitStreamState {
        &mut self.state
    }
}

impl BitStreamStorageMut for ZfpBitStream {
    fn words_mut(&mut self) -> &mut [ZfpBitStreamWord] {
        &mut self.words
    }
}

impl std::fmt::Debug for ZfpBitStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZfpBitStream")
            .field("capacity_bytes", &self.capacity())
            .field("bits_written", &self.bits_written())
            .finish()
    }
}

impl ZfpBitStream {
    /// Create a new, empty `ZfpBitStream` with at least `capacity` bytes of storage.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let nwords = capacity.div_ceil(STREAM_WORD_BYTES);
        Self {
            words: vec![0u64; nwords],
            state: BitStreamState::new(),
        }
    }

    /// Wrap an existing word buffer (takes ownership).
    #[must_use]
    pub fn from_buffer(words: Vec<ZfpBitStreamWord>) -> Self {
        Self {
            words,
            state: BitStreamState::new(),
        }
    }

    /// Wrap an existing byte buffer as word-aligned 64-bit words.
    #[must_use]
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self::from_buffer(bytes_to_words(buf))
    }

    /// Consume the bitstream, returning the underlying word buffer.
    #[must_use]
    pub fn into_words(mut self) -> Vec<ZfpBitStreamWord> {
        self.flush();
        std::mem::take(&mut self.words)
    }

    /// Read one full 64-bit word.
    pub fn read_word(&mut self) -> u64 {
        ZfpBitStreamOps::read_word(self)
    }

    /// Write one full 64-bit word; returns the word previously at that position.
    pub fn write_word(&mut self, word: u64) -> u64 {
        ZfpBitStreamMutOps::write_word(self, word)
    }

    /// Read `n` bits (0 <= n <= 64) from the stream, LSB first.
    pub fn read_bits(&mut self, n: u32) -> u64 {
        ZfpBitStreamOps::read_bits(self, n)
    }

    /// Write the low `n` bits of `value`; return the overflow (bits above `n`).
    pub fn write_bits(&mut self, value: u64, n: u32) -> u64 {
        ZfpBitStreamMutOps::write_bits(self, value, n)
    }

    /// Read a single bit (0 or 1).
    pub fn read_bit(&mut self) -> u32 {
        ZfpBitStreamOps::read_bit(self)
    }

    /// Write a single bit (must be 0 or 1); returns the bit written.
    pub fn write_bit(&mut self, bit: u32) -> u32 {
        ZfpBitStreamMutOps::write_bit(self, bit)
    }

    /// Rewind the stream to the beginning (bit position 0).
    pub fn rewind(&mut self) {
        ZfpBitStreamOps::rewind(self);
    }

    /// Position the stream for writing at `offset` bits from the beginning.
    pub fn seek_write(&mut self, offset: u64) {
        ZfpBitStreamMutOps::seek_write(self, offset);
    }

    /// Position the stream for reading at `offset` bits from the beginning.
    pub fn seek_read(&mut self, offset: u64) {
        ZfpBitStreamOps::seek_read(self, offset);
    }

    /// Return the current write bit offset (`stream_wtell`).
    #[must_use]
    pub fn write_pos(&self) -> u64 {
        ZfpBitStreamOps::write_pos(self)
    }

    /// Return the current read bit offset (`stream_rtell`).
    #[must_use]
    pub fn read_pos(&self) -> u64 {
        ZfpBitStreamOps::read_pos(self)
    }

    /// Skip `n` bits forward in the read cursor.
    pub fn skip(&mut self, n: usize) {
        ZfpBitStreamOps::skip(self, n);
    }

    /// Append `n` zero-bits to the write stream (`stream_pad`).
    pub fn pad(&mut self, n: usize) {
        ZfpBitStreamMutOps::pad(self, n);
    }

    /// Discard buffered read bits and align to the next word boundary.
    pub fn align(&mut self) -> u32 {
        ZfpBitStreamOps::align(self)
    }

    /// Flush the write buffer to the next word boundary; return padding bits written.
    pub fn flush(&mut self) -> usize {
        ZfpBitStreamMutOps::flush(self)
    }

    /// Copy `n` bits from `src` into `self` (`stream_copy`).
    pub fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: usize) {
        ZfpBitStreamMutOps::copy_from(self, src, n);
    }

    /// Total number of bits written so far, matching `stream_wtell`.
    #[must_use]
    pub fn bits_written(&self) -> usize {
        ZfpBitStreamOps::bits_written(self)
    }

    /// Index of the next word to be read/written (equivalent to `ptr - begin`).
    #[must_use]
    pub fn word_pos(&self) -> usize {
        ZfpBitStreamOps::word_pos(self)
    }

    /// The current partial-word buffer value.
    #[cfg(test)]
    #[must_use]
    pub fn buffer_value(&self) -> u64 {
        self.state.buffer
    }

    /// The number of valid bits in the partial-word buffer.
    #[cfg(test)]
    #[must_use]
    pub fn buffer_bits(&self) -> u32 {
        self.state.bits
    }

    /// Byte capacity of the stream (`stream_capacity`).
    #[must_use]
    pub fn capacity(&self) -> usize {
        ZfpBitStreamOps::capacity(self)
    }

    /// Committed byte size (`size` = `word_pos * word_bytes`).
    #[must_use]
    pub fn size(&self) -> usize {
        ZfpBitStreamOps::size(self)
    }

    /// Read the word at a given word index without moving the cursor.
    #[cfg(test)]
    #[must_use]
    pub fn word_at(&self, index: usize) -> u64 {
        self.words[index]
    }

    /// Return the committed bytes as a byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        ZfpBitStreamOps::as_bytes(self)
    }

    /// Flush and consume the stream, returning the underlying byte buffer.
    #[must_use]
    pub fn into_vec(mut self) -> Vec<u8> {
        self.flush();
        self.as_bytes().to_vec()
    }

    /// Compress the field into this bitstream using the stream's parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError`][crate::types::ZfpCompressionError] if the field
    /// type or dimensions are unsupported for the selected configuration.
    pub fn compress(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        crate::compress::compress(self, field, config)
    }

    /// Decompress from this bitstream into the field.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpDecompressionError`][crate::types::ZfpDecompressionError] if the target
    /// field type or dimensions are unsupported for the selected configuration.
    pub fn decompress(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
    ) -> Result<usize, crate::types::ZfpDecompressionError> {
        crate::decompress::decompress(self, field, config)
    }

    /// Write the header section indicated by `mask` into this bitstream.
    pub fn write_header(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
        mask: ZfpHeaderMask,
    ) -> usize {
        let mode = config.mode_bits();
        crate::header::write_header_bs(self, field, mask, mode)
    }

    /// Read the header sections indicated by `mask` from this bitstream.
    ///
    /// The returned header contains metadata only when `mask` includes
    /// [`ZfpHeaderMask::META`], and a compression config only when `mask`
    /// includes [`ZfpHeaderMask::MODE`].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpHeaderError`][crate::header::ZfpHeaderError] if a requested
    /// header section is invalid.
    pub fn read_header(
        &mut self,
        mask: ZfpHeaderMask,
    ) -> Result<crate::header::ZfpHeader, crate::header::ZfpHeaderError> {
        crate::header::read_header_bs(self, mask)
    }

    /// Compress the field into this bitstream using the given execution policy.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError`][crate::types::ZfpCompressionError] if the field
    /// type or dimensions are unsupported for the selected configuration.
    pub fn compress_with_execution(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
        execution: crate::execution::ZfpExecution,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        match execution {
            crate::execution::ZfpExecution::Serial => {
                crate::compress::compress(self, field, config)
            }
            #[cfg(feature = "rayon")]
            crate::execution::ZfpExecution::Rayon {
                threads,
                chunk_size,
            } => crate::compress::compress_rayon(self, field, config, threads, chunk_size),
            #[cfg(not(feature = "rayon"))]
            crate::execution::ZfpExecution::Rayon { .. } => {
                // Rayon feature not compiled; fall back to serial.
                crate::compress::compress(self, field, config)
            }
        }
    }

    /// Decompress from this bitstream into the field using the given execution policy.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpDecompressionError`][crate::types::ZfpDecompressionError] if the target
    /// field type or dimensions are unsupported for the selected configuration.
    ///
    /// Note: Parallel decompression is only available for fixed-rate streams.
    /// Other modes fall back to serial decompression.
    pub fn decompress_with_execution(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
        execution: crate::execution::ZfpExecution,
    ) -> Result<usize, crate::types::ZfpDecompressionError> {
        match execution {
            crate::execution::ZfpExecution::Serial => {
                crate::decompress::decompress(self, field, config)
            }
            #[cfg(feature = "rayon")]
            crate::execution::ZfpExecution::Rayon {
                threads,
                chunk_size,
            } => crate::decompress::decompress_rayon(self, field, config, threads, chunk_size),
            #[cfg(not(feature = "rayon"))]
            crate::execution::ZfpExecution::Rayon { .. } => {
                // Rayon feature not compiled; fall back to serial.
                crate::decompress::decompress(self, field, config)
            }
        }
    }
}
