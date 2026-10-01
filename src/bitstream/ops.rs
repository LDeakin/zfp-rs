use crate::bitstream::core::{
    BitStreamStorage, BitStreamStorageMut, align_impl, backing_bytes, committed_words, copy_impl,
    flush_impl, pad_impl, read_bits_impl, read_pos_impl, read_word_impl, rewind_impl,
    seek_read_impl, seek_write_impl, skip_impl, write_bits_impl, write_pos_impl, write_word_impl,
};
use crate::bitstream::{ZfpBitStream, ZfpBitStreamRef, ZfpBitStreamRefMut};
use crate::config::ZfpConfig;
use crate::field::{ZfpField, ZfpFieldMetadata, ZfpFieldMut};
use crate::types::{ZfpBitStreamWord, ZfpHeaderMask};

/// Read, cursor and inspection operations shared by every ZFP bitstream.
///
/// A stream has a single bit cursor used for both reading and writing, as in
/// the C `bitstream`. Bit offsets are `u64`, counted from the start of the
/// buffer.
///
/// This trait is sealed: it is implemented by [`ZfpBitStream`],
/// [`ZfpBitStreamRef`] and [`ZfpBitStreamRefMut`], whose methods are also
/// inherent, so it only needs importing to write code generic over streams.
pub trait ZfpBitStreamOps: BitStreamStorage {
    /// Read one full 64-bit word.
    fn read_word(&mut self) -> u64;
    /// Read `n` bits, least significant first.
    ///
    /// `n` above 64, which C does not support, reads 64 bits.
    fn read_bits(&mut self, n: u32) -> u64;
    /// Read a single bit.
    fn read_bit(&mut self) -> bool;
    /// Move the cursor to the start of the stream.
    fn rewind(&mut self);
    /// Position the stream for reading at `offset` bits from the start.
    ///
    /// An offset past the end of the buffer is kept, as in C, and reads there
    /// yield zeros.
    fn seek_read(&mut self, offset: u64);
    /// Return the current read offset in bits (`stream_rtell`).
    fn read_pos(&self) -> u64;
    /// Return the current write offset in bits (`stream_wtell`).
    fn write_pos(&self) -> u64;
    /// Skip `n` bits forward.
    fn skip(&mut self, n: u64);
    /// Discard buffered read bits up to the next word boundary; return the
    /// number of bits skipped.
    fn align(&mut self) -> u32;
    /// Size of the backing buffer in bytes (`stream_capacity`).
    fn capacity(&self) -> usize;
    /// The words before the cursor.
    fn as_words(&self) -> &[ZfpBitStreamWord];
    /// The bytes before the cursor: after [`compress`][ZfpBitStreamMutOps::compress],
    /// the whole compressed stream.
    fn as_bytes(&self) -> &[u8];
    /// The whole backing buffer, as words.
    fn backing_words(&self) -> &[ZfpBitStreamWord];
    /// The whole backing buffer, as bytes.
    fn backing_bytes(&self) -> &[u8];
    /// Whether a write has fallen past the end of the buffer since the stream
    /// was created, rewound or last positioned with `seek_write`.
    ///
    /// Such writes are dropped. [`compress`][ZfpBitStreamMutOps::compress]
    /// reports this as [`ZfpCompressionError::BufferTooSmall`][crate::ZfpCompressionError::BufferTooSmall].
    fn overflowed(&self) -> bool;
    /// Bytes up to the cursor's word, unclamped (`stream_size`).
    #[cfg(feature = "ffi")]
    #[doc(hidden)]
    fn size(&self) -> usize;

    /// Read the header sections indicated by `mask` from this bitstream.
    ///
    /// The returned header contains metadata only when `mask` includes
    /// [`ZfpHeaderMask::META`], and a compression config only when `mask`
    /// includes [`ZfpHeaderMask::MODE`]. That config never carries the
    /// encoder's rounding; see [`ZfpHeader::config`][crate::header::ZfpHeader::config].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpHeaderError`][crate::header::ZfpHeaderError] if a requested
    /// header section is invalid.
    fn read_header(
        &mut self,
        mask: ZfpHeaderMask,
    ) -> Result<crate::header::ZfpHeader, crate::header::ZfpHeaderError> {
        crate::header::read_header_bs(self, mask)
    }

    /// Decompress from this bitstream into the field.
    ///
    /// Returns the read position in bytes afterwards, which is word aligned. As
    /// in C, it exceeds the capacity if decoding skipped padding past the end
    /// of the buffer without reading it.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpDecompressionError::Truncated`][crate::types::ZfpDecompressionError::Truncated]
    /// if decoding reads a word past the end of the buffer; the field's
    /// contents are then unspecified. Returns
    /// [`ZfpDecompressionError::Field`][crate::types::ZfpDecompressionError::Field]
    /// if the field is invalid, which is reachable only from the C ABI.
    fn decompress(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
    ) -> Result<usize, crate::types::ZfpDecompressionError> {
        crate::decompress::decompress(self, field, config)
    }

    /// Decompress from this bitstream into the field using the given execution policy.
    ///
    /// Returns the read position in bytes afterwards, which is word aligned.
    /// `Rayon` decodes fixed-rate streams in independent chunks and pipelines
    /// variable-rate streams by overlapping plane reading and reconstruction.
    ///
    /// # Errors
    ///
    /// As for [`decompress`][Self::decompress].
    fn decompress_with_execution(
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

/// Write operations for writable ZFP bitstreams.
///
/// Writes past the end of the buffer are dropped and flagged; see
/// [`overflowed`][ZfpBitStreamOps::overflowed].
///
/// This trait is sealed: it is implemented by [`ZfpBitStream`] and
/// [`ZfpBitStreamRefMut`].
pub trait ZfpBitStreamMutOps: ZfpBitStreamOps + BitStreamStorageMut {
    /// Write one full 64-bit word.
    fn write_word(&mut self, word: u64);
    /// Write the low `n` bits of `value`; return `value >> n`.
    ///
    /// `n` above 64, which C does not support, writes 64 bits and returns 0.
    fn write_bits(&mut self, value: u64, n: u32) -> u64;
    /// Write a single bit.
    fn write_bit(&mut self, bit: bool);
    /// Position the stream for writing at `offset` bits from the start.
    ///
    /// An offset past the end of the buffer is kept, as in C, and writes there
    /// are dropped; see [`overflowed`][ZfpBitStreamOps::overflowed].
    fn seek_write(&mut self, offset: u64);
    /// Write `n` zero bits (`stream_pad`).
    ///
    /// Words past the end of the buffer are dropped together, so a huge `n`
    /// takes no longer than filling the buffer.
    fn pad(&mut self, n: u64);
    /// Pad with zero bits to the next word boundary, so that
    /// [`as_bytes`][ZfpBitStreamOps::as_bytes] covers everything written;
    /// return the number of bits padded.
    fn flush(&mut self) -> u32;
    /// Copy `n` bits from `src` (`stream_copy`).
    ///
    /// Once both streams are past the end of their buffers, the remaining
    /// words are skipped together, so a huge `n` takes no longer than
    /// traversing both buffers.
    fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: u64);

    /// Write the header sections indicated by `mask` into this bitstream.
    ///
    /// `metadata` is written only when `mask` includes [`ZfpHeaderMask::META`];
    /// obtain it from a field with [`ZfpField::metadata`]. Returns the number
    /// of bits written.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError::Metadata`][crate::types::ZfpCompressionError::Metadata]
    /// if `mask` includes [`ZfpHeaderMask::META`] and the field metadata cannot be
    /// encoded, or [`ZfpCompressionError::BufferTooSmall`][crate::types::ZfpCompressionError::BufferTooSmall]
    /// if the header does not fit in the stream, or
    /// [`ZfpCompressionError::Config`][crate::types::ZfpCompressionError::Config]
    /// if the mode cannot preserve the config's expert parameters. Nothing is
    /// written in any of these cases.
    fn write_header(
        &mut self,
        config: &ZfpConfig,
        metadata: &ZfpFieldMetadata,
        mask: ZfpHeaderMask,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        let mode = if mask.contains(ZfpHeaderMask::MODE) {
            config.checked_mode_bits()?
        } else {
            0
        };
        crate::header::write_header_bs(self, metadata, mask, mode)
    }

    /// Compress the field into this bitstream.
    ///
    /// Returns the size of the stream in bytes afterwards, including anything
    /// written before, such as a header. The stream is flushed, so
    /// [`as_bytes`][ZfpBitStreamOps::as_bytes] is the whole compressed stream.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError::BufferTooSmall`][crate::types::ZfpCompressionError::BufferTooSmall]
    /// if the output does not fit in the stream; size the stream with
    /// [`ZfpConfig::maximum_size`] to rule this out. Returns
    /// [`ZfpCompressionError::Field`][crate::types::ZfpCompressionError::Field] if
    /// the field is invalid, which is reachable only from the C ABI.
    fn compress(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        crate::compress::compress(self, field, config)
    }

    /// Compress the field into this bitstream using the given execution policy.
    ///
    /// Returns the size of the stream in bytes afterwards, as for
    /// [`compress`][Self::compress].
    ///
    /// # Errors
    ///
    /// As for [`compress`][Self::compress].
    fn compress_with_execution(
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
}

/// Implement [`ZfpBitStreamOps`] for each type, and make its methods inherent.
macro_rules! impl_bitstream_ops {
    ($($ty:ty),* $(,)?) => {$(
        #[inherent::inherent]
        impl ZfpBitStreamOps for $ty {
            pub fn read_word(&mut self) -> u64 {
                read_word_impl(self)
            }

            pub fn read_bits(&mut self, n: u32) -> u64 {
                read_bits_impl(self, n)
            }

            pub fn read_bit(&mut self) -> bool {
                self.get_bit() != 0
            }

            pub fn rewind(&mut self) {
                rewind_impl(self);
            }

            pub fn seek_read(&mut self, offset: u64) {
                seek_read_impl(self, offset);
            }

            pub fn read_pos(&self) -> u64 {
                read_pos_impl(self)
            }

            pub fn write_pos(&self) -> u64 {
                write_pos_impl(self)
            }

            pub fn skip(&mut self, n: u64) {
                skip_impl(self, n);
            }

            pub fn align(&mut self) -> u32 {
                align_impl(self)
            }

            pub fn capacity(&self) -> usize {
                size_of_val(BitStreamStorage::words(self))
            }

            pub fn as_words(&self) -> &[ZfpBitStreamWord] {
                committed_words(self)
            }

            pub fn as_bytes(&self) -> &[u8] {
                bytemuck::cast_slice(committed_words(self))
            }

            pub fn backing_words(&self) -> &[ZfpBitStreamWord] {
                BitStreamStorage::words(self)
            }

            pub fn backing_bytes(&self) -> &[u8] {
                backing_bytes(self)
            }

            pub fn overflowed(&self) -> bool {
                self.state().overflowed
            }

            #[cfg(feature = "ffi")]
            #[doc(hidden)]
            pub fn size(&self) -> usize {
                self.byte_len()
            }

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn read_header(
            &mut self,
            mask: ZfpHeaderMask,
        ) -> Result<crate::header::ZfpHeader, crate::header::ZfpHeaderError>;

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn decompress(
            &mut self,
            config: &ZfpConfig,
            field: &mut ZfpFieldMut,
        ) -> Result<usize, crate::types::ZfpDecompressionError>;

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn decompress_with_execution(
            &mut self,
            config: &ZfpConfig,
            field: &mut ZfpFieldMut,
            execution: crate::execution::ZfpExecution,
        ) -> Result<usize, crate::types::ZfpDecompressionError>;
        }
    )*};
}

/// Implement [`ZfpBitStreamMutOps`] for each type, and make its methods inherent.
macro_rules! impl_bitstream_mut_ops {
    ($($ty:ty),* $(,)?) => {$(
        #[inherent::inherent]
        impl ZfpBitStreamMutOps for $ty {
            pub fn write_word(&mut self, word: u64) {
                write_word_impl(self, word);
            }

            pub fn write_bits(&mut self, value: u64, n: u32) -> u64 {
                write_bits_impl(self, value, n)
            }

            pub fn write_bit(&mut self, bit: bool) {
                self.put_bit(u32::from(bit));
            }

            pub fn seek_write(&mut self, offset: u64) {
                seek_write_impl(self, offset);
            }

            pub fn pad(&mut self, n: u64) {
                pad_impl(self, n);
            }

            pub fn flush(&mut self) -> u32 {
                flush_impl(self)
            }

            pub fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: u64) {
                copy_impl(self, src, n);
            }

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn write_header(
            &mut self,
            config: &ZfpConfig,
            metadata: &ZfpFieldMetadata,
            mask: ZfpHeaderMask,
        ) -> Result<usize, crate::types::ZfpCompressionError>;

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn compress(
            &mut self,
            config: &ZfpConfig,
            field: &ZfpField,
        ) -> Result<usize, crate::types::ZfpCompressionError>;

        #[allow(clippy::missing_errors_doc, reason = "documented on the trait method")]
        pub fn compress_with_execution(
            &mut self,
            config: &ZfpConfig,
            field: &ZfpField,
            execution: crate::execution::ZfpExecution,
        ) -> Result<usize, crate::types::ZfpCompressionError>;
        }
    )*};
}

impl_bitstream_ops!(ZfpBitStream, ZfpBitStreamRef<'_>, ZfpBitStreamRefMut<'_>);
impl_bitstream_mut_ops!(ZfpBitStream, ZfpBitStreamRefMut<'_>);
