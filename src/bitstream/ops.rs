use crate::bitstream::core::{
    BitStreamStorage, WSIZE, align_impl, as_committed_bytes, backing_bytes, flush_impl, pad_impl,
    read_bit_impl, read_bits_impl, read_pos_impl, read_word_impl, rewind_impl, seek_read_impl,
    seek_write_impl, skip_impl, write_bit_impl, write_bits_impl, write_pos_impl, write_word_impl,
};
use crate::bitstream::{ZfpBitStream, ZfpBitStreamRef, ZfpBitStreamRefMut};
use crate::config::{STREAM_WORD_BYTES, ZfpConfig};
use crate::field::{ZfpField, ZfpFieldMetadata, ZfpFieldMut};
use crate::types::{ZfpBitStreamWord, ZfpHeaderMask};

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
    /// Whether a write has fallen past the end of the buffer since the stream
    /// was created, rewound or last positioned with `seek_write`.
    ///
    /// Such writes are dropped. [`compress`][ZfpBitStreamMutOps::compress]
    /// reports this as [`ZfpCompressionError::BufferTooSmall`][crate::ZfpCompressionError::BufferTooSmall].
    fn overflowed(&self) -> bool;
    /// The backing buffer's start pointer.
    #[cfg(feature = "ffi")]
    fn data_ptr(&self) -> *mut std::os::raw::c_void;

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
    /// # Errors
    ///
    /// Returns [`ZfpDecompressionError`][crate::types::ZfpDecompressionError] if the target
    /// field type or dimensions are unsupported for the selected configuration.
    fn decompress(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
    ) -> Result<usize, crate::types::ZfpDecompressionError> {
        crate::decompress::decompress(self, field, config)
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
    /// if the header does not fit in the stream. Nothing is written in either case.
    fn write_header(
        &mut self,
        config: &ZfpConfig,
        metadata: &ZfpFieldMetadata,
        mask: ZfpHeaderMask,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        let mode = config.mode_bits();
        crate::header::write_header_bs(self, metadata, mask, mode)
    }

    /// Compress the field into this bitstream using the stream's parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError`][crate::types::ZfpCompressionError] if the field
    /// is invalid, or [`BufferTooSmall`][crate::types::ZfpCompressionError::BufferTooSmall]
    /// if the output does not fit in the stream. Size the stream with
    /// [`ZfpConfig::maximum_size`] to rule the latter out.
    fn compress(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
    ) -> Result<usize, crate::types::ZfpCompressionError> {
        crate::compress::compress(self, field, config)
    }

    /// Compress the field into this bitstream using the given execution policy.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError`][crate::types::ZfpCompressionError] if the field
    /// is invalid, or [`BufferTooSmall`][crate::types::ZfpCompressionError::BufferTooSmall]
    /// if the output does not fit in the stream. Size the stream with
    /// [`ZfpConfig::maximum_size`] to rule the latter out.
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

            pub fn read_bit(&mut self) -> u32 {
                read_bit_impl(self)
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

            pub fn words(&self) -> &[ZfpBitStreamWord] {
                <Self as BitStreamStorage>::words(self)
            }

            pub fn skip(&mut self, n: usize) {
                skip_impl(self, n);
            }

            pub fn align(&mut self) -> u32 {
                align_impl(self)
            }

            #[allow(
                clippy::cast_possible_truncation,
                reason = "bitstream capacities are addressable as usize on supported targets"
            )]
            pub fn bits_written(&self) -> usize {
                self.write_pos() as usize
            }

            pub fn word_pos(&self) -> usize {
                self.state().word_pos
            }

            pub fn capacity(&self) -> usize {
                self.words().len() * STREAM_WORD_BYTES
            }

            pub fn size(&self) -> usize {
                self.state().word_pos * STREAM_WORD_BYTES
            }

            pub fn as_bytes(&self) -> &[u8] {
                as_committed_bytes(self)
            }

            pub fn backing_bytes(&self) -> &[u8] {
                backing_bytes(self)
            }

            pub fn overflowed(&self) -> bool {
                self.state().overflowed
            }

            #[cfg(feature = "ffi")]
            pub fn data_ptr(&self) -> *mut std::os::raw::c_void {
                self.words()
                    .as_ptr()
                    .cast_mut()
                    .cast::<std::os::raw::c_void>()
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
            pub fn write_word(&mut self, word: u64) -> u64 {
                write_word_impl(self, word)
            }

            pub fn write_bits(&mut self, value: u64, n: u32) -> u64 {
                write_bits_impl(self, value, n)
            }

            pub fn write_bit(&mut self, bit: u32) -> u32 {
                write_bit_impl(self, bit)
            }

            pub fn seek_write(&mut self, offset: u64) {
                seek_write_impl(self, offset);
            }

            pub fn pad(&mut self, n: usize) {
                pad_impl(self, n);
            }

            pub fn flush(&mut self) -> usize {
                flush_impl(self)
            }

            #[allow(clippy::cast_possible_truncation)]
            pub fn copy_from(&mut self, src: &mut dyn ZfpBitStreamOps, n: usize) {
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
