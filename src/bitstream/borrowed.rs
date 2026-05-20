use crate::bitstream::core::{
    BitStreamState, BitStreamStorage, BitStreamStorageMut, cast_bytes_to_words,
    cast_bytes_to_words_mut,
};
use crate::types::ZfpBitStreamWord;

/// A read-only borrowed ZFP bitstream.
pub struct ZfpBitStreamRef<'a> {
    words: &'a [ZfpBitStreamWord],
    state: BitStreamState,
}

impl std::fmt::Debug for ZfpBitStreamRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::bitstream::ZfpBitStreamOps;
        f.debug_struct("ZfpBitStreamRef")
            .field("capacity_bytes", &self.capacity())
            .finish()
    }
}

impl<'a> ZfpBitStreamRef<'a> {
    /// Borrow an existing word buffer as a read-only bitstream.
    #[must_use]
    pub fn from_words(words: &'a [ZfpBitStreamWord]) -> Self {
        Self {
            words,
            state: BitStreamState::new(),
        }
    }

    /// Borrow an aligned byte buffer as a read-only bitstream.
    ///
    /// Returns `None` if `buf` is not aligned for `ZfpBitStreamWord`.
    /// Trailing partial words are ignored, matching C `stream_open`.
    #[must_use]
    pub fn from_bytes(buf: &'a [u8]) -> Option<Self> {
        cast_bytes_to_words(buf).map(Self::from_words)
    }
}

impl BitStreamStorage for ZfpBitStreamRef<'_> {
    fn words(&self) -> &[ZfpBitStreamWord] {
        self.words
    }

    fn state(&self) -> &BitStreamState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BitStreamState {
        &mut self.state
    }
}

impl std::fmt::Debug for ZfpBitStreamRefMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::bitstream::ZfpBitStreamOps;
        f.debug_struct("ZfpBitStreamRefMut")
            .field("capacity_bytes", &self.capacity())
            .finish()
    }
}

/// A mutable borrowed ZFP bitstream.
pub struct ZfpBitStreamRefMut<'a> {
    words: &'a mut [ZfpBitStreamWord],
    state: BitStreamState,
}

impl<'a> ZfpBitStreamRefMut<'a> {
    /// Borrow an existing word buffer as a mutable bitstream.
    pub fn from_words_mut(words: &'a mut [ZfpBitStreamWord]) -> Self {
        Self {
            words,
            state: BitStreamState::new(),
        }
    }

    /// Borrow an aligned byte buffer as a mutable bitstream.
    ///
    /// Returns `None` if `buf` is not aligned for `ZfpBitStreamWord`.
    /// Trailing partial words are ignored, matching C `stream_open`.
    pub fn from_bytes_mut(buf: &'a mut [u8]) -> Option<Self> {
        cast_bytes_to_words_mut(buf).map(Self::from_words_mut)
    }

    /// Compress the field into this bitstream using the given execution policy.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpCompressionError`][crate::types::ZfpCompressionError] if the field
    /// type or dimensions are unsupported for the selected configuration.
    ///
    pub fn compress_with_execution(
        &mut self,
        config: &crate::config::ZfpConfig,
        field: &crate::field::ZfpField,
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
        config: &crate::config::ZfpConfig,
        field: &mut crate::field::ZfpFieldMut,
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

impl BitStreamStorage for ZfpBitStreamRefMut<'_> {
    fn words(&self) -> &[ZfpBitStreamWord] {
        self.words
    }

    fn state(&self) -> &BitStreamState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BitStreamState {
        &mut self.state
    }
}

impl BitStreamStorageMut for ZfpBitStreamRefMut<'_> {
    fn words_mut(&mut self) -> &mut [ZfpBitStreamWord] {
        self.words
    }
}
