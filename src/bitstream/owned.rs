use crate::bitstream::core::{
    BitStreamState, BitStreamStorage, BitStreamStorageMut, bytes_to_words, vec_with_capacity,
    zeroed_words,
};
use crate::config::STREAM_WORD_BYTES;
use crate::types::{ZfpAllocError, ZfpBitStreamWord};

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

    fn split(&mut self) -> (&[ZfpBitStreamWord], &mut BitStreamState) {
        (&self.words, &mut self.state)
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

    fn split_mut(&mut self) -> (&mut [ZfpBitStreamWord], &mut BitStreamState) {
        (&mut self.words, &mut self.state)
    }
}

impl std::fmt::Debug for ZfpBitStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZfpBitStream")
            .field("capacity_bytes", &self.capacity())
            .field("write_pos", &self.write_pos())
            .finish()
    }
}

impl ZfpBitStream {
    /// Create a new, empty `ZfpBitStream` with at least `capacity` bytes of storage.
    ///
    /// The stream never grows: writes past the end are dropped and flagged,
    /// see [`overflowed`][Self::overflowed].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpAllocError`] if the storage cannot be allocated.
    pub fn new(capacity: usize) -> Result<Self, ZfpAllocError> {
        Ok(Self::from_words(zeroed_words(
            capacity.div_ceil(STREAM_WORD_BYTES),
        )?))
    }

    /// Wrap an existing word buffer (takes ownership).
    #[must_use]
    pub fn from_words(words: Vec<ZfpBitStreamWord>) -> Self {
        Self {
            words,
            state: BitStreamState::new(),
        }
    }

    /// Copy a byte buffer into a new stream.
    ///
    /// A trailing partial word is zero-padded.
    ///
    /// # Errors
    ///
    /// Returns [`ZfpAllocError`] if the copy cannot be allocated.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, ZfpAllocError> {
        Ok(Self::from_words(bytes_to_words(buf)?))
    }

    /// Flush and consume the stream, returning the words written, as
    /// [`as_words`][Self::as_words].
    #[must_use]
    pub fn into_words(mut self) -> Vec<ZfpBitStreamWord> {
        self.flush();
        let len = self.as_words().len();
        let mut words = std::mem::take(&mut self.words);
        words.truncate(len);
        words
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

    /// Read the word at a given word index without moving the cursor.
    #[cfg(test)]
    #[must_use]
    pub fn word_at(&self, index: usize) -> u64 {
        self.words[index]
    }

    /// Flush and consume the stream, returning a copy of the bytes written, as
    /// [`as_bytes`][Self::as_bytes].
    ///
    /// # Errors
    ///
    /// Returns [`ZfpAllocError`] if the copy cannot be allocated.
    pub fn into_bytes(mut self) -> Result<Vec<u8>, ZfpAllocError> {
        self.flush();
        let bytes = self.as_bytes();
        let mut out = vec_with_capacity(bytes.len())?;
        out.extend_from_slice(bytes);
        Ok(out)
    }
}
