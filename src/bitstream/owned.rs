use crate::bitstream::core::{
    BitStreamState, BitStreamStorage, BitStreamStorageMut, bytes_to_words,
};
use crate::config::STREAM_WORD_BYTES;
use crate::types::ZfpBitStreamWord;

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

    /// Flush and consume the stream, returning the underlying byte buffer.
    #[must_use]
    pub fn into_vec(mut self) -> Vec<u8> {
        self.flush();
        self.as_bytes().to_vec()
    }
}
