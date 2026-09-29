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

    fn split(&mut self) -> (&[ZfpBitStreamWord], &mut BitStreamState) {
        (self.words, &mut self.state)
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
    #[must_use]
    pub fn from_words(words: &'a mut [ZfpBitStreamWord]) -> Self {
        Self {
            words,
            state: BitStreamState::new(),
        }
    }

    /// Borrow an aligned byte buffer as a mutable bitstream.
    ///
    /// Returns `None` if `buf` is not aligned for `ZfpBitStreamWord`.
    /// Trailing partial words are ignored, matching C `stream_open`.
    #[must_use]
    pub fn from_bytes(buf: &'a mut [u8]) -> Option<Self> {
        cast_bytes_to_words_mut(buf).map(Self::from_words)
    }
}

impl BitStreamStorage for ZfpBitStreamRefMut<'_> {
    fn words(&self) -> &[ZfpBitStreamWord] {
        self.words
    }

    fn split(&mut self) -> (&[ZfpBitStreamWord], &mut BitStreamState) {
        (&*self.words, &mut self.state)
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

    fn split_mut(&mut self) -> (&mut [ZfpBitStreamWord], &mut BitStreamState) {
        (&mut *self.words, &mut self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::{ZfpBitStreamRef, ZfpBitStreamRefMut};
    use crate::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpHeaderMask};

    fn ramp() -> Vec<f64> {
        (0..64).map(f64::from).collect()
    }

    #[test]
    fn borrowed_stream_reads_header_and_decompresses() {
        let data = ramp();
        let field = ZfpField::new(&data, [4usize, 4, 4]).unwrap();
        let config = ZfpConfig::reversible();
        let mut bs = ZfpBitStream::new(4096).unwrap();
        bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)
            .expect("write header");
        bs.compress(&config, &field).expect("compress");
        let words = bs.into_words();

        let mut bs = ZfpBitStreamRef::from_words(&words);
        let header = bs.read_header(ZfpHeaderMask::FULL).expect("read header");
        assert_eq!(header.config, Some(config));
        let mut out = vec![0f64; 64];
        let mut field = ZfpFieldMut::new(&mut out, [4usize, 4, 4]).unwrap();
        bs.decompress(&config, &mut field).expect("decompress");
        assert_eq!(out, data);
    }

    #[test]
    fn borrowed_mut_stream_round_trips_with_header() {
        let data = ramp();
        let field = ZfpField::new(&data, [4usize, 4, 4]).unwrap();
        let config = ZfpConfig::reversible();
        let mut words = vec![0u64; 512];

        let mut bs = ZfpBitStreamRefMut::from_words(&mut words);
        bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)
            .expect("write header");
        bs.compress(&config, &field).expect("compress");
        bs.flush();
        bs.rewind();
        let header = bs.read_header(ZfpHeaderMask::FULL).expect("read header");
        assert_eq!(header.config, Some(config));
        let mut out = vec![0f64; 64];
        let mut field = ZfpFieldMut::new(&mut out, [4usize, 4, 4]).unwrap();
        bs.decompress(&config, &mut field).expect("decompress");
        assert_eq!(out, data);
    }
}
