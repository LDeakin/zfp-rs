use crate::config::STREAM_WORD_BYTES;
use crate::types::ZfpBitStreamWord;

pub(crate) const WSIZE: u32 = 64;

#[derive(Clone, Copy)]
pub(crate) struct BitStreamState {
    pub(crate) word_pos: usize,
    pub(crate) buffer: u64,
    pub(crate) bits: u32,
}

impl BitStreamState {
    pub(super) const fn new() -> Self {
        Self {
            word_pos: 0,
            buffer: 0,
            bits: 0,
        }
    }
}

pub(crate) trait BitStreamStorage {
    fn words(&self) -> &[ZfpBitStreamWord];
    fn state(&self) -> &BitStreamState;
    fn state_mut(&mut self) -> &mut BitStreamState;
}

pub(crate) trait BitStreamStorageMut: BitStreamStorage {
    fn words_mut(&mut self) -> &mut [ZfpBitStreamWord];
}

pub(super) fn bytes_to_words(buf: &[u8]) -> Vec<ZfpBitStreamWord> {
    let chunks = buf.as_chunks::<{ size_of::<ZfpBitStreamWord>() }>().0;
    let mut out: Vec<ZfpBitStreamWord> = Vec::with_capacity(chunks.len());
    match bytemuck::try_cast_slice::<u8, ZfpBitStreamWord>(&buf[..chunks.len() * STREAM_WORD_BYTES])
    {
        Ok(words) => out.extend_from_slice(words),
        Err(_) => out.extend(chunks.iter().map(|c| ZfpBitStreamWord::from_ne_bytes(*c))),
    }
    out
}

pub(super) fn cast_bytes_to_words(buf: &[u8]) -> Option<&[ZfpBitStreamWord]> {
    let nbytes = (buf.len() / STREAM_WORD_BYTES) * STREAM_WORD_BYTES;
    bytemuck::try_cast_slice::<u8, ZfpBitStreamWord>(&buf[..nbytes]).ok()
}

pub(super) fn cast_bytes_to_words_mut(buf: &mut [u8]) -> Option<&mut [ZfpBitStreamWord]> {
    let nbytes = (buf.len() / STREAM_WORD_BYTES) * STREAM_WORD_BYTES;
    bytemuck::try_cast_slice_mut::<u8, ZfpBitStreamWord>(&mut buf[..nbytes]).ok()
}

fn read_word_raw<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u64 {
    let pos = stream.state().word_pos;
    // Zero past the end rather than panicking; C reads out of bounds here.
    let w = stream.words().get(pos).copied().unwrap_or(0);
    stream.state_mut().word_pos += 1;
    w
}

fn write_word_raw<S: BitStreamStorageMut + ?Sized>(stream: &mut S, value: u64) {
    let pos = stream.state().word_pos;
    stream.words_mut()[pos] = value;
    stream.state_mut().word_pos += 1;
}

pub(super) fn as_committed_bytes<S: BitStreamStorage + ?Sized>(stream: &S) -> &[u8] {
    // Clamped: a seek past the end is permitted, so `word_pos` may exceed the buffer.
    let words = stream.words();
    let end = stream.state().word_pos.min(words.len());
    bytemuck::cast_slice(&words[..end])
}

pub(super) fn backing_bytes<S: BitStreamStorage + ?Sized>(stream: &S) -> &[u8] {
    bytemuck::cast_slice(stream.words())
}

pub(super) fn read_word_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u64 {
    read_word_raw(stream)
}

pub(super) fn read_bits_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, n: u32) -> u64 {
    let mut value = stream.state().buffer;
    if stream.state().bits < n {
        loop {
            let word = read_word_raw(stream);
            {
                let state = stream.state_mut();
                state.buffer = word;
                value += state.buffer << state.bits;
                state.bits += WSIZE;
                if state.bits >= n {
                    break;
                }
            }
        }
        let state = stream.state_mut();
        state.bits -= n;
        if state.bits == 0 {
            state.buffer = 0;
        } else {
            state.buffer >>= WSIZE - state.bits;
            if n < 64 {
                value &= (2u64 << (n - 1)) - 1;
            }
        }
    } else {
        let state = stream.state_mut();
        state.bits -= n;
        state.buffer >>= n;
        if n < 64 {
            value &= (1u64 << n) - 1;
        }
    }
    value
}

pub(super) fn read_bit_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u32 {
    if stream.state().bits == 0 {
        let word = read_word_raw(stream);
        let state = stream.state_mut();
        state.buffer = word;
        state.bits = WSIZE;
    }
    let state = stream.state_mut();
    state.bits -= 1;
    let bit = (state.buffer & 1) as u32;
    state.buffer >>= 1;
    bit
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn seek_read_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, offset: u64) {
    let n = (offset % u64::from(WSIZE)) as u32;
    // Clamped to the buffer; C's `stream_rseek` stores the offset unchecked.
    let limit = stream.words().len();
    stream.state_mut().word_pos = ((offset / u64::from(WSIZE)) as usize).min(limit);
    if n != 0 {
        let word = read_word_raw(stream);
        let state = stream.state_mut();
        state.buffer = word >> n;
        state.bits = WSIZE - n;
    } else {
        let state = stream.state_mut();
        state.buffer = 0;
        state.bits = 0;
    }
}

pub(super) fn write_word_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S, word: u64) -> u64 {
    let pos = stream.state().word_pos;
    let prev = stream.words()[pos];
    write_word_raw(stream, word);
    prev
}

pub(super) fn write_bits_impl<S: BitStreamStorageMut + ?Sized>(
    stream: &mut S,
    value: u64,
    n: u32,
) -> u64 {
    {
        let state = stream.state_mut();
        state.buffer = state.buffer.wrapping_add(value << state.bits);
        state.bits += n;
    }
    if stream.state().bits >= WSIZE {
        let val = value >> 1;
        let remaining = n - 1;
        loop {
            let bits_after = stream.state().bits - WSIZE;
            let buffer = stream.state().buffer;
            stream.state_mut().bits = bits_after;
            write_word_raw(stream, buffer);
            stream.state_mut().buffer = val >> (remaining - bits_after);
            if stream.state().bits < WSIZE {
                break;
            }
        }
    }
    if stream.state().bits < 64 {
        let bits = stream.state().bits;
        stream.state_mut().buffer &= (1u64 << bits) - 1;
    }
    if n < 64 { value >> n } else { 0 }
}

pub(super) fn write_bit_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S, bit: u32) -> u32 {
    {
        let state = stream.state_mut();
        state.buffer += u64::from(bit) << state.bits;
        state.bits += 1;
    }
    if stream.state().bits == WSIZE {
        let buffer = stream.state().buffer;
        write_word_raw(stream, buffer);
        let state = stream.state_mut();
        state.buffer = 0;
        state.bits = 0;
    }
    bit
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn seek_write_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, offset: u64) {
    let n = (offset % u64::from(WSIZE)) as u32;
    // Clamped, as in `seek_read_impl`.
    let limit = stream.words().len();
    stream.state_mut().word_pos = ((offset / u64::from(WSIZE)) as usize).min(limit);
    if n != 0 {
        let pos = stream.state().word_pos;
        let Some(&buf) = stream.words().get(pos) else {
            let state = stream.state_mut();
            state.buffer = 0;
            state.bits = n;
            return;
        };
        let state = stream.state_mut();
        state.buffer = buf & ((1u64 << n) - 1);
        state.bits = n;
    } else {
        let state = stream.state_mut();
        state.buffer = 0;
        state.bits = 0;
    }
}

pub(super) fn rewind_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) {
    *stream.state_mut() = BitStreamState::new();
}

pub(super) fn write_pos_impl<S: BitStreamStorage + ?Sized>(stream: &S) -> u64 {
    // Wrapping, as in `read_pos_impl`: a wrapped read position can be seeked to.
    (stream.state().word_pos as u64)
        .wrapping_mul(u64::from(WSIZE))
        .wrapping_add(u64::from(stream.state().bits))
}

pub(super) fn read_pos_impl<S: BitStreamStorage + ?Sized>(stream: &S) -> u64 {
    // `bits` is shared with the write path, so `word_pos == 0` with `bits > 0`
    // is reachable (e.g. after one `write_bits`). C's `stream_rtell` wraps too.
    (stream.state().word_pos as u64)
        .wrapping_mul(u64::from(WSIZE))
        .wrapping_sub(u64::from(stream.state().bits))
}

pub(super) fn skip_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, n: usize) {
    // Wrapping, as in `read_pos_impl`.
    let pos = read_pos_impl(stream).wrapping_add(n as u64);
    seek_read_impl(stream, pos);
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn pad_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S, n: usize) {
    let mut bits = u64::from(stream.state().bits) + n as u64;
    while bits >= u64::from(WSIZE) {
        let buffer = stream.state().buffer;
        write_word_raw(stream, buffer);
        stream.state_mut().buffer = 0;
        bits -= u64::from(WSIZE);
    }
    stream.state_mut().bits = bits as u32;
}

pub(super) fn align_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u32 {
    let bits = stream.state().bits;
    if bits != 0 {
        skip_impl(stream, bits as usize);
    }
    bits
}

pub(super) fn flush_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S) -> usize {
    let pad = (WSIZE - stream.state().bits) % WSIZE;
    if pad != 0 {
        pad_impl(stream, pad as usize);
    }
    pad as usize
}
