use crate::config::STREAM_WORD_BYTES;
use crate::types::{ZfpAllocError, ZfpBitStreamWord};

pub(crate) const WSIZE: u32 = 64;

#[derive(Clone, Copy)]
pub struct BitStreamState {
    pub(crate) word_pos: usize,
    pub(crate) buffer: u64,
    pub(crate) bits: u32,
    /// Set when a write fell past the end of the buffer and was dropped.
    pub(crate) overflowed: bool,
}

impl BitStreamState {
    pub(super) const fn new() -> Self {
        Self {
            word_pos: 0,
            buffer: 0,
            bits: 0,
            overflowed: false,
        }
    }
}

/// Buffer and cursor access behind every stream type.
///
/// Public but unnameable outside the crate, this seals
/// [`ZfpBitStreamOps`][super::ZfpBitStreamOps] and carries the C-style bit
/// I/O the codec is written against.
pub trait BitStreamStorage {
    fn words(&self) -> &[ZfpBitStreamWord];
    fn state(&self) -> &BitStreamState;
    fn state_mut(&mut self) -> &mut BitStreamState;
    /// The buffer and the cursor at once, for [`BitReader`].
    fn split(&mut self) -> (&[ZfpBitStreamWord], &mut BitStreamState);

    /// Read a single bit as 0 or 1 (C `stream_read_bit`).
    #[inline]
    fn get_bit(&mut self) -> u32 {
        read_bit_impl(self)
    }

    /// Bytes up to the cursor's word, unclamped (C `stream_size`).
    ///
    /// Saturates for a cursor seeked so far past the end that the count
    /// overflows `usize`, which only a 32-bit target can reach.
    #[inline]
    fn byte_len(&self) -> usize {
        self.state().word_pos.saturating_mul(STREAM_WORD_BYTES)
    }
}

/// Mutable counterpart of [`BitStreamStorage`], sealing
/// [`ZfpBitStreamMutOps`][super::ZfpBitStreamMutOps].
pub trait BitStreamStorageMut: BitStreamStorage {
    fn words_mut(&mut self) -> &mut [ZfpBitStreamWord];
    /// The buffer and the cursor at once, for [`BitWriter`].
    fn split_mut(&mut self) -> (&mut [ZfpBitStreamWord], &mut BitStreamState);

    /// Write the low bit of `bit` and return it (C `stream_write_bit`).
    #[inline]
    fn put_bit(&mut self, bit: u32) -> u32 {
        write_bit_impl(self, bit)
    }
}

/// `n` zeroed words, or the error if they cannot be allocated.
///
/// Zeroed by the allocator, as `vec![0; n]` is, so a large stream costs no
/// more than the pages it touches.
pub(super) fn zeroed_words(n: usize) -> Result<Vec<ZfpBitStreamWord>, ZfpAllocError> {
    let error = ZfpAllocError {
        bytes: n.saturating_mul(STREAM_WORD_BYTES),
    };
    if n == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<ZfpBitStreamWord>(n).map_err(|_| error)?;
    // SAFETY: `layout` has a nonzero size, as `n > 0`.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(error);
    }
    #[expect(clippy::cast_ptr_alignment, reason = "`layout` is aligned for words")]
    let words = std::ptr::slice_from_raw_parts_mut(ptr.cast::<ZfpBitStreamWord>(), n);
    // SAFETY: `words` was allocated by the global allocator with the layout of
    // `[ZfpBitStreamWord; n]`, and is initialised, as zero is a valid word.
    Ok(unsafe { Box::from_raw(words) }.into_vec())
}

/// An empty vector with room for `n` elements, or the error if it cannot be
/// allocated.
pub(crate) fn vec_with_capacity<T>(n: usize) -> Result<Vec<T>, ZfpAllocError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| ZfpAllocError {
        bytes: n.saturating_mul(size_of::<T>()),
    })?;
    Ok(v)
}

pub(super) fn bytes_to_words(buf: &[u8]) -> Result<Vec<ZfpBitStreamWord>, ZfpAllocError> {
    let (chunks, tail) = buf.as_chunks::<{ size_of::<ZfpBitStreamWord>() }>();
    let mut out: Vec<ZfpBitStreamWord> = vec_with_capacity(buf.len().div_ceil(STREAM_WORD_BYTES))?;
    match bytemuck::try_cast_slice::<u8, ZfpBitStreamWord>(&buf[..chunks.len() * STREAM_WORD_BYTES])
    {
        Ok(words) => out.extend_from_slice(words),
        Err(_) => out.extend(chunks.iter().map(|c| ZfpBitStreamWord::from_ne_bytes(*c))),
    }
    if !tail.is_empty() {
        let mut last = [0u8; STREAM_WORD_BYTES];
        last[..tail.len()].copy_from_slice(tail);
        out.push(ZfpBitStreamWord::from_ne_bytes(last));
    }
    Ok(out)
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
    stream.state_mut().word_pos = pos.saturating_add(1);
    w
}

fn write_word_raw<S: BitStreamStorageMut + ?Sized>(stream: &mut S, value: u64) {
    let pos = stream.state().word_pos;
    // Past the end, drop the word and flag the overflow rather than panicking;
    // C writes out of bounds here. The cursor still advances, so the position
    // reports how much space the stream needed.
    if let Some(word) = stream.words_mut().get_mut(pos) {
        *word = value;
    } else {
        stream.state_mut().overflowed = true;
    }
    stream.state_mut().word_pos = pos.saturating_add(1);
}

pub(super) fn committed_words<S: BitStreamStorage + ?Sized>(stream: &S) -> &[ZfpBitStreamWord] {
    // Clamped: reads and dropped writes may carry `word_pos` past the buffer.
    let words = stream.words();
    let end = stream.state().word_pos.min(words.len());
    &words[..end]
}

pub(super) fn backing_bytes<S: BitStreamStorage + ?Sized>(stream: &S) -> &[u8] {
    bytemuck::cast_slice(stream.words())
}

pub(super) fn read_word_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u64 {
    read_word_raw(stream)
}

/// Read `n` bits, least significant first; `n` above 64 is read as 64, the
/// most C's `stream_read_bits` supports.
pub(super) fn read_bits_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, n: u32) -> u64 {
    let n = n.min(WSIZE);
    let mut value = stream.state().buffer;
    if stream.state().bits < n {
        loop {
            let word = read_word_raw(stream);
            {
                let state = stream.state_mut();
                state.buffer = word;
                // Wrapping, as C's unsigned addition: `put_bit` can leave bits
                // above the buffered count, which this would carry into.
                value = value.wrapping_add(state.buffer << state.bits);
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

/// The index of the word holding bit `offset`, which may be past the end of
/// the buffer.
///
/// Saturates where `usize` is narrower than the offset. No allocation exceeds
/// `isize::MAX` bytes, so that is past the end of any buffer, with room for the
/// cursor to advance without overflowing.
fn word_index(offset: u64) -> usize {
    usize::try_from(offset / u64::from(WSIZE)).unwrap_or(usize::MAX / 2)
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn seek_read_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, offset: u64) {
    let n = (offset % u64::from(WSIZE)) as u32;
    // Kept past the end of the buffer, as C's `stream_rseek` does, so the
    // cursor never moves backwards; reads there yield zeros.
    stream.state_mut().word_pos = word_index(offset);
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

pub(super) fn write_word_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S, word: u64) {
    write_word_raw(stream, word);
}

/// Write the low `n` bits of `value` and return `value >> n`; `n` above 64 is
/// written as 64, the most C's `stream_write_bits` supports.
pub(super) fn write_bits_impl<S: BitStreamStorageMut + ?Sized>(
    stream: &mut S,
    value: u64,
    n: u32,
) -> u64 {
    let n = n.min(WSIZE);
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
        // Wrapping, as C's unsigned addition: a `bit` above 1 is added whole.
        state.buffer = state.buffer.wrapping_add(u64::from(bit) << state.bits);
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
    // Kept past the end, as in `seek_read_impl`; writes there are dropped.
    stream.state_mut().word_pos = word_index(offset);
    stream.state_mut().overflowed = false;
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

pub(super) fn skip_impl<S: BitStreamStorage + ?Sized>(stream: &mut S, n: u64) {
    // Wrapping, as in `read_pos_impl`.
    let pos = read_pos_impl(stream).wrapping_add(n);
    seek_read_impl(stream, pos);
}

/// Write `n` zero bits (C `stream_pad`).
///
/// Once the cursor is past the end of the buffer, every word left would be
/// dropped, so they are skipped at once, leaving the stream as dropping them
/// one at a time would. That bounds the work by the buffer's length.
///
/// Out of line: fixed-rate blocks are padded, and inlining this into the
/// block encoders changed what else LLVM inlined there, costing reversible
/// compression 5%.
#[inline(never)]
#[allow(clippy::cast_possible_truncation)]
pub(super) fn pad_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S, n: u64) {
    let mut bits = u64::from(stream.state().bits).saturating_add(n);
    while bits >= u64::from(WSIZE) {
        if stream.state().word_pos >= stream.words().len() {
            let words = usize::try_from(bits / u64::from(WSIZE)).unwrap_or(usize::MAX);
            let state = stream.state_mut();
            state.word_pos = state.word_pos.saturating_add(words);
            state.buffer = 0;
            state.overflowed = true;
            bits %= u64::from(WSIZE);
            break;
        }
        let buffer = stream.state().buffer;
        write_word_raw(stream, buffer);
        stream.state_mut().buffer = 0;
        bits -= u64::from(WSIZE);
    }
    stream.state_mut().bits = bits as u32;
}

/// Copy `n` bits from `src` to `dst` (C `stream_copy`).
///
/// Once both cursors are past the end of their buffers, reads yield zeros and
/// writes are dropped, so the remaining whole words are skipped at once. That
/// leaves both streams exactly as copying them one at a time would, and bounds
/// the work by the buffer lengths whatever `n` is.
pub(super) fn copy_impl<D, S>(dst: &mut D, src: &mut S, n: u64)
where
    D: BitStreamStorageMut + ?Sized,
    S: BitStreamStorage + ?Sized,
{
    let mut remaining = n;
    while remaining > u64::from(WSIZE) {
        let src_done = src.state().word_pos >= src.words().len() && src.state().buffer == 0;
        let dst_done = dst.state().word_pos >= dst.words().len();
        if src_done && dst_done {
            // The words the loop below would copy, leaving the tail to it.
            let skipped = (remaining - 1) / u64::from(WSIZE);
            let words = usize::try_from(skipped).unwrap_or(usize::MAX);
            let src_state = src.state_mut();
            src_state.word_pos = src_state.word_pos.saturating_add(words);
            let dst_state = dst.state_mut();
            dst_state.word_pos = dst_state.word_pos.saturating_add(words);
            dst_state.buffer = 0;
            dst_state.overflowed = true;
            remaining -= skipped * u64::from(WSIZE);
            break;
        }
        let w = read_bits_impl(src, WSIZE);
        write_bits_impl(dst, w, WSIZE);
        remaining -= u64::from(WSIZE);
    }
    if remaining > 0 {
        #[allow(clippy::cast_possible_truncation, reason = "remaining <= 64")]
        let remaining = remaining as u32;
        let w = read_bits_impl(src, remaining);
        write_bits_impl(dst, w, remaining);
    }
}

pub(super) fn align_impl<S: BitStreamStorage + ?Sized>(stream: &mut S) -> u32 {
    let bits = stream.state().bits;
    if bits != 0 {
        skip_impl(stream, u64::from(bits));
    }
    bits
}

pub(super) fn flush_impl<S: BitStreamStorageMut + ?Sized>(stream: &mut S) -> u32 {
    let pad = (WSIZE - stream.state().bits) % WSIZE;
    if pad != 0 {
        pad_impl(stream, u64::from(pad));
    }
    pad
}

/// A stream's read cursor, copied into locals for the length of a block.
///
/// C decodes a block through a copy of the `bitstream` for the same reason: a
/// cursor behind a pointer cannot stay in registers. Reads have exactly the
/// effect `read_bits` would, and the cursor is written back on drop.
pub(crate) struct BitReader<'a> {
    words: &'a [ZfpBitStreamWord],
    state: &'a mut BitStreamState,
    word_pos: usize,
    buffer: u64,
    bits: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub(crate) fn new<S: BitStreamStorage + ?Sized>(stream: &'a mut S) -> Self {
        let (words, state) = stream.split();
        Self {
            words,
            word_pos: state.word_pos,
            buffer: state.buffer,
            bits: state.bits,
            state,
        }
    }

    /// The word after the buffered bits; zero past the end, as in `read_word_raw`.
    #[inline]
    fn next_word(&self) -> u64 {
        self.words.get(self.word_pos).copied().unwrap_or(0)
    }

    /// The next 64 bits, without consuming them.
    ///
    /// Relies on the cursor invariant `buffer < 2^bits`, with `bits < 64`.
    #[inline]
    pub(crate) fn peek(&self) -> u64 {
        self.buffer | (self.next_word() << self.bits)
    }

    /// Consume `n <= 64` bits, leaving the cursor as `read_bits(n)` would.
    #[inline]
    pub(crate) fn consume(&mut self, n: u32) {
        debug_assert!(n <= WSIZE);
        if n <= self.bits {
            self.buffer >>= n;
            self.bits -= n;
        } else {
            let word = self.next_word();
            self.word_pos = self.word_pos.saturating_add(1);
            self.bits = self.bits + WSIZE - n;
            // `word >> (WSIZE - bits)`, and zero when `bits == 0`.
            self.buffer = (word >> 1) >> (WSIZE - 1 - self.bits);
        }
    }

    /// Read `n <= 64` bits, least significant first.
    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> u64 {
        let value = self.peek();
        self.consume(n);
        if n < WSIZE {
            value & ((1u64 << n) - 1)
        } else {
            value
        }
    }
}

impl Drop for BitReader<'_> {
    #[inline]
    fn drop(&mut self) {
        self.state.word_pos = self.word_pos;
        self.state.buffer = self.buffer;
        self.state.bits = self.bits;
    }
}

/// A stream's write cursor, copied into locals for the length of a block.
///
/// The write counterpart of [`BitReader`]: writes have exactly the effect
/// `write_bits` would, including dropping words past the end of the buffer,
/// and the cursor is written back on drop.
pub(crate) struct BitWriter<'a> {
    words: &'a mut [ZfpBitStreamWord],
    state: &'a mut BitStreamState,
    word_pos: usize,
    buffer: u64,
    bits: u32,
    overflowed: bool,
}

impl<'a> BitWriter<'a> {
    #[inline]
    pub(crate) fn new<S: BitStreamStorageMut + ?Sized>(stream: &'a mut S) -> Self {
        let (words, state) = stream.split_mut();
        Self {
            words,
            word_pos: state.word_pos,
            buffer: state.buffer,
            bits: state.bits,
            overflowed: state.overflowed,
            state,
        }
    }

    #[inline]
    fn write_word(&mut self, word: u64) {
        if let Some(slot) = self.words.get_mut(self.word_pos) {
            *slot = word;
        } else {
            self.overflowed = true;
        }
        self.word_pos = self.word_pos.saturating_add(1);
    }

    /// Append the `n <= 64` bits of `value`, which must have no bits set above
    /// them.
    #[inline]
    pub(crate) fn put(&mut self, value: u64, n: u32) {
        debug_assert!(n <= WSIZE && (n == WSIZE || value >> n == 0));
        self.buffer |= value << self.bits;
        let total = self.bits + n;
        if total >= WSIZE {
            let spilled = self.bits;
            self.write_word(self.buffer);
            self.bits = total - WSIZE;
            // `value >> (WSIZE - spilled)`, and zero when `spilled == 0`.
            self.buffer = (value >> 1) >> (WSIZE - 1 - spilled);
        } else {
            self.bits = total;
        }
    }

    /// Append `n` zero bits.
    #[inline]
    pub(crate) fn put_zeros(&mut self, mut n: u32) {
        while n > WSIZE {
            self.put(0, WSIZE);
            n -= WSIZE;
        }
        self.put(0, n);
    }
}

impl Drop for BitWriter<'_> {
    #[inline]
    fn drop(&mut self) {
        self.state.word_pos = self.word_pos;
        self.state.buffer = self.buffer;
        self.state.bits = self.bits;
        self.state.overflowed = self.overflowed;
    }
}
