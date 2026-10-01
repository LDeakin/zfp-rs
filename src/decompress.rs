//! Core decompression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress].

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

#[cfg(feature = "rayon")]
use crate::bitstream::mark_overread;
use crate::bitstream::{ZfpBitStreamOps, overread, reset_overread};
use crate::config::ZfpConfig;
use crate::field::ZfpFieldMut;
use crate::field_plan::FieldPlan;
use crate::types::{ZfpDecompressionError, ZfpScalar, ZfpScalarType};
use std::ops::Range;

#[cfg(feature = "rayon")]
mod pipeline;

// ---------------------------------------------------------------------------
// Serial decompression
// ---------------------------------------------------------------------------

/// Decompress from the bitstream into the field with the given expert parameters.
pub(crate) fn decompress(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
) -> Result<usize, ZfpDecompressionError> {
    let info = field.plan()?;
    // Derived once: a fresh `&mut` retag per block would be needless work, and
    // interleaving it with shared borrows of `field` is a hazard worth avoiding.
    let base = field.data_mut().as_mut_ptr();
    // SAFETY: `FieldPlan::new` validated the buffer `base` points to.
    unsafe { decompress_planned(bs, &info, base, config) }
}

/// [`decompress`] into a field `info` planned, whose data buffer `base` points
/// to.
///
/// # Safety
/// As for [`decompress_blocks`].
unsafe fn decompress_planned(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    info: &FieldPlan,
    base: *mut u8,
    config: &ZfpConfig,
) -> Result<usize, ZfpDecompressionError> {
    reset_overread(bs);
    // SAFETY: the caller's contract.
    unsafe { decompress_blocks(bs, base, info, config, 0..info.num_blocks) };
    finish(bs)
}

/// Align the cursor after the last block, and report how many bytes decoding
/// read.
///
/// Reads past the end of the buffer yield zeros, so a stream is truncated
/// exactly when decoding loaded a word the buffer does not hold, which is where
/// C reads past its end. A seek past the end loads no word if it lands on a
/// word boundary, as in C, so a stream missing only whole words of its last
/// block's padding is not truncated, and the size then exceeds the buffer, as
/// C's does.
fn finish(bs: &mut (impl ZfpBitStreamOps + ?Sized)) -> Result<usize, ZfpDecompressionError> {
    bs.align();
    let required = bs.byte_len();
    if overread(bs) {
        let capacity = bs.capacity();
        return Err(ZfpDecompressionError::Truncated { required, capacity });
    }
    Ok(required)
}

/// Decode blocks `range` from the bitstream, in order, from its cursor.
///
/// # Safety
/// `base` must point to the start of the field's data buffer: at least
/// `checked_size_bytes()` long, aligned for the field's scalar type, and
/// writable for the duration of the call. `FieldPlan::new` validates both
/// properties, so deriving `base` from a field it accepted satisfies this.
// `FieldPlan::new` rejects a field whose buffer is not aligned for its scalar
// type, so this cast is checked once per field rather than once per block.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decompress_blocks(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    base: *mut u8,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    // SAFETY: the caller's contract, with the pointer cast to the scalar type
    // `FieldPlan::new` checked its alignment for.
    unsafe {
        match info.scalar_type {
            ZfpScalarType::I32 => {
                decompress_typed(bs, base.cast::<i32>(), info, config, range);
            }
            ZfpScalarType::I64 => {
                decompress_typed(bs, base.cast::<i64>(), info, config, range);
            }
            ZfpScalarType::F32 => {
                decompress_typed(bs, base.cast::<f32>(), info, config, range);
            }
            ZfpScalarType::F64 => {
                decompress_typed(bs, base.cast::<f64>(), info, config, range);
            }
        }
    }
}

/// [`decompress_blocks`] for one scalar type.
///
/// # Safety
/// As for [`decompress_blocks`], with `base` cast to `T`.
unsafe fn decompress_typed<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    base: *mut T,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    use crate::codec::block::{decode_lossy, decode_lossy_partial, decode_reversible};

    let dims = info.dims_enum;
    let strides = &info.strides;
    let reversible = config.is_reversible();
    for coords in info.blocks(range) {
        let (offset, lengths) = info.block_geometry(coords);
        // SAFETY: `base` is the field's whole data buffer, which
        // `FieldPlan::new` validated to be at least `checked_size_bytes()`
        // long and aligned for the scalar type. `offset` is the block origin
        // measured from the *lowest* address of the strided span, so every
        // offset the strides generate from it lands inside the buffer.
        unsafe {
            let block = base.add(offset);
            if reversible {
                decode_reversible(bs, block, dims, strides, lengths, config);
            } else if info.is_full(lengths) {
                decode_lossy(bs, block, dims, strides, config);
            } else {
                decode_lossy_partial(bs, block, dims, lengths, strides, config);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel decompression (rayon feature)
// ---------------------------------------------------------------------------

/// Parallel decompression via Rayon.
///
/// For **fixed-rate** streams, each thread creates an independent bitstream
/// view and seeks directly to its block's bit position.
/// This falls back to serial when field strides may alias (two blocks writing
/// the same element would race), when the blocks would run past the largest
/// bit offset, or when the pool or the chunks cannot be created.
///
/// Other streams use a plane-reader / reconstruction pipeline.
/// This falls back to serial when field strides may alias, with fewer than two
/// pool threads, or when the pool or the pipeline's buffers cannot be created.
#[cfg(feature = "rayon")]
pub(crate) fn decompress_rayon(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Result<usize, ZfpDecompressionError> {
    let info = field.plan()?;
    let base = field.data_mut().as_mut_ptr();
    let done = FieldPtr::new(base, &info).and_then(|shared| {
        if config.mode() == crate::types::ZfpMode::FixedRate {
            decompress_parallel(bs, &info, shared, config, threads, chunk_size)
        } else {
            decompress_pipeline(bs, &info, shared, config, threads, chunk_size)
        }
    });
    if done.is_none() {
        // SAFETY: `FieldPlan::new` validated the buffer `base` points to.
        return unsafe { decompress_planned(bs, &info, base, config) };
    }
    finish(bs)
}

/// Leave the cursor, and the overread flag, where serial decompression would.
#[cfg(feature = "rayon")]
fn park(bs: &mut (impl ZfpBitStreamOps + ?Sized), end: u64, overread: bool) {
    reset_overread(bs);
    bs.seek_read(end);
    if overread {
        mark_overread(bs);
    }
}

/// Decode every block of a fixed-rate stream in parallel chunks, then
/// [`park`] the cursor.
///
/// Returns `None`, having decoded nothing, where the blocks have no fixed size
/// or would run past the largest bit offset, or the pool or the chunks cannot
/// be created.
#[cfg(feature = "rayon")]
fn decompress_parallel(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    info: &FieldPlan,
    base: FieldPtr,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Option<()> {
    // A fixed-rate config only gives each block a fixed physical size if its
    // budget can hold the type's block header. Below 9 bits for f32 or 12 for
    // f64, zero blocks use max_bits but nonzero blocks write the full header.
    // The maximum-size calculation already accounts for this distinction.
    if config.block_bits(info.scalar_type, info.dims_enum) != config.max_bits() {
        return None;
    }
    let bits_per_block = config.max_bits();
    let start_read_bit = bs.read_pos();
    // Where serial decompression would leave the cursor. Bounding it bounds
    // every block's offset, so the per-chunk seeks cannot overflow.
    let end = u64::try_from(info.num_blocks)
        .ok()?
        .checked_mul(u64::from(bits_per_block))?
        .checked_add(start_read_bit)?;
    let ranges = crate::execution::chunk_ranges(info.num_blocks, threads, chunk_size)?;

    // Shared read-only slice of all words in the bitstream buffer.
    // `word_pos` tracks the current read cursor (reset by rewind), so we use
    // the full buffer length to cover all compressed data.
    let words = bs.backing_words();
    let run = || {
        decompress_chunks(
            words,
            &ranges,
            info,
            config,
            bits_per_block,
            start_read_bit,
            base,
        )
    };
    // The pool is built before any block is decoded, so if it cannot be, the
    // serial decoder fills the whole field.
    let overread_chunks = crate::execution::install(threads, run)?;

    park(bs, end, overread_chunks);
    Some(())
}

/// A `*mut u8` into the field's data buffer, handed to worker threads.
///
/// Carrying the real pointer rather than a `usize` round-trip keeps its
/// provenance intact, which `-Zmiri-strict-provenance` requires.
///
/// Only [`Self::new`] makes one, and only where blocks write disjoint bytes.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy)]
struct FieldPtr(*mut u8);

#[cfg(feature = "rayon")]
impl FieldPtr {
    /// Share `base`, the buffer `info` was planned for, with workers.
    ///
    /// `None` where two blocks could write the same element, as with strides
    /// that may alias, and where there are no blocks to share out.
    fn new(base: *mut u8, info: &FieldPlan) -> Option<Self> {
        (info.num_blocks != 0 && !info.strides_may_alias()).then_some(Self(base))
    }

    /// Take `self` by value so closures capture the whole `FieldPtr` rather
    /// than the bare `*mut u8` inside it: precise capture would otherwise grab
    /// `base.0`, and `&*mut u8` is not `Sync`.
    fn ptr(self) -> *mut u8 {
        self.0
    }
}

// SAFETY: chunks and pipeline batches cover disjoint ranges of block indices,
// and `FieldPtr::new` only makes one for non-aliasing strides, under which
// blocks map to disjoint byte ranges. So no two threads write the same byte and
// no thread reads a byte another thread writes.
#[cfg(feature = "rayon")]
unsafe impl Send for FieldPtr {}
#[cfg(feature = "rayon")]
unsafe impl Sync for FieldPtr {}

/// Decompress chunks of blocks in parallel, and report whether any read past
/// the end of the buffer.
///
/// Each chunk gets its own bitstream view and seeks to its first block, then
/// defers to the same `decompress_blocks` the serial path uses. Every block
/// takes exactly `bits_per_block` bits, as its decoder skips to `min_bits`, so
/// the cursor then moves from block to block.
#[cfg(feature = "rayon")]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "`decompress_rayon` checked that the offset past the last block fits `u64`"
)]
fn decompress_chunks(
    words: &[u64],
    ranges: &[Range<usize>],
    info: &FieldPlan,
    config: &ZfpConfig,
    bits_per_block: u32,
    start_read_bit: u64,
    base: FieldPtr,
) -> bool {
    use crate::bitstream::borrowed::ZfpBitStreamRef;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // `reduce` rather than `any`, which would stop decoding at the first chunk
    // that reads past the end.
    ranges
        .par_iter()
        .map(|range| {
            let mut local_bs = ZfpBitStreamRef::from_words(words);
            local_bs.seek_read(start_read_bit + range.start as u64 * u64::from(bits_per_block));
            // SAFETY: `base` is the buffer `FieldPlan::new` validated for length
            // and alignment, and chunks cover disjoint blocks, so this thread
            // writes only bytes no other thread touches.
            unsafe {
                decompress_blocks(&mut local_bs, base.ptr(), info, config, range.clone());
            }
            overread(&local_bs)
        })
        .reduce(|| false, |a, b| a | b)
}

/// Decode every block through the plane-reader / reconstruction pipeline, then
/// [`park`] the cursor.
///
/// Returns `None`, having read and written nothing, where only one thread is
/// available, or the pool or the pipeline's buffers cannot be created.
#[cfg(feature = "rayon")]
fn decompress_pipeline(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    info: &FieldPlan,
    base: FieldPtr,
    config: &ZfpConfig,
    threads: u32,
    batch_size: u32,
) -> Option<()> {
    if threads == 1 {
        return None;
    }
    let start = bs.read_pos();
    let words = bs.backing_words();
    let (end, truncated) = crate::execution::install(threads, || {
        pipeline::decode(words, start, base, info, config, batch_size)
    })??;
    park(bs, end, truncated);
    Some(())
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use crate::config::{ZfpConfig, ZfpStreamAlignment};
    use crate::execution::ZfpExecution;
    use crate::types::{ZfpDecompressionError, ZfpDimensionality, ZfpScalarType};
    use crate::{ZfpBitStream, ZfpField, ZfpFieldMut};

    /// `[1, 1]` maps index `[x, y]` to `x + y`, so the span is 15 elements.
    fn decode(bs: &mut ZfpBitStream, config: &ZfpConfig, execution: ZfpExecution) -> [u64; 15] {
        let mut out = [0f64; 15];
        let mut field = ZfpFieldMut::new_strided(&mut out, [8usize, 8], [1isize, 1]).unwrap();
        bs.rewind();
        bs.decompress_with_execution(config, &mut field, execution)
            .unwrap();
        out.map(f64::to_bits)
    }

    /// Aliasing strides make two blocks write the same element, which the
    /// parallel path cannot do without a data race. It must fall back to
    /// serial, so its output matches serial decompression exactly.
    #[test]
    fn aliasing_strides_decompress_serially() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let src: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(4096).unwrap();
        bs.compress(&config, &ZfpField::new(&src, [8usize, 8]).unwrap())
            .unwrap();

        let parallel = decode(
            &mut bs,
            &config,
            ZfpExecution::Rayon {
                threads: 4,
                chunk_size: 1,
            },
        );
        assert_eq!(parallel, decode(&mut bs, &config, ZfpExecution::Serial));
    }

    /// Block offsets past `u64::MAX` overflowed the per-block seeks; they now
    /// decompress serially.
    #[test]
    fn blocks_past_the_largest_offset_decompress_serially() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let mut bs = ZfpBitStream::new(64).unwrap();
        let decode = |bs: &mut ZfpBitStream, execution| {
            bs.seek_read(u64::MAX - 100);
            let mut out = [1f64; 64];
            let mut field = ZfpFieldMut::new(&mut out, [8usize, 8]).unwrap();
            let read = bs.decompress_with_execution(&config, &mut field, execution);
            (read, bs.read_pos(), out.map(f64::to_bits))
        };
        let serial = decode(&mut bs, ZfpExecution::Serial);
        assert!(matches!(
            serial.0,
            Err(ZfpDecompressionError::Truncated { .. })
        ));
        let parallel = ZfpExecution::Rayon {
            threads: 2,
            chunk_size: 1,
        };
        assert_eq!(decode(&mut bs, parallel), serial);
    }

    /// A truncated stream is reported, in parallel as serially, with the size
    /// the whole stream needs. The field is filled as if the missing words were
    /// zeros, and the parallel path's final seek once clamped to the buffer
    /// instead.
    #[test]
    fn truncated_stream_is_reported_serially_and_in_parallel() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let src: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(4096).unwrap();
        let size = bs
            .compress(&config, &ZfpField::new(&src, [8usize, 8]).unwrap())
            .unwrap();
        let kept = size / 2;
        let mut padded = bs.as_bytes()[..kept].to_vec();
        padded.resize(size, 0);

        let decode = |bytes: &[u8], execution| {
            let mut out = [0f64; 64];
            let mut field = ZfpFieldMut::new(&mut out, [8usize, 8]).unwrap();
            let result = ZfpBitStream::from_bytes(bytes)
                .unwrap()
                .decompress_with_execution(&config, &mut field, execution);
            (result, out.map(f64::to_bits))
        };
        let expect = decode(&padded, ZfpExecution::Serial);
        assert_eq!(expect.0, Ok(size));
        let truncated = &bs.as_bytes()[..kept];
        let error = Err(ZfpDecompressionError::Truncated {
            required: size,
            capacity: kept,
        });
        assert_eq!(decode(truncated, ZfpExecution::Serial), (error, expect.1));
        let parallel = ZfpExecution::Rayon {
            threads: 4,
            chunk_size: 1,
        };
        assert_eq!(decode(truncated, parallel), (error, expect.1));
    }
}
