//! Core compression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress].

use crate::bitstream::ZfpBitStreamMutOps;
use crate::config::ZfpConfig;
use crate::field::ZfpField;
use crate::field_plan::FieldPlan;
use crate::types::{ZFP_MIN_EXP, ZfpCompressionError, ZfpScalar, ZfpScalarType};
use std::ops::Range;

// ---------------------------------------------------------------------------
// Serial compression
// ---------------------------------------------------------------------------

/// Compress a field into the bitstream with the given expert parameters.
pub(crate) fn compress(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    field: &ZfpField,
    config: &ZfpConfig,
) -> Result<usize, ZfpCompressionError> {
    let info = plan(field)?;
    // SAFETY: `field.data()` is the buffer `FieldPlan::new` validated.
    unsafe { compress_blocks(bs, field.data().as_ptr(), &info, config, 0..info.num_blocks) };

    finish(bs)
}

/// Flush the stream and return its size, or the overflow error.
fn finish(bs: &mut (impl ZfpBitStreamMutOps + ?Sized)) -> Result<usize, ZfpCompressionError> {
    bs.flush();
    if bs.overflowed() {
        return Err(ZfpCompressionError::BufferTooSmall {
            required: bs.byte_len(),
            capacity: bs.capacity(),
        });
    }
    Ok(bs.byte_len())
}

/// Derive the block plan for a field, mapping the layout error.
fn plan(field: &ZfpField) -> Result<FieldPlan, ZfpCompressionError> {
    Ok(FieldPlan::new(
        field.scalar_type(),
        field.dims(),
        field.dimensionality(),
        field.effective_strides(),
        field.data(),
        field.checked_size_bytes().unwrap_or(usize::MAX),
    )?)
}

/// Encode blocks `range` into the bitstream, in order.
///
/// # Safety
/// `base` must point to the start of the field's data buffer: at least
/// `checked_size_bytes()` long, aligned for the field's scalar type, and
/// readable for the duration of the call. `FieldPlan::new` validates both
/// properties, so deriving `base` from a field it accepted satisfies this.
// The alignment check is per field, so this cast is not repeated per block.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn compress_blocks(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    base: *const u8,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    // SAFETY: the caller's contract, with the pointer cast to the scalar type
    // `FieldPlan::new` checked its alignment for.
    unsafe {
        match info.scalar_type {
            ZfpScalarType::I32 => compress_typed(bs, base.cast::<i32>(), info, config, range),
            ZfpScalarType::I64 => compress_typed(bs, base.cast::<i64>(), info, config, range),
            ZfpScalarType::F32 => compress_typed(bs, base.cast::<f32>(), info, config, range),
            ZfpScalarType::F64 => compress_typed(bs, base.cast::<f64>(), info, config, range),
        }
    }
}

/// [`compress_blocks`] for one scalar type.
///
/// # Safety
/// As for [`compress_blocks`], with `base` cast to `T`.
unsafe fn compress_typed<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    base: *const T,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    use crate::codec::block::{
        encode_block_strided, encode_block_strided_reversible, encode_partial_block_strided,
    };

    let dims = info.dims_enum;
    let strides = &info.strides;
    // The strided encoders check this per block; hoisting it is faster.
    let reversible = config.min_exp() < ZFP_MIN_EXP;
    for coords in info.blocks(range) {
        let (offset, lengths) = info.block_geometry(coords);
        // SAFETY: `base` is the field's whole data buffer, and `offset` is
        // the block origin measured from the *lowest* address of the strided
        // span, so every offset the strides generate from it lands inside the
        // buffer.
        unsafe {
            let block = base.add(offset);
            if reversible {
                encode_block_strided_reversible(bs, block, dims, strides, lengths, config);
            } else if info.is_full(lengths) {
                encode_block_strided(bs, block, dims, strides, config);
            } else {
                encode_partial_block_strided(bs, block, dims, &lengths, strides, config);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel compression (rayon feature)
// ---------------------------------------------------------------------------

/// Parallel compression via Rayon.
///
/// Splits block indices into C-OMP-compatible chunks, compresses each chunk
/// into a local bitstream, then concatenates results at bit-level granularity
/// to match the C OMP implementation's `stream_copy` behavior.
#[cfg(feature = "rayon")]
pub(crate) fn compress_rayon(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    field: &ZfpField,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Result<usize, ZfpCompressionError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let info = plan(field)?;
    let buf = field.data();
    let blocks = info.num_blocks;
    if blocks == 0 {
        return finish(bs);
    }

    let (chunks, chunk_starts) = compute_chunk_ranges(blocks, threads, chunk_size);

    // Each chunk returns (bits_written, words) for bit-level concatenation.
    let chunk_results: Vec<(u64, Vec<u64>)> = if threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .expect("rayon thread pool creation failed");
        pool.install(|| {
            (0..chunks)
                .into_par_iter()
                .map(|c| compress_one_chunk(&chunk_starts, &info, buf, config, blocks, c))
                .collect()
        })
    } else {
        (0..chunks)
            .into_par_iter()
            .map(|c| compress_one_chunk(&chunk_starts, &info, buf, config, blocks, c))
            .collect()
    };

    // Concatenate chunks at bit-level granularity, matching C's stream_copy.
    // Write chunks sequentially (no seeking) to avoid buffer clobbering.
    for (bits_written, chunk_words) in &chunk_results {
        append_bits(bs, chunk_words, *bits_written);
    }

    finish(bs)
}

/// Append the first `bits` bits of `words`, as `copy_from` would, but a word at
/// a time without going through a `dyn` stream. Words past the end read as
/// zero, as they do from a stream.
#[cfg(feature = "rayon")]
#[allow(clippy::cast_possible_truncation)] // `bits / 64` counts words in memory
fn append_bits(bs: &mut (impl ZfpBitStreamMutOps + ?Sized), words: &[u64], bits: u64) {
    let mut w = crate::bitstream::BitWriter::new(bs);
    let word = |i: usize| words.get(i).copied().unwrap_or(0);
    let full = (bits / 64) as usize;
    for i in 0..full {
        w.put(word(i), 64);
    }
    let rest = (bits % 64) as u32;
    if rest != 0 {
        w.put(word(full) & ((1 << rest) - 1), rest);
    }
}

/// Compress one chunk of blocks, returning (`bits_written`, `compressed_words`).
///
/// Returns the exact bit count (before flush padding) alongside the flushed
/// words, so the caller can copy at bit-level granularity across chunk boundaries.
#[cfg(feature = "rayon")]
#[allow(clippy::cast_possible_truncation)] // chunk_bits / 64 + 1 fits in usize for valid fields
fn compress_one_chunk(
    chunk_starts: &[usize],
    info: &FieldPlan,
    buf: &[u8],
    config: &ZfpConfig,
    total_blocks: usize,
    chunk: usize,
) -> (u64, Vec<u64>) {
    use crate::ZfpBitStream;

    let start = chunk_starts[chunk];
    let end = if chunk + 1 < chunk_starts.len() {
        chunk_starts[chunk + 1]
    } else {
        total_blocks
    };
    let chunk_bits = (end - start) as u64 * u64::from(config.max_bits());
    #[allow(clippy::cast_possible_truncation)]
    // chunk_bits is bounded by the field size, which fits in usize.
    let chunk_words = (chunk_bits / 64 + 1) as usize;
    let mut local_bs = ZfpBitStream::new(chunk_words * 8);
    // SAFETY: `buf` is the field's whole data buffer, validated by `FieldPlan::new`.
    unsafe { compress_blocks(&mut local_bs, buf.as_ptr(), info, config, start..end) };
    // Record bits written before flushing (flush pads to word boundary).
    let bits_written = local_bs.write_pos();
    (bits_written, local_bs.into_words())
}

/// Compute chunk ranges following C OMP semantics.
#[cfg(feature = "rayon")]
#[allow(clippy::cast_sign_loss)]
fn compute_chunk_ranges(blocks: usize, threads: u32, chunk_size: u32) -> (usize, Vec<usize>) {
    let threads = if threads > 0 {
        threads as usize
    } else {
        rayon::current_num_threads()
    };

    let chunks = if chunk_size > 0 {
        blocks.div_ceil(chunk_size as usize).min(blocks)
    } else {
        threads.min(blocks)
    };

    let chunks = chunks.max(1);
    let mut starts = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        starts.push((blocks * chunk) / chunks);
    }
    (chunks, starts)
}
