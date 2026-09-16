//! Core compression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress].

use crate::bitstream::ZfpBitStreamMutOps;
#[cfg(feature = "rayon")]
use crate::bitstream::ZfpBitStreamOps;
use crate::config::ZfpConfig;
use crate::field::ZfpField;
use crate::field_plan::{FieldPlan, PlanError};
use crate::types::{ZFP_MIN_EXP, ZfpCompressionError, ZfpScalarType};

// ---------------------------------------------------------------------------
// Serial compression
// ---------------------------------------------------------------------------

/// Compress a field into the bitstream with the given expert parameters.
pub(crate) fn compress(
    bs: &mut dyn ZfpBitStreamMutOps,
    field: &ZfpField,
    config: &ZfpConfig,
) -> Result<usize, ZfpCompressionError> {
    let info = plan(field)?;
    let buf = field.data();
    for block_idx in 0..info.num_blocks {
        compress_block(bs, buf, &info, config, block_idx);
    }

    bs.flush();
    Ok(bs.size())
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

impl From<PlanError> for ZfpCompressionError {
    fn from(e: PlanError) -> Self {
        match e {
            PlanError::NoData => ZfpCompressionError::NoData,
            PlanError::InvalidField { required, actual } => {
                ZfpCompressionError::InvalidField { required, actual }
            }
            PlanError::MisalignedData { align } => ZfpCompressionError::MisalignedData { align },
        }
    }
}

/// Encode a single block into the bitstream.
// `FieldPlan::new` rejects a field whose buffer is not aligned for its scalar
// type, so these casts are checked once per field rather than once per block.
#[allow(clippy::cast_ptr_alignment)]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn compress_block(
    bs: &mut dyn ZfpBitStreamMutOps,
    buf: &[u8],
    info: &FieldPlan,
    config: &ZfpConfig,
    block_idx: usize,
) {
    use crate::codec::block::{
        encode_block_strided_reversible, encode_block_strided_with_params,
        encode_partial_block_strided_with_params,
    };

    let (ix, iy, iz, iw) = info.block_coords(block_idx);
    let [nx, ny, nz, nw] = info.dims;
    let dim_count = info.dim_count();

    let elem_off = -info.imin
        + (ix as isize) * 4 * info.strides[0]
        + (iy as isize) * 4 * info.strides[1]
        + (iz as isize) * 4 * info.strides[2]
        + (iw as isize) * 4 * info.strides[3];

    let lx = (nx - ix * 4).min(4);
    let ly = if dim_count >= 2 {
        (ny - iy * 4).min(4)
    } else {
        0
    };
    let lz = if dim_count >= 3 {
        (nz - iz * 4).min(4)
    } else {
        0
    };
    let lw = if dim_count >= 4 {
        (nw - iw * 4).min(4)
    } else {
        0
    };
    let full = lx == 4
        && (dim_count < 2 || ly == 4)
        && (dim_count < 3 || lz == 4)
        && (dim_count < 4 || lw == 4);
    let lengths = [lx, ly, lz, lw];
    let dims = info.dims_enum;

    let byte_off = (elem_off as usize) * info.elem_size();
    let ty = info.scalar_type;

    macro_rules! encode_dispatch {
        ($($zfp_ty:path => $elem_ty:ty),* $(,)?) => {{
            match ty {
                $(
                    $zfp_ty => {
                        // SAFETY: `buf` is the field's whole data buffer, which
                        // `FieldPlan::new` validated to be at least
                        // `checked_size_bytes()` long and aligned for the scalar
                        // type. `byte_off` is the block origin measured from the
                        // *lowest* address of the strided span (`elem_off`
                        // includes the `-imin` shift), so every offset the
                        // strides generate from it lands inside `buf`.
                        let block = unsafe { buf.as_ptr().add(byte_off).cast::<$elem_ty>() };
                        unsafe {
                            if config.min_exp() < ZFP_MIN_EXP {
                                encode_block_strided_reversible(
                                    bs, block, dims, &info.strides, lengths,
                                );
                            } else if full {
                                encode_block_strided_with_params(
                                    bs, block, dims, &info.strides, config.min_bits(),
                                    config.max_bits(), config.max_prec(), config.min_exp(),
                                );
                            } else {
                                encode_partial_block_strided_with_params(
                                    bs, block, dims, &lengths, &info.strides,
                                    config.min_bits(), config.max_bits(), config.max_prec(),
                                    config.min_exp(),
                                );
                            }
                        }
                    }
                )*
            }
        }};
    }
    encode_dispatch!(
        ZfpScalarType::Int32 => i32,
        ZfpScalarType::Int64 => i64,
        ZfpScalarType::Float => f32,
        ZfpScalarType::Double => f64,
    );
}

// ---------------------------------------------------------------------------
// Parallel compression (rayon feature)
// ---------------------------------------------------------------------------

/// Compress a contiguous range of blocks into a bitstream.
#[allow(dead_code)] // used only when rayon feature is enabled
fn compress_blocks_range(
    bs: &mut dyn ZfpBitStreamMutOps,
    buf: &[u8],
    info: &FieldPlan,
    config: &ZfpConfig,
    start_block: usize,
    end_block: usize,
) {
    for block_idx in start_block..end_block {
        compress_block(bs, buf, info, config, block_idx);
    }
}

/// Parallel compression via Rayon.
///
/// Splits block indices into C-OMP-compatible chunks, compresses each chunk
/// into a local bitstream, then concatenates results at bit-level granularity
/// to match the C OMP implementation's `stream_copy` behavior.
#[cfg(feature = "rayon")]
pub(crate) fn compress_rayon(
    bs: &mut dyn ZfpBitStreamMutOps,
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
        return Ok(0);
    }

    let (chunks, chunk_starts) = compute_chunk_ranges(blocks, threads, chunk_size);

    // Each chunk returns (bits_written, words) for bit-level concatenation.
    let chunk_results: Vec<(usize, Vec<u64>)> = if threads > 0 {
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
        // Create a temporary read bitstream from the chunk's words.
        let mut src = crate::ZfpBitStreamRef::from_words(chunk_words);
        // Copy bits at current write position (sequential, no seek).
        let mut remaining = *bits_written;
        while remaining > 64 {
            let w = src.read_bits(64);
            bs.write_bits(w, 64);
            remaining -= 64;
        }
        if remaining > 0 {
            #[allow(clippy::cast_possible_truncation)]
            // remaining is strictly < 64 after the loop above.
            let bits = remaining as u32;
            let w = src.read_bits(bits);
            bs.write_bits(w, bits);
        }
    }

    bs.flush();
    Ok(bs.size())
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
) -> (usize, Vec<u64>) {
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
    compress_blocks_range(&mut local_bs, buf, info, config, start, end);
    // Record bits written before flushing (flush pads to word boundary).
    let bits_written = local_bs.bits_written();
    local_bs.flush();
    let words_written = local_bs.word_pos();
    (bits_written, local_bs.words[..words_written].to_vec())
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
