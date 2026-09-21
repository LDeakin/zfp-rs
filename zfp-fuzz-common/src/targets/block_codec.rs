//! Fuzz target: the strided block gather/scatter paths.
//!
//! `src/codec/block.rs` dispatches to eight `dim{1,2,3,4}` modules built almost
//! entirely from raw `*p.offset(x*sx + y*sy + z*sz + w*sw)` arithmetic. Those
//! offsets are driven by caller-supplied strides, and the `*_strided` entry
//! points cannot bounds-check them (see the module docs on
//! `zfp_rs::codec::block`) — so this is the widest memory-safety surface in the
//! pure-Rust crate, and the reason this target runs under `AddressSanitizer`.
//!
//! Buffers are allocated from the block's exact index span and the pointer is
//! offset so that negative strides reach backwards into the allocation, exactly
//! as `src/compress.rs` does. `ASan` redzones therefore sit immediately on both
//! sides of the span, and any off-by-one in a gather or scatter is a crash.
//!
//! The pointer is derived from the whole buffer rather than a subslice, so its
//! provenance covers the backwards offsets too — a `&src[origin..]` would not,
//! which Miri catches and `ASan` cannot.

use zfp_rs::{
    ZfpBitStream, ZfpConfig,
    codec::block::{
        decode_block_strided_with_params, decode_partial_block_strided_with_params,
        encode_block_strided_with_params, encode_partial_block_strided_with_params,
    },
    types::ZFP_MIN_EXP,
    types::ZfpDimensionality,
};

use crate::input::{ModeSpec, ScalarKind};
use crate::limits::MAX_STRIDE_GAP;
use crate::scalar::{FuzzScalar, decode_scalars};

/// Framing prefix: kind, rank, four stride bytes, four length bytes, mode.
const HEADER_LEN: usize = 15;

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    if data.len() < HEADER_LEN {
        return;
    }
    let (hdr, payload) = data.split_at(HEADER_LEN);

    let kind = ScalarKind::from_byte(hdr[0]);
    let rank = 1 + usize::from(hdr[1] % 4);
    let dims = match rank {
        1 => ZfpDimensionality::D1,
        2 => ZfpDimensionality::D2,
        3 => ZfpDimensionality::D3,
        _ => ZfpDimensionality::D4,
    };

    let strides = build_strides(rank, [hdr[2], hdr[3], hdr[4], hdr[5]]);
    // Partial-block lengths in 1..=4; a full block is all 4s.
    let mut lengths = [0usize; 4];
    for (axis, len) in lengths.iter_mut().enumerate().take(rank) {
        *len = 1 + usize::from(hdr[6 + axis] % 4);
    }
    let partial = lengths[..rank].iter().any(|&l| l != 4);
    let mode = ModeSpec::from_bytes(hdr[10], [hdr[11], hdr[12], hdr[13]], hdr[14]);

    let Some(config) = mode.to_config(kind.scalar_type(), dims) else {
        return;
    };
    // Reversible mode is signalled by `min_exp < ZFP_MIN_EXP`, and
    // `src/compress.rs` routes it to `encode_block_strided_reversible` rather
    // than the `_with_params` entry points this target drives. Passing
    // reversible parameters to `_with_params` runs the ordinary lossy codec, so
    // the config would not mean what it says. Skip those inputs.
    if config.min_exp() < ZFP_MIN_EXP {
        return;
    }

    match kind {
        ScalarKind::I32 => typed::<i32>(dims, rank, &strides, &lengths, partial, config, payload),
        ScalarKind::I64 => typed::<i64>(dims, rank, &strides, &lengths, partial, config, payload),
        ScalarKind::F32 => typed::<f32>(dims, rank, &strides, &lengths, partial, config, payload),
        ScalarKind::F64 => typed::<f64>(dims, rank, &strides, &lengths, partial, config, payload),
    }
}

/// Build a stride vector: a permutation of the natural layout, each axis with
/// an optional gap and an optional sign flip.
///
/// Strides are never zero — a zero stride would alias every element of an axis
/// onto one slot, which the codec does not support and the field constructors
/// treat as "use the default".
fn build_strides(rank: usize, bytes: [u8; 4]) -> [isize; 4] {
    let mut strides = [0isize; 4];
    let mut magnitude: isize = 1;
    for axis in 0..rank {
        let b = bytes[axis];
        let gap = 1 + isize::from(b % MAX_STRIDE_GAP);
        magnitude *= gap;
        strides[axis] = if b & 0x80 == 0 { magnitude } else { -magnitude };
        // Each successive axis must step over the whole extent of the previous
        // one, or blocks would overlap and the round-trip oracle below would
        // not hold.
        magnitude *= 4;
    }
    strides
}

/// Index span a 4^d block covers, as `(min, max)` element offsets.
fn block_span(rank: usize, strides: &[isize; 4], lengths: &[usize; 4]) -> (isize, isize) {
    let mut lo = 0isize;
    let mut hi = 0isize;
    for axis in 0..rank {
        let extent = strides[axis] * (lengths[axis].cast_signed() - 1);
        lo += extent.min(0);
        hi += extent.max(0);
    }
    (lo, hi)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // one call per encode/decode variant
fn typed<T: FuzzScalar>(
    dims: ZfpDimensionality,
    rank: usize,
    strides: &[isize; 4],
    lengths: &[usize; 4],
    partial: bool,
    config: ZfpConfig,
    payload: &[u8],
) {
    let full_lengths = [4usize; 4];
    let effective = if partial { lengths } else { &full_lengths };
    let (lo, hi) = block_span(rank, strides, effective);
    let span = (hi - lo + 1).cast_unsigned();

    // A single block never legitimately spans more than 4^4 * gap^4 elements;
    // anything larger means the stride construction went wrong.
    if span > 1 << 20 {
        return;
    }

    // The origin sits `-lo` elements into the buffer so that negative strides
    // reach backwards into the allocation, mirroring the `-info.imin` offset in
    // `src/compress.rs`.
    let origin = (-lo).cast_unsigned();
    let src: Vec<T> = decode_scalars::<T>(payload, span);

    let (min_bits, max_bits, max_prec, min_exp) = (
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.min_exp(),
    );

    let cap = 4096;
    let mut bs = ZfpBitStream::new(cap);
    // SAFETY (all three call sites below): `src`/`dst` are allocated to the
    // block's exact index span and `origin` places the pointer so that every
    // offset the strides generate lands inside the allocation.
    let written = unsafe {
        let block = src.as_ptr().add(origin);
        if partial {
            encode_partial_block_strided_with_params(
                &mut bs,
                block,
                dims,
                &effective[..rank],
                &strides[..rank],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        } else {
            encode_block_strided_with_params(
                &mut bs,
                block,
                dims,
                &strides[..rank],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        }
    };
    assert!(
        written <= cap * 8,
        "block encode wrote {written} bits into a {cap} B stream"
    );

    // Decode back into a buffer with the same span. The primary property under
    // test is memory safety — every gather and scatter offset must land inside
    // the allocation, which ASan checks — plus bit-exactness in reversible
    // mode, verified below.
    //
    // Note what is deliberately *not* asserted here: that re-encoding a decoded
    // block reproduces the original bitstream. zfp's inverse transform rounds,
    // so `encode(decode(x)) != encode(x)` for lossy modes and asserting it
    // produces false crashes.
    let mut dst: Vec<T> = vec![T::default(); span];
    bs.rewind();
    unsafe {
        let block = dst.as_mut_ptr().add(origin);
        if partial {
            decode_partial_block_strided_with_params(
                &mut bs,
                block,
                dims,
                &effective[..rank],
                &strides[..rank],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        } else {
            decode_block_strided_with_params(
                &mut bs,
                block,
                dims,
                &strides[..rank],
                min_bits,
                max_bits,
                max_prec,
                min_exp,
            )
        };
    }

    // Elements outside the block's own offsets must not be touched. This is the
    // correctness half of the target: a scatter that writes to the wrong index
    // corrupts a neighbour rather than overrunning the allocation, so ASan
    // cannot see it, but this can.
    let mut covered = vec![false; span];
    for index in block_offsets(rank, strides, effective) {
        covered[(origin.cast_signed() + index).cast_unsigned()] = true;
    }
    for (at, touched) in covered.iter().enumerate() {
        assert!(
            *touched || dst[at].to_bits_u64() == 0,
            "block decode wrote to buffer index {at}, which no stride offset covers \
             (dims={dims:?}, strides={strides:?}, lengths={effective:?})"
        );
    }
}

/// Element offsets, relative to the block origin, that a block covers.
fn block_offsets(
    rank: usize,
    strides: &[isize; 4],
    lengths: &[usize; 4],
) -> impl Iterator<Item = isize> {
    let lengths = *lengths;
    let strides = *strides;
    let counts: [usize; 4] = std::array::from_fn(|a| if a < rank { lengths[a] } else { 1 });
    (0..counts.iter().product::<usize>()).map(move |mut linear| {
        let mut offset = 0isize;
        for axis in 0..rank {
            let coord = linear % counts[axis];
            linear /= counts[axis];
            offset += strides[axis] * coord.cast_signed();
        }
        offset
    })
}
