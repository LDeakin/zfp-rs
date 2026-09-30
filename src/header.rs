//! Header read/write logic (magic, metadata, mode).

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps, exact_write_pos};
use crate::config::ZfpConfig;
use crate::config::{STREAM_WORD_BITS, STREAM_WORD_BYTES};
use crate::field::ZfpFieldMetadata;
use crate::types::{
    ZFP_MAGIC_BITS, ZFP_META_BITS, ZFP_MODE_LONG_BITS, ZFP_MODE_SHORT_BITS, ZfpCompressionError,
    ZfpHeaderMask,
};
use std::fmt;

/// Structured data decoded from a ZFP header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZfpHeader {
    /// Number of bits consumed from the bitstream.
    pub bits_read: usize,
    /// Field metadata, present only when [`ZfpHeaderMask::META`] was read.
    pub metadata: Option<ZfpFieldMetadata>,
    /// Compression configuration, present only when [`ZfpHeaderMask::MODE`] was read.
    ///
    /// Rounding is not stored in the stream, so this always uses
    /// [`ZfpRounding::Never`][crate::ZfpRounding::Never]. Apply the encoder's
    /// rounding with [`ZfpConfig::with_rounding`] before decompressing.
    pub config: Option<ZfpConfig>,
}

/// Errors that can occur while reading a ZFP header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpHeaderError {
    /// The magic header section was requested but did not match this codec.
    InvalidMagic,
    /// The metadata header section was requested but contained invalid metadata.
    InvalidMetadata,
    /// The mode header section was requested but contained invalid compression parameters.
    InvalidMode,
}

impl fmt::Display for ZfpHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid ZFP header magic"),
            Self::InvalidMetadata => write!(f, "invalid ZFP field metadata in header"),
            Self::InvalidMode => write!(f, "invalid ZFP compression mode in header"),
        }
    }
}

impl std::error::Error for ZfpHeaderError {}

/// ZFP codec version (`ZFP_CODEC` in version.h).
const ZFP_CODEC: u8 = 5;

/// Maximum value representable in a 12-bit mode field (`ZFP_MODE_SHORT_MAX`).
const MODE_SHORT_MAX: u64 = (1u64 << ZFP_MODE_SHORT_BITS) - 2;

// ---------------------------------------------------------------------------
// Borrow-safe helpers: these receive the precomputed `mode` value and return
// any new mode to apply, avoiding double-mut-borrow in bitstream methods.
// ---------------------------------------------------------------------------

/// Write header to `bs`, given the precomputed `mode_bits` for the mode section.
/// Returns bits written.
///
/// Metadata and capacity are validated before anything is written, so nothing
/// is written on failure.
pub(crate) fn write_header_bs(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    metadata: &ZfpFieldMetadata,
    mask: ZfpHeaderMask,
    mode_bits_val: u64,
) -> Result<usize, ZfpCompressionError> {
    let meta = if mask.contains(ZfpHeaderMask::META) {
        Some(metadata.to_bits()?)
    } else {
        None
    };
    let mode_size = if mode_bits_val > MODE_SHORT_MAX {
        ZFP_MODE_LONG_BITS
    } else {
        ZFP_MODE_SHORT_BITS
    };

    // At most `ZFP_HEADER_MAX_BITS`.
    let bits = [
        (mask.contains(ZfpHeaderMask::MAGIC), ZFP_MAGIC_BITS),
        (meta.is_some(), ZFP_META_BITS),
        (mask.contains(ZfpHeaderMask::MODE), mode_size),
    ]
    .into_iter()
    .filter_map(|(written, bits)| written.then_some(bits as usize))
    .sum::<usize>();
    // Not from `write_pos`, which wraps back to a small offset once the cursor
    // passes bit `u64::MAX`.
    let capacity = bs.capacity();
    let end = exact_write_pos(bs).saturating_add(bits as u128);
    if end > (capacity as u128).saturating_mul(8) {
        let required = end
            .div_ceil(u128::from(STREAM_WORD_BITS))
            .saturating_mul(STREAM_WORD_BYTES as u128);
        return Err(ZfpCompressionError::BufferTooSmall {
            required: usize::try_from(required).unwrap_or(usize::MAX),
            capacity,
        });
    }

    if mask.contains(ZfpHeaderMask::MAGIC) {
        bs.write_bits(u64::from(b'z'), 8);
        bs.write_bits(u64::from(b'f'), 8);
        bs.write_bits(u64::from(b'p'), 8);
        bs.write_bits(u64::from(ZFP_CODEC), 8);
    }
    if let Some(meta) = meta {
        bs.write_bits(meta, ZFP_META_BITS);
    }
    if mask.contains(ZfpHeaderMask::MODE) {
        bs.write_bits(mode_bits_val, mode_size);
    }
    Ok(bits)
}

/// Read header from `bs`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "the sections total at most `ZFP_HEADER_MAX_BITS`"
)]
pub(crate) fn read_header_bs(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    mask: ZfpHeaderMask,
) -> Result<ZfpHeader, ZfpHeaderError> {
    let mut bits = 0usize;
    let mut metadata = None;
    let mut config = None;

    if mask.contains(ZfpHeaderMask::MAGIC) {
        if bs.read_bits(8) != u64::from(b'z')
            || bs.read_bits(8) != u64::from(b'f')
            || bs.read_bits(8) != u64::from(b'p')
            || bs.read_bits(8) != u64::from(ZFP_CODEC)
        {
            return Err(ZfpHeaderError::InvalidMagic);
        }
        bits += ZFP_MAGIC_BITS as usize;
    }

    if mask.contains(ZfpHeaderMask::META) {
        let meta = bs.read_bits(ZFP_META_BITS);
        metadata = Some(ZfpFieldMetadata::from_bits(meta).ok_or(ZfpHeaderError::InvalidMetadata)?);
        bits += ZFP_META_BITS as usize;
    }

    if mask.contains(ZfpHeaderMask::MODE) {
        let mut mode = bs.read_bits(ZFP_MODE_SHORT_BITS);
        bits += ZFP_MODE_SHORT_BITS as usize;
        if mode > MODE_SHORT_MAX {
            let extra = ZFP_MODE_LONG_BITS - ZFP_MODE_SHORT_BITS;
            mode |= bs.read_bits(extra) << ZFP_MODE_SHORT_BITS;
            bits += extra as usize;
        }
        config = Some(ZfpConfig::from_mode_bits(mode).ok_or(ZfpHeaderError::InvalidMode)?);
    }

    Ok(ZfpHeader {
        bits_read: bits,
        metadata,
        config,
    })
}
