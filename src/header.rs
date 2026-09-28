//! Header read/write logic (magic, metadata, mode).

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::ZfpConfig;
use crate::config::{STREAM_WORD_BITS, STREAM_WORD_BYTES};
use crate::field::{ZfpField, ZfpFieldMetadata};
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
    field: &ZfpField,
    mask: ZfpHeaderMask,
    mode_bits_val: u64,
) -> Result<usize, ZfpCompressionError> {
    let meta = if mask.contains(ZfpHeaderMask::META) {
        Some(field.metadata()?)
    } else {
        None
    };
    let mode_size = if mode_bits_val > MODE_SHORT_MAX {
        ZFP_MODE_LONG_BITS
    } else {
        ZFP_MODE_SHORT_BITS
    };

    let mut bits = 0usize;
    if mask.contains(ZfpHeaderMask::MAGIC) {
        bits += ZFP_MAGIC_BITS as usize;
    }
    if meta.is_some() {
        bits += ZFP_META_BITS as usize;
    }
    if mask.contains(ZfpHeaderMask::MODE) {
        bits += mode_size as usize;
    }
    let capacity = bs.capacity();
    let end = bs.write_pos().saturating_add(bits as u64);
    if end > (capacity as u64).saturating_mul(8) {
        let required = end.div_ceil(u64::from(STREAM_WORD_BITS)) * STREAM_WORD_BYTES as u64;
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
        config = Some(ZfpConfig::from_mode(mode).ok_or(ZfpHeaderError::InvalidMode)?);
    }

    Ok(ZfpHeader {
        bits_read: bits,
        metadata,
        config,
    })
}
