//! Header read/write logic (magic, metadata, mode).

use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::ZfpConfig;
use crate::field::{ZfpField, ZfpFieldMetadata};
use crate::types::{
    ZFP_MAGIC_BITS, ZFP_META_BITS, ZFP_MODE_LONG_BITS, ZFP_MODE_SHORT_BITS, ZfpHeaderMask,
    ZfpMetadataError,
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
/// Returns bits written (0 on failure).
pub(crate) fn write_header_bs(
    bs: &mut dyn ZfpBitStreamMutOps,
    field: &ZfpField,
    mask: ZfpHeaderMask,
    mode_bits_val: u64,
) -> usize {
    let mut bits = 0usize;

    // Pre-validate metadata if needed
    if mask.contains(ZfpHeaderMask::META)
        && matches!(
            field.metadata(),
            Err(ZfpMetadataError::Null | ZfpMetadataError::DimensionTooLarge)
        )
    {
        return 0;
    }

    if mask.contains(ZfpHeaderMask::MAGIC) {
        bs.write_bits(u64::from(b'z'), 8);
        bs.write_bits(u64::from(b'f'), 8);
        bs.write_bits(u64::from(b'p'), 8);
        bs.write_bits(u64::from(ZFP_CODEC), 8);
        bits += ZFP_MAGIC_BITS as usize;
    }

    if mask.contains(ZfpHeaderMask::META) {
        let Ok(meta) = field.metadata() else { return 0 };
        bs.write_bits(meta, ZFP_META_BITS);
        bits += ZFP_META_BITS as usize;
    }

    if mask.contains(ZfpHeaderMask::MODE) {
        let size = if mode_bits_val > MODE_SHORT_MAX {
            ZFP_MODE_LONG_BITS
        } else {
            ZFP_MODE_SHORT_BITS
        };
        bs.write_bits(mode_bits_val, size);
        bits += size as usize;
    }

    bits
}

/// Read header from `bs`.
pub(crate) fn read_header_bs(
    bs: &mut dyn ZfpBitStreamOps,
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
