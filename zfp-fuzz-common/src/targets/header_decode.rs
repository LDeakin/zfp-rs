//! Fuzz target: `read_header` on arbitrary bytes.
//!
//! Headers are the first thing a decoder touches on untrusted data, and
//! `decode_metadata` is dense bit-shift logic with a `_ => return None`
//! fallthrough. Inputs are microseconds each, so this is cheap coverage of the
//! most exposed parse in the crate.
//!
//! Framing: `[0]` mask selector, `[1..]` header bytes (zero-padded to cover the
//! 148-bit maximum header). Hand-rolled so `gen_seeds` can emit real headers.

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpFieldMetadata, types::ZfpHeaderMask};

/// Bytes needed to hold `ZFP_HEADER_MAX_BITS` (148), rounded up to whole words.
const HEADER_BYTES: usize = 24;

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    let Some((&selector, payload)) = data.split_first() else {
        return;
    };

    // All eight combinations of the three header sections.
    let mask = ZfpHeaderMask::from_bits_truncate(u32::from(selector % 8));

    let mut bytes = vec![0u8; HEADER_BYTES.max(payload.len().next_multiple_of(8))];
    bytes[..payload.len()].copy_from_slice(payload);
    let mut bs = ZfpBitStream::from_bytes(&bytes);

    let Ok(header) = bs.read_header(mask) else {
        // Rejecting malformed headers is the correct outcome, not a finding.
        return;
    };

    assert_eq!(
        header.bits_read as u64,
        bs.read_pos(),
        "read_header reported {} bits but moved the read cursor to {}",
        header.bits_read,
        bs.read_pos()
    );

    // Sections must be present exactly when they were requested.
    assert_eq!(
        header.metadata.is_some(),
        mask.contains(ZfpHeaderMask::META),
        "metadata presence does not match the requested mask {mask:?}"
    );
    assert_eq!(
        header.config.is_some(),
        mask.contains(ZfpHeaderMask::MODE),
        "mode presence does not match the requested mask {mask:?}"
    );

    // A decoded section must survive a re-encode.
    if let Some(metadata) = header.metadata {
        let bits = metadata
            .to_bits()
            .expect("metadata decoded from a header must re-encode");
        assert_eq!(
            ZfpFieldMetadata::from_bits(bits),
            Some(metadata),
            "field metadata does not survive a bits round-trip"
        );
    }
    if let Some(config) = header.config {
        // Not `from_mode(c.mode_bits()) == c`: that does not hold, and should
        // not. A long-form mode word can carry parameters outside the ranges
        // the short forms cover — `min_exp` below `ZFP_MIN_EXP`, for instance,
        // is classified as reversible and re-encoded as the short reversible
        // word. The real invariant is that the encoding is idempotent, so a
        // stream written from a decoded header is stable.
        let bits = config.mode_bits();
        let decoded = ZfpConfig::from_mode(bits)
            .expect("a mode word produced by mode_bits must decode again");
        assert_eq!(
            decoded.mode_bits(),
            bits,
            "mode-word encoding is not idempotent for {config:?}"
        );
    }
}
