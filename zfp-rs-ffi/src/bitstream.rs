//! Bitstream API: C-level wrappers around `zfp_rs::ZfpBitStream`.
//!
//! Implements: `stream_open`, `stream_close`, `stream_clone`, `stream_alignment`,
//! `stream_data`, `stream_size`, `stream_capacity`, `stream_stride_block`,
//! `stream_stride_delta`, `stream_read_bit`, `stream_write_bit`,
//! `stream_read_bits`, `stream_write_bits`, `stream_rtell`, `stream_wtell`,
//! `stream_rewind`, `stream_rseek`, `stream_wseek`, `stream_skip`, `stream_pad`,
//! `stream_align`, `stream_flush`, `stream_copy`.

use crate::abi::{bitstream, bitstream_count, bitstream_offset, bitstream_size, uint, uint64};
use crate::util::{is_bitstream_mut_null, is_bitstream_null};
use std::ops::{Deref, DerefMut};
use zfp_rs::{
    STREAM_WORD_BITS, ZfpBitStream, ZfpBitStreamMutOps, ZfpBitStreamOps, ZfpBitStreamRefMut,
    ZfpConfig, ZfpExecution, ZfpField, ZfpFieldMut, ZfpHeader, ZfpHeaderError, ZfpHeaderMask,
};

pub(crate) enum ZfpBitStreamHandleInner {
    Owned(ZfpBitStream),
    BorrowedMut(ZfpBitStreamRefMut<'static>),
}

impl ZfpBitStreamHandleInner {
    pub(crate) fn as_ops(&self) -> &(dyn ZfpBitStreamOps + 'static) {
        match self {
            Self::Owned(bs) => bs,
            Self::BorrowedMut(bs) => bs,
        }
    }

    pub(crate) fn as_ops_mut(&mut self) -> &mut (dyn ZfpBitStreamMutOps + 'static) {
        match self {
            Self::Owned(bs) => bs,
            Self::BorrowedMut(bs) => bs,
        }
    }

    pub(crate) fn compress(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
    ) -> Result<usize, zfp_rs::ZfpCompressionError> {
        zfp_rs::compress_bitstream(self.as_ops_mut(), field, config)
    }

    pub(crate) fn decompress(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
    ) -> Result<usize, zfp_rs::ZfpDecompressionError> {
        zfp_rs::decompress_bitstream(self.as_ops_mut(), field, config)
    }

    pub(crate) fn compress_with_execution(
        &mut self,
        config: &ZfpConfig,
        field: &ZfpField,
        execution: ZfpExecution,
    ) -> Result<usize, zfp_rs::ZfpCompressionError> {
        match self {
            Self::Owned(bs) => bs.compress_with_execution(config, field, execution),
            Self::BorrowedMut(bs) => bs.compress_with_execution(config, field, execution),
        }
    }

    pub(crate) fn decompress_with_execution(
        &mut self,
        config: &ZfpConfig,
        field: &mut ZfpFieldMut,
        execution: ZfpExecution,
    ) -> Result<usize, zfp_rs::ZfpDecompressionError> {
        match self {
            Self::Owned(bs) => bs.decompress_with_execution(config, field, execution),
            Self::BorrowedMut(bs) => bs.decompress_with_execution(config, field, execution),
        }
    }

    pub(crate) fn read_header(&mut self, mask: ZfpHeaderMask) -> Result<ZfpHeader, ZfpHeaderError> {
        zfp_rs::read_header_bitstream(self.as_ops_mut(), mask)
    }

    fn clone_copy(&self) -> Self {
        // Deep copy of the committed bytes, matching C `stream_clone`.
        Self::Owned(ZfpBitStream::from_bytes(self.as_ops().as_bytes()))
    }
}

impl Deref for ZfpBitStreamHandleInner {
    type Target = dyn ZfpBitStreamMutOps;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(bs) => bs,
            Self::BorrowedMut(bs) => bs,
        }
    }
}

impl DerefMut for ZfpBitStreamHandleInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_ops_mut()
    }
}

/// Internal wrapper for bitstream lifetime management.
pub(crate) struct ZfpBitStreamHandle {
    pub(crate) inner: ZfpBitStreamHandleInner,
}

/// Look up the wrapper for a bitstream pointer by address.
///
/// # Safety
/// `ptr` must be a valid handle returned by `stream_open` or `stream_clone`.
/// The caller must ensure the entry has not been unregistered.
pub(crate) unsafe fn get_handle<'a>(ptr: *const bitstream) -> Option<&'a ZfpBitStreamHandle> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr.cast::<ZfpBitStreamHandle>() })
    }
}

/// Look up the mutable wrapper for a bitstream pointer by address.
///
/// # Safety
/// `ptr` must be a valid handle returned by `stream_open` or `stream_clone`.
/// The caller must ensure the entry has not been unregistered.
pub(crate) unsafe fn get_handle_mut<'a>(ptr: *mut bitstream) -> Option<&'a mut ZfpBitStreamHandle> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &mut *ptr.cast::<ZfpBitStreamHandle>() })
    }
}

// ===========================================================================
// stream_open
// ===========================================================================

/// Open a new bitstream with the given byte capacity.
///
/// Returns a pointer to the new bitstream, or null on allocation failure.
///
/// # Safety
///
/// - `buffer` may be null, indicating the bitstream should allocate its own
///   backing storage.
/// - If `buffer` is non-null, the caller must ensure:
///   - `buffer` points to a valid allocation of at least `bytes` bytes.
///   - The allocation remains valid and unaccessed by other code until the
///     returned bitstream handle is destroyed via [`stream_close`].
///   - If `buffer` is properly aligned to [`size_of::<ZfpBitStreamWord>`],
///     the bitstream will write directly to it; otherwise a private copy is
///     used and `buffer` is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_open(
    buffer: *mut std::os::raw::c_void,
    bytes: usize,
) -> *mut bitstream {
    let inner = if buffer.is_null() {
        ZfpBitStreamHandleInner::Owned(ZfpBitStream::new(bytes))
    } else if !(buffer as usize).is_multiple_of(align_of::<zfp_rs::ZfpBitStreamWord>()) {
        let slice = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), bytes) };
        ZfpBitStreamHandleInner::Owned(ZfpBitStream::from_bytes(slice))
    } else {
        let words_len = bytes / size_of::<zfp_rs::ZfpBitStreamWord>();
        // SAFETY: this C API mirrors zfp's `stream_open`: the caller owns the
        // buffer and must keep it valid until `stream_close`.
        let words = unsafe {
            std::slice::from_raw_parts_mut(buffer.cast::<zfp_rs::ZfpBitStreamWord>(), words_len)
        };
        let bs = ZfpBitStreamRefMut::from_words_mut(words);
        // SAFETY: the caller owns the backing buffer and guarantees it lives
        // for the lifetime of the returned handle (until `stream_close`).
        let bs: ZfpBitStreamRefMut<'static> = unsafe { std::mem::transmute(bs) };
        ZfpBitStreamHandleInner::BorrowedMut(bs)
    };

    Box::into_raw(Box::new(ZfpBitStreamHandle { inner })).cast::<bitstream>()
}

// ===========================================================================
// stream_close
// ===========================================================================

/// Close and free a bitstream.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open` or `stream_clone`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_close(stream: *mut bitstream) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }
    drop(Box::from_raw(stream.cast::<ZfpBitStreamHandle>()));
}

// ===========================================================================
// stream_clone
// ===========================================================================

/// Clone a bitstream by copying all committed bytes.
///
/// Returns a pointer to the new bitstream.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_clone(stream: *const bitstream) -> *mut bitstream {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return std::ptr::null_mut();
    }

    let Some(wrapper) = get_handle(stream) else {
        return std::ptr::null_mut();
    };

    Box::into_raw(Box::new(ZfpBitStreamHandle {
        inner: wrapper.inner.clone_copy(),
    }))
    .cast::<bitstream>()
}

// ===========================================================================
// stream_alignment
// ===========================================================================

/// Return the required alignment for bitstream operations.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_alignment() -> bitstream_count {
    (STREAM_WORD_BITS / 8) as usize
}

// ===========================================================================
// stream_data
// ===========================================================================

/// Return a pointer to the bitstream's data buffer.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_data(stream: *const bitstream) -> *mut std::os::raw::c_void {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return std::ptr::null_mut();
    }

    let Some(wrapper) = get_handle(stream) else {
        return std::ptr::null_mut();
    };

    wrapper.inner.data_ptr()
}

// ===========================================================================
// stream_size
// ===========================================================================

/// Return the committed byte size of the bitstream.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_size(stream: *const bitstream) -> usize {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle(stream) else {
        return 0;
    };

    wrapper.inner.size()
}

// ===========================================================================
// stream_capacity
// ===========================================================================

/// Return the total byte capacity of the bitstream.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_capacity(stream: *const bitstream) -> usize {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle(stream) else {
        return 0;
    };

    wrapper.inner.capacity()
}

// ===========================================================================
// stream_stride_block
// ===========================================================================

/// Return the stride block size in bytes (always 8 for 64-bit words).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_stride_block(_stream: *const bitstream) -> usize {
    8
}

// ===========================================================================
// stream_stride_delta
// ===========================================================================

/// Return the stride delta between blocks (always -8).
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_stride_delta(_stream: *const bitstream) -> isize {
    -8
}

// ===========================================================================
// stream_read_bit
// ===========================================================================

/// Read a single bit from the bitstream.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_read_bit(stream: *mut bitstream) -> uint {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    wrapper.inner.read_bit() as uint
}

// ===========================================================================
// stream_write_bit
// ===========================================================================

/// Write a single bit to the bitstream.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_write_bit(stream: *mut bitstream, bit: uint) -> uint {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    wrapper.inner.write_bit(bit)
}

// ===========================================================================
// stream_read_bits
// ===========================================================================

/// Read `n` bits from the bitstream.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_read_bits(stream: *mut bitstream, n: bitstream_count) -> uint64 {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    #[allow(clippy::cast_possible_truncation)]
    // FFI: bitstream_count is bounded to u16 range by the C API contract.
    wrapper.inner.read_bits(n as u32)
}

// ===========================================================================
// stream_write_bits
// ===========================================================================

/// Write the low `n` bits of `value` to the bitstream.
/// Returns the overflow (bits above `n`).
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_write_bits(
    stream: *mut bitstream,
    value: uint64,
    n: bitstream_count,
) -> uint64 {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    #[allow(clippy::cast_possible_truncation)]
    // FFI: bitstream_count is bounded to u16 range by the C API contract.
    wrapper.inner.write_bits(value, n as u32)
}

// ===========================================================================
// stream_rtell
// ===========================================================================

/// Return the current read bit offset.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_rtell(stream: *const bitstream) -> bitstream_offset {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle(stream) else {
        return 0;
    };

    wrapper.inner.read_pos()
}

// ===========================================================================
// stream_wtell
// ===========================================================================

/// Return the current write bit offset.
///
/// # Safety
/// `stream` must be a valid pointer returned by `stream_open`.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn stream_wtell(stream: *const bitstream) -> bitstream_offset {
    if is_bitstream_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle(stream) else {
        return 0;
    };

    wrapper.inner.write_pos()
}

// ===========================================================================
// stream_rewind
// ===========================================================================

/// Rewind the bitstream to the beginning.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_rewind(stream: *mut bitstream) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return;
    };

    wrapper.inner.rewind();
}

// ===========================================================================
// stream_rseek
// ===========================================================================

/// Position the read cursor at `offset` bits.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_rseek(stream: *mut bitstream, offset: bitstream_offset) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return;
    };

    wrapper.inner.seek_read(offset);
}

// ===========================================================================
// stream_wseek
// ===========================================================================

/// Position the write cursor at `offset` bits.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_wseek(stream: *mut bitstream, offset: bitstream_offset) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return;
    };

    wrapper.inner.seek_write(offset);
}

// ===========================================================================
// stream_skip
// ===========================================================================

/// Skip `n` bits forward in the read cursor.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_skip(stream: *mut bitstream, n: bitstream_size) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return;
    };

    #[allow(clippy::cast_possible_truncation)]
    // FFI: bitstream_size is bounded to u32 range by the C API contract.
    wrapper.inner.skip(n as usize);
}

// ===========================================================================
// stream_pad
// ===========================================================================

/// Append `n` zero-bits to the write stream.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_pad(stream: *mut bitstream, n: bitstream_size) {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return;
    };

    #[allow(clippy::cast_possible_truncation)]
    // FFI: bitstream_size is bounded to u32 range by the C API contract.
    wrapper.inner.pad(n as usize);
}

// ===========================================================================
// stream_align
// ===========================================================================

/// Discard buffered read bits and align to the next word boundary.
/// Returns the number of bits discarded.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_align(stream: *mut bitstream) -> bitstream_count {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    wrapper.inner.align() as bitstream_count
}

// ===========================================================================
// stream_flush
// ===========================================================================

/// Flush the write buffer to the next word boundary.
/// Returns the number of padding bits written.
///
/// # Safety
/// `stream` must be a valid mutable pointer returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_flush(stream: *mut bitstream) -> bitstream_count {
    if is_bitstream_mut_null(stream) == crate::abi::zfp_false {
        return 0;
    }

    let Some(wrapper) = get_handle_mut(stream) else {
        return 0;
    };

    wrapper.inner.flush() as bitstream_count
}

// ===========================================================================
// stream_copy
// ===========================================================================

/// Copy `n` bits from `src` into `dst`.
///
/// # Safety
/// Both `dst` and `src` must be valid mutable pointers returned by `stream_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stream_copy(dst: *mut bitstream, src: *mut bitstream, n: bitstream_size) {
    if is_bitstream_mut_null(dst) == crate::abi::zfp_false
        || is_bitstream_mut_null(src) == crate::abi::zfp_false
    {
        return;
    }

    let Some(dst_wrapper) = get_handle_mut(dst) else {
        return;
    };
    let Some(src_wrapper) = get_handle_mut(src) else {
        return;
    };

    #[allow(clippy::cast_possible_truncation)]
    // FFI: bitstream_size is bounded to u32 range by the C API contract.
    dst_wrapper
        .inner
        .copy_from(src_wrapper.inner.as_ops_mut(), n as usize);
}
