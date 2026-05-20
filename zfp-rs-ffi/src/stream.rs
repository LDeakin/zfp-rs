//! Stream API: C-level wrappers around `zfp_rs::ZfpConfig`.

use crate::abi::{
    bitstream, uint, uint64, zfp_bool, zfp_exec_policy, zfp_false, zfp_field, zfp_mode, zfp_stream,
    zfp_true, zfp_type,
};
use crate::util::c_dims_to_rust;
use zfp_rs::{
    STREAM_WORD_BITS, ZfpConfig, ZfpExecution, ZfpHeaderMask, ZfpScalarType, ZfpStreamAlignment,
};

fn params_from_c(stream: &zfp_stream) -> (uint, uint, uint, i32) {
    (
        stream.minbits,
        stream.maxbits,
        stream.maxprec,
        stream.minexp,
    )
}

fn write_params(stream: &mut zfp_stream, zfp: &ZfpConfig) {
    stream.minbits = zfp.min_bits();
    stream.maxbits = zfp.max_bits();
    stream.maxprec = zfp.max_prec();
    stream.minexp = zfp.min_exp();
}

fn stream_with_c_state(stream: &zfp_stream) -> ZfpConfig {
    ZfpConfig::expert(
        stream.minbits,
        stream.maxbits,
        stream.maxprec,
        stream.minexp,
    )
}

pub(crate) unsafe fn stream_params(
    stream: *const zfp_stream,
) -> Option<(u32, u32, u32, i32, *mut bitstream)> {
    if stream.is_null() {
        return None;
    }
    let stream = unsafe { &*stream };
    Some((
        stream.minbits,
        stream.maxbits,
        stream.maxprec,
        stream.minexp,
        stream.stream,
    ))
}

fn header_mask(mask: uint) -> ZfpHeaderMask {
    ZfpHeaderMask::from_bits_truncate(mask)
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_type_size(ty: zfp_type) -> usize {
    match ty {
        crate::abi::zfp_type_zfp_type_int32 | crate::abi::zfp_type_zfp_type_float => 4,
        crate::abi::zfp_type_zfp_type_int64 | crate::abi::zfp_type_zfp_type_double => 8,
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_open(bs: *mut bitstream) -> *mut zfp_stream {
    let zfp = ZfpConfig::new();
    let mut stream = zfp_stream {
        stream: bs,
        exec: crate::abi::zfp_execution::default(),
        ..zfp_stream::default()
    };
    write_params(&mut stream, &zfp);
    Box::into_raw(Box::new(stream))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_close(stream: *mut zfp_stream) {
    if !stream.is_null() {
        unsafe {
            free_omp_params(&mut *stream);
            drop(Box::from_raw(stream));
        }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_bit_stream(stream: *const zfp_stream) -> *mut bitstream {
    if stream.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*stream).stream }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_compression_mode(stream: *const zfp_stream) -> zfp_mode {
    if stream.is_null() {
        return zfp_mode::zfp_mode_null;
    }
    let (minbits, maxbits, maxprec, minexp) = params_from_c(unsafe { &*stream });
    crate::util::rust_mode_to_zfp(zfp_rs::compression_mode_from_params(
        minbits, maxbits, maxprec, minexp,
    ))
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_rate(stream: *const zfp_stream, dims: uint) -> f64 {
    if stream.is_null() {
        return 0.0;
    }
    let (minbits, maxbits, maxprec, minexp) = params_from_c(unsafe { &*stream });
    match c_dims_to_rust(dims) {
        Some(d) => zfp_rs::rate_from_params(minbits, maxbits, maxprec, minexp, d),
        None => 0.0,
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_precision(stream: *const zfp_stream) -> uint {
    if stream.is_null() {
        0
    } else {
        let (minbits, maxbits, maxprec, minexp) = params_from_c(unsafe { &*stream });
        zfp_rs::precision_from_params(minbits, maxbits, maxprec, minexp)
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_accuracy(stream: *const zfp_stream) -> f64 {
    if stream.is_null() {
        0.0
    } else {
        let (minbits, maxbits, maxprec, minexp) = params_from_c(unsafe { &*stream });
        zfp_rs::accuracy_from_params(minbits, maxbits, maxprec, minexp)
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_mode(stream: *const zfp_stream) -> uint64 {
    if stream.is_null() {
        0
    } else {
        let (minbits, maxbits, maxprec, minexp) = params_from_c(unsafe { &*stream });
        zfp_rs::mode_bits_from_params(minbits, maxbits, maxprec, minexp)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_params(
    stream: *const zfp_stream,
    minbits: *mut uint,
    maxbits: *mut uint,
    maxprec: *mut uint,
    minexp: *mut i32,
) {
    if stream.is_null()
        || minbits.is_null()
        || maxbits.is_null()
        || maxprec.is_null()
        || minexp.is_null()
    {
        return;
    }
    let stream = unsafe { &*stream };
    unsafe {
        *minbits = stream.minbits;
        *maxbits = stream.maxbits;
        *maxprec = stream.maxprec;
        *minexp = stream.minexp;
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_compressed_size(stream: *const zfp_stream) -> usize {
    if stream.is_null() {
        return 0;
    }
    let bs = unsafe { (*stream).stream };
    match unsafe { crate::bitstream_api::get_handle(bs) } {
        Some(handle) => handle.inner.size(),
        None => 0,
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_maximum_size(
    stream: *const zfp_stream,
    field: *const zfp_field,
) -> usize {
    if stream.is_null() || field.is_null() {
        return 0;
    }
    let zfp = stream_with_c_state(unsafe { &*stream });
    let field = unsafe { &*field };
    let Some(ty) = crate::util::zfp_type_to_rust_type(field.r#type) else {
        return 0;
    };
    let all_dims = [field.nx, field.ny, field.nz, field.nw];
    // Determine actual dimensionality (1–4) by counting non-zero trailing dims.
    // This matches C zfp's `zfp_field_dimensionality` semantics.
    let dims = match (&all_dims[1..], &all_dims[2..], &all_dims[3..]) {
        ([0, 0, 0], ..) => &all_dims[..1],
        ([_, 0, 0], ..) => &all_dims[..2],
        ([_, _, 0], ..) => &all_dims[..3],
        _ => &all_dims[..4],
    };
    zfp.maximum_size(ty, dims)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_write_header(
    stream: *mut zfp_stream,
    field: *const zfp_field,
    mask: uint,
) -> usize {
    if stream.is_null() || field.is_null() {
        return 0;
    }
    let stream_ref = unsafe { &*stream };
    let bs = match unsafe { crate::bitstream_api::get_handle_mut(stream_ref.stream) } {
        Some(handle) => &mut handle.inner,
        None => return 0,
    };
    let rust_stream = stream_with_c_state(stream_ref);
    let Some(rust_field) = crate::field::field_to_rust(unsafe { &*field }) else {
        return 0;
    };
    let mask = header_mask(mask);
    let mut bits = 0usize;

    if mask.contains(ZfpHeaderMask::MAGIC) {
        bs.write_bits(u64::from(b'z'), 8);
        bs.write_bits(u64::from(b'f'), 8);
        bs.write_bits(u64::from(b'p'), 8);
        bs.write_bits(u64::from(crate::header::ZFP_CODEC_VERSION), 8);
        bits += zfp_rs::ZFP_MAGIC_BITS as usize;
    }
    if mask.contains(ZfpHeaderMask::META) {
        let Ok(meta) = rust_field.metadata() else {
            return 0;
        };
        bs.write_bits(meta, zfp_rs::ZFP_META_BITS);
        bits += zfp_rs::ZFP_META_BITS as usize;
    }
    if mask.contains(ZfpHeaderMask::MODE) {
        let mode = rust_stream.mode_bits();
        let size = if mode > u64::from(crate::abi::ZFP_MODE_SHORT_MAX) {
            zfp_rs::ZFP_MODE_LONG_BITS
        } else {
            zfp_rs::ZFP_MODE_SHORT_BITS
        };
        bs.write_bits(mode, size);
        bits += size as usize;
    }

    bits
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_read_header(
    stream: *mut zfp_stream,
    field: *mut zfp_field,
    mask: uint,
) -> usize {
    if stream.is_null() || field.is_null() {
        return 0;
    }

    let stream_ref = unsafe { &mut *stream };
    let bs = match unsafe { crate::bitstream_api::get_handle_mut(stream_ref.stream) } {
        Some(handle) => &mut handle.inner,
        None => return 0,
    };

    let mask = header_mask(mask);

    match bs.read_header(mask) {
        Ok(header) => {
            if let Some(metadata) = header.metadata {
                crate::field::write_header_metadata(unsafe { &mut *field }, metadata);
            }
            if let Some(config) = header.config {
                write_params(stream_ref, &config);
            }
            header.bits_read
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_compress(stream: *mut zfp_stream, field: *const zfp_field) -> usize {
    if stream.is_null() || field.is_null() {
        return 0;
    }
    let stream_ref = unsafe { &*stream };
    let Some(handle) = (unsafe { crate::bitstream_api::get_handle_mut((*stream).stream) }) else {
        return 0;
    };
    let Some(rust_field) = crate::field::field_to_rust(unsafe { &*field }) else {
        return 0;
    };
    if rust_field.data().is_empty() {
        return 0;
    }

    let rust_config = stream_with_c_state(stream_ref);
    let execution = stream_execution(stream_ref);
    let bytes = handle
        .inner
        .compress_with_execution(&rust_config, &rust_field, execution)
        .unwrap_or(0);
    write_params(unsafe { &mut *stream }, &rust_config);
    bytes
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_decompress(stream: *mut zfp_stream, field: *mut zfp_field) -> usize {
    if stream.is_null() || field.is_null() {
        return 0;
    }
    if crate::util::zfp_type_to_scalar(unsafe { (*field).r#type }).is_none() {
        return 0;
    }

    let stream_ref = unsafe { &*stream };
    let Some(handle) = (unsafe { crate::bitstream_api::get_handle_mut((*stream).stream) }) else {
        return 0;
    };
    let Some(mut rust_field) = (unsafe { crate::field::field_mut_to_rust(&mut *field) }) else {
        return 0;
    };

    let rust_config = stream_with_c_state(stream_ref);
    let execution = stream_execution(stream_ref);
    let bytes = handle
        .inner
        .decompress_with_execution(&rust_config, &mut rust_field, execution)
        .unwrap_or(0);
    write_params(unsafe { &mut *stream }, &rust_config);
    bytes
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_rewind(stream: *mut zfp_stream) {
    if stream.is_null() {
        return;
    }
    let bs = unsafe { (*stream).stream };
    if let Some(handle) = unsafe { crate::bitstream_api::get_handle_mut(bs) } {
        handle.inner.rewind();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_bit_stream(stream: *mut zfp_stream, bs: *mut bitstream) {
    if !stream.is_null() {
        unsafe {
            (*stream).stream = bs;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_reversible(stream: *mut zfp_stream) {
    if stream.is_null() {
        return;
    }
    let stream = unsafe { &mut *stream };
    let zfp = ZfpConfig::reversible();
    write_params(stream, &zfp);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_rate(
    stream: *mut zfp_stream,
    rate: f64,
    ty: zfp_type,
    dims: uint,
    align: zfp_bool,
) -> f64 {
    if stream.is_null() {
        return 0.0;
    }
    let Some(rust_dims) = c_dims_to_rust(dims) else {
        return 0.0;
    };
    let rust_ty = crate::util::zfp_type_to_rust_type(ty);
    let Some(rust_ty) = rust_ty else { return 0.0 };

    let stream = unsafe { &mut *stream };
    let n = 1u32 << (2 * u32::from(rust_dims));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Rate is positive and n fits in u32, so this cast is safe.
    let mut bits = (f64::from(n) * rate + 0.5).floor() as u32;
    match rust_ty {
        ZfpScalarType::Float if bits < 1 + 8 => {
            bits = 1 + 8;
        }
        ZfpScalarType::Double if bits < 1 + 11 => {
            bits = 1 + 11;
        }
        _ => {}
    }
    let align = align == zfp_true;
    if align {
        bits = bits.next_multiple_of(STREAM_WORD_BITS);
    }
    let stream_align = if align {
        ZfpStreamAlignment::WordAligned
    } else {
        ZfpStreamAlignment::None
    };
    let zfp = ZfpConfig::fixed_rate(rate, rust_ty, rust_dims, stream_align);
    write_params(stream, &zfp);
    f64::from(bits) / f64::from(n)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_precision(
    stream: *mut zfp_stream,
    precision: uint,
) -> uint {
    if stream.is_null() {
        return 0;
    }
    let stream = unsafe { &mut *stream };
    // precision is already u32
    let zfp = ZfpConfig::fixed_precision(precision);
    write_params(stream, &zfp);
    zfp.precision()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_accuracy(stream: *mut zfp_stream, tolerance: f64) -> f64 {
    if stream.is_null() {
        return 0.0;
    }
    let stream = unsafe { &mut *stream };
    let zfp = ZfpConfig::fixed_accuracy(tolerance);
    write_params(stream, &zfp);
    zfp.accuracy()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_mode(stream: *mut zfp_stream, mode: uint64) -> zfp_mode {
    if stream.is_null() {
        return zfp_mode::zfp_mode_null;
    }
    let stream = unsafe { &mut *stream };
    let Some(zfp) = ZfpConfig::from_mode(mode) else {
        write_params(stream, &stream_with_c_state(stream));
        return zfp_mode::zfp_mode_null;
    };
    write_params(stream, &zfp);
    crate::util::rust_mode_to_zfp(zfp.compression_mode())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_params(
    stream: *mut zfp_stream,
    minbits: uint,
    maxbits: uint,
    maxprec: uint,
    minexp: i32,
) -> zfp_bool {
    if stream.is_null() {
        return zfp_false;
    }
    if minbits > maxbits || !(0 < maxprec && maxprec <= 64) {
        return zfp_false;
    }
    let zfp = ZfpConfig::expert(minbits, maxbits, maxprec, minexp);
    write_params(unsafe { &mut *stream }, &zfp);
    zfp_true
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_execution(stream: *const zfp_stream) -> zfp_exec_policy {
    if stream.is_null() {
        zfp_exec_policy::zfp_exec_serial
    } else {
        unsafe { (*stream).exec.policy }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_omp_threads(stream: *const zfp_stream) -> uint {
    if stream.is_null() {
        return 0;
    }
    let params_ptr = unsafe { (*stream).exec.params };
    if params_ptr.is_null() {
        return 0;
    }
    unsafe { (*(params_ptr as *const crate::abi::zfp_exec_params_omp)).threads }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_stream_omp_chunk_size(stream: *const zfp_stream) -> uint {
    if stream.is_null() {
        return 0;
    }
    let params_ptr = unsafe { (*stream).exec.params };
    if params_ptr.is_null() {
        return 0;
    }
    unsafe { (*(params_ptr as *const crate::abi::zfp_exec_params_omp)).chunk_size }
}

/// Helper: free OMP params associated with a stream.
#[allow(clippy::ptr_as_ptr)]
unsafe fn free_omp_params(stream: &mut zfp_stream) {
    let ptr = stream.exec.params;
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr.cast::<crate::abi::zfp_exec_params_omp>()));
        }
        stream.exec.params = std::ptr::null_mut();
    }
}

/// Helper: get [`ZfpExecution`] from the C stream.
#[allow(clippy::if_not_else, clippy::ptr_as_ptr)]
fn stream_execution(stream: &zfp_stream) -> ZfpExecution {
    match stream.exec.policy {
        zfp_exec_policy::zfp_exec_omp => {
            // SAFETY: exec.params points to a valid zfp_exec_params_omp.
            if !stream.exec.params.is_null() {
                let params =
                    unsafe { *(stream.exec.params as *const crate::abi::zfp_exec_params_omp) };
                ZfpExecution::Rayon {
                    threads: params.threads,
                    chunk_size: params.chunk_size,
                }
            } else {
                ZfpExecution::Rayon {
                    threads: 0,
                    chunk_size: 0,
                }
            }
        }
        // zfp_exec_serial and zfp_exec_cuda (unsupported) both map to Serial.
        _ => ZfpExecution::Serial,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_set_execution(
    stream: *mut zfp_stream,
    policy: zfp_exec_policy,
) -> zfp_bool {
    if stream.is_null() {
        return zfp_false;
    }
    let stream = unsafe { &mut *stream };
    match policy {
        zfp_exec_policy::zfp_exec_serial => {
            free_omp_params(stream);
            stream.exec.policy = zfp_exec_policy::zfp_exec_serial;
            zfp_true
        }
        #[cfg(feature = "rayon")]
        zfp_exec_policy::zfp_exec_omp => {
            stream.exec.policy = zfp_exec_policy::zfp_exec_omp;
            zfp_true
        }
        #[cfg(not(feature = "rayon"))]
        zfp_exec_policy::zfp_exec_omp => zfp_false,
        zfp_exec_policy::zfp_exec_cuda => zfp_false,
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::ptr_as_ptr)]
pub unsafe extern "C" fn zfp_stream_set_omp_threads(
    stream: *mut zfp_stream,
    threads: uint,
) -> zfp_bool {
    if stream.is_null() {
        return zfp_false;
    }
    #[cfg(feature = "rayon")]
    {
        let stream = unsafe { &mut *stream };
        free_omp_params(stream);
        let params = Box::new(crate::abi::zfp_exec_params_omp {
            threads,
            chunk_size: 0,
        });
        stream.exec.params = Box::into_raw(params).cast::<std::os::raw::c_void>();
        stream.exec.policy = zfp_exec_policy::zfp_exec_omp;
        zfp_true
    }
    #[cfg(not(feature = "rayon"))]
    {
        let _ = threads; // silence unused warning
        zfp_false
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::ptr_as_ptr)]
pub unsafe extern "C" fn zfp_stream_set_omp_chunk_size(
    stream: *mut zfp_stream,
    chunk_size: uint,
) -> zfp_bool {
    if stream.is_null() {
        return zfp_false;
    }
    #[cfg(feature = "rayon")]
    {
        let stream = unsafe { &mut *stream };
        let params_ptr = stream.exec.params as *mut crate::abi::zfp_exec_params_omp;
        if params_ptr.is_null() {
            // Create params if none exist
            let params = Box::new(crate::abi::zfp_exec_params_omp {
                threads: 0,
                chunk_size,
            });
            stream.exec.params = Box::into_raw(params).cast::<std::os::raw::c_void>();
        } else {
            unsafe {
                (*params_ptr).chunk_size = chunk_size;
            }
        }
        stream.exec.policy = zfp_exec_policy::zfp_exec_omp;
        zfp_true
    }
    #[cfg(not(feature = "rayon"))]
    {
        let _ = chunk_size; // silence unused warning
        zfp_false
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_flush(stream: *mut zfp_stream) -> usize {
    if stream.is_null() {
        return 0;
    }
    let bs = unsafe { (*stream).stream };
    match unsafe { crate::bitstream_api::get_handle_mut(bs) } {
        Some(handle) => handle.inner.flush(),
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_stream_align(stream: *mut zfp_stream) -> usize {
    if stream.is_null() {
        return 0;
    }
    let bs = unsafe { (*stream).stream };
    match unsafe { crate::bitstream_api::get_handle_mut(bs) } {
        Some(handle) => handle.inner.align() as usize,
        None => 0,
    }
}
