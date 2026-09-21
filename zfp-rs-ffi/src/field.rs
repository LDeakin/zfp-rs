//! Field API: C-level wrappers around `zfp_rs::ZfpField`.

use crate::abi::{
    uint, uint64, zfp_bool, zfp_false, zfp_field, zfp_true, zfp_type, zfp_type_zfp_type_int32,
    zfp_type_zfp_type_none,
};
use zfp_rs::{ZfpField, ZfpFieldMetadata};

fn alloc_field(field: zfp_field) -> *mut zfp_field {
    Box::into_raw(Box::new(field))
}

fn field_with(
    data: *mut std::ffi::c_void,
    scalar_type: zfp_type,
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
) -> zfp_field {
    zfp_field {
        r#type: scalar_type,
        nx,
        ny,
        nz,
        nw,
        sx: 0,
        sy: 0,
        sz: 0,
        sw: 0,
        data,
    }
}

pub(crate) fn active_dims(field: &zfp_field) -> [usize; 4] {
    [field.nx, field.ny, field.nz, field.nw]
}

pub(crate) fn active_strides(field: &zfp_field) -> [isize; 4] {
    [field.sx, field.sy, field.sz, field.sw]
}

pub(crate) fn field_to_rust(field: &zfp_field) -> Option<ZfpField<'static>> {
    let scalar_type = crate::util::zfp_type_to_rust_type(field.r#type)?;
    let dims = active_dims(field);
    let strides = active_strides(field);
    if field.data.is_null() {
        // SAFETY: a null pointer with a zero byte count yields an empty field.
        return Some(unsafe {
            ZfpField::from_raw(std::ptr::null(), 0, scalar_type, dims, strides)
        });
    }
    let (begin, byte_count) = span_of(field.data.cast_const(), scalar_type, &dims, &strides);
    // SAFETY: the C caller guarantees that the field data pointer spans
    // the descriptor's memory footprint.
    Some(unsafe { ZfpField::from_raw(begin, byte_count, scalar_type, dims, strides) })
}

/// Translate a C `zfp_field.data` pointer into the `(begin, byte_count)` pair
/// the Rust constructors take.
///
/// The two APIs anchor a strided field at opposite ends of its span: C's
/// `data` points at the element with index `[0, 0, 0, 0]`, so with a negative
/// stride the span runs *backwards* from it, while `ZfpField`/`ZfpFieldMut`
/// take a buffer that starts at the span's lowest address. Shifting by `imin`
/// converts one to the other; without it the constructed slice would claim
/// `-imin` elements past the end of the caller's buffer.
fn span_of(
    data: *const std::ffi::c_void,
    scalar_type: zfp_rs::types::ZfpScalarType,
    dims: &[usize; 4],
    strides: &[isize; 4],
) -> (*const u8, usize) {
    let (imin, imax) = ZfpField::field_index_span_static(dims, strides);
    let elem_size = scalar_type.size();
    let byte_count = (imax - imin + 1).cast_unsigned() * elem_size;
    // SAFETY: `imin <= 0` is the lowest element index the strides reach from
    // `data`, which the C caller guarantees is inside its buffer.
    let begin = unsafe { data.cast::<u8>().offset(imin * elem_size.cast_signed()) };
    (begin, byte_count)
}

pub(crate) unsafe fn field_mut_to_rust(
    field: &mut zfp_field,
) -> Option<zfp_rs::ZfpFieldMut<'static>> {
    let scalar_type = crate::util::zfp_type_to_scalar(field.r#type)?;
    let dims = active_dims(field);
    let strides = active_strides(field);
    if field.data.is_null() {
        return None;
    }
    let (begin, byte_count) = span_of(field.data.cast_const(), scalar_type, &dims, &strides);
    if byte_count == 0 {
        return None;
    }
    // SAFETY: the C caller guarantees that the field data pointer spans
    // the descriptor's memory footprint.
    Some(zfp_rs::ZfpFieldMut::from_raw(
        begin.cast_mut(),
        byte_count,
        scalar_type,
        dims,
        strides,
    ))
}

pub(crate) fn write_header_metadata(dst: &mut zfp_field, metadata: zfp_rs::ZfpFieldMetadata) {
    dst.r#type = crate::util::rust_type_to_zfp_type(metadata.scalar_type);
    dst.nx = metadata.dims[0];
    dst.ny = metadata.dims[1];
    dst.nz = metadata.dims[2];
    dst.nw = metadata.dims[3];
    dst.sx = 0;
    dst.sy = 0;
    dst.sz = 0;
    dst.sw = 0;
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_alloc() -> *mut zfp_field {
    alloc_field(zfp_field::default())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_1d(
    data: *mut std::ffi::c_void,
    scalar_type: zfp_type,
    nx: usize,
) -> *mut zfp_field {
    alloc_field(field_with(data, scalar_type, nx, 0, 0, 0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_2d(
    data: *mut std::ffi::c_void,
    scalar_type: zfp_type,
    nx: usize,
    ny: usize,
) -> *mut zfp_field {
    alloc_field(field_with(data, scalar_type, nx, ny, 0, 0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_3d(
    data: *mut std::ffi::c_void,
    scalar_type: zfp_type,
    nx: usize,
    ny: usize,
    nz: usize,
) -> *mut zfp_field {
    alloc_field(field_with(data, scalar_type, nx, ny, nz, 0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_4d(
    data: *mut std::ffi::c_void,
    scalar_type: zfp_type,
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
) -> *mut zfp_field {
    alloc_field(field_with(data, scalar_type, nx, ny, nz, nw))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_free(field: *mut zfp_field) {
    if !field.is_null() {
        unsafe {
            drop(Box::from_raw(field));
        }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_pointer(field: *const zfp_field) -> *mut std::ffi::c_void {
    if field.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*field).data }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_begin(field: *const zfp_field) -> *mut std::ffi::c_void {
    if field.is_null() {
        return std::ptr::null_mut();
    }
    let rust = unsafe { field_to_rust(&*field) };
    let Some(rust) = rust else {
        return std::ptr::null_mut();
    };
    rust.begin()
        .map_or(std::ptr::null_mut(), |ptr| ptr as *mut std::ffi::c_void)
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_type(field: *const zfp_field) -> zfp_type {
    if field.is_null() {
        zfp_type_zfp_type_none
    } else {
        unsafe { (*field).r#type }
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_precision(field: *const zfp_field) -> uint {
    if field.is_null() {
        0
    } else {
        field_to_rust(unsafe { &*field }).map_or(0, |field| field.precision() as uint)
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_dimensionality(field: *const zfp_field) -> uint {
    if field.is_null() {
        0
    } else {
        field_to_rust(unsafe { &*field }).map_or(0, |field| field.dimensionality().into())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_size(field: *const zfp_field, dims: *mut usize) -> usize {
    if field.is_null() {
        return 0;
    }
    let size = active_dims(unsafe { &*field });
    if !dims.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(size.as_ptr(), dims, 4);
        }
    }
    field_to_rust(unsafe { &*field }).map_or(0, |field| field.num_elements())
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_size_bytes(field: *const zfp_field) -> usize {
    if field.is_null() {
        0
    } else {
        field_to_rust(unsafe { &*field }).map_or(0, |field| field.size_bytes())
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_blocks(field: *const zfp_field) -> usize {
    if field.is_null() {
        0
    } else {
        field_to_rust(unsafe { &*field }).map_or(0, |field| field.num_blocks())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_stride(
    field: *const zfp_field,
    strides: *mut isize,
) -> zfp_bool {
    if field.is_null() {
        return zfp_false;
    }
    let stride = active_strides(unsafe { &*field });
    if !strides.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(stride.as_ptr(), strides, 4);
        }
    }
    if field_to_rust(unsafe { &*field }).is_some_and(|field| field.is_contiguous()) {
        zfp_false
    } else {
        zfp_true
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_is_contiguous(field: *const zfp_field) -> zfp_bool {
    if field.is_null() {
        zfp_false
    } else if field_to_rust(unsafe { &*field }).is_some_and(|field| field.is_contiguous()) {
        zfp_true
    } else {
        zfp_false
    }
}

#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn zfp_field_metadata(field: *const zfp_field) -> uint64 {
    if field.is_null() {
        0
    } else {
        field_to_rust(unsafe { &*field })
            .map_or(u64::MAX, |field| field.metadata().unwrap_or(u64::MAX))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_pointer(
    field: *mut zfp_field,
    ptr: *mut std::os::raw::c_void,
) {
    if !field.is_null() {
        unsafe {
            (*field).data = ptr;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_type(field: *mut zfp_field, ty: zfp_type) -> zfp_type {
    if field.is_null() {
        return zfp_type_zfp_type_int32;
    }
    if crate::util::zfp_type_to_scalar(ty).is_some() {
        unsafe {
            (*field).r#type = ty;
        }
        ty
    } else {
        zfp_type_zfp_type_none
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_size_1d(field: *mut zfp_field, nx: usize) {
    if !field.is_null() {
        unsafe {
            (*field).nx = nx;
            (*field).ny = 0;
            (*field).nz = 0;
            (*field).nw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_size_2d(field: *mut zfp_field, nx: usize, ny: usize) {
    if !field.is_null() {
        unsafe {
            (*field).nx = nx;
            (*field).ny = ny;
            (*field).nz = 0;
            (*field).nw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_size_3d(
    field: *mut zfp_field,
    nx: usize,
    ny: usize,
    nz: usize,
) {
    if !field.is_null() {
        unsafe {
            (*field).nx = nx;
            (*field).ny = ny;
            (*field).nz = nz;
            (*field).nw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_size_4d(
    field: *mut zfp_field,
    nx: usize,
    ny: usize,
    nz: usize,
    nw: usize,
) {
    if !field.is_null() {
        unsafe {
            (*field).nx = nx;
            (*field).ny = ny;
            (*field).nz = nz;
            (*field).nw = nw;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_stride_1d(field: *mut zfp_field, sx: isize) {
    if !field.is_null() {
        unsafe {
            (*field).sx = sx;
            (*field).sy = 0;
            (*field).sz = 0;
            (*field).sw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_stride_2d(field: *mut zfp_field, sx: isize, sy: isize) {
    if !field.is_null() {
        unsafe {
            (*field).sx = sx;
            (*field).sy = sy;
            (*field).sz = 0;
            (*field).sw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_stride_3d(
    field: *mut zfp_field,
    sx: isize,
    sy: isize,
    sz: isize,
) {
    if !field.is_null() {
        unsafe {
            (*field).sx = sx;
            (*field).sy = sy;
            (*field).sz = sz;
            (*field).sw = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_stride_4d(
    field: *mut zfp_field,
    sx: isize,
    sy: isize,
    sz: isize,
    sw: isize,
) {
    if !field.is_null() {
        unsafe {
            (*field).sx = sx;
            (*field).sy = sy;
            (*field).sz = sz;
            (*field).sw = sw;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zfp_field_set_metadata(
    field: *mut zfp_field,
    metadata: uint64,
) -> zfp_bool {
    if field.is_null() {
        return zfp_false;
    }
    let Some(metadata) = ZfpFieldMetadata::from_bits(metadata) else {
        return zfp_false;
    };
    unsafe {
        write_header_metadata(&mut *field, metadata);
    }
    zfp_true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::zfp_type_zfp_type_double;

    /// `zfp_field_3d(.., 5, 0, 5)` is a 1-D field: `zfp_field_dimensionality`
    /// stops at `ny == 0`, so the codec walks 5 elements from `data` and `nz`
    /// is inert. `span_of` must ignore it too — folding `sz = -100` into the
    /// span would hand `from_raw` a slice starting 3200 bytes *below* the
    /// caller's buffer, which is instant UB even before the codec reads it.
    #[test]
    fn inert_axes_do_not_move_the_span_off_the_callers_buffer() {
        let mut buf = [1.5f64; 5];
        let field = field_with(
            buf.as_mut_ptr().cast(),
            zfp_type_zfp_type_double,
            5,
            0,
            5,
            0,
        );
        let field = zfp_field {
            sx: 1,
            sz: -100,
            ..field
        };

        let (begin, byte_count) = span_of(
            field.data.cast_const(),
            zfp_rs::types::ZfpScalarType::Double,
            &active_dims(&field),
            &active_strides(&field),
        );
        assert_eq!(begin, buf.as_ptr().cast::<u8>());
        assert_eq!(byte_count, std::mem::size_of_val(&buf));

        let rust = field_to_rust(&field).expect("a double field converts");
        assert_eq!(rust.begin().unwrap(), buf.as_ptr().cast::<u8>());
        assert_eq!(rust.size_bytes(), byte_count);
    }
}
