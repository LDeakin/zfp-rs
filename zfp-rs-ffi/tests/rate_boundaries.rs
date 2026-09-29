use zfp_rs_ffi as ffi;

fn params(stream: *const ffi::zfp_stream) -> (u32, u32, u32, i32) {
    unsafe {
        (
            (*stream).minbits,
            (*stream).maxbits,
            (*stream).maxprec,
            (*stream).minexp,
        )
    }
}

/// Rates for which C's conversion to `uint` is undefined, and a word-aligned
/// budget that C's rounding wraps around to zero.
#[test]
fn invalid_ffi_rates_return_zero_without_changing_stream() {
    let stream = unsafe { ffi::zfp_stream_open(std::ptr::null_mut()) };
    assert!(!stream.is_null());
    let initial = params(stream);
    // Rounds to 2^32 bits per 1-D block, one more than `uint` holds.
    let too_large = (f64::from(u32::MAX) + 1.0) / 4.0;
    let cases = [
        -0.5,
        -1.0,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::INFINITY,
        too_large,
        f64::MAX,
    ]
    .into_iter()
    .flat_map(|rate| [(rate, ffi::zfp_false), (rate, ffi::zfp_true)])
    .chain([(f64::from(u32::MAX) / 4.0, ffi::zfp_true)]);
    for (rate, align) in cases {
        let set = unsafe {
            ffi::zfp_stream_set_rate(stream, rate, ffi::zfp_type_zfp_type_double, 1, align)
        };
        assert_eq!(set, 0.0, "rate={rate:?} align={align}");
        assert_eq!(params(stream), initial, "rate={rate:?} align={align}");
    }
    unsafe { ffi::zfp_stream_close(stream) };
}
