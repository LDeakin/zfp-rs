//! `zfp_block_maximum_size` against the reference implementation.

use zfp_rs_ffi::zfp_type;

#[test]
fn matches_c_for_every_input() {
    let types = [
        (zfp_type::zfp_type_none, 0u32),
        (zfp_type::zfp_type_int32, 1),
        (zfp_type::zfp_type_int64, 2),
        (zfp_type::zfp_type_float, 3),
        (zfp_type::zfp_type_double, 4),
    ];
    // dims 0 and 5 are out of range and must return 0, as in C.
    for (rs_ty, c_ty) in types {
        for dims in 0..=5u32 {
            for reversible in [0, 1] {
                let rs = zfp_rs_ffi::zfp_block_maximum_size(rs_ty, dims, reversible);
                let c = unsafe { zfp_sys::zfp_block_maximum_size(c_ty, dims, reversible) };
                assert_eq!(rs, c, "ty={c_ty} dims={dims} reversible={reversible}");
            }
        }
    }
}
