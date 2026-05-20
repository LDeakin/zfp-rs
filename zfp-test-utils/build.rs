fn main() {
    // Compile the C test utility functions for generating smooth random arrays.
    // These are used by the end-to-end integration tests to produce byte-identical
    // test data to the upstream C test suite.
    cc::Build::new()
        .file("../zfp/tests/utils/genSmoothRandNums.c")
        .file("../zfp/tests/utils/fixedpoint96.c")
        .file("../zfp/tests/utils/rand64.c")
        .include("../zfp/include")
        .include("../zfp/tests")
        .include("../zfp")
        // Upstream helper functions keep an unused `amplitude` arg in dotProd* signatures.
        .flag_if_supported("-Wno-unused-parameter")
        .compile("zfp_test_utils");

    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=../zfp/tests/utils/genSmoothRandNums.c");
    println!("cargo:rerun-if-changed=../zfp/tests/utils/fixedpoint96.c");
    println!("cargo:rerun-if-changed=../zfp/tests/utils/rand64.c");
}
