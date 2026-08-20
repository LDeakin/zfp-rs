#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    zfp_fuzz_common::targets::bitstream_ops::run(data);
});
