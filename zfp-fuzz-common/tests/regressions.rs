//! Replays committed fuzz crash inputs through the same target bodies the
//! libFuzzer harnesses call.
//!
//! This runs on **stable** as part of `cargo test --workspace`, so a bug the
//! fuzzer found stays fixed without anyone needing a nightly toolchain or
//! cargo-fuzz installed.
//!
//! To add a case: minimize the artifact with `just fuzz_tmin <target> <file>`
//! and commit it to `fuzz/regressions/<target>/<description>-<hash>.bin`.
//!
//! Caveat worth knowing when triaging: this harness runs without
//! `AddressSanitizer`, so a crash that is purely a heap overflow into a live
//! neighbouring allocation reproduces here as a silent pass. For those, add an
//! explicit assertion inside the target body so the case fails on stable too.

use std::path::PathBuf;

fn replay(target: &str, run: fn(&[u8])) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuzz")
        .join("regressions")
        .join(target);
    if !dir.is_dir() {
        return; // no regressions recorded for this target yet
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    paths.sort();

    for path in paths {
        let data = std::fs::read(&path).expect("read regression input");
        eprintln!("replaying {} ({} bytes)", path.display(), data.len());
        run(&data);
    }
}

macro_rules! regression_tests {
    ($($target:ident),* $(,)?) => {
        $(
            #[test]
            fn $target() {
                replay(
                    stringify!($target),
                    zfp_fuzz_common::targets::$target::run,
                );
            }
        )*
    };
}

regression_tests! {
    roundtrip,
    decompress_stream,
    block_codec,
    header_decode,
    config_mode,
    bitstream_ops,
}
