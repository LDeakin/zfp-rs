# Default: run all CI checks
default: fmt clippy build test

# Check formatting (no-op if already clean)
fmt:
	cargo fmt --all -- --check
	cargo fmt --manifest-path zfp-round-tests/Cargo.toml --all -- --check

# Lint with Clippy (denies all warnings)
clippy:
	cargo clippy --features ffi,internals --workspace --all-targets -- -D warnings

# Build the entire workspace
build:
	cargo build --workspace

# Run all tests (with and without ffi feature)
test:
	cargo test --features ffi,internals --workspace
	cargo test -p zfp-rs

# Cross-validate the rounding modes against C (own workspace; needs cmake)
test_rounding:
	cargo test --manifest-path zfp-round-tests/Cargo.toml
	cargo test -p zfp-rs-ffi --features round-first
	cargo test -p zfp-rs-ffi --features round-last
	cargo test -p zfp-rs-ffi --features round-tight-error
	cargo test -p zfp-rs-ffi --features round-last,tight-error

# Lint the rounding test crate (it is outside the workspace, so `just clippy` misses it)
round_clippy:
	cargo clippy --manifest-path zfp-round-tests/Cargo.toml --all-targets -- -D warnings

# Run benchmarks
bench:
	cargo bench -p zfp-benchmarks

# Generate SVG plots and CSV from existing benchmark results only
bench_plot:
	uv run scripts/plot_benchmarks.py

# Build all fuzz targets (requires nightly + `cargo install cargo-fuzz`)
fuzz_build:
	cargo +nightly fuzz build

# Run one fuzz target: `just fuzz roundtrip 60`
fuzz target time='60':
	mkdir -p fuzz/corpus/{{target}}
	cargo +nightly fuzz run {{target}} fuzz/corpus/{{target}} fuzz/seeds/{{target}} -- -max_total_time={{time}} -max_len=4096 -rss_limit_mb=2048 -malloc_limit_mb=1024 -timeout=10

# Briefly run every fuzz target (mirrors the CI smoke job)
fuzz_smoke:
	for t in $(cargo +nightly fuzz list); do just fuzz $t 30; done

# Replay all committed crash regressions (stable; also runs in `just test`)
fuzz_regressions:
	cargo test -p zfp-fuzz-common --test regressions

# Minimize a crash input before committing it as a regression
fuzz_tmin target input:
	cargo +nightly fuzz tmin {{target}} {{input}}

# Minimize every corpus
fuzz_cmin:
	for t in $(cargo +nightly fuzz list); do cargo +nightly fuzz cmin $t; done

# Regenerate the committed seed corpora
fuzz_seeds:
	cargo run -p zfp-fuzz-common --bin gen_seeds

# Coverage report for a fuzz target (checks the fuzzer is not stuck)
fuzz_coverage target:
	cargo +nightly fuzz coverage {{target}}

# Lint the fuzz crate (it is outside the workspace, so `just clippy` misses it)
fuzz_clippy:
	cargo +nightly clippy --manifest-path fuzz/Cargo.toml --all-targets -- -D warnings

# Check the strided codec for provenance UB under Miri (needs the miri component)
miri:
	MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test -p zfp-fuzz-common --test miri
