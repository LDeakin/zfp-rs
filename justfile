# Default: run all CI checks
default: fmt clippy build test

# Check formatting (no-op if already clean)
fmt:
	cargo fmt --all -- --check

# Lint with Clippy (denies all warnings)
clippy:
	cargo clippy --features ffi --workspace --all-targets -- -D warnings

# Build the entire workspace
build:
	cargo build --workspace

# Run all tests (with and without ffi feature)
test:
	cargo test --features ffi --workspace
	cargo test --workspace

# Run benchmarks
bench:
	cargo bench -p zfp-benchmarks

# Generate SVG plots and CSV from existing benchmark results only
bench_plot:
	uv run scripts/plot_benchmarks.py
