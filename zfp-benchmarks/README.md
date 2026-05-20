## Benchmarks

The `zfp-benchmarks` package compares the safe Rust API (`zfp-rs`), the FFI wrapper (`zfp-rs-ffi`), and the upstream C implementation (`zfp-sys`) across all scalar types and compression modes.

```bash
cargo bench -p zfp-benchmarks --bench api_compare
```

Generate local SVG plots from Criterion output:

```bash
scripts/plot_benchmarks.py
```
