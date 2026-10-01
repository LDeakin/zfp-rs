## Benchmarks

The `zfp-benchmarks` package compares the safe Rust API (`zfp-rs`), the FFI wrapper (`zfp-rs-ffi`), and the upstream C implementation (`zfp-sys`) across all scalar types and compression modes.

```bash
cargo bench -p zfp-benchmarks
```

This runs two benchmark binaries that write to the same Criterion group, `api_compare`:

| Binary           | Variants                                              |
| ---------------- | ----------------------------------------------------- |
| `api_compare`    | `zfp-rs`, `zfp-rs-rayon*`, `zfp-rs-ffi`, `zfp-rs-ffi-omp*` |
| `api_compare_c`  | `zfp-sys`, `zfp-sys-omp*`                             |

They are separate because `zfp-sys` and `zfp-rs-ffi` both define the C symbols `zfp_*` and `stream_*`.
Linked into one binary, every `zfp-sys` call binds to the `zfp-rs-ffi` definitions and the "C" results are really the Rust implementation.
`api_compare_c` asserts at startup that `zfp-sys` resolves into `libzfp`, and `api_compare` must stay the only binary that links `zfp-rs-ffi`.

Fixed accuracy is not benchmarked for `i32` and `i64`; see `Case::is_supported`.
zfp ignores the tolerance for integers and encodes them at full precision, so those cases would measure full-precision compression under the name of an accuracy mode.

Both binaries take their Criterion settings from `common::criterion_config`, so the results are comparable.
The settings favour stable means for charting over speed.

`just bench_f32` runs only the f32 benchmarks, about a quarter of the full run.
It passes a Criterion filter, a regex on the benchmark ID, so it applies to both binaries.
Filter on another scalar type the same way, for example `cargo bench -p zfp-benchmarks -- '/f64_d'`.

Run a single binary with `--bench`, for example `cargo bench -p zfp-benchmarks --bench api_compare_c`.
Run them through cargo, as the C library is a shared library that is found through the `LD_LIBRARY_PATH` that cargo sets.

Generate local SVG plots from Criterion output, which needs both binaries to have run:

```bash
scripts/plot_benchmarks.py
```

This writes `docs/benchmarks/api_compare_compress.svg` and `docs/benchmarks/api_compare_decompress.svg`, each on a single set of axes.
A block of bars per case, such as Fixed Rate 2D, holds a group for each scalar type that has results.
It plots every result in `target/criterion`, including stale ones, so clear that directory first to chart only a filtered run such as `just bench_f32`.
