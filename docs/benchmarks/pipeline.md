# Variable-rate decompression pipeline

`ZfpExecution::Rayon` now uses a plane-reader / reconstruction pipeline for variable-rate streams. Fixed-rate streams retain independent chunk decoding. The compressed stream is unchanged; no index is generated or required.

## Current implementation

Variable-rate Rayon decompression now uses a bounded queue of reusable plane buffers. One reader parses headers and walks bit planes; each ready batch is dispatched immediately for transpose, rounding, reordering, inverse transform, float conversion and scatter. Finished batches return their buffers to the reader. Up to 16 buffers target about 1 MiB of plane storage by default, with no join-and-swap barrier between batches. A producer waiting for a buffer can help pending Rayon tasks.

`threads` is the total pool size, including the reader. `chunk_size` sets blocks per batch for variable-rate decoding; zero targets 64 KiB per batch. Aliased strides, fewer than two pool threads, or a failed pool/buffer allocation use serial decoding. Negative strides, partial blocks, borrowed streams, rounding, and truncation retain serial behavior. Compression and fixed-rate chunk-size semantics are unchanged.

```rust
let execution = zfp_rs::ZfpExecution::Rayon {
    threads: 4,
    chunk_size: 0,
};
bs.decompress_with_execution(&config, &mut field, execution)?;
```

For repeated calls, install them in a caller-owned Rayon pool and pass `threads: 0`; otherwise a pool is created and destroyed on each call. `ZfpExecution::Serial` remains the library-wide default and is available for workloads where parallel decoding is slower.

See the [production queue evaluation](pipeline_queue_production.md) for the replacement's performance, including regressions.

## Original two-buffer measurement

The measurements below evaluate the original two-buffer pipeline, using two 256 KiB batches with a join-and-swap barrier. They are retained as historical results; they do not measure the current queue.

2026-10-02, Intel Core Ultra 7 268V, Rust 1.98.1, release defaults, no target-cpu=native. Baseline source: 2011d5a. Cases use the api_compare generators, precision 16, accuracy 2^-8, rate 8, and 1,048,576 elements. Shapes are [1048576], [1024,1024], [128,128,64], and [64,64,16,16]. These are synthetic workloads, not a representative application corpus.

Serial runs are pinned to core 3. Reused pool workers are pinned in order to CPUs 3,1,0,2,4,5,6,7: N=2/4 use P-cores; N=8 includes four E-cores. Rayon schedules the reader and reconstruction tasks within that pool. CPU affinity constrains each worker; the producer has no permanent CPU assignment.

For each case, interleave serial/pipeline2, serial/pipeline4, serial/pipeline8, repeated three times. Each timed sample runs 30 decodes after one warm-up; report the fastest serial and parallel sample within each thread-count pair. The first 16 cases (all i32 cases, i64 1-D cases, and i64 2-D fixed-rate) were repeated with 50 iterations because the first sweep showed severe scheduling interruptions. The table uses those complete replacement sets; both the first sweep and repeated raw samples are retained. Individual timing spreads remain in the CSV, so interpret these as best observed throughput, not latency guarantees. Setup constructs only the selected case. Output bytes and final cursor are checked against serial on every process invocation; reversible output is also checked against input. Two plane-batch allocations per decode are included.

Elapsed loop time measures latency and determines speedup. perf stat records user cycles and instructions for both cpu_core and cpu_atom PMUs, with --no-scale to avoid scaling the hybrid PMU counts across threads assigned to the other core type. Subtract a matching zero-iteration process and divide by iterations; sum the P/E counts for total CPU work. Counter ratios below use medians, not the fastest time sample. A speedup can coexist with increased total cycles. These work counts are diagnostic; scheduling, idle spinning, PMU coverage and setup subtraction add noise.

## Results

Fixed-rate rows are controls using the existing independent-chunk implementation; the remaining rows measure the new pipeline. All speedups below are measured, not extrapolated.

| Dimensions | Median speedup: 2 threads | 4 threads | 8 threads | 4-thread range |
|---|---:|---:|---:|---:|
| 1-D | 0.96× | 1.01× | 0.98× | 0.90–1.36× |
| 2-D | 1.21× | 1.27× | 1.22× | 1.02–2.12× |
| 3-D | 1.17× | 2.11× | 2.02× | 1.40–2.73× |
| 4-D | 1.24× | 2.52× | 2.51× | 2.00–3.36× |

The pipeline is most useful where reconstruction is substantial. Keep serial execution available for small blocks or arrays, and measure on the target data. More threads do not necessarily help once plane reading dominates; eight threads also introduces slower E-cores on this machine. The batch-size sweep informed keeping the 256 KiB default, but no per-case tuning is used in the following table.

### Per-case results

Times are milliseconds per complete field for the four-thread comparison. Work ratios are pipeline/serial at four threads, summed across threads. Values below 1 in the speedup columns mean a slowdown.

| Case | Serial ms | Pipeline 4 ms | Speedup 2 | Speedup 4 | Speedup 8 | CPU cycles 4 / serial | Instructions 4 / serial |
|---|---:|---:|---:|---:|---:|---:|---:|
| i32_d1_fixed_rate | 6.530 | 1.706 | 1.85× | 3.83× | 5.37× | 1.00 | 1.00 |
| i32_d1_fixed_precision | 6.252 | 5.122 | 1.20× | 1.22× | 1.09× | 1.32 | 0.99 |
| i32_d1_reversible | 10.487 | 10.011 | 1.00× | 1.05× | 0.99× | 1.18 | 1.04 |
| i32_d2_fixed_rate | 3.650 | 0.955 | 1.93× | 3.82× | 5.73× | 1.00 | 1.00 |
| i32_d2_fixed_precision | 3.354 | 1.579 | 1.67× | 2.12× | 1.84× | 1.04 | 0.96 |
| i32_d2_reversible | 4.646 | 3.655 | 1.24× | 1.27× | 1.13× | 1.14 | 1.07 |
| i32_d3_fixed_rate | 2.439 | 0.629 | 1.90× | 3.88× | 6.17× | 1.01 | 1.00 |
| i32_d3_fixed_precision | 2.046 | 0.749 | 1.12× | 2.73× | 2.74× | 1.16 | 1.02 |
| i32_d3_reversible | 2.389 | 1.165 | 1.10× | 2.05× | 1.89× | 1.42 | 1.01 |
| i32_d4_fixed_rate | 2.659 | 0.684 | 1.90× | 3.89× | 5.18× | 1.00 | 1.00 |
| i32_d4_fixed_precision | 2.020 | 0.797 | 1.08× | 2.53× | 2.84× | 1.20 | 1.03 |
| i32_d4_reversible | 2.633 | 0.918 | 1.18× | 2.87× | 2.64× | 1.14 | 1.03 |
| i64_d1_fixed_rate | 7.987 | 2.083 | 1.91× | 3.83× | 5.22× | 0.99 | 1.00 |
| i64_d1_fixed_precision | 3.110 | 2.282 | 1.46× | 1.36× | 1.17× | 2.12 | 0.99 |
| i64_d1_reversible | 33.280 | 36.793 | 0.89× | 0.90× | 0.83× | 1.17 | 1.10 |
| i64_d2_fixed_rate | 4.533 | 1.188 | 1.90× | 3.82× | 6.00× | 0.99 | 1.00 |
| i64_d2_fixed_precision | 1.318 | 1.040 | 1.00× | 1.27× | 1.23× | 2.59 | 1.04 |
| i64_d2_reversible | 9.120 | 8.528 | 1.02× | 1.07× | 1.02× | 1.19 | 1.08 |
| i64_d3_fixed_rate | 3.500 | 0.876 | 1.95× | 3.99× | 6.59× | 0.96 | 1.00 |
| i64_d3_fixed_precision | 1.164 | 0.519 | 1.10× | 2.24× | 2.04× | 1.32 | 0.95 |
| i64_d3_reversible | 4.540 | 2.134 | 1.17× | 2.13× | 2.09× | 1.33 | 1.05 |
| i64_d4_fixed_rate | 3.372 | 0.845 | 1.95× | 3.99× | 5.85× | 1.03 | 1.00 |
| i64_d4_fixed_precision | 1.252 | 0.526 | 1.07× | 2.38× | 2.30× | 1.24 | 1.06 |
| i64_d4_reversible | 4.548 | 1.360 | 1.18× | 3.35× | 3.25× | 1.05 | 1.02 |
| f32_d1_fixed_rate | 11.635 | 3.618 | 1.92× | 3.22× | 7.33× | 1.00 | 1.00 |
| f32_d1_fixed_precision | 16.823 | 16.889 | 0.97× | 1.00× | 0.98× | 1.18 | 1.03 |
| f32_d1_fixed_accuracy | 17.163 | 17.050 | 0.97× | 1.01× | 0.97× | 1.17 | 1.03 |
| f32_d1_reversible | 30.413 | 30.430 | 0.96× | 1.00× | 0.96× | 1.10 | 1.07 |
| f32_d2_fixed_rate | 4.996 | 1.292 | 1.78× | 3.87× | 7.21× | 0.99 | 0.99 |
| f32_d2_fixed_precision | 6.397 | 4.850 | 1.23× | 1.32× | 1.26× | 1.11 | 1.00 |
| f32_d2_fixed_accuracy | 7.114 | 5.664 | 1.18× | 1.26× | 1.21× | 1.10 | 1.00 |
| f32_d2_reversible | 9.692 | 9.469 | 0.99× | 1.02× | 1.01× | 1.15 | 1.08 |
| f32_d3_fixed_rate | 2.525 | 0.639 | 1.91× | 3.95× | 6.24× | 0.99 | 1.00 |
| f32_d3_fixed_precision | 2.862 | 1.359 | 1.46× | 2.11× | 2.00× | 1.15 | 1.02 |
| f32_d3_fixed_accuracy | 3.491 | 1.677 | 1.49× | 2.08× | 1.96× | 1.14 | 1.01 |
| f32_d3_reversible | 4.089 | 2.920 | 1.34× | 1.40× | 1.38× | 1.30 | 1.02 |
| f32_d4_fixed_rate | 2.574 | 0.649 | 1.91× | 3.96× | 6.03× | 1.01 | 1.00 |
| f32_d4_fixed_precision | 2.661 | 1.060 | 1.27× | 2.51× | 2.39× | 1.15 | 1.03 |
| f32_d4_fixed_accuracy | 3.495 | 1.550 | 1.40× | 2.25× | 2.11× | 1.13 | 1.03 |
| f32_d4_reversible | 3.732 | 1.862 | 1.44× | 2.00× | 1.98× | 1.10 | 1.05 |
| f64_d1_fixed_rate | 10.585 | 3.119 | 1.88× | 3.39× | 5.05× | 1.00 | 1.00 |
| f64_d1_fixed_precision | 17.070 | 16.870 | 0.96× | 1.01× | 0.98× | 1.18 | 1.04 |
| f64_d1_fixed_accuracy | 17.361 | 16.972 | 0.96× | 1.02× | 0.98× | 1.19 | 1.04 |
| f64_d1_reversible | 40.983 | 42.931 | 0.94× | 0.95× | 0.94× | 1.11 | 1.07 |
| f64_d2_fixed_rate | 5.233 | 1.362 | 1.95× | 3.84× | 6.05× | 1.00 | 1.00 |
| f64_d2_fixed_precision | 6.638 | 4.864 | 1.31× | 1.36× | 1.28× | 1.22 | 1.04 |
| f64_d2_fixed_accuracy | 7.402 | 5.768 | 1.25× | 1.28× | 1.22× | 1.20 | 1.04 |
| f64_d2_reversible | 11.878 | 11.658 | 0.97× | 1.02× | 1.02× | 1.22 | 1.10 |
| f64_d3_fixed_rate | 3.517 | 0.898 | 1.92× | 3.92× | 6.36× | 1.01 | 1.00 |
| f64_d3_fixed_precision | 3.714 | 1.622 | 1.17× | 2.29× | 2.14× | 1.35 | 1.06 |
| f64_d3_fixed_accuracy | 4.325 | 2.055 | 1.18× | 2.10× | 2.11× | 1.36 | 1.06 |
| f64_d3_reversible | 6.744 | 4.417 | 1.50× | 1.53× | 1.51× | 1.26 | 1.06 |
| f64_d4_fixed_rate | 3.686 | 0.933 | 1.93× | 3.95× | 6.02× | 1.00 | 1.00 |
| f64_d4_fixed_precision | 3.777 | 1.124 | 1.20× | 3.36× | 3.29× | 1.15 | 1.02 |
| f64_d4_fixed_accuracy | 4.295 | 1.574 | 1.30× | 2.73× | 2.62× | 1.29 | 1.03 |
| f64_d4_reversible | 6.556 | 3.172 | 1.56× | 2.07× | 2.00× | 1.09 | 1.04 |

## Creating a pool on each call

These six representative cases include pool creation and destruction inside every timed decode (20 iterations, best of three). Threads may move within the selected CPU set in this configuration; reused-pool workers above are individually pinned. Frequency and scheduling differences mean the two tables cannot isolate pool-creation cost by direct subtraction.

| Case | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|
| f32_d1_fixed_precision | 1.02× | 1.11× | 1.08× |
| f32_d2_fixed_precision | 1.34× | 1.49× | 1.38× |
| f32_d3_fixed_precision | 1.41× | 2.24× | 1.89× |
| f64_d4_fixed_precision | 1.13× | 2.84× | 2.44× |
| f64_d4_reversible | 1.48× | 2.30× | 1.74× |
| i64_d3_fixed_precision | 1.01× | 1.82× | 1.44× |

The large 3-D/4-D cases still benefit when creating a pool per call. Reusing a pool is preferable for repeated operations; no small-array latency claim is made by these one-million-element measurements.

## Serial decoder regression check

The same harness was built against pristine HEAD and the new source with identical ffi/rayon features. Across all 56 serial cases, zero-subtracted retired instructions changed by +0.008% geometrically averaged, with a largest absolute change of 0.261%. This is within the measurement notes’ 1% noise guideline. No serial performance regression was identified; wall-time differences alone were not used to claim one.

## Verification

- Exact serial/pipeline output and cursor comparisons for all scalar types, dimensions, modes, all rounding settings, nonzero starting bit offsets, negative and aliasing strides, partial blocks, multiple batches, and reversible NaNs/infinities/subnormals/signed zero.
- Truncation tests exercise the pipeline with one- and three-block batches; error results and cursor semantics match serial.
- Nested calls in reused one-, two-, and four-thread pools complete correctly.
- Full zfp-rs tests pass with default features and with rayon,ffi,internals. Workspace tests and Clippy pass; Rayon Clippy and all-feature rustdoc pass.

## Reproduce

```sh
cargo build --release -p zfp-benchmarks --example pipeline
python3 scripts/bench_pipeline.py --output /tmp/zfp-pipeline-results
```

Use --cpus for the local CPU topology and --events for the available PMUs. To include per-call pool construction, add --fresh. To select cases, pass quoted case names after --cases. The example also runs directly without perf or affinity settings on other platforms.

[pipeline.csv](pipeline.csv) includes individual thread-count timings and sample spreads; [pipeline_fresh.csv](pipeline_fresh.csv) records the per-call pool comparisons. Raw counters, logs, tuning data, and pristine-HEAD comparisons were retained in /tmp/zfp-pipeline-results for this run.
