//! Quick pipeline/serial comparison with exact output and cursor checks.
//! Set ZFP_BENCH_CPUS=3,1,0,2,4,5,6,7 to pin pool threads on the measurement laptop.
//! Without that variable, the operating system schedules threads normally.
#![allow(dead_code, unused_macros)]
#[path = "../benches/common/mod.rs"]
mod common;
use common::*;
use std::hint::black_box;
use zfp_rs::{ZfpBitStream, ZfpField, ZfpFieldMut};
#[cfg(target_os = "linux")]
fn pin(cpus: &[usize]) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set),
            0
        );
    }
}
#[cfg(not(target_os = "linux"))]
fn pin(_cpus: &[usize]) {}

fn selected_cpus() -> Vec<usize> {
    std::env::var("ZFP_BENCH_CPUS")
        .ok()
        .map(|s| s.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_default()
}
fn run<T: BenchScalar + Send>(case: Case, iters: usize, threads: usize, batch: u32, fresh: bool) {
    let cpus = selected_cpus();
    assert!(cpus.is_empty() || threads <= cpus.len());
    if !cpus.is_empty() {
        pin(&cpus[..1]);
    }
    let (data, shape) = T::generate(case.dims);
    let dims = dims4(&shape);
    let config = rust_config::<T>(case);
    let mut bs = ZfpBitStream::new(config.maximum_size(T::RUST_TYPE, dims).unwrap()).unwrap();
    let bytes = bs
        .compress(&config, &ZfpField::new(&data, dims).unwrap())
        .unwrap();
    let mut output = vec![T::default(); data.len()];
    bs.rewind();
    bs.decompress(&config, &mut ZfpFieldMut::new(&mut output, dims).unwrap())
        .unwrap();
    let expected = bytemuck::cast_slice::<T, u8>(&output).to_vec();
    let cursor = bs.read_pos();
    let pool = if threads > 0 && !fresh {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .start_handler({
                    let cpus = cpus.clone();
                    move |i| {
                        if let Some(cpu) = cpus.get(i..=i) {
                            pin(cpu);
                        }
                    }
                })
                .build()
                .unwrap(),
        )
    } else {
        None
    };
    let execution = if threads == 0 {
        zfp_rs::ZfpExecution::Serial
    } else {
        zfp_rs::ZfpExecution::Rayon {
            threads: if fresh { threads as u32 } else { 0 },
            chunk_size: batch,
        }
    };
    // For per-call pool runs, permit all selected CPUs (Rayon inherits affinity).
    if fresh && threads > 0 && !cpus.is_empty() {
        pin(&cpus[..threads]);
    }
    let mut decode = || {
        bs.rewind();
        let mut field = ZfpFieldMut::new(&mut output, dims).unwrap();
        black_box(
            bs.decompress_with_execution(
                black_box(&config),
                black_box(&mut field),
                black_box(execution),
            )
            .unwrap(),
        );
    };
    let mut measure = || {
        decode();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            decode();
        }
        start.elapsed().as_nanos()
    };
    let ns = match pool {
        Some(ref pool) => pool.install(measure),
        None => measure(),
    };
    assert_eq!(bs.read_pos(), cursor);
    assert_eq!(bytemuck::cast_slice::<T, u8>(&output), expected.as_slice());
    if case.mode == ModeKind::Reversible {
        assert_eq!(bytemuck::cast_slice::<T, u8>(&data), expected.as_slice());
    }
    println!(
        "{} threads={} batch={} iterations={} ns={} bytes={}",
        case.label(),
        threads,
        batch,
        iters,
        ns,
        bytes
    );
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "usage: pipeline CASE ITERATIONS THREADS [BATCH_SIZE] [fresh]; THREADS=0 selects serial; set ZFP_BENCH_CPUS for affinity"
    );
    let filter = &args[1];
    let iters: usize = args[2].parse().unwrap();
    let threads: usize = args[3].parse().unwrap();
    let batch: u32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);
    let fresh = args.get(5).is_some_and(|s| s == "fresh");
    let mut matched = 0;
    for &scalar in SCALARS {
        for dims in 1..=4 {
            for &mode in MODES {
                let case = Case { scalar, dims, mode };
                if !case.is_supported() || case.label() != *filter {
                    continue;
                }
                matched += 1;
                match scalar {
                    ScalarKind::I32 => run::<i32>(case, iters, threads, batch, fresh),
                    ScalarKind::I64 => run::<i64>(case, iters, threads, batch, fresh),
                    ScalarKind::F32 => run::<f32>(case, iters, threads, batch, fresh),
                    ScalarKind::F64 => run::<f64>(case, iters, threads, batch, fresh),
                }
            }
        }
    }
    assert_eq!(matched, 1, "case filter must select exactly one case");
}
