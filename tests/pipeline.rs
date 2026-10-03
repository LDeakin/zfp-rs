//! Pipeline output and cursor must match serial decoding, including fallback.
use zfp_rs::{
    ZfpBitStream, ZfpBitStreamRef, ZfpConfig, ZfpExecution, ZfpField, ZfpFieldMut, ZfpRounding,
    ZfpScalar,
};

fn compare<T: ZfpScalar>(data: &[T], dims: [usize; 4], config: ZfpConfig, batch: u32) {
    let field = ZfpField::new(data, dims).unwrap();
    let capacity = config.maximum_size(T::SCALAR_TYPE, dims).unwrap() + 16;
    let mut bs = ZfpBitStream::new(capacity).unwrap();
    bs.write_bits(0x123, 13);
    bs.compress(&config, &field).unwrap();
    let words = bs.as_words();
    let nx = dims[0] as isize;
    let ny = dims[1].max(1) as isize;
    let nz = dims[2].max(1) as isize;
    let layouts = [
        [1, nx, nx * ny, nx * ny * nz],
        [-1, nx, nx * ny, nx * ny * nz],
        // May alias across blocks: must fall back to serial.
        [1, 1, 1, 1],
    ];
    for strides in layouts {
        let decode = |execution| {
            let mut reader = ZfpBitStreamRef::from_words(words);
            reader.seek_read(13);
            let mut output = vec![T::default(); data.len() + 8];
            let mut field = ZfpFieldMut::new_strided(&mut output, dims, strides).unwrap();
            let result = reader.decompress_with_execution(&config, &mut field, execution);
            (
                result,
                reader.read_pos(),
                bytemuck::cast_slice::<T, u8>(&output).to_vec(),
            )
        };
        let serial = decode(ZfpExecution::Serial);
        for threads in [1, 2, 4] {
            assert_eq!(
                decode(ZfpExecution::Rayon {
                    threads,
                    chunk_size: batch
                }),
                serial,
                "dims={dims:?}, config={config:?}, threads={threads}, batch={batch}, strides={strides:?}"
            );
        }
    }
}

/// A field of every scalar type, with a zero block every `zero_every` values.
struct Samples {
    dims: [usize; 4],
    ints: Vec<i32>,
    longs: Vec<i64>,
    floats: Vec<f32>,
    doubles: Vec<f64>,
}

impl Samples {
    fn new(dims: [usize; 4], zero_every: Option<usize>) -> Self {
        let len = dims.iter().filter(|&&n| n != 0).product();
        let ints: Vec<i32> = (0..len)
            .map(|i| {
                if zero_every.is_some_and(|n| i % n == 0) {
                    0
                } else {
                    (i as i32).wrapping_mul(1_103_515_245)
                }
            })
            .collect();
        let longs: Vec<i64> = ints.iter().map(|&i| i64::from(i) << 24).collect();
        let floats: Vec<f32> = ints.iter().map(|&i| i as f32 / 1e6).collect();
        let doubles: Vec<f64> = longs.iter().map(|&i| i as f64 / 1e9).collect();
        Self {
            dims,
            ints,
            longs,
            floats,
            doubles,
        }
    }

    /// For modes that integers do not support.
    fn compare_floats(&self, config: ZfpConfig, batch: u32) {
        compare(&self.floats, self.dims, config, batch);
        compare(&self.doubles, self.dims, config, batch);
    }

    fn compare_all(&self, config: ZfpConfig, batch: u32) {
        compare(&self.ints, self.dims, config, batch);
        compare(&self.longs, self.dims, config, batch);
        self.compare_floats(config, batch);
    }
}

#[test]
fn pipeline_matches_serial_for_types_dimensions_modes_and_rounding() {
    let roundings = [
        ZfpRounding::Never,
        ZfpRounding::First { tight_error: false },
        ZfpRounding::First { tight_error: true },
        ZfpRounding::Last { tight_error: false },
        ZfpRounding::Last { tight_error: true },
    ];
    for dims in [[33, 0, 0, 0], [9, 7, 0, 0], [9, 5, 6, 0], [9, 5, 6, 3]] {
        let samples = Samples::new(dims, Some(13));
        for rounding in roundings {
            for config in [
                ZfpConfig::fixed_precision(16),
                ZfpConfig::fixed_accuracy(0.00390625),
                ZfpConfig::reversible(),
                ZfpConfig::expert(1, 1, 64, -1074).unwrap(),
                ZfpConfig::expert(0, 1, 64, -1074).unwrap(),
                ZfpConfig::expert(100, 200, 24, -20).unwrap(),
            ] {
                samples.compare_all(config.with_rounding(rounding), 3);
            }
        }
    }
}

#[test]
fn pipeline_preserves_reversible_float_bits_and_zero_blocks() {
    let values = [
        0.0f64,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(1),
        f64::MAX,
        -1.0,
    ];
    let data: Vec<f64> = (0..9 * 7 * 5 * 3)
        .map(|i| {
            if (i / 256) % 2 == 0 {
                0.0
            } else {
                values[i % values.len()]
            }
        })
        .collect();
    let floats: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    for batch in [0, 1, 7, u32::MAX] {
        compare(&data, [9, 7, 5, 3], ZfpConfig::reversible(), batch);
        compare(&floats, [9, 7, 5, 3], ZfpConfig::reversible(), batch);
    }
}

#[cfg(feature = "rayon")]
#[test]
fn pipeline_reuses_current_pool_and_nested_calls_complete() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            rayon::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|_| {
                        let data: Vec<i32> = (0..4096).collect();
                        let config = ZfpConfig::reversible();
                        let mut bs = ZfpBitStream::new(
                            config
                                .maximum_size(zfp_rs::ZfpScalarType::I32, [64, 64])
                                .unwrap(),
                        )
                        .unwrap();
                        bs.compress(&config, &ZfpField::new(&data, [64, 64]).unwrap())
                            .unwrap();
                        bs.rewind();
                        let mut output = vec![0; 4096];
                        bs.decompress_with_execution(
                            &config,
                            &mut ZfpFieldMut::new(&mut output, [64, 64]).unwrap(),
                            ZfpExecution::Rayon {
                                threads: 0,
                                chunk_size: 1,
                            },
                        )
                        .unwrap();
                        assert_eq!(output, data);
                    });
                }
            })
        });
    }
}

#[test]
fn queued_buffers_turn_over_for_all_types_and_dimensions() {
    for dims in [[257, 0, 0, 0], [33, 17, 0, 0], [17, 13, 9, 0], [9, 9, 9, 5]] {
        let samples = Samples::new(dims, None);
        for config in [ZfpConfig::fixed_precision(16), ZfpConfig::reversible()] {
            samples.compare_all(config, 1);
        }
        samples.compare_floats(ZfpConfig::fixed_accuracy(0.00390625), 1);
    }
}

/// A default batch holds a few hundred blocks, so these 3,750 take several.
#[test]
fn default_batches_match_serial_for_a_large_field() {
    let samples = Samples::new([300, 200, 0, 0], None);
    samples.compare_all(ZfpConfig::fixed_precision(16), 0);
    samples.compare_all(ZfpConfig::reversible(), 0);
    samples.compare_floats(ZfpConfig::fixed_accuracy(0.00390625), 0);
}
