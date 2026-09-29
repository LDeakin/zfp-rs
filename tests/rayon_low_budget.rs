#![cfg(feature = "rayon")]

use zfp_rs::{
    ZfpBitStream, ZfpBitStreamOps, ZfpBitStreamRef, ZfpConfig, ZfpExecution, ZfpField, ZfpFieldMut,
    ZfpHeaderMask, ZfpRounding, ZfpScalar,
};

type Decoded = (usize, u64, Vec<u8>);

fn mixed_blocks<T: ZfpScalar>(blocks: usize, nonzero: [T; 4], zero_first: bool) -> Vec<T> {
    (0..blocks)
        .flat_map(|block| {
            if (block % 2 == 0) == zero_first {
                [T::default(); 4]
            } else {
                nonzero
            }
        })
        .collect()
}

fn decode_two<T: ZfpScalar>(
    bs: &mut impl ZfpBitStreamOps,
    config: &ZfpConfig,
    lengths: [usize; 2],
    ends: [usize; 2],
    execution: ZfpExecution,
) -> [Decoded; 2] {
    std::array::from_fn(|stream| {
        let header = bs.read_header(ZfpHeaderMask::FULL).unwrap();
        assert_eq!(
            header.config.unwrap().with_rounding(config.rounding()),
            *config
        );
        let metadata = header.metadata.unwrap();
        assert_eq!(metadata.scalar_type, T::SCALAR_TYPE);
        assert_eq!(metadata.dims, [lengths[stream], 0, 0, 0]);

        let mut out = vec![T::default(); lengths[stream]];
        let mut field = ZfpFieldMut::new(&mut out, [lengths[stream]]).unwrap();
        let read = bs
            .decompress_with_execution(config, &mut field, execution)
            .unwrap();
        let position = bs.read_pos();
        assert_eq!(read, ends[stream]);
        assert_eq!(position, (ends[stream] as u64) * 8);
        (read, position, bytemuck::cast_slice(&out).to_vec())
    })
}

fn exercise<T: ZfpScalar>(first: &[T], second: &[T], budget: u32, rounding: ZfpRounding) {
    let config = ZfpConfig::try_expert(budget, budget, 64, -1074)
        .unwrap()
        .with_rounding(rounding);
    let mut encoded = ZfpBitStream::new(4096);
    let mut ends = [0; 2];
    for (index, input) in [first, second].into_iter().enumerate() {
        let field = ZfpField::new(input, [input.len()]).unwrap();
        encoded
            .write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)
            .unwrap();
        ends[index] = encoded.compress(&config, &field).unwrap();
    }
    let words = encoded.as_words().to_vec();
    let lengths = [first.len(), second.len()];
    let parallel = ZfpExecution::Rayon {
        threads: 4,
        chunk_size: 1,
    };

    let mut owned = ZfpBitStream::from_words(words.clone());
    let serial = decode_two::<T>(&mut owned, &config, lengths, ends, ZfpExecution::Serial);
    owned.rewind();
    assert_eq!(
        decode_two::<T>(&mut owned, &config, lengths, ends, parallel),
        serial
    );

    let mut borrowed = ZfpBitStreamRef::from_words(&words);
    assert_eq!(
        decode_two::<T>(&mut borrowed, &config, lengths, ends, ZfpExecution::Serial),
        serial
    );
    borrowed.rewind();
    assert_eq!(
        decode_two::<T>(&mut borrowed, &config, lengths, ends, parallel),
        serial
    );
}

#[test]
fn low_budget_float_blocks_match_serial_across_rounding_headers_and_streams() {
    let roundings = [
        ZfpRounding::Never,
        ZfpRounding::First { tight_error: false },
        ZfpRounding::First { tight_error: true },
        ZfpRounding::Last { tight_error: false },
        ZfpRounding::Last { tight_error: true },
    ];
    let f32_first = mixed_blocks(16, [0.25f32, 1.0, 2.0, -0.5], false);
    let f32_second = mixed_blocks(12, [-2.0f32, 0.5, -0.25, 1.0], true);
    let f64_first = mixed_blocks(16, [0.25f64, 1.0, 2.0, -0.5], false);
    let f64_second = mixed_blocks(12, [-2.0f64, 0.5, -0.25, 1.0], true);

    for rounding in roundings {
        for budget in 1..=9 {
            exercise(&f32_first, &f32_second, budget, rounding);
        }
        for budget in 1..=12 {
            exercise(&f64_first, &f64_second, budget, rounding);
        }
    }
}
