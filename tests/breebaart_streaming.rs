use gammachirp_rs::Error;
use gammachirp_rs::breebaart2001::{
    EiConfig, EiDelayConvention, EiStream, EiUnit, HybridBinauralConfig, HybridBinauralStream,
    MonauralConfig, MonauralStream, breebaart2001_ei, breebaart2001_monaural, hybrid_binaural,
};
use gammachirp_rs::gcfb_v234::{ControlMode, DcgcEvent, GainReference};
use ndarray::{Array2, Array3};
use proptest::prelude::*;

fn close(actual: f64, expected: f64) {
    let tolerance = 2e-9 + 2e-8 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.16e} != {expected:.16e} (tolerance {tolerance:.3e})"
    );
}

fn close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        close(actual, expected);
    }
}

fn channel_major(values: &[i16], channels: usize, samples: usize) -> Array2<f64> {
    Array2::from_shape_fn((channels, samples), |(channel, sample)| {
        let index = channel * samples + sample;
        values[index] as f64 / 8_192.0
    })
}

fn collect_ei_stream(
    left: &Array2<f64>,
    right: &Array2<f64>,
    units: &[EiUnit],
    config: EiConfig,
) -> (Array3<f64>, usize, usize) {
    let mut stream = EiStream::new(left.nrows(), 8_000.0, units, config).unwrap();
    let latency = stream.latency_samples();
    let maximum_buffer = stream.max_buffered_samples();
    let mut events = Vec::new();
    for sample in 0..left.ncols() {
        let left_sample = left.column(sample).to_vec();
        let right_sample = right.column(sample).to_vec();
        if let Some(event) = stream.process_sample(&left_sample, &right_sample).unwrap() {
            events.push(event);
        }
        assert!(stream.buffered_samples() <= maximum_buffer);
    }
    events.extend(stream.finish().unwrap());
    assert_eq!(events.len(), left.ncols());
    let mut output = Array3::zeros((units.len(), left.nrows(), left.ncols()));
    for (expected_index, event) in events.into_iter().enumerate() {
        assert_eq!(event.sample_index, expected_index);
        for unit in 0..units.len() {
            for channel in 0..left.nrows() {
                output[[unit, channel, expected_index]] = event.activity[[unit, channel]];
            }
        }
    }
    (output, latency, maximum_buffer)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn monaural_stream_matches_batch(
        channels in 1usize..6,
        samples in 1usize..100,
        values in prop::collection::vec(any::<i16>(), 600),
        time_ms in 1u16..80,
        sensitivity in 0.0f64..0.01,
    ) {
        let input = channel_major(&values[..channels * samples], channels, samples);
        let config = MonauralConfig {
            integration_time_constant_seconds: time_ms as f64 / 1_000.0,
            sensitivity,
            ..MonauralConfig::streaming()
        };
        let batch = breebaart2001_monaural(&input, 8_000.0, &config).unwrap();
        let mut stream = MonauralStream::new(channels, 8_000.0, config).unwrap();
        for sample in 0..samples {
            let input_sample = input.column(sample).to_vec();
            let event = stream
                .process_sample(&input_sample)
                .unwrap();
            prop_assert_eq!(event.sample_index, sample);
            for channel in 0..channels {
                prop_assert_eq!(event.output[channel], batch[[channel, sample]]);
            }
        }
        prop_assert_eq!(stream.samples_processed(), samples);
    }

    #[test]
    fn ei_stream_matches_batch_with_noise_and_both_delay_conventions(
        channels in 1usize..5,
        samples in 1usize..90,
        left_values in prop::collection::vec(any::<i16>(), 450),
        right_values in prop::collection::vec(any::<i16>(), 450),
        delay_samples in -4i8..5,
        iid_tenths in -80i16..81,
        convention_selector in any::<bool>(),
        noise_std in 0.0f64..1.5,
        seed in any::<u64>(),
        time_ms in 1u16..80,
    ) {
        let left = channel_major(&left_values[..channels * samples], channels, samples);
        let right = channel_major(&right_values[..channels * samples], channels, samples);
        let units = [
            EiUnit::default(),
            EiUnit::new(delay_samples as f64 / 8_000.0, iid_tenths as f64 / 10.0),
        ];
        let config = EiConfig {
            integration_time_constant_seconds: time_ms as f64 / 1_000.0,
            delay_convention: if convention_selector {
                EiDelayConvention::PaperSymmetric
            } else {
                EiDelayConvention::AmtOneSidedInteger
            },
            internal_noise_std_mu: noise_std,
            noise_seed: seed,
            ..EiConfig::streaming()
        };
        let batch = breebaart2001_ei(&left, &right, 8_000.0, &units, &config).unwrap();
        let (streamed, _, _) = collect_ei_stream(&left, &right, &units, config);
        prop_assert_eq!(streamed, batch);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 6,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn hybrid_stream_matches_batch(
        values in prop::collection::vec(any::<i16>(), 8..70),
        channels in 2usize..5,
        mode in 0u8..3,
        update in 1usize..5,
        symmetric_delay in any::<bool>(),
        corrected in any::<bool>(),
        threshold_noise in any::<bool>(),
        internal_noise in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let left: Vec<f64> = values.iter().map(|value| *value as f64 / 32_768.0).collect();
        let right: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let adjacent = values[(index + 1) % values.len()] as f64;
                (0.7 * *value as f64 + 0.3 * adjacent) / 32_768.0
            })
            .collect();
        let mut config = HybridBinauralConfig::streaming();
        config.filterbank.fs = 8_000.0;
        config.filterbank.num_ch = channels;
        config.filterbank.f_range = [180.0, 1_800.0];
        config.filterbank.out_mid_crct = if corrected { "ELC" } else { "No" }.into();
        config.filterbank.ctrl = match mode {
            0 => ControlMode::Static,
            1 => ControlMode::Level,
            _ => ControlMode::Dynamic,
        };
        config.filterbank.num_update_asym_cmp = update;
        config.filterbank.gain_ref = GainReference::Db(50.0);
        config.peripheral.absolute_threshold_noise_level_db_spl =
            threshold_noise.then_some(20.0);
        config.peripheral.absolute_threshold_noise_seed = seed;
        config.ei.delay_convention = if symmetric_delay {
            EiDelayConvention::PaperSymmetric
        } else {
            EiDelayConvention::AmtOneSidedInteger
        };
        config.ei.internal_noise_std_mu = if internal_noise { 0.5 } else { 0.0 };
        config.ei.noise_seed = seed ^ 0xa5a5_5a5a_1234_5678;
        let units = [EiUnit::default(), EiUnit::new(1.0 / 8_000.0, 2.5)];

        let batch = hybrid_binaural(&left, &right, &units, config.clone()).unwrap();
        let mut stream = HybridBinauralStream::new(&units, config).unwrap();
        prop_assert_eq!(stream.center_frequencies_hz(), &batch.center_frequencies_hz);
        prop_assert_eq!(stream.units(), units.as_slice());
        let maximum_buffer = stream.max_buffered_ei_samples();
        let mut ei_events = Vec::new();
        for sample in 0..left.len() {
            let step = stream.process_sample(left[sample], right[sample]).unwrap();
            prop_assert_eq!(step.sample_index, sample);
            prop_assert!(stream.buffered_ei_samples() <= maximum_buffer);
            close_slice(
                step.left_filterbank.scgc_smpl.as_slice().unwrap(),
                &batch.left_filterbank.scgc_smpl.column(sample).to_vec(),
            );
            close_slice(
                step.right_filterbank.scgc_smpl.as_slice().unwrap(),
                &batch.right_filterbank.scgc_smpl.column(sample).to_vec(),
            );
            let Some(DcgcEvent::Sample {
                dcgc_out: left_dcgc,
                lvl_db: left_level,
                frat_val: left_ratio,
                fr2: left_center,
                ..
            }) = &step.left_filterbank.event else {
                panic!("hybrid stream must force sample-domain GCFB output");
            };
            let Some(DcgcEvent::Sample {
                dcgc_out: right_dcgc,
                lvl_db: right_level,
                frat_val: right_ratio,
                fr2: right_center,
                ..
            }) = &step.right_filterbank.event else {
                panic!("hybrid stream must force sample-domain GCFB output");
            };
            close_slice(
                left_dcgc.as_slice().unwrap(),
                &batch.left_filterbank.dcgc_out.column(sample).to_vec(),
            );
            close_slice(
                right_dcgc.as_slice().unwrap(),
                &batch.right_filterbank.dcgc_out.column(sample).to_vec(),
            );
            close_slice(
                step.left_internal.as_slice().unwrap(),
                &batch.left_internal.column(sample).to_vec(),
            );
            close_slice(
                step.right_internal.as_slice().unwrap(),
                &batch.right_internal.column(sample).to_vec(),
            );
            if mode == 2 {
                close_slice(
                    left_level.as_ref().unwrap().as_slice().unwrap(),
                    &batch.left_filterbank.gc_resp.lvl_db.column(sample).to_vec(),
                );
                close_slice(
                    right_level.as_ref().unwrap().as_slice().unwrap(),
                    &batch.right_filterbank.gc_resp.lvl_db.column(sample).to_vec(),
                );
                close_slice(
                    left_ratio.as_ref().unwrap().as_slice().unwrap(),
                    &batch.left_filterbank.gc_resp.frat_val.column(sample).to_vec(),
                );
                close_slice(
                    right_ratio.as_ref().unwrap().as_slice().unwrap(),
                    &batch.right_filterbank.gc_resp.frat_val.column(sample).to_vec(),
                );
                close_slice(
                    left_center.as_ref().unwrap().as_slice().unwrap(),
                    &batch.left_filterbank.gc_resp.fr2.column(sample).to_vec(),
                );
                close_slice(
                    right_center.as_ref().unwrap().as_slice().unwrap(),
                    &batch.right_filterbank.gc_resp.fr2.column(sample).to_vec(),
                );
            } else {
                prop_assert!(left_level.is_none() && left_ratio.is_none() && left_center.is_none());
                prop_assert!(right_level.is_none() && right_ratio.is_none() && right_center.is_none());
            }
            if let Some(event) = step.ei_event {
                ei_events.push(event);
            }
        }
        prop_assert_eq!(stream.samples_processed(), left.len());
        ei_events.extend(stream.finish().unwrap());
        prop_assert_eq!(ei_events.len(), left.len());
        for (sample, event) in ei_events.into_iter().enumerate() {
            prop_assert_eq!(event.sample_index, sample);
            for unit in 0..units.len() {
                close_slice(
                    event.activity.row(unit).as_slice().unwrap(),
                    &batch.ei_map.slice(ndarray::s![unit, .., sample]).to_vec(),
                );
            }
        }
    }
}

#[test]
fn streams_reject_acausal_modes_and_invalid_samples_without_advancing() {
    assert!(MonauralStream::new(2, 8_000.0, MonauralConfig::default()).is_err());
    assert!(EiStream::new(2, 8_000.0, &[EiUnit::default()], EiConfig::default()).is_err());
    assert!(
        HybridBinauralStream::new(&[EiUnit::default()], HybridBinauralConfig::default(),).is_err()
    );

    let mut monaural = MonauralStream::new(2, 8_000.0, MonauralConfig::streaming()).unwrap();
    assert!(matches!(
        monaural.process_sample(&[0.0, f64::NAN]),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(monaural.samples_processed(), 0);

    let mut ei = EiStream::new(2, 8_000.0, &[EiUnit::default()], EiConfig::streaming()).unwrap();
    assert!(ei.process_sample(&[0.0], &[0.0]).is_err());
    assert_eq!(ei.samples_processed(), 0);
    let after_error = ei
        .process_sample(&[1.0, 2.0], &[0.5, 1.5])
        .unwrap()
        .unwrap();
    let mut fresh = EiStream::new(2, 8_000.0, &[EiUnit::default()], EiConfig::streaming()).unwrap();
    let expected = fresh
        .process_sample(&[1.0, 2.0], &[0.5, 1.5])
        .unwrap()
        .unwrap();
    assert_eq!(after_error.activity, expected.activity);

    let mut hybrid_config = HybridBinauralConfig::streaming();
    hybrid_config.filterbank.fs = 8_000.0;
    hybrid_config.filterbank.num_ch = 2;
    hybrid_config.filterbank.f_range = [200.0, 1_500.0];
    hybrid_config.filterbank.out_mid_crct = "No".into();
    hybrid_config.filterbank.ctrl = ControlMode::Static;
    let mut hybrid = HybridBinauralStream::new(&[EiUnit::default()], hybrid_config).unwrap();
    assert!(hybrid.process_sample(f64::NAN, 0.0).is_err());
    assert_eq!(hybrid.samples_processed(), 0);

    assert!(
        EiStream::new(2, 8_000.0, &[EiUnit::default()], EiConfig::streaming(),)
            .unwrap()
            .finish()
            .is_err()
    );
}

#[test]
fn hybrid_stream_is_terminal_after_an_ear_processing_error() {
    let mut config = HybridBinauralConfig::streaming();
    config.filterbank.fs = 8_000.0;
    config.filterbank.num_ch = 2;
    config.filterbank.f_range = [200.0, 1_500.0];
    config.filterbank.out_mid_crct = "No".into();
    config.filterbank.ctrl = ControlMode::Dynamic;
    config.peripheral.absolute_threshold_noise_level_db_spl = None;
    config.ei.internal_noise_std_mu = 0.0;

    let mut stream = HybridBinauralStream::new(&[EiUnit::default()], config).unwrap();
    // The passive gammachirp starts at zero, so this first pair is accepted;
    // the large right-ear value reaches the level estimator on the next step.
    stream.process_sample(0.0, 1e300).unwrap();
    assert!(stream.process_sample(0.0, 0.0).is_err());
    assert!(matches!(
        stream.process_sample(0.0, 0.0),
        Err(Error::Numerical(message)) if message.contains("cannot continue")
    ));
    assert_eq!(stream.samples_processed(), 1);
    assert!(matches!(stream.finish(), Err(Error::Numerical(_))));
}

#[test]
fn symmetric_delay_buffer_remains_bounded_for_a_long_stream() {
    let units = [
        EiUnit::new(-4.5 / 8_000.0, -2.0),
        EiUnit::default(),
        EiUnit::new(4.5 / 8_000.0, 2.0),
    ];
    let mut stream = EiStream::new(3, 8_000.0, &units, EiConfig::streaming()).unwrap();
    let bound = stream.max_buffered_samples();
    assert!(bound < 64);
    for sample in 0..10_000 {
        let input = [sample as f64, sample as f64 + 1.0, sample as f64 + 2.0];
        stream.process_sample(&input, &input).unwrap();
        assert!(stream.buffered_samples() <= bound);
    }
    let tail = stream.finish().unwrap();
    assert!(tail.len() < bound);
}
