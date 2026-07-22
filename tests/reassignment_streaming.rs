use std::f64::consts::PI;

use gammachirp_rs::gcfb_v234::{
    BandwidthConsensusConfig, BandwidthConsensusStream, BandwidthConsensusStreamConfig,
    BandwidthConsensusStreamFrame, ControlMode, DcgcEvent, DynHpaf, GcParam, ReassignmentStream,
    gcfb_v234_with_bandwidth_consensus, gcfb_v234_with_phase_reassignment, reassign_gcfb_v234,
};
use ndarray::Array2;
use num_complex::Complex64;
use proptest::prelude::*;

fn close(actual: f64, expected: f64) {
    let tolerance = 2e-9 + 2e-8 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.16e} != {expected:.16e} (tolerance {tolerance:.3e})"
    );
}

fn linear_weights(axis: &[f64], value: f64) -> Option<[(usize, f64); 2]> {
    if axis.len() < 2 || !value.is_finite() || value < axis[0] || value > axis[axis.len() - 1] {
        return None;
    }
    match axis.binary_search_by(|candidate| candidate.total_cmp(&value)) {
        Ok(index) => Some([(index, 1.0), (index, 0.0)]),
        Err(upper) if upper > 0 && upper < axis.len() => {
            let lower = upper - 1;
            let upper_weight = (value - axis[lower]) / (axis[upper] - axis[lower]);
            Some([(lower, 1.0 - upper_weight), (upper, upper_weight)])
        }
        _ => None,
    }
}

struct CollectedStream {
    energy_map: Array2<f64>,
    complex_map: Array2<Complex64>,
    t_hat: Array2<f64>,
    f_hat: Array2<f64>,
    coordinate_mask: Array2<bool>,
    scgc_output: Array2<f64>,
    dcgc_output: Array2<f64>,
    maximum_buffer: usize,
}

fn collect_stream(signal: &[f64], parameters: GcParam) -> CollectedStream {
    let mut stream = ReassignmentStream::new(parameters).unwrap();
    let channels = stream.gc_param().num_ch;
    let time_axis: Vec<f64> = (0..signal.len())
        .map(|sample| sample as f64 / stream.gc_param().fs)
        .collect();
    let (frequency_axis, _) =
        gammachirp_rs::gcfb_v234::utils::freq2erb(stream.gc_param().fr1.as_slice().unwrap());
    let frequency_axis = frequency_axis.to_vec();
    let maximum_buffer = stream.max_buffered_samples();
    let mut t_hat = Array2::from_elem((channels, signal.len()), f64::NAN);
    let mut f_hat = t_hat.clone();
    let mut coordinate_mask = Array2::from_elem((channels, signal.len()), false);
    let mut source_energy = Array2::zeros((channels, signal.len()));
    let mut phase = Array2::from_elem((channels, signal.len()), Complex64::new(0.0, 0.0));
    let mut filterbank_output = Array2::zeros((channels, signal.len()));
    let mut dcgc_output = Array2::zeros((channels, signal.len()));
    for (sample_index, &sample) in signal.iter().enumerate() {
        let step = stream.process_sample(sample).unwrap();
        assert_eq!(step.filterbank.sample_index, sample_index);
        assert_eq!(stream.latency_samples(), 0);
        assert!(stream.buffered_samples() <= maximum_buffer);
        filterbank_output
            .column_mut(sample_index)
            .assign(&step.filterbank.scgc_smpl);
        let Some(gammachirp_rs::gcfb_v234::DcgcEvent::Sample { dcgc_out, .. }) =
            step.filterbank.event.as_ref()
        else {
            panic!("causal reassignment must contain a sample-domain dcGC event");
        };
        dcgc_output.column_mut(sample_index).assign(dcgc_out);
        t_hat.column_mut(sample_index).assign(&step.t_hat);
        f_hat.column_mut(sample_index).assign(&step.f_hat);
        coordinate_mask
            .column_mut(sample_index)
            .assign(&step.coordinate_mask);
        source_energy
            .column_mut(sample_index)
            .assign(&step.source_energy);
        phase
            .column_mut(sample_index)
            .assign(&step.phase_contribution);
    }
    assert_eq!(stream.samples_processed(), signal.len());

    let mut energy_map = Array2::zeros((channels, signal.len()));
    let mut complex_map = Array2::from_elem((channels, signal.len()), Complex64::new(0.0, 0.0));
    for ch in 0..channels {
        let maximum = source_energy.row(ch).iter().copied().fold(0.0, f64::max);
        let threshold = maximum * 1e-8;
        for sample in 0..signal.len() {
            if !coordinate_mask[[ch, sample]] || source_energy[[ch, sample]] < threshold {
                continue;
            }
            let Some(time_weights) = linear_weights(&time_axis, t_hat[[ch, sample]]) else {
                continue;
            };
            let frequency_hz = f_hat[[ch, sample]];
            if frequency_hz <= 0.0 {
                continue;
            }
            let (erb, _) = gammachirp_rs::gcfb_v234::utils::freq2erb(&[frequency_hz]);
            let Some(frequency_weights) = linear_weights(&frequency_axis, erb[0]) else {
                continue;
            };
            for (time_bin, time_weight) in time_weights {
                for (frequency_bin, frequency_weight) in frequency_weights {
                    let weight = time_weight * frequency_weight;
                    energy_map[[frequency_bin, time_bin]] += source_energy[[ch, sample]] * weight;
                    complex_map[[frequency_bin, time_bin]] += phase[[ch, sample]] * weight;
                }
            }
        }
    }
    CollectedStream {
        energy_map,
        complex_map,
        t_hat,
        f_hat,
        coordinate_mask,
        scgc_output: filterbank_output,
        dcgc_output,
        maximum_buffer,
    }
}

fn cosine_similarity(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum();
    let left_norm: f64 = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm: f64 = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    dot / (left_norm * right_norm)
}

fn complex_correlation(left: &Array2<Complex64>, right: &Array2<Complex64>) -> f64 {
    let dot: Complex64 = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right.conj())
        .sum();
    let left_norm: f64 = left
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let right_norm: f64 = right
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    dot.norm() / (left_norm * right_norm)
}

fn dominant_complex_correlation(
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
    energy: &Array2<f64>,
) -> f64 {
    let maximum = energy.iter().copied().fold(0.0, f64::max);
    let mut left = left.clone();
    let mut right = right.clone();
    for ((left, right), &energy) in left.iter_mut().zip(&mut right).zip(energy) {
        if energy < 0.01 * maximum {
            *left = Complex64::new(0.0, 0.0);
            *right = Complex64::new(0.0, 0.0);
        }
    }
    complex_correlation(&left, &right)
}

struct CollectedConsensusStream {
    scale_energy_maps: Vec<Array2<f64>>,
    agreement: Array2<f64>,
    consensus_mask: Array2<bool>,
    salience: Array2<f64>,
    scgc_output: Array2<f64>,
    dcgc_output: Array2<f64>,
    window_samples: usize,
    maximum_scale_buffer: usize,
}

fn assign_consensus_frame(
    frame: BandwidthConsensusStreamFrame,
    scale_energy_maps: &mut [Array2<f64>],
    agreement: &mut Array2<f64>,
    consensus_mask: &mut Array2<bool>,
    salience: &mut Array2<f64>,
) {
    let sample = frame.sample_index;
    for (map, column) in scale_energy_maps
        .iter_mut()
        .zip(&frame.scale_energy_columns)
    {
        map.column_mut(sample).assign(column);
    }
    agreement.column_mut(sample).assign(&frame.agreement);
    consensus_mask
        .column_mut(sample)
        .assign(&frame.consensus_mask);
    salience.column_mut(sample).assign(&frame.salience);
}

fn collect_consensus_stream(
    signal: &[f64],
    parameters: GcParam,
    config: BandwidthConsensusStreamConfig,
) -> CollectedConsensusStream {
    let mut stream = BandwidthConsensusStream::new(parameters, config).unwrap();
    let channels = stream.gc_param().num_ch;
    let scales = stream.scales().len();
    let baseline_index = stream.baseline_index();
    let window_samples = stream.window_samples();
    let maximum_scale_buffer = stream.max_buffered_scale_samples();
    let mut scale_energy_maps = (0..scales)
        .map(|_| Array2::zeros((channels, signal.len())))
        .collect::<Vec<_>>();
    let mut agreement = Array2::zeros((channels, signal.len()));
    let mut consensus_mask = Array2::from_elem((channels, signal.len()), false);
    let mut salience = Array2::zeros((channels, signal.len()));
    let mut scgc_output = Array2::zeros((channels, signal.len()));
    let mut dcgc_output = Array2::zeros((channels, signal.len()));
    let mut next_frame = 0;
    for (sample_index, &sample) in signal.iter().enumerate() {
        let step = stream.process_sample(sample).unwrap();
        assert_eq!(step.baseline_index, baseline_index);
        assert_eq!(step.scale_steps.len(), scales);
        assert_eq!(step.baseline().filterbank.sample_index, sample_index);
        scgc_output
            .column_mut(sample_index)
            .assign(&step.baseline().filterbank.scgc_smpl);
        let Some(gammachirp_rs::gcfb_v234::DcgcEvent::Sample { dcgc_out, .. }) =
            step.baseline().filterbank.event.as_ref()
        else {
            panic!("bandwidth consensus must contain a sample-domain baseline event");
        };
        dcgc_output.column_mut(sample_index).assign(dcgc_out);
        assert!(stream.buffered_target_samples() <= window_samples);
        assert_eq!(stream.max_buffered_target_samples(), window_samples + 1);
        assert!(stream.buffered_scale_samples() <= maximum_scale_buffer);
        if let Some(frame) = step.consensus {
            assert_eq!(frame.sample_index, next_frame);
            assign_consensus_frame(
                frame,
                &mut scale_energy_maps,
                &mut agreement,
                &mut consensus_mask,
                &mut salience,
            );
            next_frame += 1;
        }
    }
    assert_eq!(stream.samples_processed(), signal.len());
    for frame in stream.finish().unwrap() {
        assert_eq!(frame.sample_index, next_frame);
        assign_consensus_frame(
            frame,
            &mut scale_energy_maps,
            &mut agreement,
            &mut consensus_mask,
            &mut salience,
        );
        next_frame += 1;
    }
    assert_eq!(next_frame, signal.len());
    CollectedConsensusStream {
        scale_energy_maps,
        agreement,
        consensus_mask,
        salience,
        scgc_output,
        dcgc_output,
        window_samples,
        maximum_scale_buffer,
    }
}

fn interior_cosine_similarity(left: &Array2<f64>, right: &Array2<f64>, edge: usize) -> f64 {
    let end = left.ncols().saturating_sub(edge);
    let mut dot = 0.0;
    let mut left_power = 0.0;
    let mut right_power = 0.0;
    for ch in 0..left.nrows() {
        for sample in edge.min(end)..end {
            dot += left[[ch, sample]] * right[[ch, sample]];
            left_power += left[[ch, sample]].powi(2);
            right_power += right[[ch, sample]].powi(2);
        }
    }
    dot / (left_power * right_power).sqrt()
}

fn interior_mask_iou(left: &Array2<bool>, right: &Array2<bool>, edge: usize) -> f64 {
    let end = left.ncols().saturating_sub(edge);
    let mut intersection = 0;
    let mut union = 0;
    for ch in 0..left.nrows() {
        for sample in edge.min(end)..end {
            intersection += usize::from(left[[ch, sample]] && right[[ch, sample]]);
            union += usize::from(left[[ch, sample]] || right[[ch, sample]]);
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 6,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn causal_stream_is_similar_to_batch_reassignment(
        channels in 3usize..7,
        mode in 0u8..3,
        corrected in any::<bool>(),
        impaired in any::<bool>(),
        update in 1usize..8,
        frequency_hz in 350.0f64..1_350.0,
        amplitude in 0.05f64..0.8,
        phase in -PI..PI,
    ) {
        let samples = 1_536;
        let signal: Vec<f64> = (0..samples)
            .map(|sample| {
                amplitude
                    * (2.0 * PI * frequency_hz * sample as f64 / 8_000.0 + phase).cos()
            })
            .collect();
        let ctrl = match mode {
            0 => ControlMode::Static,
            1 => ControlMode::Level,
            _ => ControlMode::Dynamic,
        };
        let parameters = GcParam {
            fs: 8_000.0,
            num_ch: channels,
            f_range: [250.0, 1_800.0],
            out_mid_crct: if corrected { "ELC" } else { "No" }.into(),
            ctrl,
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            hloss_type: if impaired { "HL3" } else { "NH" }.into(),
            num_update_asym_cmp: update,
            ..GcParam::default()
        };
        let (batch_output, batch) =
            gcfb_v234_with_phase_reassignment(&signal, parameters.clone()).unwrap();
        let batch_energy = reassign_gcfb_v234(&signal, &batch_output).unwrap();
        prop_assert_eq!(&batch_energy.energy_map, &batch.reassignment.energy_map);
        prop_assert_eq!(&batch_energy.t_hat, &batch.reassignment.t_hat);
        prop_assert_eq!(&batch_energy.f_hat, &batch.reassignment.f_hat);
        let stream = collect_stream(&signal, parameters);

        for (actual, expected) in stream.scgc_output.iter().zip(&batch_output.scgc_smpl) {
            close(*actual, *expected);
        }
        for (actual, expected) in stream.dcgc_output.iter().zip(&batch_output.dcgc_out) {
            close(*actual, *expected);
        }

        let energy_similarity =
            cosine_similarity(&stream.energy_map, &batch.reassignment.energy_map);
        prop_assert!(
            energy_similarity >= 0.55,
            "causal/batch energy-map cosine similarity was {energy_similarity}"
        );
        let energy_ratio = stream.energy_map.sum() / batch.reassignment.energy_map.sum();
        prop_assert!(
            (0.35..=1.65).contains(&energy_ratio),
            "causal/batch retained-energy ratio was {energy_ratio}"
        );

        let phase_similarity = dominant_complex_correlation(
            &stream.complex_map,
            &batch.complex_map,
            &batch.reassignment.energy_map,
        );
        prop_assert!(
            phase_similarity >= 0.30,
            "causal/batch dominant complex correlation was {phase_similarity}"
        );

        let start = stream.maximum_buffer.min(samples / 3);
        let end = samples - start;
        let mut weight_sum = 0.0;
        let mut time_error = 0.0;
        let mut frequency_error = 0.0;
        for ch in 0..channels {
            for sample in start..end {
                let weight = batch.unreassigned_energy_map[[ch, sample]];
                if weight == 0.0 || !stream.coordinate_mask[[ch, sample]] {
                    continue;
                }
                weight_sum += weight;
                time_error += weight
                    * (stream.t_hat[[ch, sample]] - batch.reassignment.t_hat[[ch, sample]]).abs();
                frequency_error += weight
                    * (stream.f_hat[[ch, sample]] - batch.reassignment.f_hat[[ch, sample]]).abs();
            }
        }
        prop_assert!(weight_sum > 0.0);
        let mean_time_error = time_error / weight_sum;
        let mean_frequency_error = frequency_error / weight_sum;
        let maximum_channel_spacing = batch
            .reassignment
            .frequency_axis_hz
            .windows(2)
            .into_iter()
            .map(|window| window[1] - window[0])
            .fold(0.0, f64::max);
        prop_assert!(
            mean_time_error <= stream.maximum_buffer as f64 / 8_000.0,
            "causal/batch energy-weighted time error was {mean_time_error} seconds"
        );
        prop_assert!(
            mean_frequency_error <= maximum_channel_spacing,
            "causal/batch energy-weighted frequency error was {mean_frequency_error} Hz"
        );
    }
}

#[test]
fn dynamic_stream_frequency_is_the_actual_causal_phase_increment() {
    let sample_rate = 8_000.0;
    let mut stream = ReassignmentStream::new(GcParam {
        fs: sample_rate,
        num_ch: 4,
        f_range: [250.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Dynamic,
        dyn_hpaf: DynHpaf {
            str_prc: "sample-base".into(),
            ..DynHpaf::default()
        },
        num_update_asym_cmp: 1,
        ..GcParam::default()
    })
    .unwrap();
    let mut previous = [Complex64::new(0.0, 0.0); 4];
    let mut comparisons = 0;
    for sample in 0..512 {
        let amplitude = 0.05 + 0.45 * sample as f64 / 511.0;
        let input = amplitude * (2.0 * PI * 700.0 * sample as f64 / sample_rate).cos();
        let step = stream.process_sample(input).unwrap();
        for (ch, previous) in previous.iter_mut().enumerate() {
            if step.coordinate_mask[ch] && previous.norm_sqr() > 0.0 {
                let increment = step.coefficient[ch] * previous.conj();
                let expected = sample_rate * increment.im.atan2(increment.re) / (2.0 * PI);
                close(step.f_hat[ch], expected);
                comparisons += 1;
            }
            *previous = step.coefficient[ch];
        }
    }
    assert!(comparisons > 1_000);
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn rolling_bandwidth_consensus_is_similar_to_batch_consensus(
        channels in 3usize..7,
        mode in 0u8..3,
        corrected in any::<bool>(),
        impaired in any::<bool>(),
        update in 1usize..8,
        frequency_hz in 350.0f64..1_350.0,
        amplitude in 0.05f64..0.8,
        phase in -PI..PI,
        window_samples in 256usize..385,
    ) {
        let samples = 1_536;
        let signal: Vec<f64> = (0..samples)
            .map(|sample| {
                amplitude
                    * (2.0 * PI * frequency_hz * sample as f64 / 8_000.0 + phase).cos()
            })
            .collect();
        let ctrl = match mode {
            0 => ControlMode::Static,
            1 => ControlMode::Level,
            _ => ControlMode::Dynamic,
        };
        let parameters = GcParam {
            fs: 8_000.0,
            num_ch: channels,
            f_range: [250.0, 1_800.0],
            out_mid_crct: if corrected { "ELC" } else { "No" }.into(),
            ctrl,
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            hloss_type: if impaired { "HL3" } else { "NH" }.into(),
            num_update_asym_cmp: update,
            ..GcParam::default()
        };
        let batch_config = BandwidthConsensusConfig::default();
        let batch_result = gcfb_v234_with_bandwidth_consensus(
            &signal,
            parameters.clone(),
            &batch_config,
        );
        if let Err(error) = &batch_result {
            prop_assert!(
                matches!(error, gammachirp_rs::Error::Unsupported(_)),
                "unexpected bandwidth-consensus error: {error}"
            );
            prop_assume!(false);
        }
        let (batch_output, batch) = batch_result.unwrap();
        let stream = collect_consensus_stream(
            &signal,
            parameters,
            BandwidthConsensusStreamConfig {
                scales: batch_config.scales.clone(),
                relative_support_floor: batch_config.relative_support_floor,
                required_agreement: batch_config.required_agreement,
                window_samples: Some(window_samples),
            },
        );

        for (actual, expected) in stream.scgc_output.iter().zip(&batch_output.scgc_smpl) {
            close(*actual, *expected);
        }
        for (actual, expected) in stream.dcgc_output.iter().zip(&batch_output.dcgc_out) {
            close(*actual, *expected);
        }
        prop_assert_eq!(stream.scale_energy_maps.len(), batch.analyses.len());
        for (scale, analysis) in stream.scale_energy_maps.iter().zip(&batch.analyses) {
            let similarity = cosine_similarity(scale, &analysis.reassignment.energy_map);
            prop_assert!(
                similarity >= 0.30,
                "rolling/batch scale energy-map cosine similarity was {similarity}"
            );
            let ratio = scale.sum() / analysis.reassignment.energy_map.sum();
            prop_assert!(
                (0.25..=1.75).contains(&ratio),
                "rolling/batch scale retained-energy ratio was {ratio}"
            );
        }

        let edge = stream.window_samples.min(samples / 3);
        let salience_similarity =
            interior_cosine_similarity(&stream.salience, &batch.salience_map, edge);
        prop_assert!(
            salience_similarity >= 0.50,
            "rolling/batch salience cosine similarity was {salience_similarity}"
        );
        let mask_iou = interior_mask_iou(&stream.consensus_mask, &batch.consensus_mask, edge);
        prop_assert!(
            mask_iou >= 0.50,
            "rolling/batch consensus-mask intersection over union was {mask_iou}"
        );
        let tone_channel = batch_output
            .gc_param
            .fr1
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left - frequency_hz)
                    .abs()
                    .total_cmp(&(*right - frequency_hz).abs())
            })
            .unwrap()
            .0;
        let tone_agreement = stream
            .agreement
            .row(tone_channel)
            .iter()
            .skip(edge)
            .take(samples - 2 * edge)
            .copied()
            .fold(0.0, f64::max);
        prop_assert_eq!(tone_agreement, 1.0);
        prop_assert!(stream
            .agreement
            .iter()
            .all(|agreement| (0.0..=1.0).contains(agreement)));
        prop_assert!(stream.maximum_scale_buffer > 0);
    }
}

fn assert_scaled_consensus_frame(
    baseline: &BandwidthConsensusStreamFrame,
    scaled: &BandwidthConsensusStreamFrame,
    scale: f64,
) {
    assert_eq!(baseline.sample_index, scaled.sample_index);
    assert_eq!(baseline.consensus_mask, scaled.consensus_mask);
    for (actual, expected) in scaled.agreement.iter().zip(&baseline.agreement) {
        close(*actual, *expected);
    }
    for (actual, expected) in scaled.salience.iter().zip(&baseline.salience) {
        close(*actual, *expected);
    }
    for (actual, expected) in scaled
        .normalization_maxima
        .iter()
        .zip(&baseline.normalization_maxima)
    {
        close(*actual, scale.powi(2) * expected);
    }
    for (actual_columns, expected_columns) in scaled
        .scale_energy_columns
        .iter()
        .zip(&baseline.scale_energy_columns)
    {
        for (actual, expected) in actual_columns.iter().zip(expected_columns) {
            close(*actual, scale.powi(2) * expected);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 10,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn rolling_consensus_is_invariant_to_uniform_amplitude_scaling(
        values in prop::collection::vec(any::<i16>(), 1..160),
        scale in 0.25f64..3.0,
        window_samples in 1usize..48,
    ) {
        let parameters = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [250.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            ..GcParam::default()
        };
        let config = BandwidthConsensusStreamConfig {
            scales: vec![0.9, 1.0],
            window_samples: Some(window_samples),
            ..BandwidthConsensusStreamConfig::default()
        };
        let mut baseline = BandwidthConsensusStream::new(parameters.clone(), config.clone()).unwrap();
        let mut scaled = BandwidthConsensusStream::new(parameters, config).unwrap();
        for value in values {
            let sample = value as f64 / 32_768.0;
            let low = baseline.process_sample(sample).unwrap();
            let high = scaled.process_sample(scale * sample).unwrap();
            prop_assert_eq!(low.consensus.is_some(), high.consensus.is_some());
            if let (Some(low), Some(high)) = (&low.consensus, &high.consensus) {
                assert_scaled_consensus_frame(low, high, scale);
            }
        }
        let low_tail = baseline.finish().unwrap();
        let high_tail = scaled.finish().unwrap();
        prop_assert_eq!(low_tail.len(), high_tail.len());
        for (low, high) in low_tail.iter().zip(&high_tail) {
            assert_scaled_consensus_frame(low, high, scale);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 12,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn causal_stream_scales_phase_linearly_and_energy_quadratically(
        values in prop::collection::vec(any::<i16>(), 1..160),
        scale in 0.25f64..3.0,
    ) {
        let parameters = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [250.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            ..GcParam::default()
        };
        let mut baseline = ReassignmentStream::new(parameters.clone()).unwrap();
        let mut scaled = ReassignmentStream::new(parameters).unwrap();
        for value in values {
            let sample = value as f64 / 32_768.0;
            let low = baseline.process_sample(sample).unwrap();
            let high = scaled.process_sample(scale * sample).unwrap();
            prop_assert_eq!(&low.coordinate_mask, &high.coordinate_mask);
            for ch in 0..low.source_energy.len() {
                close(high.source_energy[ch], scale.powi(2) * low.source_energy[ch]);
                close(
                    high.phase_contribution[ch].re,
                    scale * low.phase_contribution[ch].re,
                );
                close(
                    high.phase_contribution[ch].im,
                    scale * low.phase_contribution[ch].im,
                );
                if low.coordinate_mask[ch] {
                    close(high.t_hat[ch], low.t_hat[ch]);
                    close(high.f_hat[ch], low.f_hat[ch]);
                }
            }
        }
    }
}

#[test]
fn phase_power_and_memory_bound_hold_for_an_indefinite_stream_prefix() {
    let mut stream = ReassignmentStream::new(GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [250.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        ..GcParam::default()
    })
    .unwrap();
    let bound = stream.max_buffered_samples();
    assert!(bound > 0);
    for sample in 0..(10 * bound) {
        let input = (2.0 * PI * 700.0 * sample as f64 / 8_000.0).sin();
        let step = stream.process_sample(input).unwrap();
        assert_eq!(step.filterbank.sample_index, sample);
        assert!(stream.buffered_samples() <= bound);
        for ch in 0..step.source_energy.len() {
            assert!(step.source_energy[ch].is_finite() && step.source_energy[ch] >= 0.0);
            if step.coordinate_mask[ch] {
                close(
                    step.phase_contribution[ch].norm_sqr(),
                    step.source_energy[ch],
                );
                assert!(step.t_hat[ch].is_finite());
                assert!(step.f_hat[ch].is_finite());
            }
        }
    }
    assert_eq!(stream.samples_processed(), 10 * bound);
    assert_eq!(stream.buffered_samples(), bound);
}

#[test]
fn frame_mode_is_rejected_and_invalid_input_does_not_advance() {
    let frame_parameters = GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [250.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Dynamic,
        dyn_hpaf: DynHpaf {
            str_prc: "frame-base".into(),
            ..DynHpaf::default()
        },
        ..GcParam::default()
    };
    assert!(ReassignmentStream::new(frame_parameters).is_err());

    let mut stream = ReassignmentStream::new(GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [250.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        ..GcParam::default()
    })
    .unwrap();
    assert!(stream.process_sample(f64::NAN).is_err());
    assert_eq!(stream.samples_processed(), 0);
    assert_eq!(stream.buffered_samples(), 0);
    assert_eq!(
        stream.process_sample(1.0).unwrap().filterbank.sample_index,
        0
    );
}

#[test]
fn rolling_consensus_memory_is_bounded_for_an_indefinite_prefix() {
    let window_samples = 37;
    let mut stream = BandwidthConsensusStream::new(
        GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [250.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            ..GcParam::default()
        },
        BandwidthConsensusStreamConfig {
            window_samples: Some(window_samples),
            ..BandwidthConsensusStreamConfig::default()
        },
    )
    .unwrap();
    let scale_bound = stream.max_buffered_scale_samples();
    let mut next_frame = 0;
    for sample in 0..(20 * window_samples) {
        let input = (2.0 * PI * 700.0 * sample as f64 / 8_000.0).sin();
        let step = stream.process_sample(input).unwrap();
        assert_eq!(step.scale_steps.len(), 3);
        assert!(stream.buffered_target_samples() <= window_samples);
        assert_eq!(stream.max_buffered_target_samples(), window_samples + 1);
        assert!(stream.buffered_scale_samples() <= scale_bound);
        if let Some(frame) = step.consensus {
            assert_eq!(frame.sample_index, next_frame);
            next_frame += 1;
        }
    }
    assert_eq!(stream.samples_processed(), 20 * window_samples);
    assert_eq!(next_frame, 19 * window_samples + 1);
}

#[test]
fn rolling_consensus_validates_configuration_and_input_atomically() {
    let parameters = GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [250.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        ..GcParam::default()
    };
    for config in [
        BandwidthConsensusStreamConfig {
            scales: vec![1.0],
            ..BandwidthConsensusStreamConfig::default()
        },
        BandwidthConsensusStreamConfig {
            scales: vec![0.8, 1.2],
            ..BandwidthConsensusStreamConfig::default()
        },
        BandwidthConsensusStreamConfig {
            window_samples: Some(0),
            ..BandwidthConsensusStreamConfig::default()
        },
    ] {
        assert!(BandwidthConsensusStream::new(parameters.clone(), config).is_err());
    }

    let mut stream = BandwidthConsensusStream::new(
        parameters,
        BandwidthConsensusStreamConfig {
            window_samples: Some(8),
            ..BandwidthConsensusStreamConfig::default()
        },
    )
    .unwrap();
    assert_eq!(stream.scale_metadata().len(), stream.scales().len());
    let baseline_peaks =
        &stream.scale_metadata()[stream.baseline_index()].nominal_peak_frequencies_hz;
    let peak_fft_len = stream.scale_metadata()[stream.baseline_index()].peak_grid_fft_len;
    let peak_spacing = stream.scale_metadata()[stream.baseline_index()].peak_grid_spacing_hz;
    assert!(peak_fft_len >= 65_536);
    assert!(peak_fft_len.is_power_of_two());
    for (scale, metadata) in stream.scales().iter().zip(stream.scale_metadata()) {
        close(metadata.scale, *scale);
        assert_eq!(&metadata.nominal_peak_frequencies_hz, baseline_peaks);
        assert_eq!(metadata.peak_grid_fft_len, peak_fft_len);
        assert_eq!(metadata.peak_grid_spacing_hz, peak_spacing);
    }
    assert!(stream.process_sample(f64::NAN).is_err());
    assert_eq!(stream.samples_processed(), 0);
    assert_eq!(stream.buffered_target_samples(), 0);
    assert_eq!(
        stream
            .process_sample(1.0)
            .unwrap()
            .baseline()
            .filterbank
            .sample_index,
        0
    );

    let derived = BandwidthConsensusStream::new(
        GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [250.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            ..GcParam::default()
        },
        BandwidthConsensusStreamConfig::default(),
    )
    .unwrap();
    assert_eq!(
        derived.window_samples(),
        derived.max_buffered_scale_samples()
    );
}

#[test]
fn rolling_consensus_tolerates_unreachable_discrete_peak_bins() {
    let sample_rate = 16_000.0;
    let samples = 64_usize;
    let mut stream = BandwidthConsensusStream::new(
        GcParam {
            fs: sample_rate,
            num_ch: 16,
            f_range: [100.0, 6_000.0],
            out_mid_crct: "ELC".into(),
            ctrl: ControlMode::Dynamic,
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            lvl_est: gammachirp_rs::gcfb_v234::gcfb_v234::LvlEst {
                rms2spldb: 100.0,
                ..Default::default()
            },
            num_update_asym_cmp: 3,
            ..GcParam::default()
        },
        BandwidthConsensusStreamConfig::default(),
    )
    .unwrap();
    let mut held_centers = None;
    let mut update_ratios = None;
    let mut ratio_changed_while_center_was_held = false;
    for sample in 0..samples {
        let input = 0.2 * (2.0 * PI * 1_000.0 * sample as f64 / sample_rate).cos();
        let step = stream.process_sample(input).unwrap();
        let Some(DcgcEvent::Sample {
            frat_val: Some(ratios),
            fr2: Some(centers),
            ..
        }) = step.baseline().filterbank.event.as_ref()
        else {
            panic!("dynamic consensus baseline must emit sample-domain centers and ratios");
        };
        if sample.is_multiple_of(3) {
            held_centers = Some(centers.clone());
            update_ratios = Some(ratios.clone());
        } else {
            assert_eq!(centers, held_centers.as_ref().unwrap());
            ratio_changed_while_center_was_held |= ratios != update_ratios.as_ref().unwrap();
        }
        if let Some(frame) = step.consensus {
            assert!(
                frame
                    .salience
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            );
        }
    }
    assert!(ratio_changed_while_center_was_held);
    assert_eq!(stream.samples_processed(), samples);
}
