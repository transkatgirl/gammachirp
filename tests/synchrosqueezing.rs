use std::f64::consts::PI;

use gammachirp_rs::Error;
use gammachirp_rs::gcfb_v234::{
    ControlMode, DynHpaf, GainReference, GcParam, SynchrosqueezingMode, SynchrosqueezingStream,
    gcfb_v234, gcfb_v234_with_synchrosqueezing, reassign_gcfb_v234, synchrosqueeze_gcfb_v234,
};
use ndarray::Array1;
use num_complex::Complex64;

fn static_parameters() -> GcParam {
    GcParam {
        fs: 8_000.0,
        num_ch: 8,
        f_range: [200.0, 1_800.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        gain_ref: GainReference::Db(50.0),
        ..GcParam::default()
    }
}

fn sample_parameters() -> GcParam {
    GcParam {
        ctrl: ControlMode::Dynamic,
        dyn_hpaf: DynHpaf {
            str_prc: "sample-base".into(),
            ..DynHpaf::default()
        },
        ..static_parameters()
    }
}

fn tone(frequency_hz: f64, samples: usize, scale: f64) -> Vec<f64> {
    (0..samples)
        .map(|sample| scale * (2.0 * PI * frequency_hz * sample as f64 / 8_000.0).cos())
        .collect()
}

fn close(actual: f64, expected: f64) {
    let tolerance = 2e-10 + 2e-8 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.16e} != {expected:.16e} (tolerance {tolerance:.3e})"
    );
}

fn close_complex(actual: Complex64, expected: Complex64) {
    close(actual.re, expected.re);
    close(actual.im, expected.im);
}

fn nearest(axis: &[f64], frequency_hz: f64) -> Option<usize> {
    if !frequency_hz.is_finite()
        || frequency_hz <= 0.0
        || frequency_hz < axis[0]
        || frequency_hz > axis[axis.len() - 1]
    {
        return None;
    }
    Some(
        axis.iter()
            .enumerate()
            .min_by(|(left_index, left), (right_index, right)| {
                let left_distance = (**left - frequency_hz).abs();
                let right_distance = (**right - frequency_hz).abs();
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| left_index.cmp(right_index))
            })
            .unwrap()
            .0,
    )
}

#[test]
fn batch_wrapper_preserves_filterbank_and_reuses_reassignment_frequencies() {
    let signal = tone(700.0, 320, 0.2);
    let parameters = static_parameters();
    let direct = gcfb_v234(&signal, parameters.clone()).unwrap();
    let (wrapped, squeezed) = gcfb_v234_with_synchrosqueezing(&signal, parameters).unwrap();
    assert_eq!(wrapped.scgc_smpl, direct.scgc_smpl);
    assert_eq!(wrapped.dcgc_out, direct.dcgc_out);

    let reassigned = reassign_gcfb_v234(&signal, &direct).unwrap();
    assert_eq!(squeezed.validity_mask, reassigned.validity_mask);
    for (&actual, &expected) in squeezed.f_hat.iter().zip(&reassigned.f_hat) {
        if actual.is_nan() || expected.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual, expected);
        }
    }
    assert_eq!(squeezed.mode, SynchrosqueezingMode::Fixed);
    assert_eq!(squeezed.time_axis.len(), signal.len());
    for (sample, &time) in squeezed.time_axis.iter().enumerate() {
        assert_eq!(time, sample as f64 / 8_000.0);
    }

    let target = nearest(squeezed.frequency_axis_hz.as_slice().unwrap(), 700.0).unwrap();
    let dominant = (0..squeezed.energy_map.nrows())
        .max_by(|&left, &right| {
            squeezed
                .energy_map
                .row(left)
                .sum()
                .total_cmp(&squeezed.energy_map.row(right).sum())
        })
        .unwrap();
    assert_eq!(dominant, target);
}

#[test]
fn batch_complex_and_energy_maps_scale_linearly_and_quadratically() {
    let low_signal = tone(850.0, 256, 0.1);
    let scale = 2.5;
    let high_signal: Vec<f64> = low_signal.iter().map(|sample| scale * sample).collect();
    let low = synchrosqueeze_gcfb_v234(
        &low_signal,
        &gcfb_v234(&low_signal, static_parameters()).unwrap(),
    )
    .unwrap();
    let high = synchrosqueeze_gcfb_v234(
        &high_signal,
        &gcfb_v234(&high_signal, static_parameters()).unwrap(),
    )
    .unwrap();
    assert_eq!(low.validity_mask, high.validity_mask);
    for (&actual, &reference) in high.complex_map.iter().zip(&low.complex_map) {
        close_complex(actual, reference * scale);
    }
    for (&actual, &reference) in high.energy_map.iter().zip(&low.energy_map) {
        close(actual, reference * scale.powi(2));
    }
    close(high.source_energy, low.source_energy * scale.powi(2));
    close(high.discarded_energy, low.discarded_energy * scale.powi(2));
}

#[test]
fn subnormal_signals_keep_relative_floors_scale_safe() {
    let signal = tone(700.0, 128, 1e-160);
    let output = gcfb_v234(&signal, static_parameters()).unwrap();
    let reassigned = reassign_gcfb_v234(&signal, &output).unwrap();
    let squeezed = synchrosqueeze_gcfb_v234(&signal, &output).unwrap();

    assert!(reassigned.source_energy > 0.0);
    assert!(squeezed.source_energy > 0.0);
}

#[test]
fn sample_dynamic_batch_is_conditional_and_energy_conserving() {
    let signal = tone(650.0, 256, 0.15);
    let (_, squeezed) = gcfb_v234_with_synchrosqueezing(&signal, sample_parameters()).unwrap();
    assert_eq!(squeezed.mode, SynchrosqueezingMode::SampleConditional);
    assert_eq!(squeezed.energy_map.dim(), (8, signal.len()));
    close(
        squeezed.source_energy,
        squeezed.retained_energy() + squeezed.discarded_energy,
    );
}

#[test]
fn dynamic_batch_stream_and_existing_output_share_peak_locked_analysis() {
    let mut parameters = sample_parameters();
    parameters.num_update_asym_cmp = 3;
    let signal: Vec<f64> = tone(650.0, 192, 0.15)
        .into_iter()
        .enumerate()
        .map(|(sample, value)| value * (0.2 + 0.8 * sample as f64 / 191.0))
        .collect();
    let ordinary = gcfb_v234(&signal, parameters.clone()).unwrap();
    let (batch, squeezed) = gcfb_v234_with_synchrosqueezing(&signal, parameters.clone()).unwrap();
    let existing = synchrosqueeze_gcfb_v234(&signal, &ordinary).unwrap();

    assert_eq!(squeezed.f_hat, existing.f_hat);
    assert_eq!(squeezed.complex_map, existing.complex_map);
    assert_eq!(squeezed.energy_map, existing.energy_map);

    let mut different_signal = signal.clone();
    different_signal[0] += 0.5;
    assert!(matches!(
        synchrosqueeze_gcfb_v234(&different_signal, &ordinary),
        Err(Error::InvalidParameter(_))
    ));

    let mut stream = SynchrosqueezingStream::new(parameters).unwrap();
    for (sample_index, &sample) in signal.iter().enumerate() {
        let step = stream.process_sample(sample).unwrap();
        for channel in 0..batch.gc_param.num_ch {
            close(
                step.filterbank.scgc_smpl[channel],
                batch.scgc_smpl[[channel, sample_index]],
            );
        }
        let gammachirp_rs::gcfb_v234::DcgcEvent::Sample {
            dcgc_out,
            fr2: Some(fr2),
            ..
        } = step.filterbank.event.unwrap()
        else {
            panic!("dynamic synchrosqueezing stream must emit peak-locked samples");
        };
        for channel in 0..batch.gc_param.num_ch {
            close(dcgc_out[channel], batch.dcgc_out[[channel, sample_index]]);
            close(fr2[channel], batch.gc_resp.fr2[[channel, sample_index]]);
        }
    }
}

#[test]
fn level_mode_is_supported_by_batch_and_stream() {
    let parameters = GcParam {
        ctrl: ControlMode::Level,
        ..static_parameters()
    };
    let signal = tone(500.0, 128, 0.1);
    let (_, squeezed) = gcfb_v234_with_synchrosqueezing(&signal, parameters.clone()).unwrap();
    assert_eq!(squeezed.mode, SynchrosqueezingMode::Fixed);
    assert!(squeezed.source_energy > 0.0);

    let mut stream = SynchrosqueezingStream::new(parameters).unwrap();
    let step = stream.process_sample(signal[0]).unwrap();
    assert_eq!(step.energy_column.len(), 8);
}

#[test]
fn stream_columns_match_their_public_source_diagnostics() {
    let mut stream = SynchrosqueezingStream::new(static_parameters()).unwrap();
    let axis = stream.frequency_axis_hz().clone();
    let gains = stream.gc_resp().gain_factor.clone();
    let maximum_buffer = stream.max_buffered_samples();
    for (sample_index, sample) in tone(700.0, 384, 0.2).into_iter().enumerate() {
        let step = stream.process_sample(sample).unwrap();
        assert_eq!(step.filterbank.sample_index, sample_index);
        assert!(stream.buffered_samples() <= maximum_buffer);
        let mut expected_complex = Array1::from_elem(axis.len(), Complex64::new(0.0, 0.0));
        let mut expected_energy = Array1::zeros(axis.len());
        let mut unresolved = 0.0;
        let mut boundary = 0.0;
        for source_channel in 0..axis.len() {
            let energy = step.source_energy[source_channel];
            if energy == 0.0 {
                continue;
            }
            if !step.validity_mask[source_channel] {
                unresolved += energy;
            } else if let Some(target) =
                nearest(axis.as_slice().unwrap(), step.f_hat[source_channel])
            {
                expected_complex[target] +=
                    step.coefficient[source_channel] * gains[source_channel];
                expected_energy[target] += energy;
            } else {
                boundary += energy;
            }
        }
        for (&actual, &expected) in step.complex_column.iter().zip(&expected_complex) {
            close_complex(actual, expected);
        }
        for (&actual, &expected) in step.energy_column.iter().zip(&expected_energy) {
            close(actual, expected);
        }
        close(step.frequency_unresolved_energy, unresolved);
        close(step.boundary_discarded_energy, boundary);
        close(
            step.source_energy.sum(),
            step.retained_energy() + step.discarded_energy,
        );
    }
    assert_eq!(stream.samples_processed(), 384);
    assert_eq!(stream.latency_samples(), 0);

    let before = stream.samples_processed();
    assert!(matches!(
        stream.process_sample(f64::NAN),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(stream.samples_processed(), before);
}

#[test]
fn dynamic_stream_reports_unresolved_startup_then_valid_frequencies() {
    let mut stream = SynchrosqueezingStream::new(sample_parameters()).unwrap();
    let mut saw_unresolved = false;
    let mut saw_retained = false;
    for sample in std::iter::once(1.0).chain(std::iter::repeat_n(0.0, 511)) {
        let step = stream.process_sample(sample).unwrap();
        saw_unresolved |= step.frequency_unresolved_energy > 0.0;
        saw_retained |= step.retained_energy() > 0.0;
    }
    assert!(saw_unresolved);
    assert!(saw_retained);
}

#[test]
fn batch_and_stream_reject_dynamic_frame_mode() {
    let frame = GcParam {
        ctrl: ControlMode::Dynamic,
        dyn_hpaf: DynHpaf {
            str_prc: "frame-base".into(),
            ..DynHpaf::default()
        },
        ..static_parameters()
    };
    let signal = vec![0.0; 64];
    assert!(matches!(
        gcfb_v234_with_synchrosqueezing(&signal, frame.clone()),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        SynchrosqueezingStream::new(frame),
        Err(Error::Unsupported(_))
    ));
}
