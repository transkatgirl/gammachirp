use gammachirp_rs::{Error, gcfb_v211, gcfb_v234};
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

fn test_signal(values: &[i16]) -> Vec<f64> {
    values
        .iter()
        .map(|&value| value as f64 / 32_768.0)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn v211_stream_matches_batch(
        values in prop::collection::vec(any::<i16>(), 1..100),
        channels in 2usize..7,
        mode in 0u8..3,
        update in 1usize..9,
        corrected in any::<bool>(),
        gain_ref in 35.0f64..75.0,
        alternate_coefficients in any::<bool>(),
    ) {
        let signal = test_signal(&values);
        let ctrl = match mode {
            0 => gcfb_v211::ControlMode::Static,
            1 => gcfb_v211::ControlMode::Level,
            _ => gcfb_v211::ControlMode::Dynamic,
        };
        let param = gcfb_v211::GcParam {
            fs: 8_000.0,
            num_ch: channels,
            f_range: [180.0, 1_800.0],
            out_mid_crct: if corrected { "ELC" } else { "No" }.into(),
            ctrl,
            num_update_asym_cmp: update,
            gain_ref_db: gain_ref,
            b1: if alternate_coefficients { [1.70, 0.02] } else { [1.81, 0.0] },
            c1: if alternate_coefficients { [-2.70, -0.02] } else { [-2.96, 0.0] },
            frat: if alternate_coefficients {
                [[0.48, 0.005], [0.01, 0.0002]]
            } else {
                [[0.4660, 0.0], [0.0109, 0.0]]
            },
            b2: if alternate_coefficients {
                [[2.0, 0.05], [0.0, 0.0]]
            } else {
                [[2.17, 0.0], [0.0, 0.0]]
            },
            c2: if alternate_coefficients {
                [[2.0, -0.05], [0.0, 0.0]]
            } else {
                [[2.2, 0.0], [0.0, 0.0]]
            },
            ..gcfb_v211::GcParam::default()
        };
        let batch = gcfb_v211::gcfb_v211(&signal, param.clone()).unwrap();
        let mut stream = gcfb_v211::GcfbStream::new(param).unwrap();
        assert_eq!(stream.samples_processed(), 0);
        assert_eq!(stream.gc_param().num_ch, channels);
        for (sample_index, &sample) in signal.iter().enumerate() {
            let step = stream.process_sample(sample).unwrap();
            assert_eq!(step.sample_index, sample_index);
            close_slice(
                step.pgc_out.as_slice().unwrap(),
                &batch.pgc_out.column(sample_index).to_vec(),
            );
            close_slice(
                step.cgc_out.as_slice().unwrap(),
                &batch.cgc_out.column(sample_index).to_vec(),
            );
            if ctrl == gcfb_v211::ControlMode::Dynamic {
                close_slice(
                    step.lvl_db.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.lvl_db.column(sample_index).to_vec(),
                );
                close_slice(
                    step.frat_val.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.frat_val.column(sample_index).to_vec(),
                );
                close_slice(
                    step.fr2.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.fr2.column(sample_index).to_vec(),
                );
            } else {
                assert!(step.lvl_db.is_none());
                assert!(step.frat_val.is_none());
                assert!(step.fr2.is_none());
            }
        }
        assert_eq!(stream.samples_processed(), signal.len());
        close_slice(
            stream.gc_resp().gain_factor.as_slice().unwrap(),
            batch.gc_resp.gain_factor.as_slice().unwrap(),
        );
    }

    #[test]
    fn v234_sample_stream_matches_batch(
        values in prop::collection::vec(any::<i16>(), 1..90),
        channels in 2usize..6,
        mode in 0u8..3,
        update in 1usize..8,
        corrected in any::<bool>(),
        impaired in any::<bool>(),
        explicit_gain in any::<bool>(),
        alternate_coefficients in any::<bool>(),
    ) {
        let signal = test_signal(&values);
        let ctrl = match mode {
            0 => gcfb_v234::ControlMode::Static,
            1 => gcfb_v234::ControlMode::Level,
            _ => gcfb_v234::ControlMode::Dynamic,
        };
        let param = gcfb_v234::GcParam {
            fs: 8_000.0,
            num_ch: channels,
            f_range: [180.0, 1_800.0],
            out_mid_crct: if corrected { "ELC" } else { "No" }.into(),
            ctrl,
            num_update_asym_cmp: update,
            gain_ref: if explicit_gain {
                gcfb_v234::GainReference::Db(55.0)
            } else {
                gcfb_v234::GainReference::NormalizeIoFunction
            },
            dyn_hpaf: gcfb_v234::DynHpaf {
                str_prc: "sample-base".into(),
                ..gcfb_v234::DynHpaf::default()
            },
            hloss_type: if impaired { "HL3" } else { "NH" }.into(),
            b1: if alternate_coefficients { [1.70, 0.02] } else { [1.81, 0.0] },
            c1: if alternate_coefficients { [-2.70, -0.02] } else { [-2.96, 0.0] },
            frat: if alternate_coefficients {
                [[0.48, 0.005], [0.01, 0.0002]]
            } else {
                [[0.4660, 0.0], [0.0109, 0.0]]
            },
            b2: if alternate_coefficients {
                [[2.0, 0.05], [0.0, 0.0]]
            } else {
                [[2.17, 0.0], [0.0, 0.0]]
            },
            c2: if alternate_coefficients {
                [[2.0, -0.05], [0.0, 0.0]]
            } else {
                [[2.2, 0.0], [0.0, 0.0]]
            },
            ..gcfb_v234::GcParam::default()
        };
        let batch = gcfb_v234::gcfb_v234(&signal, param.clone()).unwrap();
        let mut stream = gcfb_v234::GcfbStream::new(param).unwrap();
        for (sample_index, &sample) in signal.iter().enumerate() {
            let step = stream.process_sample(sample).unwrap();
            close_slice(
                step.scgc_smpl.as_slice().unwrap(),
                &batch.scgc_smpl.column(sample_index).to_vec(),
            );
            let Some(gcfb_v234::DcgcEvent::Sample {
                sample_index: event_index,
                dcgc_out,
                lvl_db,
                frat_val,
                fr2,
            }) = step.event else {
                panic!("sample processing must emit a sample event");
            };
            assert_eq!(event_index, sample_index);
            close_slice(
                dcgc_out.as_slice().unwrap(),
                &batch.dcgc_out.column(sample_index).to_vec(),
            );
            if ctrl == gcfb_v234::ControlMode::Dynamic {
                close_slice(
                    lvl_db.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.lvl_db.column(sample_index).to_vec(),
                );
                close_slice(
                    frat_val.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.frat_val.column(sample_index).to_vec(),
                );
                close_slice(
                    fr2.as_ref().unwrap().as_slice().unwrap(),
                    &batch.gc_resp.fr2.column(sample_index).to_vec(),
                );
            } else {
                assert!(lvl_db.is_none() && frat_val.is_none() && fr2.is_none());
            }
        }
        close_slice(
            stream.gc_resp().gain_factor.as_slice().unwrap(),
            batch.gc_resp.gain_factor.as_slice().unwrap(),
        );
        assert!(stream.finish().unwrap().is_empty());
    }

    #[test]
    fn v234_frame_stream_matches_batch(
        values in prop::collection::vec(any::<i16>(), 1..100),
        channels in 2usize..6,
        shift_selector in 0u8..2,
        corrected in any::<bool>(),
        impaired in any::<bool>(),
        explicit_gain in any::<bool>(),
        alternate_coefficients in any::<bool>(),
    ) {
        let signal = test_signal(&values);
        let shift = if shift_selector == 0 { 4 } else { 8 };
        let param = gcfb_v234::GcParam {
            fs: 8_000.0,
            num_ch: channels,
            f_range: [180.0, 1_800.0],
            out_mid_crct: if corrected { "ELC" } else { "No" }.into(),
            ctrl: gcfb_v234::ControlMode::Dynamic,
            gain_ref: if explicit_gain {
                gcfb_v234::GainReference::Db(45.0)
            } else {
                gcfb_v234::GainReference::NormalizeIoFunction
            },
            dyn_hpaf: gcfb_v234::DynHpaf {
                str_prc: "frame-base".into(),
                t_frame: 16.0 / 8_000.0,
                t_shift: shift as f64 / 8_000.0,
                ..gcfb_v234::DynHpaf::default()
            },
            hloss_type: if impaired { "HL3" } else { "NH" }.into(),
            b1: if alternate_coefficients { [1.70, 0.02] } else { [1.81, 0.0] },
            c1: if alternate_coefficients { [-2.70, -0.02] } else { [-2.96, 0.0] },
            frat: if alternate_coefficients {
                [[0.48, 0.005], [0.01, 0.0002]]
            } else {
                [[0.4660, 0.0], [0.0109, 0.0]]
            },
            b2: if alternate_coefficients {
                [[2.0, 0.05], [0.0, 0.0]]
            } else {
                [[2.17, 0.0], [0.0, 0.0]]
            },
            c2: if alternate_coefficients {
                [[2.0, -0.05], [0.0, 0.0]]
            } else {
                [[2.2, 0.0], [0.0, 0.0]]
            },
            ..gcfb_v234::GcParam::default()
        };
        let batch = gcfb_v234::gcfb_v234(&signal, param.clone()).unwrap();
        let mut stream = gcfb_v234::GcfbStream::new(param).unwrap();
        let mut events = Vec::new();
        for (sample_index, &sample) in signal.iter().enumerate() {
            let step = stream.process_sample(sample).unwrap();
            close_slice(
                step.scgc_smpl.as_slice().unwrap(),
                &batch.scgc_smpl.column(sample_index).to_vec(),
            );
            assert!(stream.buffered_frame_samples() <= 16);
            if let Some(event) = step.event {
                events.push(event);
            }
        }
        events.extend(stream.finish().unwrap());
        assert_eq!(events.len(), signal.len() / shift + 1);
        assert_eq!(events.len(), batch.dcgc_out.ncols());
        for (expected_frame, event) in events.into_iter().enumerate() {
            let gcfb_v234::DcgcEvent::Frame {
                frame_index,
                center_index,
                dcgc_out,
                lvl_db,
                pgc_frame,
                scgc_frame,
                asym_func_gain,
            } = event else {
                panic!("frame processing must emit frame events");
            };
            assert_eq!(frame_index, expected_frame);
            assert_eq!(center_index, expected_frame * shift);
            close_slice(dcgc_out.as_slice().unwrap(), &batch.dcgc_out.column(expected_frame).to_vec());
            close_slice(lvl_db.as_slice().unwrap(), &batch.gc_resp.lvl_db_frame.column(expected_frame).to_vec());
            close_slice(pgc_frame.as_slice().unwrap(), &batch.gc_resp.pgc_frame.column(expected_frame).to_vec());
            close_slice(scgc_frame.as_slice().unwrap(), &batch.gc_resp.scgc_frame.column(expected_frame).to_vec());
            close_slice(asym_func_gain.as_slice().unwrap(), &batch.gc_resp.asym_func_gain.column(expected_frame).to_vec());
        }
    }
}

#[test]
fn non_finite_samples_do_not_advance_either_stream() {
    let mut v211 = gcfb_v211::GcfbStream::new(gcfb_v211::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ..gcfb_v211::GcParam::default()
    })
    .unwrap();
    assert!(matches!(
        v211.process_sample(f64::NAN),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(v211.samples_processed(), 0);
    let after_error = v211.process_sample(1.0).unwrap();
    let mut fresh = gcfb_v211::GcfbStream::new(v211.gc_param().clone()).unwrap();
    let expected = fresh.process_sample(1.0).unwrap();
    close_slice(
        after_error.cgc_out.as_slice().unwrap(),
        expected.cgc_out.as_slice().unwrap(),
    );

    let mut v234 = gcfb_v234::GcfbStream::new(gcfb_v234::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ctrl: gcfb_v234::ControlMode::Static,
        ..gcfb_v234::GcParam::default()
    })
    .unwrap();
    assert!(v234.process_sample(f64::INFINITY).is_err());
    assert_eq!(v234.samples_processed(), 0);
    let after_error = v234.process_sample(1.0).unwrap();
    let mut fresh = gcfb_v234::GcfbStream::new(v234.gc_param().clone()).unwrap();
    let expected = fresh.process_sample(1.0).unwrap();
    close_slice(
        after_error.scgc_smpl.as_slice().unwrap(),
        expected.scgc_smpl.as_slice().unwrap(),
    );
}

#[test]
fn zero_length_passive_impulses_match_the_batch_zero_operator() {
    let v211_param = gcfb_v211::GcParam {
        fs: 10.0,
        num_ch: 2,
        f_range: [1.0, 2.0],
        out_mid_crct: "No".into(),
        c1: [0.0, 0.0],
        ctrl: gcfb_v211::ControlMode::Static,
        ..gcfb_v211::GcParam::default()
    };
    let v211_batch = gcfb_v211::gcfb_v211(&[1.0], v211_param.clone()).unwrap();
    let v211_stream = gcfb_v211::GcfbStream::new(v211_param)
        .unwrap()
        .process_sample(1.0)
        .unwrap();
    close_slice(
        v211_stream.cgc_out.as_slice().unwrap(),
        v211_batch.cgc_out.column(0).as_slice().unwrap(),
    );

    let v234_param = gcfb_v234::GcParam {
        fs: 10.0,
        num_ch: 2,
        f_range: [1.0, 2.0],
        out_mid_crct: "No".into(),
        c1: [0.0, 0.0],
        ctrl: gcfb_v234::ControlMode::Static,
        dyn_hpaf: gcfb_v234::DynHpaf {
            t_frame: 0.2,
            t_shift: 0.1,
            ..gcfb_v234::DynHpaf::default()
        },
        ..gcfb_v234::GcParam::default()
    };
    let v234_batch = gcfb_v234::gcfb_v234(&[1.0], v234_param.clone()).unwrap();
    let v234_stream = gcfb_v234::GcfbStream::new(v234_param)
        .unwrap()
        .process_sample(1.0)
        .unwrap();
    let Some(gcfb_v234::DcgcEvent::Sample { dcgc_out, .. }) = v234_stream.event else {
        panic!("static processing must emit a sample event");
    };
    close_slice(
        dcgc_out.as_slice().unwrap(),
        v234_batch.dcgc_out.column(0).as_slice().unwrap(),
    );
}

#[test]
fn dynamic_filter_update_errors_make_streams_terminal() {
    let mut v211 = gcfb_v211::GcfbStream::new(gcfb_v211::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ctrl: gcfb_v211::ControlMode::Dynamic,
        // The 50 dB reference ratio is valid, but the zero-input runtime
        // level produces a negative center.
        frat: [[-49.0, 0.0], [1.0, 0.0]],
        ..gcfb_v211::GcParam::default()
    })
    .unwrap();
    assert!(matches!(
        v211.process_sample(0.0),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        v211.process_sample(0.0),
        Err(Error::Numerical(message)) if message.contains("cannot continue")
    ));
    assert_eq!(v211.samples_processed(), 0);

    let mut v234 = gcfb_v234::GcfbStream::new(gcfb_v234::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ctrl: gcfb_v234::ControlMode::Dynamic,
        dyn_hpaf: gcfb_v234::DynHpaf {
            str_prc: "sample-base".into(),
            ..gcfb_v234::DynHpaf::default()
        },
        ..gcfb_v234::GcParam::default()
    })
    .unwrap();
    // The passive impulse starts at zero; the oversized finite value reaches
    // the dynamic level calculation on the following sample.
    v234.process_sample(1e300).unwrap();
    assert!(matches!(
        v234.process_sample(0.0),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        v234.process_sample(0.0),
        Err(Error::Numerical(message)) if message.contains("cannot continue")
    ));
    assert_eq!(v234.samples_processed(), 1);
    assert!(matches!(v234.finish(), Err(Error::Numerical(_))));
}

#[test]
fn frame_events_are_delayed_and_finish_flushes_the_tail() {
    let param = gcfb_v234::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ctrl: gcfb_v234::ControlMode::Dynamic,
        dyn_hpaf: gcfb_v234::DynHpaf {
            str_prc: "frame-base".into(),
            t_frame: 16.0 / 8_000.0,
            t_shift: 8.0 / 8_000.0,
            ..gcfb_v234::DynHpaf::default()
        },
        ..gcfb_v234::GcParam::default()
    };
    let mut stream = gcfb_v234::GcfbStream::new(param).unwrap();
    for sample_index in 0..7 {
        assert!(
            stream
                .process_sample(sample_index as f64 / 10.0)
                .unwrap()
                .event
                .is_none()
        );
    }
    let first = stream.process_sample(0.7).unwrap().event.unwrap();
    assert!(matches!(
        first,
        gcfb_v234::DcgcEvent::Frame {
            frame_index: 0,
            center_index: 0,
            ..
        }
    ));
    let tail = stream.finish().unwrap();
    assert_eq!(tail.len(), 1);
    assert!(matches!(
        tail[0],
        gcfb_v234::DcgcEvent::Frame {
            frame_index: 1,
            center_index: 8,
            ..
        }
    ));
}

#[test]
fn one_sample_frame_stream_and_invalid_frame_configuration() {
    let valid = gcfb_v234::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ctrl: gcfb_v234::ControlMode::Dynamic,
        dyn_hpaf: gcfb_v234::DynHpaf {
            str_prc: "frame-base".into(),
            t_frame: 8.0 / 8_000.0,
            t_shift: 4.0 / 8_000.0,
            ..gcfb_v234::DynHpaf::default()
        },
        ..gcfb_v234::GcParam::default()
    };
    assert!(
        gcfb_v234::GcfbStream::new(valid.clone())
            .unwrap()
            .finish()
            .is_err()
    );
    let mut stream = gcfb_v234::GcfbStream::new(valid).unwrap();
    assert!(stream.process_sample(1.0).unwrap().event.is_none());
    assert_eq!(stream.finish().unwrap().len(), 1);

    let invalid = gcfb_v234::GcParam {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        dyn_hpaf: gcfb_v234::DynHpaf {
            str_prc: "frame-base".into(),
            t_frame: 15.0 / 8_000.0,
            t_shift: 4.0 / 8_000.0,
            ..gcfb_v234::DynHpaf::default()
        },
        ..gcfb_v234::GcParam::default()
    };
    assert!(gcfb_v234::GcfbStream::new(invalid.clone()).is_err());

    // Frame geometry is irrelevant when the control mode emits samples.
    for ctrl in [
        gcfb_v234::ControlMode::Static,
        gcfb_v234::ControlMode::Level,
    ] {
        let mut stream = gcfb_v234::GcfbStream::new(gcfb_v234::GcParam {
            ctrl,
            ..invalid.clone()
        })
        .unwrap();
        assert!(matches!(
            stream.process_sample(0.0).unwrap().event,
            Some(gcfb_v234::DcgcEvent::Sample { .. })
        ));
    }
}
