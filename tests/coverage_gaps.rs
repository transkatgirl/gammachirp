//! Deterministic coverage for public workflows that are not exercised by the
//! checked-in Python parity fixtures.

use std::{fs, time::SystemTime};

use approx::assert_relative_eq;
use gammachirpy::{
    Error,
    gcfb_v211::{
        gammachirp::{self, Carrier, Normalization},
        gcfb_v211::{self as fb211, AcfStatus, ControlMode, GcParam as Param211, SmoothSpecParam},
        utils::{self as utils211, FrequencyScale},
    },
    gcfb_v234::{
        gcfb_v234::{self as fb234, DynHpaf, EmParam, GainReference, GcParam as Param234},
        utils::{self as utils234, Floor, ParamTransFunc},
    },
};
use ndarray::{Array1, Array2};

fn prepared_v234() -> Param234 {
    fb234::set_param(Param234 {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ..Param234::default()
    })
    .unwrap()
    .0
}

#[test]
fn smooth_spectrum_covers_both_windows_and_reports_frame_times() {
    let input = Array2::from_shape_fn((2, 480), |(channel, _)| channel as f64 + 1.0);

    for (method, window_seconds) in [(1, 0.025), (2, 0.010)] {
        let mut param = SmoothSpecParam::new(8_000.0);
        param.method = method;
        let (smoothed, param) = fb211::cal_smooth_spec(&input, param).unwrap();

        assert_eq!(smoothed.dim(), (2, 13));
        assert_eq!(param.temporal_positions.len(), 13);
        assert_relative_eq!(param.t_shift, 0.005, epsilon = 1e-15);
        assert_relative_eq!(param.t_win, window_seconds, epsilon = 1e-15);
        assert_relative_eq!(param.temporal_positions[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(param.temporal_positions[12], 0.06, epsilon = 1e-15);

        // Frame four is far enough from both edges for either window, so a
        // normalized smoother must preserve each constant channel exactly.
        assert_relative_eq!(smoothed[[0, 4]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(smoothed[[1, 4]], 2.0, epsilon = 1e-12);
    }

    let mut invalid = SmoothSpecParam::new(8_000.0);
    invalid.method = 3;
    assert!(fb211::cal_smooth_spec(&input, invalid).is_err());
}

#[test]
fn envelope_modulation_loss_and_analysis_cover_success_and_validation() {
    let param = prepared_v234();
    let frames = Array2::from_shape_fn((4, 16), |(channel, _)| channel as f64 + 1.0);

    let (reduced, em) = fb234::gcfb_v23_env_mod_loss(&frames, &param, EmParam::default()).unwrap();
    assert_eq!(reduced.dim(), frames.dim());
    assert_eq!(em.fs, param.dyn_hpaf.fs);
    assert_eq!(em.fb_fr1, param.fr1);
    assert_eq!(em.fb_reduce_db.len(), param.num_ch);
    assert_eq!(em.fb_f_cutoff.len(), param.num_ch);
    for channel in 0..frames.nrows() {
        for frame in 0..frames.ncols() - 1 {
            assert_relative_eq!(
                reduced[[channel, frame]],
                frames[[channel, frame]],
                epsilon = 1e-12
            );
        }
        assert_eq!(reduced[[channel, frames.ncols() - 1]], 0.0);
    }

    let (analysis, analyzed) =
        fb234::gcfb_v23_ana_env_mod(&reduced, &param, EmParam::default()).unwrap();
    assert_eq!(analysis.dim(), (4, 9, 16));
    assert_eq!(analyzed.fs, param.dyn_hpaf.fs);
    assert!(analysis.iter().all(|value| value.is_finite()));

    let mut sample_param = param.clone();
    sample_param.dyn_hpaf.str_prc = "sample-base".into();
    assert!(fb234::gcfb_v23_env_mod_loss(&frames, &sample_param, EmParam::default()).is_err());
    assert!(fb234::gcfb_v23_ana_env_mod(&frames, &sample_param, EmParam::default()).is_err());

    let wrong_audiogram = EmParam {
        reduce_db: Array1::zeros(6),
        ..EmParam::default()
    };
    assert!(fb234::gcfb_v23_env_mod_loss(&frames, &param, wrong_audiogram).is_err());

    let aliased_cutoff = EmParam {
        f_cutoff: Array1::from_elem(7, param.dyn_hpaf.fs / 2.0),
        ..EmParam::default()
    };
    assert!(fb234::gcfb_v23_env_mod_loss(&frames, &param, aliased_cutoff).is_err());
    assert!(
        fb234::gcfb_v23_env_mod_loss(
            &Array2::zeros((3, frames.ncols())),
            &param,
            EmParam::default(),
        )
        .is_err()
    );
}

#[test]
fn valid_wav_with_an_odd_unknown_chunk_is_read_as_mono_pcm() {
    let samples = [i16::MIN, -16_384, 0, 16_384, i16::MAX];
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&0_u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"JUNK");
    wav.extend_from_slice(&3_u32.to_le_bytes());
    wav.extend_from_slice(&[1, 2, 3, 0]);
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&8_000_u32.to_le_bytes());
    wav.extend_from_slice(&16_000_u32.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&((samples.len() * 2) as u32).to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    let riff_size = (wav.len() - 8) as u32;
    wav[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "gammachirpy-valid-{}-{nonce}.wav",
        std::process::id()
    ));
    fs::write(&path, wav).unwrap();
    let result = utils211::audioread(&path);
    fs::remove_file(&path).unwrap();

    let (decoded, sample_rate) = result.unwrap();
    assert_eq!(sample_rate, 8_000);
    assert_eq!(decoded.len(), samples.len());
    for (actual, sample) in decoded.iter().zip(samples) {
        assert_relative_eq!(*actual, sample as f64 / 32_768.0, epsilon = 1e-15);
    }

    let missing = path.with_extension("missing.wav");
    assert!(matches!(utils211::audioread(missing), Err(Error::Io(_))));
}

#[test]
fn public_signal_utilities_cover_empty_inputs_aliases_and_rejections() {
    assert!(utils211::rms(&[]).is_nan());
    assert!(utils211::eqlz2meddis_hc_level(&[0.0, 0.0], 60.0).is_err());
    assert_eq!(utils211::fftfilt(&[], &[1.0, 2.0]).to_vec(), vec![0.0, 0.0]);
    assert!(utils211::isrow(&[1.0, 2.0]));
    assert!(utils211::iscolumn(&Array2::<f64>::zeros((3, 1))));
    assert!(!utils211::iscolumn(&Array2::<f64>::zeros((1, 3))));
    assert!(utils211::rceps(&[]).is_err());

    for scale in [
        FrequencyScale::Linear,
        FrequencyScale::Mel,
        FrequencyScale::Log,
    ] {
        let (frequencies, positions) =
            utils211::equal_freq_scale(scale, 4, [100.0, 1_600.0]).unwrap();
        assert_eq!(frequencies.len(), 4);
        assert_eq!(positions.len(), 4);
        assert_relative_eq!(frequencies[0], 100.0, epsilon = 1e-10);
        assert_relative_eq!(frequencies[3], 1_600.0, epsilon = 1e-9);
    }
    assert!(utils211::equal_freq_scale(FrequencyScale::Linear, 1, [100.0, 200.0]).is_err());

    let (gaussian, name) = utils211::taper_window(9, "Gaussian", Some(3), 2.5).unwrap();
    assert_eq!(name, "Gauss");
    assert_eq!(gaussian.len(), 9);
    assert_relative_eq!(gaussian[4], 1.0, epsilon = 1e-15);
    assert!(utils211::taper_window(0, "Hamming", None, 1.0).is_err());

    for invalid in [(0, Some(1)), (5, Some(1)), (4, Some(3))] {
        assert!(utils211::set_frame4time_sequence(&[1.0], invalid.0, invalid.1).is_err());
    }

    let (no_power, no_frequency, no_db) = utils211::out_mid_crct("NO", 0, 8_000.0).unwrap();
    assert_eq!(no_power.to_vec(), vec![1.0]);
    assert_eq!(no_frequency.to_vec(), vec![0.0]);
    assert_eq!(no_db.to_vec(), vec![0.0]);
    assert!(utils211::out_mid_crct("unknown", 32, 8_000.0).is_err());

    let forward = utils211::out_mid_crct_filt("ELC", 8_000.0, 0).unwrap();
    let inverse = utils211::out_mid_crct_filt("ELC", 8_000.0, 1).unwrap();
    assert_eq!(forward.len(), inverse.len());
    assert_ne!(forward, inverse);
    assert!(utils211::out_mid_crct_filt("ELC", 8_000.0, 3).is_err());
}

#[test]
fn transfer_function_selectors_filters_and_noise_floor_are_covered() {
    for (selector, expected) in [
        ("FreeField2EarDrum_Moore16", "FreeField"),
        ("DiffuseField2EarDrum_Moore16", "DiffuseField"),
        ("ITUField2EarDrum", "ITU"),
    ] {
        let (frequencies, values, canonical) =
            utils234::trans_func_field2eardrum_set(selector).unwrap();
        assert_eq!(canonical, expected);
        assert_eq!(frequencies.len(), values.len());
        assert!(!frequencies.is_empty());
    }
    assert!(utils234::trans_func_field2eardrum_set("free-field").is_err());

    let outside = utils234::interp1(&[0.0, 1.0], &[2.0, 4.0], &[-1.0, 2.0], false).unwrap();
    assert!(outside.iter().all(|value| value.is_nan()));
    assert!(utils234::interp1(&[0.0, 0.0], &[1.0, 2.0], &[0.0], true).is_err());

    let invalid_transfer = ParamTransFunc {
        n_frq_rsl: 1,
        ..ParamTransFunc::default()
    };
    assert!(utils234::trans_func_field2cochlea(&invalid_transfer).is_err());

    for (kind, forward) in [("ELC", true), ("EarDrum", true), ("EarDrum", false)] {
        let (filter, metadata) = utils234::mk_filter_field2cochlea(kind, 8_000.0, forward).unwrap();
        assert!(!filter.is_empty());
        assert!(filter.iter().all(|value| value.is_finite()));
        assert!(metadata.name_filter.contains(kind));
    }
    assert!(utils234::mk_filter_field2cochlea("Unknown", 8_000.0, true).is_err());
    assert!(utils234::hl2spl(123.0, 0.0).is_err());
    assert!(utils234::hl2pin_cochlea(123.0, 0.0).is_err());

    let input = Array2::from_shape_vec((2, 2), vec![0.0, 0.5, -0.5, 1.0]).unwrap();
    let first = utils234::eqlz_gcfb2rms1_at_0db(&input, Floor::NoiseFloor);
    let second = utils234::eqlz_gcfb2rms1_at_0db(&input, Floor::NoiseFloor);
    let without_noise = utils234::eqlz_gcfb2rms1_at_0db(&input, Floor::None);
    assert_eq!(
        first, second,
        "the documented noise floor must be reproducible"
    );
    assert_ne!(first, without_noise);
    assert!(first.iter().all(|value| value.is_finite()));
}

#[test]
fn public_filterbank_modes_and_wrappers_have_smoke_coverage() {
    let signal = [1.0, 0.0, -0.25, 0.0, 0.125, 0.0, 0.0, 0.0];

    let level211 = fb211::gcfb_v211(
        &signal,
        Param211 {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 1_500.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Level,
            ..Param211::default()
        },
    )
    .unwrap();
    assert_eq!(level211.cgc_out.dim(), (4, signal.len()));
    assert!(level211.cgc_out.iter().all(|value| value.is_finite()));

    let corrected211 = fb211::gcfb_v211(
        &signal,
        Param211 {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 1_500.0],
            out_mid_crct: "ELC".into(),
            ..Param211::default()
        },
    )
    .unwrap();
    assert_eq!(corrected211.cgc_out.dim(), (4, signal.len()));
    assert!(fb211::gcfb_v211(&[], Param211::default()).is_err());

    let coefficients = fb211::make_asym_cmp_filters_v2(8_000.0, &[500.0], &[2.17], &[2.2]).unwrap();
    let mut state = AcfStatus::new(&coefficients);
    let filtered = fb211::acfilterbank(&coefficients, &mut state, &[1.0], true).unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(state.count, 1);

    let level234 = fb234::gcfb_v234(
        &signal,
        Param234 {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 1_500.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Level,
            gain_ref: GainReference::Db(50.0),
            ..Param234::default()
        },
    )
    .unwrap();
    assert_eq!(level234.dcgc_out.dim(), (4, signal.len()));
    assert!(level234.gc_resp.cgc_ref.is_some());
    assert!(fb234::gcfb_v234(&[], Param234::default()).is_err());

    let (frames, centers) = fb234::set_frame4time_sequence(&signal, 4, Some(2)).unwrap();
    assert_eq!(frames.dim(), (4, 5));
    assert_eq!(centers.to_vec(), vec![0, 2, 4, 6, 8]);
    assert!(fb234::set_frame4time_sequence(&signal, 5, Some(1)).is_err());

    let synthesized = fb234::gcfb_v23_synth_snd(
        &Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 3.0, 5.0]).unwrap(),
        &Param234 {
            out_mid_crct: "No".into(),
            ..Param234::default()
        },
    )
    .unwrap();
    assert_eq!(synthesized.to_vec(), vec![-30.0, -60.0]);
    assert!(fb234::gcfb_v23_synth_snd(&Array2::zeros((0, 2)), &Param234::default()).is_err());
    assert!(fb234::gcfb_v23_synth_snd(&Array2::zeros((2, 0)), &Param234::default()).is_err());
}

#[test]
fn gammachirp_public_validation_rejects_malformed_broadcasts() {
    assert!(
        gammachirp::gammachirp(
            &[],
            8_000.0,
            4.0,
            1.81,
            -2.96,
            0.0,
            Carrier::Cosine,
            Normalization::None,
        )
        .is_err()
    );
    assert!(
        gammachirp::gammachirp_frsp(
            &[500.0, 1_000.0],
            8_000.0,
            4.0,
            &[1.81, 1.81, 1.81],
            &[-2.96],
            0.0,
            256,
        )
        .is_err()
    );
    assert!(
        gammachirp::gammachirp_frsp(&[500.0], 8_000.0, 4.0, &[0.0], &[-2.96], 0.0, 256,).is_err()
    );
}

#[test]
fn prepared_parameter_variants_cover_hamming_and_sample_modes() {
    let hamming = fb234::set_param(Param234 {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        dyn_hpaf: DynHpaf {
            name_win: "hamming".into(),
            ..DynHpaf::default()
        },
        ..Param234::default()
    })
    .unwrap()
    .0;
    assert_eq!(hamming.dyn_hpaf.val_win.len(), hamming.dyn_hpaf.len_frame);
    assert_relative_eq!(hamming.dyn_hpaf.val_win.sum(), 1.0, epsilon = 1e-12);

    let sample = fb234::set_param(Param234 {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        dyn_hpaf: DynHpaf {
            str_prc: "sample-base".into(),
            ..DynHpaf::default()
        },
        ..Param234::default()
    })
    .unwrap()
    .0;
    assert_eq!(sample.dyn_hpaf.str_prc, "sample-base");
    assert!(sample.dyn_hpaf.val_win.is_empty());
}
