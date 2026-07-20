//! Deterministic coverage for public workflows that are not exercised by the
//! checked-in Python parity fixtures.

use std::{fs, time::SystemTime};

use approx::assert_relative_eq;
use gammachirp_rs::{
    Error,
    gcfb_v234::{
        gammachirp::{self, Carrier, Normalization},
        gcfb_v234::{
            self as fb234, AcfStatus, ControlMode, DynHpaf, EmParam, GainReference,
            GcParam as Param234, SmoothSpecParam,
        },
        utils::{self as utils234, Floor, FrequencyScale, ParamTransFunc},
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

fn compact_v234() -> Param234 {
    Param234 {
        fs: 8_000.0,
        num_ch: 4,
        f_range: [200.0, 1_500.0],
        out_mid_crct: "No".into(),
        ..Param234::default()
    }
}

fn read_wav_bytes(label: &str, bytes: &[u8]) -> gammachirp_rs::Result<(Array1<f64>, u32)> {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "gammachirpy-{label}-{}-{nonce}.wav",
        std::process::id()
    ));
    fs::write(&path, bytes).unwrap();
    let result = utils234::audioread(&path);
    fs::remove_file(path).unwrap();
    result
}

fn riff_wave(chunks: &[u8]) -> Vec<u8> {
    let mut wav = Vec::from(&b"RIFF"[..]);
    wav.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(chunks);
    wav
}

#[test]
fn smooth_spectrum_covers_both_windows_and_reports_frame_times() {
    let input = Array2::from_shape_fn((2, 480), |(channel, _)| channel as f64 + 1.0);

    for (method, window_seconds) in [(1, 0.025), (2, 0.010)] {
        let mut param = SmoothSpecParam::new(8_000.0);
        param.method = method;
        let (smoothed, param) = fb234::cal_smooth_spec(&input, param).unwrap();

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
    assert!(fb234::cal_smooth_spec(&input, invalid).is_err());
}

#[test]
fn envelope_modulation_loss_and_analysis_cover_success_and_validation() {
    let param = prepared_v234();
    let frames = Array2::from_shape_fn((4, 16), |(channel, frame)| {
        let baseline = 1.25 + 0.35 * channel as f64;
        let modulation = (0.15 + 0.04 * channel as f64)
            * (2.0 * std::f64::consts::PI * (channel + 1) as f64 * frame as f64 / 16.0).sin();
        baseline + modulation + if frame == channel + 4 { 0.2 } else { 0.0 }
    });
    let reductions = Array1::from(vec![0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    let cutoffs = Array1::from(vec![32.0, 48.0, 64.0, 96.0, 128.0, 160.0, 192.0]);
    let requested = EmParam {
        reduce_db: reductions.clone(),
        f_cutoff: cutoffs.clone(),
        ..EmParam::default()
    };
    let zero_reduction = EmParam {
        reduce_db: Array1::zeros(7),
        f_cutoff: cutoffs.clone(),
        ..EmParam::default()
    };

    let (reduced, em) = fb234::gcfb_v23_env_mod_loss(&frames, &param, requested).unwrap();
    let (unattenuated, _) = fb234::gcfb_v23_env_mod_loss(&frames, &param, zero_reduction).unwrap();
    assert_eq!(reduced.dim(), frames.dim());
    assert_eq!(em.fs, param.dyn_hpaf.fs);
    assert_eq!(em.fb_fr1, param.fr1);
    assert_eq!(em.fb_reduce_db.len(), param.num_ch);
    assert_eq!(em.fb_f_cutoff.len(), param.num_ch);

    let (audiogram_erb, _) = utils234::freq2erb(param.hloss.f_audgram_list.as_slice().unwrap());
    let (filterbank_erb, _) = utils234::freq2erb(param.fr1.as_slice().unwrap());
    let expected_reductions = utils234::interp1(
        audiogram_erb.as_slice().unwrap(),
        reductions.as_slice().unwrap(),
        filterbank_erb.as_slice().unwrap(),
        true,
    )
    .unwrap();
    let expected_cutoffs = utils234::interp1(
        audiogram_erb.as_slice().unwrap(),
        cutoffs.as_slice().unwrap(),
        filterbank_erb.as_slice().unwrap(),
        true,
    )
    .unwrap();
    for channel in 0..param.num_ch {
        assert_relative_eq!(
            em.fb_reduce_db[channel],
            expected_reductions[channel],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            em.fb_f_cutoff[channel],
            expected_cutoffs[channel],
            epsilon = 1e-12
        );
    }

    for channel in 0..frames.nrows() {
        let dc = (frames
            .row(channel)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / frames.ncols() as f64)
            .sqrt();
        let gain = 10_f64.powf(-em.fb_reduce_db[channel] / 20.0);
        for frame in 0..frames.ncols() - 1 {
            assert_relative_eq!(
                reduced[[channel, frame]],
                dc + gain * (unattenuated[[channel, frame]] - dc),
                epsilon = 1e-12
            );
            assert!(
                (reduced[[channel, frame]] - dc).abs()
                    <= (unattenuated[[channel, frame]] - dc).abs() + 1e-12
            );
        }
        assert_eq!(reduced[[channel, frames.ncols() - 1]], 0.0);
        assert_eq!(unattenuated[[channel, frames.ncols() - 1]], 0.0);
    }

    let (analysis, analyzed) = fb234::gcfb_v23_ana_env_mod(&reduced, &param, em.clone()).unwrap();
    assert_eq!(analysis.dim(), (4, 9, 16));
    assert_eq!(analyzed.fs, param.dyn_hpaf.fs);
    assert!(analysis.iter().all(|value| value.is_finite()));
    for channel in 0..frames.nrows() {
        let direct =
            fb234::gcfb_v23_env_mod_fb(reduced.row(channel).as_slice().unwrap(), &analyzed)
                .unwrap();
        for modulation_channel in 0..direct.nrows() {
            for frame in 0..direct.ncols() {
                assert_relative_eq!(
                    analysis[[channel, modulation_channel, frame]],
                    direct[[modulation_channel, frame]],
                    epsilon = 1e-12
                );
            }
        }
    }

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

    let invalid_frames = [
        Array2::zeros((3, frames.ncols())),
        Array2::zeros((5, frames.ncols())),
        Array2::zeros((frames.nrows(), 0)),
    ];
    for invalid in &invalid_frames {
        assert!(fb234::gcfb_v23_env_mod_loss(invalid, &param, EmParam::default()).is_err());
        assert!(fb234::gcfb_v23_ana_env_mod(invalid, &param, EmParam::default()).is_err());
    }

    for invalid_value in [f64::NAN, f64::INFINITY] {
        let mut invalid = frames.clone();
        invalid[[0, 0]] = invalid_value;
        assert!(fb234::gcfb_v23_env_mod_loss(&invalid, &param, EmParam::default()).is_err());
        assert!(fb234::gcfb_v23_ana_env_mod(&invalid, &param, EmParam::default()).is_err());
    }

    let mut wrong_num_ch = param.clone();
    wrong_num_ch.num_ch -= 1;
    assert!(fb234::gcfb_v23_env_mod_loss(&frames, &wrong_num_ch, EmParam::default()).is_err());
    assert!(fb234::gcfb_v23_ana_env_mod(&frames, &wrong_num_ch, EmParam::default()).is_err());

    let mut wrong_fr1 = param.clone();
    wrong_fr1.fr1 = wrong_fr1.fr1.slice(ndarray::s![..3]).to_owned();
    assert!(fb234::gcfb_v23_env_mod_loss(&frames, &wrong_fr1, EmParam::default()).is_err());
    assert!(fb234::gcfb_v23_ana_env_mod(&frames, &wrong_fr1, EmParam::default()).is_err());
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
    wav.extend_from_slice(b"ignored trailing bytes");

    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "gammachirpy-valid-{}-{nonce}.wav",
        std::process::id()
    ));
    fs::write(&path, wav).unwrap();
    let result = utils234::audioread(&path);
    fs::remove_file(&path).unwrap();

    let (decoded, sample_rate) = result.unwrap();
    assert_eq!(sample_rate, 8_000);
    assert_eq!(decoded.len(), samples.len());
    for (actual, sample) in decoded.iter().zip(samples) {
        assert_relative_eq!(*actual, sample as f64 / 32_768.0, epsilon = 1e-15);
    }

    let missing = path.with_extension("missing.wav");
    assert!(matches!(utils234::audioread(missing), Err(Error::Io(_))));
}

#[test]
fn wav_reader_rejects_malformed_data_inside_the_riff_boundary() {
    let mut physically_truncated = riff_wave(&[]);
    physically_truncated[4..8].copy_from_slice(&20_u32.to_le_bytes());

    let partial_header = riff_wave(b"JUN");

    let mut missing_unknown_payload_chunks = Vec::from(&b"JUNK"[..]);
    missing_unknown_payload_chunks.extend_from_slice(&4_u32.to_le_bytes());
    let missing_unknown_payload = riff_wave(&missing_unknown_payload_chunks);

    let mut missing_padding_chunks = Vec::from(&b"JUNK"[..]);
    missing_padding_chunks.extend_from_slice(&3_u32.to_le_bytes());
    missing_padding_chunks.extend_from_slice(&[1, 2, 3]);
    let missing_padding = riff_wave(&missing_padding_chunks);

    for (label, wav) in [
        ("truncated-riff", physically_truncated),
        ("partial-header", partial_header),
        ("missing-unknown-payload", missing_unknown_payload),
        ("missing-padding", missing_padding),
    ] {
        assert!(matches!(read_wav_bytes(label, &wav), Err(Error::Wav(_))));
    }
}

#[test]
fn filterbanks_reject_non_finite_waveform_samples() {
    for invalid in [f64::NAN, f64::INFINITY] {
        let signal = [1.0, invalid, 0.0];
        assert!(matches!(
            fb234::gcfb_v234(&signal, compact_v234()),
            Err(Error::InvalidParameter(_))
        ));
    }
}

#[test]
fn filterbank_preparation_rejects_non_finite_user_parameters() {
    let mut v234_gain = compact_v234();
    v234_gain.gain_cmpnst_db = f64::NAN;
    let mut v234_reference = compact_v234();
    v234_reference.gain_ref = GainReference::Db(f64::INFINITY);
    let mut v234_static_level = compact_v234();
    v234_static_level.level_db_scgcfb = f64::NEG_INFINITY;
    let mut v234_coefficient = compact_v234();
    v234_coefficient.c2[0][0] = f64::NAN;
    let mut v234_level_estimation = compact_v234();
    v234_level_estimation.lvl_est.pwr[1] = f64::INFINITY;
    let mut v234_negative_decay = compact_v234();
    v234_negative_decay.lvl_est.decay_hl = -1.0;
    for invalid in [
        v234_gain,
        v234_reference,
        v234_static_level,
        v234_coefficient,
        v234_level_estimation,
        v234_negative_decay,
    ] {
        assert!(matches!(
            fb234::set_param(invalid),
            Err(Error::InvalidParameter(_))
        ));
    }
}

#[test]
fn parameter_preparation_overwrites_derived_caches_without_validating_them() {
    let mut v234 = compact_v234();
    v234.fr1 = Array1::from(vec![f64::NAN]);
    v234.hloss.compression_health = Array1::from(vec![f64::NAN]);
    v234.dyn_hpaf.len_frame = usize::MAX;
    v234.dyn_hpaf.len_shift = usize::MAX;
    v234.dyn_hpaf.fs = f64::NAN;
    v234.dyn_hpaf.val_win = Array1::from(vec![f64::NAN]);
    v234.lvl_est.exp_decay_val = f64::NAN;
    v234.lvl_est.erb_space1 = f64::NAN;
    v234.lvl_est.n_ch_shift = isize::MAX;
    v234.lvl_est.n_ch_lvl_est = Array1::from(vec![usize::MAX]);
    v234.lvl_est.lvl_lin_min_lim = f64::NAN;
    v234.lvl_est.lvl_lin_ref = f64::NAN;
    let (v234, _) = fb234::set_param(v234).unwrap();
    assert_eq!(v234.fr1.len(), v234.num_ch);
    assert!(v234.fr1.iter().all(|value| value.is_finite()));
    assert!(
        v234.hloss
            .compression_health
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(v234.dyn_hpaf.fs.is_finite());
    assert_eq!(v234.dyn_hpaf.val_win.len(), v234.dyn_hpaf.len_frame);
    assert_eq!(v234.lvl_est.n_ch_lvl_est.len(), v234.num_ch);
    assert!(v234.lvl_est.exp_decay_val.is_finite());
    assert!(v234.lvl_est.lvl_lin_min_lim.is_finite());
    assert!(v234.lvl_est.lvl_lin_ref.is_finite());
}

#[test]
fn compression_health_accepts_only_the_closed_unit_interval() {
    for health in [0.0, 1.0] {
        let mut param = compact_v234();
        param.hloss_type = "HL3".into();
        param.hloss_compression_health = Some(health);
        assert!(fb234::set_param(param).is_ok());
    }

    for health in [-f64::EPSILON, 1.0 + f64::EPSILON, f64::NAN] {
        let mut param = compact_v234();
        param.hloss_type = "HL3".into();
        param.hloss_compression_health = Some(health);
        assert!(matches!(
            fb234::set_param(param),
            Err(Error::InvalidParameter(_))
        ));
    }
}

#[test]
fn public_signal_utilities_cover_empty_inputs_aliases_and_rejections() {
    assert!(utils234::rms(&[]).is_nan());
    assert!(utils234::eqlz2meddis_hc_level(&[0.0, 0.0], Some(60.0), None).is_err());
    for level in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(utils234::eqlz2meddis_hc_level(&[1.0, -1.0], Some(level), None).is_err());
        assert!(utils234::eqlz2meddis_hc_level(&[1.0, -1.0], Some(level), None).is_err());
        assert!(utils234::eqlz2meddis_hc_level(&[1.0, -1.0], None, Some(level)).is_err());
    }
    assert!(utils234::eqlz2meddis_hc_level(&[1.0, -1.0], Some(f64::NAN), Some(94.0)).is_ok());
    assert_eq!(utils234::fftfilt(&[], &[1.0, 2.0]).to_vec(), vec![0.0, 0.0]);
    assert!(utils234::isrow(&[1.0, 2.0]));
    assert!(utils234::iscolumn(&Array2::<f64>::zeros((3, 1))));
    assert!(!utils234::iscolumn(&Array2::<f64>::zeros((1, 3))));
    assert!(utils234::rceps(&[]).is_err());

    for scale in [
        FrequencyScale::Erb,
        FrequencyScale::Linear,
        FrequencyScale::Mel,
        FrequencyScale::Log,
    ] {
        let (frequencies, positions) =
            utils234::equal_freq_scale(scale, 4, [100.0, 1_600.0]).unwrap();
        assert_eq!(frequencies.len(), 4);
        assert_eq!(positions.len(), 4);
        assert_eq!(frequencies[0], 100.0);
        assert_eq!(frequencies[3], 1_600.0);
    }
    assert!(utils234::equal_freq_scale(FrequencyScale::Linear, 1, [100.0, 200.0]).is_err());

    let (gaussian, name) = utils234::taper_window(9, "Gaussian", Some(3), 2.5).unwrap();
    assert_eq!(name, "Gauss");
    assert_eq!(gaussian.len(), 9);
    assert_relative_eq!(gaussian[4], 1.0, epsilon = 1e-15);
    assert!(utils234::taper_window(0, "Hamming", None, 1.0).is_err());

    for invalid in [(0, Some(1)), (5, Some(1)), (4, Some(3))] {
        assert!(utils234::set_frame4time_sequence(&[1.0], invalid.0, invalid.1).is_err());
    }

    let (no_power, no_frequency, no_db) = utils234::out_mid_crct("NO", 0, 8_000.0).unwrap();
    assert_eq!(no_power.to_vec(), vec![1.0]);
    assert_eq!(no_frequency.to_vec(), vec![0.0]);
    assert_eq!(no_db.to_vec(), vec![0.0]);
    assert!(utils234::out_mid_crct("unknown", 32, 8_000.0).is_err());

    let forward = utils234::out_mid_crct_filt("ELC", 8_000.0, 0).unwrap();
    let inverse = utils234::out_mid_crct_filt("ELC", 8_000.0, 1).unwrap();
    assert_eq!(forward.len(), inverse.len());
    assert_ne!(forward, inverse);
    assert!(utils234::out_mid_crct_filt("ELC", 8_000.0, 3).is_err());
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
    for level in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(utils234::hl2spl(1_000.0, level).is_err());
        assert!(utils234::hl2pin_cochlea(1_000.0, level).is_err());
    }

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

    let coefficients = fb234::make_asym_cmp_filters_v2(8_000.0, &[500.0], &[2.17], &[2.2]).unwrap();
    let mut state = AcfStatus::new(&coefficients);
    let filtered = fb234::acfilterbank(&coefficients, &mut state, &[1.0], true).unwrap();
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
