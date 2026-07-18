//! Frame-based GCFB v2.34 with hearing-loss characteristics.

use ndarray::{Array1, Array2, Array3, Axis, s};

use super::utils::{self, FrequencyScale};
use crate::gcfb_v211::gammachirp::{self, Carrier, Normalization};
pub use crate::gcfb_v211::gcfb_v211::{ControlMode, LvlEst};
use crate::{Error, Result, dsp};

pub use crate::gcfb_v211::gcfb_v211::{
    AcfCoef, AcfStatus, AsymCmpResponse, CgcResponse, SmoothSpecParam, acfilterbank,
    asym_cmp_frsp_v2, cal_smooth_spec, cmprs_gc_frsp, fp2_to_fr1, fr1_to_fp2,
    make_asym_cmp_filters_v2,
};

#[derive(Clone, Debug)]
pub struct DynHpaf {
    pub str_prc: String,
    pub t_frame: f64,
    pub t_shift: f64,
    pub len_frame: usize,
    pub len_shift: usize,
    pub fs: f64,
    pub name_win: String,
    pub val_win: Array1<f64>,
}

impl Default for DynHpaf {
    fn default() -> Self {
        Self {
            str_prc: "frame-base".into(),
            t_frame: 0.001,
            t_shift: 0.0005,
            len_frame: 0,
            len_shift: 0,
            fs: 0.0,
            name_win: "hanning".into(),
            val_win: Array1::zeros(0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HLoss {
    pub f_audgram_list: Array1<f64>,
    pub type_name: String,
    pub hearing_level_db: Array1<f64>,
    pub pin_loss_db_act: Array1<f64>,
    pub pin_loss_db_act_init: Array1<f64>,
    pub pin_loss_db_pas: Array1<f64>,
    pub compression_health: Array1<f64>,
    pub compression_health_initval: Array1<f64>,
    pub af_gain_cmpnst_db: Array1<f64>,
    pub hl_val_pin_cochlea_db: Array1<f64>,
    pub fb_fr1: Array1<f64>,
    pub fb_hearing_level_db: Array1<f64>,
    pub fb_pin_cochlea_db: Array1<f64>,
    pub fb_pin_loss_db_act: Array1<f64>,
    pub fb_pin_loss_db_pas: Array1<f64>,
    pub fb_compression_health: Array1<f64>,
    pub fb_af_gain_cmpnst_db: Array1<f64>,
}

impl Default for HLoss {
    fn default() -> Self {
        Self {
            f_audgram_list: Array1::from(vec![125., 250., 500., 1000., 2000., 4000., 8000.]),
            type_name: "NH_NormalHearing".into(),
            hearing_level_db: Array1::zeros(7),
            pin_loss_db_act: Array1::zeros(7),
            pin_loss_db_act_init: Array1::zeros(7),
            pin_loss_db_pas: Array1::zeros(7),
            compression_health: Array1::ones(7),
            compression_health_initval: Array1::ones(7),
            af_gain_cmpnst_db: Array1::zeros(7),
            hl_val_pin_cochlea_db: Array1::zeros(7),
            fb_fr1: Array1::zeros(0),
            fb_hearing_level_db: Array1::zeros(0),
            fb_pin_cochlea_db: Array1::zeros(0),
            fb_pin_loss_db_act: Array1::zeros(0),
            fb_pin_loss_db_pas: Array1::zeros(0),
            fb_compression_health: Array1::zeros(0),
            fb_af_gain_cmpnst_db: Array1::zeros(0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GainReference {
    Db(f64),
    NormalizeIoFunction,
}

#[derive(Clone, Debug)]
pub struct GcParam {
    pub fs: f64,
    pub num_ch: usize,
    /// Requested channel-frequency interval `[low, high]`, satisfying
    /// `0 < low < high < fs / 2`.
    pub f_range: [f64; 2],
    pub out_mid_crct: String,
    pub n: f64,
    pub b1: [f64; 2],
    pub c1: [f64; 2],
    pub frat: [[f64; 2]; 2],
    pub b2: [[f64; 2]; 2],
    pub c2: [[f64; 2]; 2],
    pub ctrl: ControlMode,
    pub gain_cmpnst_db: f64,
    pub gain_ref: GainReference,
    pub level_db_scgcfb: f64,
    pub lvl_est: LvlEst,
    pub num_update_asym_cmp: usize,
    pub dyn_hpaf: DynHpaf,
    pub hloss_type: String,
    pub hloss_compression_health: Option<f64>,
    pub hloss_hearing_level_db: Option<Array1<f64>>,
    pub hloss: HLoss,
    pub fr1: Array1<f64>,
    pub meddis_hc_level_rms0db_spldb: f64,
}

impl Default for GcParam {
    fn default() -> Self {
        Self {
            fs: 48000.,
            num_ch: 100,
            f_range: [100., 6000.],
            out_mid_crct: "ELC".into(),
            n: 4.,
            b1: [1.81, 0.],
            c1: [-2.96, 0.],
            frat: [[0.466, 0.], [0.0109, 0.]],
            b2: [[2.17, 0.], [0., 0.]],
            c2: [[2.2, 0.], [0., 0.]],
            ctrl: ControlMode::Dynamic,
            gain_cmpnst_db: -1.,
            gain_ref: GainReference::NormalizeIoFunction,
            level_db_scgcfb: 50.,
            lvl_est: LvlEst::default(),
            num_update_asym_cmp: 1,
            dyn_hpaf: DynHpaf::default(),
            hloss_type: "NH".into(),
            hloss_compression_health: None,
            hloss_hearing_level_db: None,
            hloss: HLoss::default(),
            fr1: Array1::zeros(0),
            meddis_hc_level_rms0db_spldb: 30.,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GcResp {
    pub fr1: Array1<f64>,
    pub fr2: Array2<f64>,
    pub erb_space1: f64,
    pub ef: Array1<f64>,
    pub b1_val: Array1<f64>,
    pub c1_val: Array1<f64>,
    pub fp1: Array1<f64>,
    pub fp2: Array1<f64>,
    pub b2_val: Array1<f64>,
    pub c2_val: Array1<f64>,
    pub frat_val: Array2<f64>,
    pub frat0_val: Array1<f64>,
    pub frat1_val: Array1<f64>,
    pub pc_hpaf: Array1<f64>,
    pub frat0_pc: Array1<f64>,
    pub lvl_db: Array2<f64>,
    pub lvl_db_frame: Array2<f64>,
    pub pgc_frame: Array2<f64>,
    pub scgc_frame: Array2<f64>,
    pub asym_func_gain: Array2<f64>,
    pub gain_factor: Array1<f64>,
    pub cgc_ref: Option<CgcResponse>,
}

#[derive(Clone, Debug)]
pub struct GcfbOutput {
    pub dcgc_out: Array2<f64>,
    pub scgc_smpl: Array2<f64>,
    pub gc_param: GcParam,
    pub gc_resp: GcResp,
}

fn validate_prepared_frequencies(fs: f64, frequencies: &[f64], name: &str) -> Result<()> {
    if frequencies
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0 || *frequency >= fs / 2.0)
    {
        return Err(Error::InvalidParameter(format!(
            "{name} must be finite, positive, and below Nyquist"
        )));
    }
    Ok(())
}

fn initial_asymmetric_ratio_and_centers(
    param: &GcParam,
    response: &GcResp,
) -> (Array1<f64>, Array1<f64>) {
    let ratios = if param.ctrl == ControlMode::Static {
        let level = param.level_db_scgcfb;
        Array1::from_iter((0..param.num_ch).map(|ch| {
            response.frat0_pc[ch] + response.frat1_val[ch] * (level - response.pc_hpaf[ch])
        }))
    } else {
        Array1::from_elem(param.num_ch, param.lvl_est.frat)
    };
    let centers = &ratios * &response.fp1;
    (ratios, centers)
}

pub fn set_param(param: GcParam) -> Result<(GcParam, GcResp)> {
    set_param_with_preserved_hearing_loss(param, None)
}

fn set_param_with_preserved_hearing_loss(
    mut param: GcParam,
    preserved_hearing_loss: Option<&HLoss>,
) -> Result<(GcParam, GcResp)> {
    if !param.fs.is_finite()
        || param.fs <= 0.
        || param.num_ch < 2
        || param.f_range.iter().any(|value| !value.is_finite())
        || param.f_range[0] <= 0.0
        || param.f_range[1] <= param.f_range[0]
        || param.f_range[1] >= param.fs / 2.0
        || param.num_update_asym_cmp == 0
        || !param.dyn_hpaf.t_frame.is_finite()
        || param.dyn_hpaf.t_frame <= 0.0
        || !param.dyn_hpaf.t_shift.is_finite()
        || param.dyn_hpaf.t_shift <= 0.0
    {
        return Err(Error::InvalidParameter(
            "v2.34 requires a finite positive sample rate, at least two channels, a frequency range satisfying 0 < low < high < fs / 2, and positive frame and update periods"
                .into(),
        ));
    }
    param.dyn_hpaf.len_frame = (param.dyn_hpaf.t_frame * param.fs).trunc() as usize;
    param.dyn_hpaf.len_shift = (param.dyn_hpaf.t_shift * param.fs).trunc() as usize;
    if param.dyn_hpaf.len_frame == 0 || param.dyn_hpaf.len_shift == 0 {
        return Err(Error::InvalidParameter(
            "dynamic frame and shift must contain at least one sample".into(),
        ));
    }
    param.dyn_hpaf.t_frame = param.dyn_hpaf.len_frame as f64 / param.fs;
    param.dyn_hpaf.t_shift = param.dyn_hpaf.len_shift as f64 / param.fs;
    param.dyn_hpaf.fs = 1. / param.dyn_hpaf.t_shift;
    let processing = param.dyn_hpaf.str_prc.to_ascii_lowercase();
    let frame_processing = processing.contains("frame");
    let sample_processing = processing.contains("sample");
    if frame_processing == sample_processing {
        return Err(Error::InvalidParameter(
            "dynamic processing mode must select exactly one of frame-base or sample-base".into(),
        ));
    }
    param.dyn_hpaf.str_prc = if frame_processing {
        "frame-base".into()
    } else {
        "sample-base".into()
    };
    if frame_processing {
        let window_name = param.dyn_hpaf.name_win.to_ascii_lowercase();
        let win = if window_name.contains("hann") {
            dsp::hanning(param.dyn_hpaf.len_frame)
        } else if window_name.contains("hamm") {
            dsp::hamming(param.dyn_hpaf.len_frame)
        } else {
            return Err(Error::InvalidParameter(
                "frame window must be hanning or hamming".into(),
            ));
        };
        let sum: f64 = win.iter().sum();
        param.dyn_hpaf.val_win = Array1::from_iter(win.into_iter().map(|v| v / sum));
    }
    let (fr1, erb_grid) =
        utils::equal_freq_scale(FrequencyScale::Erb, param.num_ch, param.f_range)?;
    validate_prepared_frequencies(
        param.fs,
        fr1.as_slice().unwrap(),
        "channel grid frequencies",
    )?;
    param.fr1 = fr1.clone();
    let erb_space = erb_grid
        .windows(2)
        .into_iter()
        .map(|w| w[1] - w[0])
        .sum::<f64>()
        / (param.num_ch - 1) as f64;
    let (erb, _) = utils::freq2erb(fr1.as_slice().unwrap());
    let (erb1k, _) = utils::freq2erb(&[1000.]);
    let ef = erb.mapv(|v| v / erb1k[0] - 1.);
    let b1 = ef.mapv(|v| param.b1[0] + param.b1[1] * v);
    let c1 = ef.mapv(|v| param.c1[0] + param.c1[1] * v);
    let (_, width) = utils::freq2erb(fr1.as_slice().unwrap());
    let fp1 =
        Array1::from_iter((0..param.num_ch).map(|i| fr1[i] + c1[i] * width[i] * b1[i] / param.n));
    validate_prepared_frequencies(param.fs, fp1.as_slice().unwrap(), "derived filter centers")?;
    let b2 = ef.mapv(|v| param.b2[0][0] + param.b2[0][1] * v);
    let c2 = ef.mapv(|v| param.c2[0][0] + param.c2[0][1] * v);
    let frat0 = ef.mapv(|v| param.frat[0][0] + param.frat[0][1] * v);
    let frat1 = ef.mapv(|v| param.frat[1][0] + param.frat[1][1] * v);
    let pc = Array1::from_iter((0..param.num_ch).map(|i| (1. - frat0[i]) / frat1[i]));
    let frat0_pc = Array1::from_iter((0..param.num_ch).map(|i| frat0[i] + frat1[i] * pc[i]));
    let mut response = GcResp {
        fr1: fr1.clone(),
        fr2: Array2::zeros((param.num_ch, 0)),
        erb_space1: erb_space,
        ef,
        b1_val: b1,
        c1_val: c1,
        fp1,
        fp2: Array1::zeros(param.num_ch),
        b2_val: b2,
        c2_val: c2,
        frat_val: Array2::zeros((param.num_ch, 0)),
        frat0_val: frat0,
        frat1_val: frat1,
        pc_hpaf: pc,
        frat0_pc,
        lvl_db: Array2::zeros((param.num_ch, 0)),
        lvl_db_frame: Array2::zeros((0, 0)),
        pgc_frame: Array2::zeros((0, 0)),
        scgc_frame: Array2::zeros((0, 0)),
        asym_func_gain: Array2::zeros((0, 0)),
        gain_factor: Array1::ones(param.num_ch),
        cgc_ref: None,
    };
    let (_, initial_centers) = initial_asymmetric_ratio_and_centers(&param, &response);
    validate_prepared_frequencies(
        param.fs,
        initial_centers.as_slice().unwrap(),
        "initial asymmetric filter centers",
    )?;
    if let Some(hearing_loss) = preserved_hearing_loss {
        param.hloss = hearing_loss.clone();
    } else {
        param = gcfb_v23_hearing_loss(param, &response)?;
    }
    let shift = (param.lvl_est.lct_erb / erb_space).round() as isize;
    param.lvl_est.exp_decay_val =
        (-1. / (param.lvl_est.decay_hl * param.fs / 1000.) * 2_f64.ln()).exp();
    param.lvl_est.erb_space1 = erb_space;
    param.lvl_est.n_ch_shift = shift;
    param.lvl_est.n_ch_lvl_est = Array1::from_iter(
        (0..param.num_ch)
            .map(|ch| (ch as isize + shift).clamp(0, param.num_ch as isize - 1) as usize),
    );
    param.lvl_est.lvl_lin_min_lim = 10_f64.powf(-param.lvl_est.rms2spldb / 20.);
    param.lvl_est.lvl_lin_ref = 10_f64.powf((param.lvl_est.ref_db - param.lvl_est.rms2spldb) / 20.);
    response.fr1 = param.fr1.clone();
    Ok((param, response))
}

fn hearing_pattern(
    kind: &str,
    manual: Option<&Array1<f64>>,
) -> Result<(&'static str, Array1<f64>)> {
    let key = kind.split('_').next().unwrap_or(kind).to_ascii_uppercase();
    let pair = match key.as_str() {
        "NH" => ("NH_NormalHearing", vec![0.; 7]),
        "HL0" => (
            "HLval_ManualSet",
            manual
                .ok_or_else(|| {
                    Error::InvalidParameter("HL0 requires hloss_hearing_level_db".into())
                })?
                .to_vec(),
        ),
        "HL1" => ("HL1_Example", vec![10., 4., 10., 13., 48., 58., 79.]),
        "HL2" => (
            "HL2_Tsuiki2002_80yr",
            vec![23.5, 24.3, 26.8, 27.9, 32.9, 48.3, 68.5],
        ),
        "HL3" => (
            "HL3_ISO7029_70yr_male",
            vec![8., 8., 9., 10., 19., 43., 59.],
        ),
        "HL4" => (
            "HL4_ISO7029_70yr_female",
            vec![8., 8., 9., 10., 16., 24., 41.],
        ),
        "HL5" => ("HL5_ISO7029_60yr_male", vec![5., 5., 6., 7., 12., 28., 39.]),
        "HL6" => (
            "HL6_ISO7029_60yr_female",
            vec![5., 5., 6., 7., 11., 16., 26.],
        ),
        "HL7" => (
            "HL7_Example_Otosclerosis",
            vec![50., 55., 50., 50., 40., 25., 20.],
        ),
        "HL8" => (
            "HL8_Example_NoiseInduced",
            vec![15., 10., 15., 10., 10., 40., 20.],
        ),
        _ => {
            return Err(Error::InvalidParameter(
                "hearing-loss type must be NH or HL0..HL8".into(),
            ));
        }
    };
    if pair.1.len() != 7 || pair.1.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err(Error::InvalidParameter(
            "audiogram must contain seven finite non-negative values".into(),
        ));
    }
    Ok((pair.0, Array1::from(pair.1)))
}

pub fn gcfb_v23_hearing_loss(mut param: GcParam, response: &GcResp) -> Result<GcParam> {
    let (name, hearing) =
        hearing_pattern(&param.hloss_type, param.hloss_hearing_level_db.as_ref())?;
    let mut loss = HLoss {
        type_name: name.into(),
        hearing_level_db: hearing,
        ..HLoss::default()
    };
    let default_health = if name == "NH_NormalHearing" { 1. } else { 0.5 };
    loss.compression_health =
        Array1::from_elem(7, param.hloss_compression_health.unwrap_or(default_health));
    loss.compression_health_initval = loss.compression_health.clone();
    let mut act = Array1::zeros(7);
    let mut act_initial = Array1::zeros(7);
    let mut passive = Array1::zeros(7);
    let mut gain = Array1::zeros(7);
    let mut hl_pin = Array1::zeros(7);
    for i in 0..7 {
        let f = loss.f_audgram_list[i];
        let hl0 = utils::hl2pin_cochlea(f, 0.)?;
        let (_, io_normal) = asym_func_in_out_scalar(&param, response, f, 1., hl0);
        let mut health = loss.compression_health[i];
        let reduction =
            gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, health, io_normal)?;
        act[i] = reduction - hl0;
        act_initial[i] = act[i];
        passive[i] = (loss.hearing_level_db[i] - act[i]).max(0.);
        if passive[i] < f64::EPSILON * 1e4 {
            act[i] = loss.hearing_level_db[i];
            let health_values: Vec<f64> = (0..10).map(|j| 1. - j as f64 * 0.1).collect();
            let mut active_values = Vec::new();
            for &h in &health_values {
                active_values.push(
                    gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, h, io_normal)? - hl0,
                );
            }
            health =
                dsp::interp1(&active_values, &health_values, &[act[i]], true)?[0].clamp(0., 1.);
            act[i] = gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, health, io_normal)?
                - hl0;
            passive[i] = loss.hearing_level_db[i] - act[i];
        }
        loss.compression_health[i] = health;
        hl_pin[i] = utils::hl2pin_cochlea(f, 0.)? + loss.hearing_level_db[i];
        let (_, io) = asym_func_in_out_scalar(&param, response, f, health, hl_pin[i]);
        gain[i] = io;
    }
    loss.pin_loss_db_act = act.clone();
    loss.pin_loss_db_act_init = act_initial;
    loss.pin_loss_db_pas = passive.clone();
    loss.af_gain_cmpnst_db = gain.clone();
    loss.hl_val_pin_cochlea_db = hl_pin.clone();
    loss.fb_fr1 = response.fr1.clone();
    let (aud_erb, _) = utils::freq2erb(loss.f_audgram_list.as_slice().unwrap());
    let (fb_erb, _) = utils::freq2erb(response.fr1.as_slice().unwrap());
    let x = aud_erb.as_slice().unwrap();
    let q = fb_erb.as_slice().unwrap();
    loss.fb_hearing_level_db =
        utils::interp1(x, loss.hearing_level_db.as_slice().unwrap(), q, true)?;
    loss.fb_pin_cochlea_db = utils::interp1(x, hl_pin.as_slice().unwrap(), q, true)?;
    loss.fb_pin_loss_db_act = utils::interp1(x, act.as_slice().unwrap(), q, true)?;
    loss.fb_pin_loss_db_pas = utils::interp1(x, passive.as_slice().unwrap(), q, true)?;
    loss.fb_compression_health =
        utils::interp1(x, loss.compression_health.as_slice().unwrap(), q, true)?
            .mapv(|v| v.clamp(0., 1.));
    loss.fb_af_gain_cmpnst_db = utils::interp1(x, gain.as_slice().unwrap(), q, true)?;
    param.hloss = loss;
    Ok(param)
}

pub fn cal_asym_func(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    compression_health: f64,
    pin_db: f64,
) -> f64 {
    let ch = param
        .fr1
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - fr1query).abs().total_cmp(&(b.1 - fr1query).abs()))
        .unwrap()
        .0;
    let frat = response.frat0_pc[ch] + response.frat1_val[ch] * (pin_db - response.pc_hpaf[ch]);
    let fr2 = frat * response.fp1[ch];
    let (_, w) = utils::freq2erb(&[fr2]);
    (compression_health
        * response.c2_val[ch]
        * (response.fp1[ch] - fr2).atan2(response.b2_val[ch] * w[0]))
    .exp()
}

fn asym_func_in_out_scalar(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    health: f64,
    pin: f64,
) -> (f64, f64) {
    let v = cal_asym_func(param, response, fr1query, health, pin);
    let norm = cal_asym_func(param, response, fr1query, health, 100.);
    let db = 20. * (v / norm).log10();
    (db, db + pin)
}

pub fn gcfb_v23_asym_func_in_out(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    compression_health: f64,
    pin_db: &[f64],
) -> (Array1<f64>, Array1<f64>) {
    let mut af = Array1::zeros(pin_db.len());
    let mut io = Array1::zeros(pin_db.len());
    for (i, &pin) in pin_db.iter().enumerate() {
        (af[i], io[i]) =
            asym_func_in_out_scalar(param, response, fr1query, compression_health, pin);
    }
    (af, io)
}

pub fn gcfb_v23_asym_func_in_out_inv_io_func(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    health: f64,
    io_db: f64,
) -> Result<f64> {
    let pins: Vec<f64> = (0..=2700).map(|i| -120. + i as f64 * 0.1).collect();
    let (_, ios) = gcfb_v23_asym_func_in_out(param, response, fr1query, health, &pins);
    Ok(dsp::interp1(ios.as_slice().unwrap(), &pins, &[io_db], true)?[0])
}

pub fn gcfb_v23_frame_base(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
) -> Result<Array2<f64>> {
    let (channels, samples) = pgc.dim();
    let channel_vectors_match = [
        param.fr1.len(),
        param.hloss.fb_compression_health.len(),
        param.lvl_est.n_ch_lvl_est.len(),
        response.b1_val.len(),
        response.c1_val.len(),
        response.fp1.len(),
        response.b2_val.len(),
        response.c2_val.len(),
        response.frat0_pc.len(),
        response.frat1_val.len(),
        response.pc_hpaf.len(),
    ]
    .into_iter()
    .all(|len| len == channels);
    if channels == 0
        || samples == 0
        || scgc.dim() != pgc.dim()
        || param.num_ch != channels
        || !channel_vectors_match
        || param.dyn_hpaf.val_win.len() != param.dyn_hpaf.len_frame
        || param
            .lvl_est
            .n_ch_lvl_est
            .iter()
            .any(|&source| source >= channels)
    {
        return Err(Error::InvalidParameter(
            "frame processing requires non-empty, equally shaped channel matrices and matching prepared parameters"
                .into(),
        ));
    }
    let decay = param
        .lvl_est
        .exp_decay_val
        .powf(param.dyn_hpaf.len_shift as f64);
    let c2: Array1<f64> = &param.hloss.fb_compression_health * param.lvl_est.c2;
    let static_response = cmprs_gc_frsp(
        param.fr1.as_slice().unwrap(),
        param.fs,
        param.n,
        response.b1_val.as_slice().unwrap(),
        response.c1_val.as_slice().unwrap(),
        &[param.lvl_est.frat],
        &[param.lvl_est.b2],
        c2.as_slice().unwrap(),
        2048,
    )?;
    let first_frames = dsp::frame_sequence(
        pgc.row(0).as_slice().unwrap(),
        param.dyn_hpaf.len_frame,
        param.dyn_hpaf.len_shift,
    )?
    .0
    .ncols();
    let mut out = Array2::zeros((channels, first_frames));
    response.lvl_db_frame = Array2::zeros((channels, first_frames));
    response.pgc_frame = Array2::zeros((channels, first_frames));
    response.scgc_frame = Array2::zeros((channels, first_frames));
    response.asym_func_gain = Array2::zeros((channels, first_frames));
    for ch in 0..channels {
        let (pf, _) = dsp::frame_sequence(
            pgc.row(ch).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let (sf, _) = dsp::frame_sequence(
            scgc.row(ch).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let source = param.lvl_est.n_ch_lvl_est[ch];
        let (l1, _) = dsp::frame_sequence(
            pgc.row(source).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let (l2, _) = dsp::frame_sequence(
            scgc.row(source).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let weighted = |m: &Array2<f64>, frame: usize| -> f64 {
            param
                .dyn_hpaf
                .val_win
                .iter()
                .enumerate()
                .map(|(i, w)| w * m[[i, frame]].powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut level1 = Array1::from_iter((0..first_frames).map(|f| weighted(&l1, f)));
        let mut level2 = Array1::from_iter((0..first_frames).map(|f| weighted(&l2, f)));
        for f in 1..first_frames {
            level1[f] = level1[f].max(level1[f - 1] * decay);
            level2[f] = level2[f].max(level2[f - 1] * decay);
        }
        for frame in 0..first_frames {
            response.pgc_frame[[ch, frame]] = weighted(&pf, frame);
            response.scgc_frame[[ch, frame]] = weighted(&sf, frame);
            let total = param.lvl_est.weight
                * param.lvl_est.lvl_lin_ref
                * (level1[frame] / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[0])
                + (1. - param.lvl_est.weight)
                    * param.lvl_est.lvl_lin_ref
                    * (level2[frame] / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[1]);
            let level_db = 20. * total.max(param.lvl_est.lvl_lin_min_lim).log10()
                + param.lvl_est.rms2spldb
                - 3.;
            response.lvl_db_frame[[ch, frame]] = level_db;
            let (af, _) = asym_func_in_out_scalar(
                param,
                response,
                param.fr1[ch],
                param.hloss.fb_compression_health[ch],
                level_db,
            );
            let gain = 10_f64.powf(af / 20.);
            response.asym_func_gain[[ch, frame]] = gain;
            out[[ch, frame]] =
                gain * static_response.norm_fct_fp2[ch] * response.scgc_frame[[ch, frame]];
        }
    }
    Ok(out)
}

pub fn gcfb_v23_sample_base(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
) -> Result<Array2<f64>> {
    let (channels, samples) = pgc.dim();
    let channel_vectors_match = [
        param.fr1.len(),
        param.hloss.fb_compression_health.len(),
        param.lvl_est.n_ch_lvl_est.len(),
        response.b1_val.len(),
        response.c1_val.len(),
        response.fp1.len(),
        response.b2_val.len(),
        response.c2_val.len(),
        response.frat0_pc.len(),
        response.frat1_val.len(),
        response.pc_hpaf.len(),
    ]
    .into_iter()
    .all(|len| len == channels);
    if channels == 0
        || samples == 0
        || scgc.dim() != pgc.dim()
        || param.num_ch != channels
        || param.num_update_asym_cmp == 0
        || !channel_vectors_match
        || param
            .lvl_est
            .n_ch_lvl_est
            .iter()
            .any(|&source| source >= channels)
    {
        return Err(Error::InvalidParameter(
            "sample processing requires non-empty, equally shaped channel matrices and matching prepared parameters"
                .into(),
        ));
    }
    let mut out = Array2::zeros((channels, samples));
    response.fr2 = Array2::zeros((channels, samples));
    response.frat_val = Array2::zeros((channels, samples));
    response.lvl_db = Array2::zeros((channels, samples));
    let (_, centers) = initial_asymmetric_ratio_and_centers(param, response);
    let mut coef = make_asym_cmp_filters_v2(
        param.fs,
        centers.as_slice().unwrap(),
        response.b2_val.as_slice().unwrap(),
        response.c2_val.as_slice().unwrap(),
    )?;
    let mut status = AcfStatus::new(&coef);
    let mut previous = Array2::<f64>::zeros((channels, 2));
    for sample in 0..samples {
        for ch in 0..channels {
            let source = param.lvl_est.n_ch_lvl_est[ch];
            let a = pgc[[source, sample]]
                .max(0.)
                .max(previous[[ch, 0]] * param.lvl_est.exp_decay_val);
            let b = scgc[[source, sample]]
                .max(0.)
                .max(previous[[ch, 1]] * param.lvl_est.exp_decay_val);
            previous[[ch, 0]] = a;
            previous[[ch, 1]] = b;
            let total = param.lvl_est.weight
                * param.lvl_est.lvl_lin_ref
                * (a / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[0])
                + (1. - param.lvl_est.weight)
                    * param.lvl_est.lvl_lin_ref
                    * (b / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[1]);
            let db =
                20. * total.max(param.lvl_est.lvl_lin_min_lim).log10() + param.lvl_est.rms2spldb;
            response.lvl_db[[ch, sample]] = db;
            let ratio = response.frat0_pc[ch]
                + param.hloss.fb_compression_health[ch]
                    * response.frat1_val[ch]
                    * (db - response.pc_hpaf[ch]);
            response.frat_val[[ch, sample]] = ratio;
            response.fr2[[ch, sample]] = response.fp1[ch] * ratio;
        }
        if sample % param.num_update_asym_cmp == 0 {
            let centers = response.fr2.column(sample).to_vec();
            coef = make_asym_cmp_filters_v2(
                param.fs,
                &centers,
                response.b2_val.as_slice().unwrap(),
                response.c2_val.as_slice().unwrap(),
            )?;
        }
        let value = status.process(&coef, &pgc.column(sample).to_vec(), false)?;
        out.column_mut(sample).assign(&value);
    }
    Ok(out)
}

pub fn gcfb_v234(snd_in: &[f64], gc_param: GcParam) -> Result<GcfbOutput> {
    gcfb_v234_internal(snd_in, gc_param, None)
}

pub(crate) fn gcfb_v234_with_preserved_hearing_loss(
    snd_in: &[f64],
    gc_param: GcParam,
    hearing_loss: &HLoss,
) -> Result<GcfbOutput> {
    gcfb_v234_internal(snd_in, gc_param, Some(hearing_loss))
}

fn gcfb_v234_internal(
    snd_in: &[f64],
    gc_param: GcParam,
    preserved_hearing_loss: Option<&HLoss>,
) -> Result<GcfbOutput> {
    if snd_in.is_empty() {
        return Err(Error::InvalidParameter(
            "input sound cannot be empty".into(),
        ));
    }
    let (param, mut response) =
        set_param_with_preserved_hearing_loss(gc_param, preserved_hearing_loss)?;
    let snd = if param.out_mid_crct.eq_ignore_ascii_case("no") {
        snd_in.to_vec()
    } else {
        let (fir, _) = utils::mk_filter_field2cochlea(&param.out_mid_crct, param.fs, true)?;
        dsp::lfilter(fir.as_slice().unwrap(), &[1.], snd_in)?
    };
    let channels = param.num_ch;
    let samples = snd.len();
    let mut pgc = Array2::zeros((channels, samples));
    let mut scgc = Array2::zeros((channels, samples));
    let (initial_ratios, fixed_centers) = initial_asymmetric_ratio_and_centers(&param, &response);
    let fixed_c2: Array1<f64>;
    if param.ctrl == ControlMode::Static {
        let level = param.level_db_scgcfb;
        response.fr2 = Array2::zeros((channels, 1));
        response.fr2.column_mut(0).assign(&fixed_centers);
        response.frat_val = Array2::zeros((channels, 1));
        response.frat_val.column_mut(0).assign(&initial_ratios);
        response.lvl_db = Array2::from_elem((channels, 1), level);
        for ch in 0..channels {
            response.fp2[ch] = fr1_to_fp2(
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
                response.b2_val[ch],
                response.c2_val[ch],
                initial_ratios[ch],
                response.fr1[ch],
            )?
            .0;
        }
        fixed_c2 = response.c2_val.clone();
    } else {
        fixed_c2 = &param.hloss.fb_compression_health * param.lvl_est.c2;
    }
    let level_b2 = [param.lvl_est.b2];
    let fixed_b2 = if param.ctrl == ControlMode::Static {
        response.b2_val.as_slice().unwrap()
    } else {
        &level_b2
    };
    let fixed_coef = make_asym_cmp_filters_v2(
        param.fs,
        fixed_centers.as_slice().unwrap(),
        fixed_b2,
        fixed_c2.as_slice().unwrap(),
    )?;
    for ch in 0..channels {
        let impulse = gammachirp::gammachirp(
            &[response.fr1[ch]],
            param.fs,
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            0.,
            Carrier::Cosine,
            Normalization::Peak,
        )?;
        let filtered = utils::fftfilt(impulse.gc.row(0).as_slice().unwrap(), &snd);
        pgc.row_mut(ch).assign(&filtered);
        let mut value = filtered.to_vec();
        for section in 0..4 {
            value = dsp::lfilter(
                &fixed_coef.bz.slice(s![ch, .., section]).to_vec(),
                &fixed_coef.ap.slice(s![ch, .., section]).to_vec(),
                &value,
            )?;
        }
        scgc.row_mut(ch).assign(&Array1::from(value));
    }
    let mut dcgc = match param.ctrl {
        ControlMode::Static => scgc.clone(),
        ControlMode::Dynamic if param.dyn_hpaf.str_prc.contains("frame") => {
            gcfb_v23_frame_base(&pgc, &scgc, &param, &mut response)?
        }
        ControlMode::Dynamic if param.dyn_hpaf.str_prc.contains("sample") => {
            gcfb_v23_sample_base(&pgc, &scgc, &param, &mut response)?
        }
        ControlMode::Dynamic => {
            return Err(Error::InvalidParameter(
                "dynamic processing mode must be frame-base or sample-base".into(),
            ));
        }
        ControlMode::Level => scgc.clone(),
    };
    match param.gain_ref {
        GainReference::NormalizeIoFunction => {
            for ch in 0..channels {
                let gain = 10_f64.powf(-param.hloss.fb_af_gain_cmpnst_db[ch] / 20.);
                dcgc.row_mut(ch).mapv_inplace(|v| v * gain);
                response.gain_factor[ch] = gain;
            }
        }
        GainReference::Db(db) => {
            let ratios = Array1::from_iter((0..channels).map(|ch| {
                response.frat0_pc[ch] + response.frat1_val[ch] * (db - response.pc_hpaf[ch])
            }));
            let reference = cmprs_gc_frsp(
                response.fr1.as_slice().unwrap(),
                param.fs,
                param.n,
                response.b1_val.as_slice().unwrap(),
                response.c1_val.as_slice().unwrap(),
                ratios.as_slice().unwrap(),
                response.b2_val.as_slice().unwrap(),
                response.c2_val.as_slice().unwrap(),
                1024,
            )?;
            response.gain_factor = reference
                .norm_fct_fp2
                .mapv(|v| 10_f64.powf(param.gain_cmpnst_db / 20.) * v);
            for ch in 0..channels {
                dcgc.row_mut(ch)
                    .mapv_inplace(|v| v * response.gain_factor[ch]);
            }
            response.cgc_ref = Some(reference);
        }
    }
    Ok(GcfbOutput {
        dcgc_out: dcgc,
        scgc_smpl: scgc,
        gc_param: param,
        gc_resp: response,
    })
}

pub fn set_frame4time_sequence(
    snd: &[f64],
    len_win: usize,
    len_shift: Option<usize>,
) -> Result<(Array2<f64>, Array1<usize>)> {
    dsp::frame_sequence(snd, len_win, len_shift.unwrap_or(len_win / 2))
}

pub fn gcfb_v23_synth_snd(gc_smpl: &Array2<f64>, param: &GcParam) -> Result<Array1<f64>> {
    let mean = gc_smpl.mean_axis(Axis(0)).ok_or_else(|| {
        Error::InvalidParameter("cannot synthesize empty filterbank output".into())
    })?;
    if mean.is_empty() {
        return Err(Error::InvalidParameter(
            "cannot synthesize empty filterbank output".into(),
        ));
    }
    if param.out_mid_crct.eq_ignore_ascii_case("no") {
        Ok(mean.mapv(|v| -15. * v))
    } else {
        let (fir, _) = utils::mk_filter_field2cochlea(&param.out_mid_crct, param.fs, false)?;
        // The legacy ELC inverse is linear phase. Pad before shifting so the
        // compensated output includes the FIR tail even for short signals.
        // The other correction filters are minimum phase and need no fixed
        // ELC delay compensation.
        let delay = if param.out_mid_crct.eq_ignore_ascii_case("ELC") {
            fir.len().saturating_sub(1) / 2
        } else {
            0
        };
        let mut padded = mean.to_vec();
        padded.resize(padded.len() + delay, 0.);
        let filtered = dsp::lfilter(fir.as_slice().unwrap(), &[1.], &padded)?;
        let out: Vec<f64> = (0..mean.len())
            .map(|i| -15. * filtered[i + delay])
            .collect();
        Ok(Array1::from(out))
    }
}

#[derive(Clone, Debug)]
pub struct EmParam {
    pub reduce_db: Array1<f64>,
    pub f_cutoff: Array1<f64>,
    pub fc_mod_list: Array1<f64>,
    pub fs: f64,
    pub fb_fr1: Array1<f64>,
    pub fb_reduce_db: Array1<f64>,
    pub fb_f_cutoff: Array1<f64>,
}
impl Default for EmParam {
    fn default() -> Self {
        Self {
            reduce_db: Array1::from_elem(7, 0.),
            f_cutoff: Array1::from_elem(7, 128.),
            fc_mod_list: Array1::from(vec![1., 2., 4., 8., 16., 32., 64., 128., 256.]),
            fs: 0.,
            fb_fr1: Array1::zeros(0),
            fb_reduce_db: Array1::zeros(0),
            fb_f_cutoff: Array1::zeros(0),
        }
    }
}

pub fn gcfb_v23_env_mod_loss(
    frames: &Array2<f64>,
    param: &GcParam,
    mut em: EmParam,
) -> Result<(Array2<f64>, EmParam)> {
    if !param.dyn_hpaf.str_prc.contains("frame") {
        return Err(Error::InvalidParameter(
            "envelope modulation loss requires frame processing".into(),
        ));
    }
    if em.reduce_db.len() != 7 || em.f_cutoff.len() != 7 {
        return Err(Error::InvalidParameter(
            "envelope parameters require seven audiogram values".into(),
        ));
    }
    em.fs = param.dyn_hpaf.fs;
    if !em.fs.is_finite()
        || em.fs <= 0.0
        || em.reduce_db.iter().any(|value| !value.is_finite())
        || em
            .f_cutoff
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= em.fs / 2.0)
        || frames.nrows() != param.fr1.len()
    {
        return Err(Error::InvalidParameter(
            "envelope modulation parameters must be finite, cutoffs must be below Nyquist, and frame channels must match the filterbank".into(),
        ));
    }
    let (aud, _) = utils::freq2erb(param.hloss.f_audgram_list.as_slice().unwrap());
    let (fb, _) = utils::freq2erb(param.fr1.as_slice().unwrap());
    em.fb_fr1 = param.fr1.clone();
    em.fb_reduce_db = utils::interp1(
        aud.as_slice().unwrap(),
        em.reduce_db.as_slice().unwrap(),
        fb.as_slice().unwrap(),
        true,
    )?;
    em.fb_f_cutoff = utils::interp1(
        aud.as_slice().unwrap(),
        em.f_cutoff.as_slice().unwrap(),
        fb.as_slice().unwrap(),
        true,
    )?;
    if em
        .fb_f_cutoff
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= em.fs / 2.0)
    {
        return Err(Error::InvalidParameter(
            "interpolated envelope cutoffs must be finite and below Nyquist".into(),
        ));
    }
    let mut out = Array2::zeros(frames.dim());
    for ch in 0..frames.nrows() {
        let env = frames.row(ch);
        let dc = (env.iter().map(|v| v * v).sum::<f64>() / env.len() as f64).sqrt();
        let hp: Vec<f64> = env.iter().map(|v| v - dc).collect();
        let (b, a) = dsp::first_order_lowpass(em.fb_f_cutoff[ch], em.fs);
        let filtered = dsp::lfilter(&b, &a, &hp)?;
        let gain = 10_f64.powf(-em.fb_reduce_db[ch] / 20.);
        for i in 0..frames.ncols().saturating_sub(1) {
            out[[ch, i]] = dc + gain * filtered[i + 1];
        }
    }
    Ok((out, em))
}

pub fn gcfb_v23_env_mod_fb(env: &[f64], em: &EmParam) -> Result<Array2<f64>> {
    if !em.fs.is_finite()
        || em.fs <= 0.
        || em.fc_mod_list.iter().any(|frequency| {
            !frequency.is_finite() || *frequency <= 0.0 || *frequency >= em.fs / 2.0
        })
    {
        return Err(Error::InvalidParameter(
            "modulation filterbank frequencies must be finite, positive, and below Nyquist".into(),
        ));
    }
    let mut out = Array2::zeros((em.fc_mod_list.len(), env.len()));
    for (ch, &fc) in em.fc_mod_list.iter().enumerate() {
        if ch == 0 {
            let (b, a) = dsp::third_order_butterworth_lowpass(fc, em.fs);
            out.row_mut(ch)
                .assign(&Array1::from(dsp::lfilter(&b, &a, env)?));
        } else {
            let warp = 2. * std::f64::consts::PI * fc / em.fs;
            let w0 = (warp / 2.).tan();
            let b0 = w0;
            let a0 = 1. + b0 + w0 * w0;
            let b = [b0 / a0, 0., -b0 / a0];
            let a = [1., (2. * w0 * w0 - 2.) / a0, (1. - b0 + w0 * w0) / a0];
            out.row_mut(ch)
                .assign(&Array1::from(dsp::lfilter(&b, &a, env)?));
        }
    }
    Ok(out)
}

pub fn gcfb_v23_ana_env_mod(
    frames: &Array2<f64>,
    param: &GcParam,
    mut em: EmParam,
) -> Result<(Array3<f64>, EmParam)> {
    if !param.dyn_hpaf.str_prc.contains("frame") {
        return Err(Error::InvalidParameter(
            "modulation analysis requires frame processing".into(),
        ));
    }
    if em.fs == 0. {
        em.fs = param.dyn_hpaf.fs;
    }
    let mut out = Array3::zeros((frames.nrows(), em.fc_mod_list.len(), frames.ncols()));
    for ch in 0..frames.nrows() {
        let bank = gcfb_v23_env_mod_fb(frames.row(ch).as_slice().unwrap(), &em)?;
        out.slice_mut(s![ch, .., ..]).assign(&bank);
    }
    Ok((out, em))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn nh_parameters_match_expected_health() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 10,
            ..GcParam::default()
        };
        let (p, r) = set_param(p).unwrap();
        assert!(
            p.hloss
                .fb_compression_health
                .iter()
                .all(|v| (*v - 1.).abs() < 1e-12)
        );
        assert_relative_eq!(r.fr1[0], 100., epsilon = 1e-9);
    }
    #[test]
    fn frame_output_shape() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200., 2000.],
            ..GcParam::default()
        };
        let out = gcfb_v234(&[1., 0., 0., 0., 0., 0., 0., 0.], p).unwrap();
        assert_eq!(out.dcgc_out.nrows(), 4);
        assert_eq!(out.dcgc_out.ncols(), 1);
        assert!(out.dcgc_out.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn hl3_audiogram_is_split_into_active_and_passive_loss() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 10,
            hloss_type: "HL3".into(),
            hloss_compression_health: Some(0.5),
            ..GcParam::default()
        };
        let (p, _) = set_param(p).unwrap();
        for i in 0..p.hloss.hearing_level_db.len() {
            assert_relative_eq!(
                p.hloss.pin_loss_db_act[i] + p.hloss.pin_loss_db_pas[i],
                p.hloss.hearing_level_db[i],
                epsilon = 0.35
            );
        }
        assert!(
            p.hloss
                .fb_compression_health
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        );
    }

    #[test]
    fn sample_histories_are_channel_major_and_update_cadence_changes_filtering() {
        let signal: Vec<f64> = (0..96)
            .map(|sample| {
                let envelope = match sample {
                    0..=23 => 0.02,
                    24..=47 => 1.0,
                    48..=71 => 0.12,
                    _ => 0.55,
                };
                let transient = match sample {
                    8 => 1.0,
                    56 => -0.7,
                    _ => 0.0,
                };
                envelope * (2.0 * std::f64::consts::PI * 700.0 * sample as f64 / 8_000.0).sin()
                    + transient
            })
            .collect();
        let run = |num_update_asym_cmp| {
            gcfb_v234(
                &signal,
                GcParam {
                    fs: 8_000.0,
                    out_mid_crct: "No".into(),
                    num_ch: 4,
                    f_range: [200.0, 2_000.0],
                    num_update_asym_cmp,
                    dyn_hpaf: DynHpaf {
                        str_prc: "sample-base".into(),
                        ..DynHpaf::default()
                    },
                    ..GcParam::default()
                },
            )
            .unwrap()
        };
        let every_sample = run(1);
        let every_eighth_sample = run(8);

        for output in [&every_sample, &every_eighth_sample] {
            assert_eq!(output.dcgc_out.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.fr2.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.frat_val.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.lvl_db.dim(), (4, signal.len()));
            assert!(output.dcgc_out.iter().all(|value| value.is_finite()));
            for history in [
                &output.gc_resp.fr2,
                &output.gc_resp.frat_val,
                &output.gc_resp.lvl_db,
            ] {
                assert!(history.rows().into_iter().any(|row| {
                    row.windows(2)
                        .into_iter()
                        .any(|pair| (pair[1] - pair[0]).abs() > 1e-12)
                }));
            }
            for ch in 0..4 {
                for sample in 0..signal.len() {
                    assert_relative_eq!(
                        output.gc_resp.fr2[[ch, sample]],
                        output.gc_resp.fp1[ch] * output.gc_resp.frat_val[[ch, sample]],
                        epsilon = 1e-12
                    );
                }
            }
        }

        assert_eq!(
            every_sample.gc_resp.lvl_db,
            every_eighth_sample.gc_resp.lvl_db
        );
        assert_eq!(
            every_sample.gc_resp.frat_val,
            every_eighth_sample.gc_resp.frat_val
        );
        assert_eq!(every_sample.gc_resp.fr2, every_eighth_sample.gc_resp.fr2);
        assert!(
            every_sample
                .dcgc_out
                .iter()
                .zip(&every_eighth_sample.dcgc_out)
                .any(|(left, right)| (left - right).abs() > 1e-12)
        );
    }

    #[test]
    fn static_processing_populates_response_metadata() {
        let p = GcParam {
            fs: 8000.0,
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200.0, 1500.0],
            ctrl: ControlMode::Static,
            ..GcParam::default()
        };
        let out = gcfb_v234(&[1.0, 0.0, 0.0, 0.0], p).unwrap();

        assert_eq!(out.gc_resp.fr2.dim(), (4, 1));
        assert_eq!(out.gc_resp.frat_val.dim(), (4, 1));
        assert_eq!(out.gc_resp.lvl_db.dim(), (4, 1));
        assert!(out.gc_resp.fp2.iter().all(|value| *value > 0.0));
        for ch in 0..4 {
            assert_relative_eq!(
                out.gc_resp.fr2[[ch, 0]],
                out.gc_resp.frat_val[[ch, 0]] * out.gc_resp.fp1[ch],
                epsilon = 1e-10
            );
            assert_relative_eq!(out.gc_resp.lvl_db[[ch, 0]], 50.0);
        }
    }

    #[test]
    fn invalid_dynamic_processing_and_window_names_are_rejected() {
        let invalid_processing = GcParam {
            dyn_hpaf: DynHpaf {
                str_prc: "fram-base".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        };
        assert!(set_param(invalid_processing).is_err());

        let invalid_window = GcParam {
            dyn_hpaf: DynHpaf {
                name_win: "blackman".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        };
        assert!(set_param(invalid_window).is_err());
    }

    #[test]
    fn v234_channel_range_is_strictly_sub_nyquist() {
        let nyquist = 4_000.0_f64;
        let below_nyquist = f64::from_bits(nyquist.to_bits() - 1);
        let above_nyquist = f64::from_bits(nyquist.to_bits() + 1);

        let (_, response) = set_param(GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [100.0, below_nyquist],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        })
        .unwrap();
        assert_eq!(response.fr1[3], below_nyquist);

        for high in [nyquist, above_nyquist] {
            let error = set_param(GcParam {
                fs: 8_000.0,
                num_ch: 4,
                f_range: [100.0, high],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            })
            .unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message.contains("0 < low < high < fs / 2")
            ));
        }
    }

    #[test]
    fn v234_processes_a_range_above_the_former_fs_over_three_limit() {
        let output = gcfb_v234(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            GcParam {
                fs: 8_000.0,
                num_ch: 4,
                f_range: [2_800.0, 3_200.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
        )
        .unwrap();

        assert_eq!(output.scgc_smpl.dim(), (4, 8));
        assert!(output.scgc_smpl.iter().all(|value| value.is_finite()));
        assert!(output.dcgc_out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn v234_rejects_invalid_mode_specific_initial_centers_during_preparation() {
        let static_param = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [1_000.0, 2_000.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            frat: [[10.0, 0.0], [0.0109, 0.0]],
            ..GcParam::default()
        };
        let dynamic_param = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [1_000.0, 2_000.0],
            out_mid_crct: "No".into(),
            lvl_est: LvlEst {
                frat: 10.0,
                ..LvlEst::default()
            },
            ..GcParam::default()
        };

        for param in [static_param, dynamic_param] {
            let error = set_param(param).unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message == "initial asymmetric filter centers must be finite, positive, and below Nyquist"
            ));
        }
    }

    #[test]
    fn v234_dynamic_centers_remain_runtime_validated() {
        let (param, mut response) = set_param(GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 1_500.0],
            out_mid_crct: "No".into(),
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        })
        .unwrap();
        response.frat0_pc.fill(100.0);
        let pgc = Array2::ones((4, 1));
        let scgc = Array2::ones((4, 1));

        assert!(gcfb_v23_sample_base(&pgc, &scgc, &param, &mut response).is_err());
    }

    #[test]
    fn v234_validates_derived_filter_centers_without_a_fixed_frequency_floor() {
        for param in [
            GcParam {
                num_ch: 4,
                f_range: [20.0, 1600.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
            GcParam {
                c1: [f64::NAN, 0.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
            GcParam {
                c1: [1000.0, 0.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
        ] {
            let error = set_param(param).unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message == "derived filter centers must be finite, positive, and below Nyquist"
            ));
        }

        let valid = GcParam {
            num_ch: 4,
            f_range: [39.0, 1600.0],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        };
        assert!(set_param(valid).is_ok());
    }

    #[test]
    fn modulation_filterbank_rejects_degenerate_or_aliased_frequencies() {
        for frequency in [0.0, 1000.0, 1500.0, f64::NAN] {
            let em = EmParam {
                fs: 2000.0,
                fc_mod_list: Array1::from(vec![frequency]),
                ..EmParam::default()
            };
            assert!(gcfb_v23_env_mod_fb(&[1.0, 0.0, 0.0], &em).is_err());
        }
    }

    #[test]
    fn hearing_pattern_rejects_ambiguous_names_and_non_finite_levels() {
        assert!(hearing_pattern("HL10", None).is_err());
        assert!(hearing_pattern("NHL3", None).is_err());
        assert!(hearing_pattern("HL3_ISO7029_70yr_male", None).is_ok());

        let manual = Array1::from(vec![0., 0., 0., f64::NAN, 0., 0., 0.]);
        assert!(hearing_pattern("HL0", Some(&manual)).is_err());
    }

    #[test]
    fn dynamic_processing_rejects_empty_or_mismatched_matrices() {
        let p = GcParam {
            fs: 8_000.,
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200., 2_000.],
            ..GcParam::default()
        };
        let (param, mut response) = set_param(p).unwrap();

        assert!(
            gcfb_v23_frame_base(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                &param,
                &mut response,
            )
            .is_err()
        );
        assert!(
            gcfb_v23_frame_base(
                &Array2::zeros((4, 8)),
                &Array2::zeros((3, 8)),
                &param,
                &mut response,
            )
            .is_err()
        );
        assert!(
            gcfb_v23_sample_base(
                &Array2::zeros((4, 8)),
                &Array2::zeros((3, 8)),
                &param,
                &mut response,
            )
            .is_err()
        );
    }

    #[test]
    fn corrected_synthesis_preserves_short_signals() {
        let input = Array2::from_shape_vec((1, 8), vec![1., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        for correction in ["ELC", "EarDrum"] {
            let param = GcParam {
                fs: 8_000.,
                out_mid_crct: correction.into(),
                ..GcParam::default()
            };
            let output = gcfb_v23_synth_snd(&input, &param).unwrap();
            assert_eq!(output.len(), input.ncols());
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| value.abs() > 1e-12));
        }
    }
}
