//! Dynamic compressive gammachirp filterbank (GCFB v2.11).

use std::f64::consts::PI;

use ndarray::{Array1, Array2, Array3, s};
use num_complex::Complex64;

use super::{
    gammachirp::{self, Carrier, Normalization},
    utils::{self, FrequencyScale},
};
use crate::{Error, Result, dsp};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ControlMode {
    #[default]
    Static,
    Dynamic,
    Level,
}

#[derive(Clone, Debug)]
pub struct LvlEst {
    pub lct_erb: f64,
    pub decay_hl: f64,
    pub b2: f64,
    pub c2: f64,
    pub frat: f64,
    pub rms2spldb: f64,
    pub weight: f64,
    pub ref_db: f64,
    pub pwr: [f64; 2],
    pub exp_decay_val: f64,
    pub erb_space1: f64,
    pub n_ch_shift: isize,
    pub n_ch_lvl_est: Array1<usize>,
    pub lvl_lin_min_lim: f64,
    pub lvl_lin_ref: f64,
}

impl Default for LvlEst {
    fn default() -> Self {
        Self {
            lct_erb: 1.5,
            decay_hl: 0.5,
            b2: 2.17,
            c2: 2.2,
            frat: 1.08,
            rms2spldb: 30.0,
            weight: 0.5,
            ref_db: 50.0,
            pwr: [1.5, 0.5],
            exp_decay_val: 0.0,
            erb_space1: 0.0,
            n_ch_shift: 0,
            n_ch_lvl_est: Array1::zeros(0),
            lvl_lin_min_lim: 0.0,
            lvl_lin_ref: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GcParam {
    pub fs: f64,
    pub num_ch: usize,
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
    pub gain_ref_db: f64,
    pub level_db_scgcfb: f64,
    pub lvl_est: LvlEst,
    pub num_update_asym_cmp: usize,
}

impl Default for GcParam {
    fn default() -> Self {
        Self {
            fs: 48_000.0,
            num_ch: 100,
            f_range: [100.0, 6000.0],
            out_mid_crct: "ELC".into(),
            n: 4.0,
            b1: [1.81, 0.0],
            c1: [-2.96, 0.0],
            frat: [[0.4660, 0.0], [0.0109, 0.0]],
            b2: [[2.17, 0.0], [0.0, 0.0]],
            c2: [[2.20, 0.0], [0.0, 0.0]],
            ctrl: ControlMode::Static,
            gain_cmpnst_db: -1.0,
            gain_ref_db: 50.0,
            level_db_scgcfb: 50.0,
            lvl_est: LvlEst::default(),
            num_update_asym_cmp: 1,
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
    pub lvl_db: Array2<f64>,
    pub gain_factor: Array1<f64>,
    pub cgc_ref: Option<CgcResponse>,
}

#[derive(Clone, Debug)]
pub struct GcfbOutput {
    pub cgc_out: Array2<f64>,
    pub pgc_out: Array2<f64>,
    pub gc_param: GcParam,
    pub gc_resp: GcResp,
}

#[derive(Clone, Debug)]
pub struct AcfCoef {
    pub fs: f64,
    /// AR coefficients, indexed `[channel, coefficient, section]`.
    pub ap: Array3<f64>,
    /// MA coefficients, indexed `[channel, coefficient, section]`.
    pub bz: Array3<f64>,
}

#[derive(Clone, Debug)]
pub struct AcfStatus {
    input_history: Array2<f64>,
    output_history: Array3<f64>,
    pub count: usize,
}

impl AcfStatus {
    pub fn new(coefficients: &AcfCoef) -> Self {
        let (channels, taps, sections) = coefficients.bz.dim();
        Self {
            input_history: Array2::zeros((channels, taps)),
            output_history: Array3::zeros((channels, taps, sections)),
            count: 0,
        }
    }

    pub fn process(
        &mut self,
        coefficients: &AcfCoef,
        input: &[f64],
        reverse: bool,
    ) -> Result<Array1<f64>> {
        let (channels, taps, sections) = coefficients.bz.dim();
        if input.len() != channels
            || coefficients.ap.dim() != (channels, taps, sections)
            || taps != 3
            || self.input_history.dim() != (channels, taps)
            || self.output_history.dim() != (channels, taps, sections)
        {
            return Err(Error::InvalidParameter(
                "AC filter input, coefficient, and state dimensions do not match".into(),
            ));
        }
        self.count += 1;
        for (ch, &sample) in input.iter().enumerate().take(channels) {
            self.input_history[[ch, 0]] = self.input_history[[ch, 1]];
            self.input_history[[ch, 1]] = self.input_history[[ch, 2]];
            self.input_history[[ch, 2]] = sample;
        }
        let section_order: Vec<usize> = if reverse {
            (0..sections).rev().collect()
        } else {
            (0..sections).collect()
        };
        let mut current = self.input_history.clone();
        let mut latest = Array1::zeros(channels);
        for section in section_order {
            for ch in 0..channels {
                let forward = (0..3)
                    .map(|k| coefficients.bz[[ch, 2 - k, section]] * current[[ch, k]])
                    .sum::<f64>();
                let feedback = coefficients.ap[[ch, 2, section]]
                    * self.output_history[[ch, 1, section]]
                    + coefficients.ap[[ch, 1, section]] * self.output_history[[ch, 2, section]];
                latest[ch] = (forward - feedback) / coefficients.ap[[ch, 0, section]];
                self.output_history[[ch, 0, section]] = self.output_history[[ch, 1, section]];
                self.output_history[[ch, 1, section]] = self.output_history[[ch, 2, section]];
                self.output_history[[ch, 2, section]] = latest[ch];
            }
            current = self.output_history.slice(s![.., .., section]).to_owned();
        }
        Ok(latest)
    }
}

#[derive(Clone, Debug)]
pub struct AsymCmpResponse {
    pub acf_frsp: Array2<f64>,
    pub freq: Array1<f64>,
    pub asym_func: Array2<f64>,
}

#[derive(Clone, Debug)]
pub struct CgcResponse {
    pub fr1: Array1<f64>,
    pub pgc_frsp: Array2<f64>,
    pub cgc_frsp: Array2<f64>,
    pub cgc_nrm_frsp: Array2<f64>,
    pub acf_frsp: Array2<f64>,
    pub asym_func: Array2<f64>,
    pub fp1: Array1<f64>,
    pub fr2: Array1<f64>,
    pub fp2: Array1<f64>,
    pub val_fp2: Array1<f64>,
    pub norm_fct_fp2: Array1<f64>,
    pub freq: Array1<f64>,
}

/// Fill derived v2.11 parameters and channel responses.
pub fn set_param(mut param: GcParam) -> Result<(GcParam, GcResp)> {
    if !param.fs.is_finite()
        || param.fs <= 0.0
        || param.num_ch < 2
        || param.num_update_asym_cmp == 0
        || !param.n.is_finite()
        || param.n <= 0.0
        || param.f_range.iter().any(|value| !value.is_finite())
        || param.f_range[1] >= param.fs / 2.0
    {
        return Err(Error::InvalidParameter(
            "v2.11 requires finite parameters, positive order and sample rate, and a frequency range below Nyquist".into(),
        ));
    }
    let (fr1, erb_rate1) =
        utils::equal_freq_scale(FrequencyScale::Erb, param.num_ch, param.f_range)?;
    let erb_space1 = erb_rate1
        .windows(2)
        .into_iter()
        .map(|w| w[1] - w[0])
        .sum::<f64>()
        / (param.num_ch - 1) as f64;
    let (erb_rate, _) = utils::freq2erb(fr1.as_slice().unwrap());
    let (erb_1k, _) = utils::freq2erb(&[1000.0]);
    let ef = erb_rate.mapv(|v| v / erb_1k[0] - 1.0);
    let b1_val = ef.mapv(|v| param.b1[0] + param.b1[1] * v);
    let c1_val = ef.mapv(|v| param.c1[0] + param.c1[1] * v);
    let (_, erb_width) = utils::freq2erb(fr1.as_slice().unwrap());
    let fp1 = Array1::from_iter(
        (0..param.num_ch).map(|i| fr1[i] + c1_val[i] * erb_width[i] * b1_val[i] / param.n),
    );
    let b2_val = ef.mapv(|v| param.b2[0][0] + param.b2[0][1] * v);
    let c2_val = ef.mapv(|v| param.c2[0][0] + param.c2[0][1] * v);
    let shift = (param.lvl_est.lct_erb / erb_space1).round() as isize;
    param.lvl_est.exp_decay_val =
        (-1.0 / (param.lvl_est.decay_hl * param.fs / 1000.0) * 2_f64.ln()).exp();
    param.lvl_est.erb_space1 = erb_space1;
    param.lvl_est.n_ch_shift = shift;
    param.lvl_est.n_ch_lvl_est = Array1::from_iter(
        (0..param.num_ch)
            .map(|ch| (ch as isize + shift).clamp(0, param.num_ch as isize - 1) as usize),
    );
    param.lvl_est.lvl_lin_min_lim = 10_f64.powf(-param.lvl_est.rms2spldb / 20.0);
    param.lvl_est.lvl_lin_ref =
        10_f64.powf((param.lvl_est.ref_db - param.lvl_est.rms2spldb) / 20.0);
    let response = GcResp {
        fr1,
        fr2: Array2::zeros((param.num_ch, 0)),
        erb_space1,
        ef,
        b1_val,
        c1_val,
        fp1,
        fp2: Array1::zeros(param.num_ch),
        b2_val,
        c2_val,
        frat_val: Array2::zeros((param.num_ch, 0)),
        lvl_db: Array2::zeros((param.num_ch, 0)),
        gain_factor: Array1::ones(param.num_ch),
        cgc_ref: None,
    };
    Ok((param, response))
}

/// Compute coefficients for the four-section asymmetric compensation filterbank.
pub fn make_asym_cmp_filters_v2(fs: f64, frs: &[f64], b: &[f64], c: &[f64]) -> Result<AcfCoef> {
    if !fs.is_finite()
        || fs <= 0.0
        || frs.is_empty()
        || frs
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0 || *frequency >= fs / 2.0)
    {
        return Err(Error::InvalidParameter(
            "filterbank requires finite positive frequencies below Nyquist".into(),
        ));
    }
    let b = broadcast(b, frs.len(), "b")?;
    let c = broadcast(c, frs.len(), "c")?;
    if b.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || c.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter(
            "asymmetric-filter bandwidths must be finite and positive; chirp coefficients must be finite"
                .into(),
        ));
    }
    let (_, erbw) = utils::freq2erb(frs);
    let mut ap = Array3::zeros((frs.len(), 3, 4));
    let mut bz = Array3::zeros((frs.len(), 3, 4));
    for ch in 0..frs.len() {
        let p0: f64 = 2.0;
        let p1 = 1.7818 * (1.0 - 0.0791 * b[ch]) * (1.0 - 0.1655 * c[ch].abs());
        let p2 = 0.5689 * (1.0 - 0.1620 * b[ch]) * (1.0 - 0.0857 * c[ch].abs());
        let p4: f64 = 1.0724;
        for section in 0..4 {
            let r = (-p1 * (p0 / p4).powi(section as i32) * 2.0 * PI * b[ch] * erbw[ch] / fs).exp();
            let delta = (p0 * p4).powi(section as i32) * p2 * c[ch] * b[ch] * erbw[ch];
            let phi = 2.0 * PI * (frs[ch] + delta).max(0.0) / fs;
            let psi = 2.0 * PI * (frs[ch] - delta).max(0.0) / fs;
            let a = [1.0, -2.0 * r * phi.cos(), r * r];
            let mut z = [1.0, -2.0 * r * psi.cos(), r * r];
            if !r.is_finite() || r >= 1.0 || a.iter().chain(&z).any(|value| !value.is_finite()) {
                return Err(Error::InvalidParameter(
                    "asymmetric-filter parameters must produce finite, stable sections".into(),
                ));
            }
            let v = Complex64::from_polar(1.0, 2.0 * PI * frs[ch] / fs);
            let powers = [Complex64::new(1.0, 0.0), v, v * v];
            let numerator = (0..3).map(|i| powers[i] * a[i]).sum::<Complex64>();
            let denominator = (0..3).map(|i| powers[i] * z[i]).sum::<Complex64>();
            let normalization = (numerator / denominator).norm();
            if !normalization.is_finite() {
                return Err(Error::InvalidParameter(
                    "asymmetric-filter normalization must be finite".into(),
                ));
            }
            for i in 0..3 {
                ap[[ch, i, section]] = a[i];
                z[i] *= normalization;
                bz[[ch, i, section]] = z[i];
            }
        }
    }
    Ok(AcfCoef { fs, ap, bz })
}

pub fn asym_cmp_frsp_v2(
    frs: &[f64],
    fs: f64,
    b: &[f64],
    c: &[f64],
    n_frq_rsl: usize,
    num_filt: usize,
) -> Result<AsymCmpResponse> {
    if !fs.is_finite()
        || fs <= 0.0
        || frs.is_empty()
        || frs
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0 || *frequency >= fs / 2.0)
        || n_frq_rsl < 64
        || !(1..=4).contains(&num_filt)
    {
        return Err(Error::InvalidParameter(
            "asymmetric response requires frequencies below Nyquist, >=64 bins, and 1..=4 sections"
                .into(),
        ));
    }
    let b = broadcast(b, frs.len(), "b")?;
    let c = broadcast(c, frs.len(), "c")?;
    if b.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || c.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter(
            "asymmetric-response bandwidths must be finite and positive; chirp coefficients must be finite"
                .into(),
        ));
    }
    let (_, erbw) = utils::freq2erb(frs);
    let freq = Array1::from_iter((0..n_frq_rsl).map(|i| i as f64 / n_frq_rsl as f64 * fs / 2.0));
    let mut acf = Array2::ones((frs.len(), n_frq_rsl));
    for ch in 0..frs.len() {
        let p0: f64 = 2.0;
        let p1 = 1.7818 * (1.0 - 0.0791 * b[ch]) * (1.0 - 0.1655 * c[ch].abs());
        let p2 = 0.5689 * (1.0 - 0.1620 * b[ch]) * (1.0 - 0.0857 * c[ch].abs());
        let p4: f64 = 1.0724;
        for section in 0..num_filt {
            let r = (-p1 * (p0 / p4).powi(section as i32) * 2.0 * PI * b[ch] * erbw[ch] / fs).exp();
            let delta = (p0 * p4).powi(section as i32) * p2 * c[ch] * b[ch] * erbw[ch];
            let phi = 2.0 * PI * (frs[ch] + delta).max(0.0) / fs;
            let psi = 2.0 * PI * (frs[ch] - delta).max(0.0) / fs;
            let a = [1.0, -2.0 * r * phi.cos(), r * r];
            let z = [1.0, -2.0 * r * psi.cos(), r * r];
            if !r.is_finite() || r >= 1.0 || a.iter().chain(&z).any(|value| !value.is_finite()) {
                return Err(Error::InvalidParameter(
                    "asymmetric-response parameters must describe finite, stable sections".into(),
                ));
            }
            let magnitude = |f: f64| {
                let cs1 = (2.0 * PI * f / fs).cos();
                let cs2 = (4.0 * PI * f / fs).cos();
                let mag2 = |q: [f64; 3]| {
                    q[0] * q[0]
                        + q[1] * q[1]
                        + q[2] * q[2]
                        + 2.0 * q[1] * (q[0] + q[2]) * cs1
                        + 2.0 * q[0] * q[2] * cs2
                };
                (mag2(z) / mag2(a)).sqrt()
            };
            let norm = magnitude(frs[ch]);
            if !norm.is_finite() || norm <= 0.0 {
                return Err(Error::InvalidParameter(
                    "asymmetric-response normalization must be finite and positive".into(),
                ));
            }
            for bin in 0..n_frq_rsl {
                let value = magnitude(freq[bin]) / norm;
                if !value.is_finite() {
                    return Err(Error::InvalidParameter(
                        "asymmetric response must remain finite".into(),
                    ));
                }
                acf[[ch, bin]] *= value;
            }
        }
    }
    let mut asym = Array2::zeros((frs.len(), n_frq_rsl));
    for ch in 0..frs.len() {
        for bin in 0..n_frq_rsl {
            let value = (c[ch] * (freq[bin] - frs[ch]).atan2(b[ch] * erbw[ch])).exp();
            if !value.is_finite() {
                return Err(Error::InvalidParameter(
                    "asymmetric function must remain finite".into(),
                ));
            }
            asym[[ch, bin]] = value;
        }
    }
    Ok(AsymCmpResponse {
        acf_frsp: acf,
        freq,
        asym_func: asym,
    })
}

pub fn cmprs_gc_frsp(
    fr1: &[f64],
    fs: f64,
    n: f64,
    b1: &[f64],
    c1: &[f64],
    frat: &[f64],
    b2: &[f64],
    c2: &[f64],
    n_frq_rsl: usize,
) -> Result<CgcResponse> {
    let b1 = broadcast(b1, fr1.len(), "b1")?;
    let c1 = broadcast(c1, fr1.len(), "c1")?;
    let frat = broadcast(frat, fr1.len(), "frat")?;
    let b2 = broadcast(b2, fr1.len(), "b2")?;
    let c2 = broadcast(c2, fr1.len(), "c2")?;
    let pgc = gammachirp::gammachirp_frsp(fr1, fs, n, &b1, &c1, 0.0, n_frq_rsl)?;
    let (_, widths) = utils::freq2erb(fr1);
    let fp1 = Array1::from_iter((0..fr1.len()).map(|i| fr1[i] + c1[i] * widths[i] * b1[i] / n));
    let fr2 = Array1::from_iter((0..fr1.len()).map(|i| frat[i] * fp1[i]));
    let acf = asym_cmp_frsp_v2(fr2.as_slice().unwrap(), fs, &b2, &c2, n_frq_rsl, 4)?;
    let cgc_frsp = &pgc.amp_frsp * &acf.asym_func;
    let mut values = Array1::zeros(fr1.len());
    let mut peaks = Array1::zeros(fr1.len());
    let mut normalized = cgc_frsp.clone();
    for ch in 0..fr1.len() {
        let row = cgc_frsp.row(ch);
        let (index, value) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        values[ch] = *value;
        peaks[ch] = pgc.freq[index];
        normalized.row_mut(ch).mapv_inplace(|v| v / value);
    }
    let norm = values.mapv(|v| 1.0 / v);
    Ok(CgcResponse {
        fr1: Array1::from(fr1.to_vec()),
        pgc_frsp: pgc.amp_frsp,
        cgc_frsp,
        cgc_nrm_frsp: normalized,
        acf_frsp: acf.acf_frsp,
        asym_func: acf.asym_func,
        fp1,
        fr2,
        fp2: peaks,
        val_fp2: values,
        norm_fct_fp2: norm,
        freq: pgc.freq,
    })
}

pub fn fr1_to_fp2(
    n: f64,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    frat: f64,
    fr1: f64,
) -> Result<(f64, f64)> {
    let (_, w1) = utils::freq2erb(&[fr1]);
    let fp1 = fr1 + c1 * w1[0] * b1 / n;
    let fr2 = frat * fp1;
    let (_, w2) = utils::freq2erb(&[fr2]);
    let bw1 = b1 * w1[0];
    let bw2 = b2 * w2[0];
    let coefficients = [
        -n,
        c1 * bw1 + c2 * bw2 + n * fr1 + 2.0 * n * fr2,
        -2.0 * fr2 * (c1 * bw1 + n * fr1) - n * (bw2 * bw2 + fr2 * fr2) - 2.0 * c2 * bw2 * fr1,
        c2 * bw2 * (bw1 * bw1 + fr1 * fr1) + (c1 * bw1 + n * fr1) * (bw2 * bw2 + fr2 * fr2),
    ];
    let roots = dsp::polynomial_real_roots(&coefficients);
    let peak = roots
        .into_iter()
        .min_by(|a, b| (a - fp1).abs().total_cmp(&(b - fp1).abs()))
        .ok_or_else(|| Error::InvalidParameter("could not find real compressive-GC peak".into()))?;
    Ok((peak, fr2))
}

pub fn fp2_to_fr1(
    n: f64,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    frat: f64,
    fp2: f64,
) -> Result<(f64, f64)> {
    let (_, a0v) = utils::freq2erb(&[0.0]);
    let (_, w1v) = utils::freq2erb(&[1.0]);
    let a0 = a0v[0];
    let a1 = w1v[0] - a0;
    let beta1 = frat * (1.0 + c1 * b1 * a1 / n);
    let beta0 = frat * c1 * b1 * a0 / n;
    let z1 = a1 * beta1;
    let z0 = a1 * beta0 + a0;
    let k1 = (b2 * b2 * z1 * z1 + beta1 * beta1) * (c1 * b1 * a1 + n)
        + (c2 * b2 * z1) * (b1 * b1 * a1 * a1 + 1.0);
    let k2 = (b2 * b2 * z1 * z1 + beta1 * beta1) * (c1 * b1 * a0 - n * fp2)
        + (2.0 * b2 * b2 * z1 * z0 - 2.0 * beta1 * (fp2 - beta0)) * (c1 * b1 * a1 + n)
        + (c2 * b2 * z1) * (2.0 * b1 * b1 * a1 * a0 - 2.0 * fp2)
        + (c2 * b2 * z0) * (b1 * b1 * a1 * a1 + 1.0);
    let k3 = (2.0 * b2 * b2 * z1 * z0 - 2.0 * beta1 * (fp2 - beta0)) * (c1 * b1 * a0 - n * fp2)
        + (b2 * b2 * z0 * z0 + (fp2 - beta0).powi(2)) * (c1 * b1 * a1 + n)
        + (c2 * b2 * z1) * (b1 * b1 * a0 * a0 + fp2 * fp2)
        + (c2 * b2 * z0) * (2.0 * b1 * b1 * a1 * a0 - 2.0 * fp2);
    let k4 = (b2 * b2 * z0 * z0 + (fp2 - beta0).powi(2)) * (c1 * b1 * a0 - n * fp2)
        + (c2 * b2 * z0) * (b1 * b1 * a0 * a0 + fp2 * fp2);
    let roots = dsp::polynomial_real_roots(&[k1, k2, k3, k4]);
    let fr1 = roots
        .into_iter()
        .min_by(|a, b| {
            let pa = *a + c1 * b1 * (a1 * *a + a0) / n;
            let pb = *b + c1 * b1 * (a1 * *b + a0) / n;
            (pa - fp2).abs().total_cmp(&(pb - fp2).abs())
        })
        .ok_or_else(|| {
            Error::InvalidParameter("could not find real passive-GC frequency".into())
        })?;
    Ok((fr1, fr1 + c1 * b1 * (a1 * fr1 + a0) / n))
}

/// Run the v2.11 filterbank.
pub fn gcfb_v211(snd_in: &[f64], gc_param: GcParam) -> Result<GcfbOutput> {
    if snd_in.is_empty() {
        return Err(Error::InvalidParameter(
            "input sound cannot be empty".into(),
        ));
    }
    let (param, mut response) = set_param(gc_param)?;
    let snd = if param.out_mid_crct.eq_ignore_ascii_case("no") {
        snd_in.to_vec()
    } else {
        // A frequency-table FIR approximation is exposed by the utility API.
        let coefficients = utils::out_mid_crct_filt(&param.out_mid_crct, param.fs, 2)?;
        dsp::lfilter(coefficients.as_slice().unwrap(), &[1.0], snd_in)?
    };
    let channels = param.num_ch;
    let samples = snd.len();
    let mut pgc_out = Array2::zeros((channels, samples));
    let mut level_out = Array2::zeros((channels, samples));
    let static_frat = Array1::from_iter((0..channels).map(|ch| {
        param.frat[0][0]
            + param.frat[0][1] * response.ef[ch]
            + (param.frat[1][0] + param.frat[1][1] * response.ef[ch]) * param.level_db_scgcfb
    }));
    let signal_coefficients = if param.ctrl == ControlMode::Static {
        let centers = Array1::from_iter((0..channels).map(|ch| static_frat[ch] * response.fp1[ch]));
        response.fr2 = Array2::from_shape_vec((channels, 1), centers.to_vec()).unwrap();
        for ch in 0..channels {
            response.fp2[ch] = fr1_to_fp2(
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
                response.b2_val[ch],
                response.c2_val[ch],
                static_frat[ch],
                response.fr1[ch],
            )?
            .0;
        }
        make_asym_cmp_filters_v2(
            param.fs,
            centers.as_slice().unwrap(),
            response.b2_val.as_slice().unwrap(),
            response.c2_val.as_slice().unwrap(),
        )?
    } else {
        let centers = response.fp1.mapv(|v| param.lvl_est.frat * v);
        make_asym_cmp_filters_v2(
            param.fs,
            centers.as_slice().unwrap(),
            &[param.lvl_est.b2],
            &[param.lvl_est.c2],
        )?
    };
    let mut cgc_out = Array2::zeros((channels, samples));
    for ch in 0..channels {
        let impulse = gammachirp::gammachirp(
            &[response.fr1[ch]],
            param.fs,
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            0.0,
            Carrier::Cosine,
            Normalization::Peak,
        )?;
        let filtered = utils::fftfilt(impulse.gc.row(0).as_slice().unwrap(), &snd);
        pgc_out.row_mut(ch).assign(&filtered);
        let mut section_out = filtered.to_vec();
        for section in 0..4 {
            let b = signal_coefficients.bz.slice(s![ch, .., section]).to_vec();
            let a = signal_coefficients.ap.slice(s![ch, .., section]).to_vec();
            section_out = dsp::lfilter(&b, &a, &section_out)?;
        }
        if param.ctrl == ControlMode::Static {
            cgc_out.row_mut(ch).assign(&Array1::from(section_out));
        } else {
            level_out.row_mut(ch).assign(&Array1::from(section_out));
        }
    }
    if param.ctrl == ControlMode::Level {
        cgc_out.assign(&level_out);
    }
    if param.ctrl == ControlMode::Dynamic {
        response.fr2 = Array2::zeros((channels, samples));
        response.frat_val = Array2::zeros((channels, samples));
        response.lvl_db = Array2::zeros((channels, samples));
        let mut previous = Array2::<f64>::zeros((channels, 2));
        let initial_centers = response.fp1.mapv(|v| param.lvl_est.frat * v);
        let mut coefficients = make_asym_cmp_filters_v2(
            param.fs,
            initial_centers.as_slice().unwrap(),
            response.b2_val.as_slice().unwrap(),
            response.c2_val.as_slice().unwrap(),
        )?;
        let mut status = AcfStatus::new(&coefficients);
        for sample in 0..samples {
            let mut levels = Array1::zeros(channels);
            for ch in 0..channels {
                let source_ch = param.lvl_est.n_ch_lvl_est[ch];
                let p = pgc_out[[source_ch, sample]]
                    .max(0.0)
                    .max(previous[[ch, 0]] * param.lvl_est.exp_decay_val);
                let q = level_out[[source_ch, sample]]
                    .max(0.0)
                    .max(previous[[ch, 1]] * param.lvl_est.exp_decay_val);
                previous[[ch, 0]] = p;
                previous[[ch, 1]] = q;
                let total = param.lvl_est.weight
                    * param.lvl_est.lvl_lin_ref
                    * (p / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[0])
                    + (1.0 - param.lvl_est.weight)
                        * param.lvl_est.lvl_lin_ref
                        * (q / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[1]);
                levels[ch] = 20.0 * total.max(param.lvl_est.lvl_lin_min_lim).log10()
                    + param.lvl_est.rms2spldb;
                response.lvl_db[[ch, sample]] = levels[ch];
                let ratio = param.frat[0][0]
                    + param.frat[0][1] * response.ef[ch]
                    + (param.frat[1][0] + param.frat[1][1] * response.ef[ch]) * levels[ch];
                response.frat_val[[ch, sample]] = ratio;
                response.fr2[[ch, sample]] = response.fp1[ch] * ratio;
            }
            if sample % param.num_update_asym_cmp == 0 {
                let centers = response.fr2.column(sample).to_vec();
                coefficients = make_asym_cmp_filters_v2(
                    param.fs,
                    &centers,
                    response.b2_val.as_slice().unwrap(),
                    response.c2_val.as_slice().unwrap(),
                )?;
            }
            let input = pgc_out.column(sample).to_vec();
            let output = status.process(&coefficients, &input, false)?;
            cgc_out.column_mut(sample).assign(&output);
        }
        let reference_ratio = Array1::from_iter((0..channels).map(|ch| {
            param.frat[0][0]
                + param.frat[0][1] * response.ef[ch]
                + (param.frat[1][0] + param.frat[1][1] * response.ef[ch]) * param.gain_ref_db
        }));
        let reference = cmprs_gc_frsp(
            response.fr1.as_slice().unwrap(),
            param.fs,
            param.n,
            response.b1_val.as_slice().unwrap(),
            response.c1_val.as_slice().unwrap(),
            reference_ratio.as_slice().unwrap(),
            response.b2_val.as_slice().unwrap(),
            response.c2_val.as_slice().unwrap(),
            1024,
        )?;
        response.gain_factor = reference
            .norm_fct_fp2
            .mapv(|v| 10_f64.powf(param.gain_cmpnst_db / 20.0) * v);
        for ch in 0..channels {
            cgc_out
                .row_mut(ch)
                .mapv_inplace(|v| v * response.gain_factor[ch]);
        }
        response.cgc_ref = Some(reference);
    }
    Ok(GcfbOutput {
        cgc_out,
        pgc_out,
        gc_param: param,
        gc_resp: response,
    })
}

pub fn acfilterbank(
    coefficients: &AcfCoef,
    status: &mut AcfStatus,
    input: &[f64],
    reverse: bool,
) -> Result<Array1<f64>> {
    status.process(coefficients, input, reverse)
}

#[derive(Clone, Debug)]
pub struct SmoothSpecParam {
    pub fs: f64,
    pub method: u8,
    pub t_shift: f64,
    pub t_win: f64,
    pub temporal_positions: Array1<f64>,
}

impl SmoothSpecParam {
    pub fn new(fs: f64) -> Self {
        Self {
            fs,
            method: 1,
            t_shift: 0.0,
            t_win: 0.0,
            temporal_positions: Array1::zeros(0),
        }
    }
}

pub fn cal_smooth_spec(
    fb_out: &Array2<f64>,
    mut param: SmoothSpecParam,
) -> Result<(Array2<f64>, SmoothSpecParam)> {
    let (window_secs, window) = match param.method {
        1 => {
            let n = (0.025 * param.fs) as usize;
            (0.025, dsp::hamming(n))
        }
        2 => {
            let n = (0.010 * param.fs) as usize;
            (0.010, dsp::hanning(n))
        }
        _ => {
            return Err(Error::InvalidParameter(
                "smoothing method must be 1 or 2".into(),
            ));
        }
    };
    param.t_shift = 0.005;
    param.t_win = window_secs;
    let shift = (param.t_shift * param.fs) as usize;
    let sum: f64 = window.iter().sum();
    let window: Vec<f64> = window.iter().map(|v| v / sum).collect();
    let mut result: Option<Array2<f64>> = None;
    for ch in 0..fb_out.nrows() {
        let (frames, centers) =
            dsp::frame_sequence(fb_out.row(ch).as_slice().unwrap(), window.len(), shift)?;
        if result.is_none() {
            result = Some(Array2::zeros((fb_out.nrows(), frames.ncols())));
            param.temporal_positions = centers.mapv(|v| v as f64 / param.fs);
        }
        for frame in 0..frames.ncols() {
            result.as_mut().unwrap()[[ch, frame]] = window
                .iter()
                .enumerate()
                .map(|(i, w)| w * frames[[i, frame]])
                .sum();
        }
    }
    Ok((result.unwrap_or_else(|| Array2::zeros((0, 0))), param))
}

pub fn set_frame4time_sequence(
    snd: &[f64],
    len_frame: usize,
    shift_frame: Option<usize>,
) -> Result<(Array2<f64>, Array1<isize>)> {
    utils::set_frame4time_sequence(snd, len_frame, shift_frame)
}

fn broadcast(values: &[f64], len: usize, name: &str) -> Result<Vec<f64>> {
    match values.len() {
        1 => Ok(vec![values[0]; len]),
        n if n == len => Ok(values.to_vec()),
        _ => Err(Error::InvalidParameter(format!(
            "{name} must contain one value or one per channel"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn default_channel_grid_matches_python_reference() {
        let param = GcParam {
            out_mid_crct: "No".into(),
            ..GcParam::default()
        };
        let (_, response) = set_param(param).unwrap();
        assert_relative_eq!(response.fr1[0], 100.0, epsilon = 1e-10);
        assert_relative_eq!(response.fr1[99], 6000.0, epsilon = 1e-9);
        assert_relative_eq!(response.fp1[0], 52.45947034, epsilon = 1e-8);
    }

    #[test]
    fn acf_coefficients_match_python_reference() {
        let c = make_asym_cmp_filters_v2(48000.0, &[1000.0], &[2.17], &[2.2]).unwrap();
        assert_relative_eq!(c.ap[[0, 1, 0]], -1.9071557202043983, epsilon = 1e-12);
        assert_relative_eq!(c.bz[[0, 1, 0]], -2.311126394035969, epsilon = 1e-12);
    }

    #[test]
    fn small_static_filterbank_has_expected_shape_and_finite_output() {
        let param = GcParam {
            num_ch: 4,
            f_range: [200.0, 2000.0],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        };
        let output = gcfb_v211(&[1.0, 0.0, 0.0, 0.0, 0.0], param).unwrap();
        assert_eq!(output.cgc_out.dim(), (4, 5));
        assert!(output.cgc_out.iter().all(|v| v.is_finite()));
        assert_eq!(output.gc_resp.fr2.dim(), (4, 1));
        assert!(output.gc_resp.fp2.iter().all(|value| *value > 0.0));
        for ch in 0..4 {
            let expected = fr1_to_fp2(
                output.gc_param.n,
                output.gc_resp.b1_val[ch],
                output.gc_resp.c1_val[ch],
                output.gc_resp.b2_val[ch],
                output.gc_resp.c2_val[ch],
                output.gc_resp.fr2[[ch, 0]] / output.gc_resp.fp1[ch],
                output.gc_resp.fr1[ch],
            )
            .unwrap()
            .0;
            assert_relative_eq!(output.gc_resp.fp2[ch], expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn small_dynamic_filterbank_updates_column_major_coefficients() {
        let param = GcParam {
            num_ch: 4,
            f_range: [200.0, 2000.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Dynamic,
            ..GcParam::default()
        };
        let mut signal = vec![0.0; 32];
        signal[0] = 1.0;
        let output = gcfb_v211(&signal, param).unwrap();
        assert_eq!(output.cgc_out.dim(), (4, 32));
        assert!(output.cgc_out.iter().all(|v| v.is_finite()));
        assert_eq!(output.gc_resp.fr2.dim(), (4, 32));
    }

    #[test]
    fn acf_state_rejects_a_coefficient_bank_with_different_dimensions() {
        let one_channel = make_asym_cmp_filters_v2(8000.0, &[500.0], &[2.17], &[2.2]).unwrap();
        let two_channels =
            make_asym_cmp_filters_v2(8000.0, &[500.0, 1000.0], &[2.17], &[2.2]).unwrap();
        let mut status = AcfStatus::new(&one_channel);

        assert!(status.process(&two_channels, &[1.0, 2.0], false).is_err());
        assert_eq!(status.count, 0);
    }

    #[test]
    fn asymmetric_filters_reject_non_positive_or_unstable_bandwidths() {
        for bandwidth in [0.0, -1.0, 20.0] {
            assert!(make_asym_cmp_filters_v2(8_000.0, &[500.0], &[bandwidth], &[2.2]).is_err());
            assert!(asym_cmp_frsp_v2(&[500.0], 8_000.0, &[bandwidth], &[2.2], 256, 4,).is_err());
        }

        let param = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 2_000.0],
            out_mid_crct: "No".into(),
            b2: [[0.0, 0.0], [0.0, 0.0]],
            ..GcParam::default()
        };
        assert!(gcfb_v211(&[1.0, 0.0, 0.0], param).is_err());
    }

    #[test]
    fn v211_rejects_a_channel_range_at_or_above_nyquist() {
        let param = GcParam {
            fs: 1000.0,
            num_ch: 2,
            f_range: [400.0, 600.0],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        };
        assert!(gcfb_v211(&[1.0, 0.0], param).is_err());
    }
}
