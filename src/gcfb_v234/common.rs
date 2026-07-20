//! Common gammachirp and asymmetric-filter primitives used by GCFB v2.34.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, Array3, s};
use num_complex::Complex64;

use super::{gammachirp, utils};
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
    /// Magnitude of the implemented, peak-normalized cosine pGC FIR.
    pub pgc_frsp: Array2<f64>,
    /// Magnitude of the implemented pGC FIR followed by the digital HP-AF.
    pub cgc_frsp: Array2<f64>,
    pub cgc_nrm_frsp: Array2<f64>,
    /// Magnitude of the implemented four-section digital HP-AF.
    pub acf_frsp: Array2<f64>,
    /// Ideal continuous asymmetric function, retained as theoretical metadata.
    pub asym_func: Array2<f64>,
    pub fp1: Array1<f64>,
    pub fr2: Array1<f64>,
    pub fp2: Array1<f64>,
    pub val_fp2: Array1<f64>,
    pub norm_fct_fp2: Array1<f64>,
    pub freq: Array1<f64>,
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

/// Sample the realized digital compressive-gammachirp cascade.
///
/// Unlike [`gammachirp::gammachirp_frsp`], this response measures the finite
/// cosine pGC FIR and the implemented four-section HP-AF. Peak metadata is the
/// maximum on the requested frequency grid. Power-of-two grids use an FFT;
/// other grid sizes retain exact DFT sampling and may be slower.
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
    if n_frq_rsl < 256 {
        return Err(Error::InvalidParameter(
            "compressive response requires at least 256 frequency bins".into(),
        ));
    }
    let fft_len = n_frq_rsl.checked_mul(2).ok_or_else(|| {
        Error::Unsupported("compressive-response transform length overflow".into())
    })?;
    let freq =
        Array1::from_iter((0..n_frq_rsl).map(|bin| bin as f64 / n_frq_rsl as f64 * fs / 2.0));
    let mut pgc_frsp = Array2::zeros((fr1.len(), n_frq_rsl));
    let mut fp1 = Array1::zeros(fr1.len());
    for ch in 0..fr1.len() {
        let impulse = gammachirp::gammachirp(
            &[fr1[ch]],
            fs,
            n,
            b1[ch],
            c1[ch],
            0.0,
            gammachirp::Carrier::Cosine,
            gammachirp::Normalization::Peak,
        )?;
        fp1[ch] = impulse.fps[0];
        let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
        for (sample, &value) in impulse.gc.row(0).iter().take(impulse.len_gc[0]).enumerate() {
            // Folding is exact at the DFT frequencies and supports unusually
            // low requested resolutions without truncating the FIR.
            spectrum[sample % fft_len].re += value;
        }
        dsp::transform(&mut spectrum, false);
        for bin in 0..n_frq_rsl {
            pgc_frsp[[ch, bin]] = spectrum[bin].norm();
        }
    }
    let fr2 = Array1::from_iter((0..fr1.len()).map(|i| frat[i] * fp1[i]));
    let acf = asym_cmp_frsp_v2(fr2.as_slice().unwrap(), fs, &b2, &c2, n_frq_rsl, 4)?;
    let cgc_frsp = &pgc_frsp * &acf.acf_frsp;
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
        peaks[ch] = freq[index];
        normalized.row_mut(ch).mapv_inplace(|v| v / value);
    }
    let norm = values.mapv(|v| 1.0 / v);
    Ok(CgcResponse {
        fr1: Array1::from(fr1.to_vec()),
        pgc_frsp,
        cgc_frsp,
        cgc_nrm_frsp: normalized,
        acf_frsp: acf.acf_frsp,
        asym_func: acf.asym_func,
        fp1,
        fr2,
        fp2: peaks,
        val_fp2: values,
        norm_fct_fp2: norm,
        freq,
    })
}

#[cfg(test)]
mod response_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn compressive_response_matches_the_implemented_fir_and_iir_cascade() {
        let fs = 48_000.0;
        let bins = 300; // Exercise the exact non-radix-2 DFT path.
        let response = cmprs_gc_frsp(
            &[1_000.0],
            fs,
            4.0,
            &[1.81],
            &[-2.96],
            &[1.08],
            &[2.17],
            &[2.2],
            bins,
        )
        .unwrap();
        let impulse = gammachirp::gammachirp(
            &[1_000.0],
            fs,
            4.0,
            1.81,
            -2.96,
            0.0,
            gammachirp::Carrier::Cosine,
            gammachirp::Normalization::Peak,
        )
        .unwrap();
        let coefficients =
            make_asym_cmp_filters_v2(fs, &[response.fr2[0]], &[2.17], &[2.2]).unwrap();
        for bin in [0, 25, 75, 150, 299] {
            let omega = -2.0 * PI * bin as f64 / (2 * bins) as f64;
            let z1 = Complex64::from_polar(1.0, omega);
            let z2 = z1 * z1;
            let passive = impulse
                .gc
                .row(0)
                .iter()
                .take(impulse.len_gc[0])
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |sum, (sample, &value)| {
                    sum + value * z1.powu(sample as u32)
                })
                .norm();
            let mut asymmetric = 1.0;
            for section in 0..4 {
                let numerator = coefficients.bz[[0, 0, section]]
                    + coefficients.bz[[0, 1, section]] * z1
                    + coefficients.bz[[0, 2, section]] * z2;
                let denominator = coefficients.ap[[0, 0, section]]
                    + coefficients.ap[[0, 1, section]] * z1
                    + coefficients.ap[[0, 2, section]] * z2;
                asymmetric *= (numerator / denominator).norm();
            }
            assert_relative_eq!(response.pgc_frsp[[0, bin]], passive, epsilon = 3e-11);
            assert_relative_eq!(response.acf_frsp[[0, bin]], asymmetric, epsilon = 1e-10);
            assert_relative_eq!(
                response.cgc_frsp[[0, bin]],
                passive * asymmetric,
                epsilon = 2e-10
            );
        }
        let (peak_bin, &peak) = response
            .cgc_frsp
            .row(0)
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap();
        assert_eq!(response.fp2[0], response.freq[peak_bin]);
        assert_eq!(response.val_fp2[0], peak);
        assert_relative_eq!(response.norm_fct_fp2[0] * peak, 1.0, epsilon = 1e-14);
        assert_relative_eq!(
            response
                .cgc_nrm_frsp
                .row(0)
                .iter()
                .copied()
                .fold(0.0, f64::max),
            1.0,
            epsilon = 1e-14
        );
    }
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

fn broadcast(values: &[f64], len: usize, name: &str) -> Result<Vec<f64>> {
    match values.len() {
        1 => Ok(vec![values[0]; len]),
        n if n == len => Ok(values.to_vec()),
        _ => Err(Error::InvalidParameter(format!(
            "{name} must contain one value or one per channel"
        ))),
    }
}
