//! Passive gammachirp impulse and frequency responses.

use std::f64::consts::PI;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::utils::{freq2erb, nextpow2};
use crate::{Error, Result};

/// Carrier used to construct a gammachirp impulse response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Carrier {
    #[default]
    Cosine,
    Sine,
    Envelope,
}

/// Peak-spectrum normalization mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Normalization {
    #[default]
    None,
    Peak,
}

#[derive(Clone, Debug)]
pub struct Gammachirp {
    pub gc: Array2<f64>,
    pub len_gc: Array1<usize>,
    pub fps: Array1<f64>,
    pub inst_freq: Array2<f64>,
}

#[derive(Clone, Debug)]
pub struct FrequencyResponse {
    pub amp_frsp: Array2<f64>,
    pub freq: Array1<f64>,
    pub f_peak: Array1<f64>,
    pub grp_dly: Array2<f64>,
    pub phs_frsp: Array2<f64>,
}

/// Generate passive gammachirp impulse responses for `frs`.
pub fn gammachirp(
    frs: &[f64],
    sr: f64,
    order_g: f64,
    coef_erbw: f64,
    coef_c: f64,
    phase: f64,
    carrier: Carrier,
    normalization: Normalization,
) -> Result<Gammachirp> {
    if frs.is_empty()
        || !sr.is_finite()
        || sr <= 0.0
        || !order_g.is_finite()
        || order_g <= 0.0
        || !coef_erbw.is_finite()
        || coef_erbw <= 0.0
        || !coef_c.is_finite()
        || !phase.is_finite()
        || frs
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0 || *f >= sr / 2.0)
    {
        return Err(Error::InvalidParameter(
            "gammachirp parameters must be finite, with positive frequencies below Nyquist".into(),
        ));
    }
    let (_, erbw) = freq2erb(frs);
    let (_, erbw_1khz) = freq2erb(&[1000.0]);
    let len_gc_1khz = (40.0 * order_g / coef_erbw + 200.0) * sr / 16000.0;
    let lengths = Array1::from_iter(
        erbw.iter()
            .map(|w| (len_gc_1khz * erbw_1khz[0] / w).trunc() as usize),
    );
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    let mut gc = Array2::zeros((frs.len(), max_len));
    let mut inst_freq = Array2::zeros((frs.len(), max_len));
    let (fps, _) = fr2fpeak(order_g, coef_erbw, coef_c, frs);

    for channel in 0..frs.len() {
        let len = lengths[channel];
        if len < 2 {
            continue;
        }
        let adjusted_phase = phase
            + if carrier == Carrier::Sine {
                -PI / 2.0
            } else {
                0.0
            }
            + coef_c * (frs[channel] / 1000.0).ln();
        let mut envelope = vec![0.0; len];
        for (sample, value) in envelope.iter_mut().enumerate().skip(1) {
            let t = sample as f64 / sr;
            *value = t.powf(order_g - 1.0) * (-2.0 * PI * coef_erbw * erbw[channel] * t).exp();
        }
        let maximum = envelope.iter().copied().fold(0.0, f64::max);
        for sample in 1..len {
            let t = sample as f64 / sr;
            let phase_t = 2.0 * PI * frs[channel] * t + coef_c * t.ln() + adjusted_phase;
            let carrier_value = match carrier {
                Carrier::Envelope => 1.0,
                Carrier::Cosine | Carrier::Sine => phase_t.cos(),
            };
            gc[[channel, sample]] = envelope[sample] / maximum * carrier_value;
            inst_freq[[channel, sample]] = frs[channel] + coef_c / (2.0 * PI * t);
        }
        if normalization == Normalization::Peak {
            let points = 1usize << nextpow2(len);
            let peak_bin =
                ((fps[channel] / sr * 2.0 * points as f64).round() as usize).min(points - 1);
            let frequency = peak_bin as f64 / points as f64 * sr / 2.0;
            let response = (0..len).fold(Complex64::new(0.0, 0.0), |sum, n| {
                sum + Complex64::from_polar(gc[[channel, n]], -2.0 * PI * frequency * n as f64 / sr)
            });
            let gain = response.norm();
            if gain > 0.0 {
                for sample in 0..max_len {
                    gc[[channel, sample]] /= gain;
                }
            }
        }
    }
    Ok(Gammachirp {
        gc,
        len_gc: lengths,
        fps,
        inst_freq,
    })
}

/// Analytic frequency response of passive gammachirp filters.
pub fn gammachirp_frsp(
    frs: &[f64],
    sr: f64,
    order_g: f64,
    coef_erbw: &[f64],
    coef_c: &[f64],
    phase: f64,
    n_frq_rsl: usize,
) -> Result<FrequencyResponse> {
    if frs.is_empty()
        || n_frq_rsl < 256
        || !sr.is_finite()
        || sr <= 0.0
        || !order_g.is_finite()
        || order_g <= 0.0
        || !phase.is_finite()
        || frs
            .iter()
            .any(|f| !f.is_finite() || *f <= 0.0 || *f >= sr / 2.0)
    {
        return Err(Error::InvalidParameter(
            "frequency response requires finite positive frequencies below Nyquist, a positive order, and at least 256 bins".into(),
        ));
    }
    let b = broadcast(coef_erbw, frs.len(), "coef_erbw")?;
    let c = broadcast(coef_c, frs.len(), "coef_c")?;
    if b.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || c.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter(
            "gammachirp bandwidths must be finite and positive; chirp coefficients must be finite"
                .into(),
        ));
    }
    let (_, erbw) = freq2erb(frs);
    let freq = Array1::from_iter((0..n_frq_rsl).map(|i| i as f64 / n_frq_rsl as f64 * sr / 2.0));
    let mut amp = Array2::zeros((frs.len(), n_frq_rsl));
    let mut delay = Array2::zeros((frs.len(), n_frq_rsl));
    let mut response_phase = Array2::zeros((frs.len(), n_frq_rsl));
    let mut peaks = Array1::zeros(frs.len());
    for ch in 0..frs.len() {
        let bh = b[ch] * erbw[ch];
        let cn = c[ch] / order_g;
        peaks[ch] = frs[ch] + b[ch] * erbw[ch] * c[ch] / order_g;
        for (bin, &frequency) in freq.iter().enumerate() {
            let fd = frequency - frs[ch];
            amp[[ch, bin]] = ((1.0 + cn * cn) / (1.0 + (fd / bh).powi(2))).powf(order_g / 2.0)
                * (c[ch] * ((fd / bh).atan() - cn.atan())).exp();
            delay[[ch, bin]] = (order_g * bh + c[ch] * fd) / (bh * bh + fd * fd) / (2.0 * PI);
            response_phase[[ch, bin]] = -order_g * (fd / bh).atan()
                - c[ch] / 2.0 * ((2.0 * PI * bh).powi(2) + (2.0 * PI * fd).powi(2)).ln()
                + phase;
        }
    }
    Ok(FrequencyResponse {
        amp_frsp: amp,
        freq,
        f_peak: peaks,
        grp_dly: delay,
        phs_frsp: response_phase,
    })
}

/// Convert asymptotic frequencies to peak frequencies.
pub fn fr2fpeak(n: f64, b: f64, c: f64, fr: &[f64]) -> (Array1<f64>, Array1<f64>) {
    let (_, erb_width) = freq2erb(fr);
    let peak = Array1::from_iter(fr.iter().zip(&erb_width).map(|(&f, &w)| f + c * w * b / n));
    (peak, erb_width)
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
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn peak_frequency_matches_reference_formula() {
        let (peak, width) = fr2fpeak(4.0, 1.81, -2.96, &[1000.0]);
        assert_relative_eq!(width[0], 132.639, epsilon = 1e-9);
        assert_relative_eq!(peak[0], 822.3433234, epsilon = 1e-7);
    }

    #[test]
    fn response_is_normalized_at_analytic_peak() {
        let response =
            gammachirp_frsp(&[1000.0], 48000.0, 4.0, &[1.019], &[-2.0], 0.0, 1024).unwrap();
        let max = response.amp_frsp.row(0).iter().copied().fold(0.0, f64::max);
        // The analytic response is unity at f_peak; the sampled grid misses
        // that frequency slightly, matching the Python implementation.
        assert_relative_eq!(max, 0.9976977824713469, epsilon = 1e-12);
    }

    #[test]
    fn digital_gammachirp_rejects_frequencies_at_or_above_nyquist() {
        assert!(
            gammachirp(
                &[4000.0],
                8000.0,
                4.0,
                1.019,
                -2.0,
                0.0,
                Carrier::Cosine,
                Normalization::None,
            )
            .is_err()
        );
        assert!(gammachirp_frsp(&[5000.0], 8000.0, 4.0, &[1.019], &[-2.0], 0.0, 256).is_err());
    }
}
