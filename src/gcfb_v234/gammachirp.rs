//! Passive gammachirp impulse and frequency responses used by GCFB v2.34.

use std::f64::consts::PI;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::utils::{freq2erb, nextpow2};
use crate::{Error, Result, dsp};

const MINIMUM_REALIZED_PEAK_FFT_LEN: usize = 65_536;
const MAXIMUM_PEAK_REFINEMENT_ITERATIONS: usize = 128;

/// Carrier used to construct a gammachirp impulse response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Carrier {
    #[default]
    Cosine,
    Sine,
    Envelope,
}

/// FIR normalization mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Normalization {
    /// Return the generated coefficients without gain normalization.
    #[default]
    None,
    /// Normalize the continuous-frequency peak of the returned finite FIR to
    /// unity.
    Peak,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormalizationPolicy {
    None,
    RealizedPeak,
    ReferencePeak,
}

#[derive(Clone, Debug)]
pub struct Gammachirp {
    pub gc: Array2<f64>,
    pub len_gc: Array1<usize>,
    /// The theoretical continuous-time peak frequency of each gammachirp.
    ///
    /// FIR truncation, sampling, and a real carrier can move the realized
    /// digital peak away from this frequency.
    pub fps: Array1<f64>,
    pub inst_freq: Array2<f64>,
}

#[derive(Clone, Debug)]
pub struct FrequencyResponse {
    /// Analytic continuous-time magnitude, normalized at the theoretical peak.
    pub amp_frsp: Array2<f64>,
    pub freq: Array1<f64>,
    pub f_peak: Array1<f64>,
    pub grp_dly: Array2<f64>,
    /// Unwrapped analytic phase for the same carrier convention as
    /// [`gammachirp`], including `arg Γ(n + jc)` and the 1 kHz phase reference.
    pub phs_frsp: Array2<f64>,
}

// Lanczos coefficients for g = 7.  GCFB only evaluates Γ(n + jc) with n > 0,
// so the reflection branch (and its attendant phase-cut bookkeeping) is not
// needed here.
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

fn complex_log_gamma_positive_real(z: Complex64) -> Complex64 {
    debug_assert!(z.re > 0.0);
    let shifted = z - 1.0;
    let mut series = Complex64::new(LANCZOS_COEFFICIENTS[0], 0.0);
    for (index, &coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;
    Complex64::new(0.5 * (2.0 * PI).ln(), 0.0) + (shifted + 0.5) * t.ln() - t + series.ln()
}

/// Generate passive gammachirp impulse responses for `frs`.
///
/// [`Normalization::Peak`] gives the returned, finite sampled FIR unit gain at
/// its actual continuous-DTFT maximum. [`Gammachirp::fps`] remains the
/// theoretical continuous-time peak frequency and is not changed to report the
/// realized FIR maximum.
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
    let policy = match normalization {
        Normalization::None => NormalizationPolicy::None,
        Normalization::Peak => NormalizationPolicy::RealizedPeak,
    };
    gammachirp_with_normalization(frs, sr, order_g, coef_erbw, coef_c, phase, carrier, policy)
}

/// Generate the Python-compatible peak-normalized FIR used to calibrate the
/// GCFB pipeline. Its gain is measured at the FFT bin nearest theoretical
/// [`Gammachirp::fps`], matching the reference implementation.
pub(crate) fn gammachirp_reference_peak(
    frs: &[f64],
    sr: f64,
    order_g: f64,
    coef_erbw: f64,
    coef_c: f64,
    phase: f64,
    carrier: Carrier,
) -> Result<Gammachirp> {
    gammachirp_with_normalization(
        frs,
        sr,
        order_g,
        coef_erbw,
        coef_c,
        phase,
        carrier,
        NormalizationPolicy::ReferencePeak,
    )
}

#[allow(clippy::too_many_arguments)]
fn gammachirp_with_normalization(
    frs: &[f64],
    sr: f64,
    order_g: f64,
    coef_erbw: f64,
    coef_c: f64,
    phase: f64,
    carrier: Carrier,
    normalization: NormalizationPolicy,
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
            if normalization == NormalizationPolicy::RealizedPeak {
                return Err(Error::Numerical(
                    "realized gammachirp FIR peak is zero or non-finite".into(),
                ));
            }
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
        let gain = match normalization {
            NormalizationPolicy::None => None,
            NormalizationPolicy::ReferencePeak => {
                let points = 1usize << nextpow2(len);
                let peak_bin =
                    ((fps[channel] / sr * 2.0 * points as f64).round() as usize).min(points - 1);
                let frequency = peak_bin as f64 / points as f64 * sr / 2.0;
                let response = (0..len).fold(Complex64::new(0.0, 0.0), |sum, n| {
                    sum + Complex64::from_polar(
                        gc[[channel, n]],
                        -2.0 * PI * frequency * n as f64 / sr,
                    )
                });
                (response.norm() > 0.0).then(|| response.norm())
            }
            NormalizationPolicy::RealizedPeak => {
                let row = gc.row(channel);
                let impulse = row.as_slice().expect("gammachirp rows are contiguous");
                Some(match carrier {
                    Carrier::Envelope => envelope_dc_gain(&impulse[..len])?,
                    Carrier::Cosine | Carrier::Sine => realized_peak_gain(&impulse[..len], sr)?,
                })
            }
        };
        if let Some(gain) = gain {
            for sample in 0..max_len {
                gc[[channel, sample]] /= gain;
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

fn envelope_dc_gain(impulse: &[f64]) -> Result<f64> {
    let gain = impulse.iter().sum::<f64>();
    if !gain.is_finite() || gain <= 0.0 {
        return Err(Error::Numerical(
            "realized gammachirp FIR peak is zero or non-finite".into(),
        ));
    }
    Ok(gain)
}

fn realized_peak_gain(impulse: &[f64], sample_rate: f64) -> Result<f64> {
    let fft_len = impulse
        .len()
        .max(MINIMUM_REALIZED_PEAK_FFT_LEN)
        .checked_next_power_of_two()
        .ok_or_else(|| Error::Unsupported("realized-peak FFT length overflow".into()))?;
    let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    for (value, &coefficient) in spectrum.iter_mut().zip(impulse) {
        value.re = coefficient;
    }
    dsp::fft(&mut spectrum, false);

    let mut peak_bin = 0;
    let mut peak_power = 0.0;
    for (bin, response) in spectrum[..=fft_len / 2].iter().enumerate() {
        let power = response.norm_sqr();
        if !power.is_finite() {
            return Err(Error::Numerical(
                "realized gammachirp FIR peak is zero or non-finite".into(),
            ));
        }
        if power > peak_power {
            peak_bin = bin;
            peak_power = power;
        }
    }
    if peak_power <= 0.0 {
        return Err(Error::Numerical(
            "realized gammachirp FIR peak is zero or non-finite".into(),
        ));
    }

    let peak_log_power = if peak_bin == 0 || peak_bin == fft_len / 2 {
        fir_log_power_and_derivative(
            impulse,
            peak_bin as f64 * sample_rate / fft_len as f64,
            sample_rate,
        )?
        .0
    } else {
        refine_realized_peak(impulse, peak_bin, fft_len, sample_rate)?
    };
    let gain = (0.5 * peak_log_power).exp();
    if !gain.is_finite() || gain <= 0.0 {
        return Err(Error::Numerical(
            "realized gammachirp FIR peak is zero or non-finite".into(),
        ));
    }
    Ok(gain)
}

fn refine_realized_peak(
    impulse: &[f64],
    peak_bin: usize,
    fft_len: usize,
    sample_rate: f64,
) -> Result<f64> {
    let spacing = sample_rate / fft_len as f64;
    let selected_frequency = peak_bin as f64 * spacing;
    let selected = fir_log_power_and_derivative(impulse, selected_frequency, sample_rate)?;
    if selected.1 == 0.0 {
        return Ok(selected.0);
    }

    let direction = if selected.1 > 0.0 { 1.0 } else { -1.0 };
    let mut previous = (selected_frequency, selected.1);
    let mut bracket = None;
    for step in 1..=MAXIMUM_PEAK_REFINEMENT_ITERATIONS * 8 {
        let frequency = selected_frequency + direction * step as f64 * spacing / 8.0;
        if frequency <= 0.0 || frequency >= sample_rate / 2.0 {
            break;
        }
        let value = fir_log_power_and_derivative(impulse, frequency, sample_rate)?;
        if value.1 == 0.0 {
            return Ok(value.0);
        }
        if direction > 0.0 && previous.1 > 0.0 && value.1 < 0.0 {
            bracket = Some((previous, (frequency, value.1)));
            break;
        }
        if direction < 0.0 && value.1 > 0.0 && previous.1 < 0.0 {
            bracket = Some(((frequency, value.1), previous));
            break;
        }
        previous = (frequency, value.1);
    }
    let (mut lower, mut upper) = bracket.ok_or_else(|| {
        Error::Numerical(
            "could not bracket the continuous-DTFT maximum of the realized gammachirp FIR".into(),
        )
    })?;

    let tolerance = 32.0 * f64::EPSILON * sample_rate.max(1.0);
    for _ in 0..MAXIMUM_PEAK_REFINEMENT_ITERATIONS {
        let midpoint = lower.0 + (upper.0 - lower.0) * 0.5;
        if midpoint == lower.0 || midpoint == upper.0 || upper.0 - lower.0 <= tolerance {
            return Ok(fir_log_power_and_derivative(impulse, midpoint, sample_rate)?.0);
        }
        let middle = fir_log_power_and_derivative(impulse, midpoint, sample_rate)?;
        if middle.1 == 0.0 {
            return Ok(middle.0);
        }
        if middle.1 > 0.0 {
            lower = (midpoint, middle.1);
        } else {
            upper = (midpoint, middle.1);
        }
    }
    Err(Error::Numerical(
        "continuous-DTFT realized-peak refinement did not converge".into(),
    ))
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let corrected = value - *compensation;
    let next = *sum + corrected;
    *compensation = (next - *sum) - corrected;
    *sum = next;
}

fn fir_log_power_and_derivative(
    impulse: &[f64],
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<(f64, f64)> {
    let radians_per_hz = 2.0 * PI / sample_rate;
    let mut response_re = 0.0;
    let mut response_im = 0.0;
    let mut derivative_re = 0.0;
    let mut derivative_im = 0.0;
    let mut response_re_compensation = 0.0;
    let mut response_im_compensation = 0.0;
    let mut derivative_re_compensation = 0.0;
    let mut derivative_im_compensation = 0.0;
    for (sample, &coefficient) in impulse.iter().enumerate() {
        let phase = radians_per_hz * frequency_hz * sample as f64;
        let (sin, cos) = phase.sin_cos();
        kahan_add(
            &mut response_re,
            &mut response_re_compensation,
            coefficient * cos,
        );
        kahan_add(
            &mut response_im,
            &mut response_im_compensation,
            -coefficient * sin,
        );
        let scale = coefficient * radians_per_hz * sample as f64;
        kahan_add(
            &mut derivative_re,
            &mut derivative_re_compensation,
            -scale * sin,
        );
        kahan_add(
            &mut derivative_im,
            &mut derivative_im_compensation,
            -scale * cos,
        );
    }
    let power = response_re.mul_add(response_re, response_im * response_im);
    if !power.is_finite() || power <= 0.0 {
        return Err(Error::Numerical(
            "realized gammachirp FIR peak is zero or non-finite".into(),
        ));
    }
    let derivative = 2.0 * response_re.mul_add(derivative_re, response_im * derivative_im) / power;
    if !derivative.is_finite() {
        return Err(Error::Numerical(
            "realized gammachirp FIR peak derivative is non-finite".into(),
        ));
    }
    Ok((power.ln(), derivative))
}

/// Analytic continuous-time frequency response of passive gammachirp filters.
///
/// Magnitude is normalized at the theoretical peak. Phase includes all
/// frequency-independent constants used by [`gammachirp`], so it can be
/// compared modulo 2π with the generated cosine filter's positive-frequency
/// response. The finite sampled FIR can still differ slightly in magnitude and
/// phase because it is truncated and discretized.
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
        let phase_constant = complex_log_gamma_positive_real(Complex64::new(order_g, c[ch])).im
            + c[ch] * (frs[ch] / 1000.0).ln()
            + phase;
        peaks[ch] = frs[ch] + b[ch] * erbw[ch] * c[ch] / order_g;
        for (bin, &frequency) in freq.iter().enumerate() {
            let fd = frequency - frs[ch];
            amp[[ch, bin]] = ((1.0 + cn * cn) / (1.0 + (fd / bh).powi(2))).powf(order_g / 2.0)
                * (c[ch] * ((fd / bh).atan() - cn.atan())).exp();
            delay[[ch, bin]] = (order_g * bh + c[ch] * fd) / (bh * bh + fd * fd) / (2.0 * PI);
            response_phase[[ch, bin]] = -order_g * (fd / bh).atan()
                - c[ch] / 2.0 * ((2.0 * PI * bh).powi(2) + (2.0 * PI * fd).powi(2)).ln()
                + phase_constant;
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

    fn assert_unit_realized_peak(generated: &Gammachirp, channel: usize, carrier: Carrier) {
        let impulse = generated.gc.row(channel);
        let impulse = &impulse.as_slice().unwrap()[..generated.len_gc[channel]];
        let gain = match carrier {
            Carrier::Envelope => impulse.iter().sum(),
            Carrier::Cosine | Carrier::Sine => realized_peak_gain(impulse, 48_000.0).unwrap(),
        };
        assert_relative_eq!(gain, 1.0, epsilon = 3e-13);
    }

    fn wrapped_phase_difference(left: f64, right: f64) -> f64 {
        (left - right + PI).rem_euclid(2.0 * PI) - PI
    }

    #[test]
    fn complex_log_gamma_matches_independent_quadrature_references() {
        // Frozen from direct numerical integration of
        // Γ(z) = integral_0^infinity t^(z-1) exp(-t) dt.  The integral fixes
        // the phase modulo 2π; the Lanczos logarithm retains its continuous
        // branch when that phase crosses the principal cut.
        let cases = [
            (
                Complex64::new(4.0, -2.0),
                1.250_835_619_356_806_6,
                -2.610_195_801_048_895_3,
            ),
            (
                Complex64::new(4.0, -2.96),
                0.662_943_947_704_853_2,
                2.273_624_343_701_705_6,
            ),
            (
                Complex64::new(3.5, 1.25),
                0.949_607_693_146_144_1,
                1.412_559_376_552_109_6,
            ),
        ];
        for (z, expected_real, expected_imag) in cases {
            let actual = complex_log_gamma_positive_real(z);
            assert_relative_eq!(actual.re, expected_real, epsilon = 2e-13);
            assert!(wrapped_phase_difference(actual.im, expected_imag).abs() < 2e-13);
        }
    }

    #[test]
    fn analytic_phase_matches_generated_cosine_gammachirp_near_its_main_lobe() {
        let fs = 48_000.0;
        let fr = 1_000.0;
        let b = 1.019;
        let c = -2.0;
        let generated = gammachirp(
            &[fr],
            fs,
            4.0,
            b,
            c,
            0.37,
            Carrier::Cosine,
            Normalization::None,
        )
        .unwrap();
        let analytic = gammachirp_frsp(&[fr], fs, 4.0, &[b], &[c], 0.37, 4096).unwrap();
        let peak_bin = analytic
            .amp_frsp
            .row(0)
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        let frequency = analytic.freq[peak_bin];
        let discrete = generated
            .gc
            .row(0)
            .iter()
            .take(generated.len_gc[0])
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |sum, (sample, &value)| {
                sum + Complex64::from_polar(value, -2.0 * PI * frequency * sample as f64 / fs)
            });
        assert!(
            wrapped_phase_difference(discrete.arg(), analytic.phs_frsp[[0, peak_bin]]).abs() < 2e-3
        );
    }

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
    fn public_peak_normalizes_realized_cosine_sine_and_envelope_peaks() {
        for carrier in [Carrier::Cosine, Carrier::Sine, Carrier::Envelope] {
            let generated = gammachirp(
                &[100.0, 1_000.0],
                48_000.0,
                4.0,
                1.019,
                -2.0,
                0.37,
                carrier,
                Normalization::Peak,
            )
            .unwrap();
            assert_ne!(generated.len_gc[0], generated.len_gc[1]);
            for channel in 0..2 {
                assert_unit_realized_peak(&generated, channel, carrier);
                assert!(
                    generated
                        .gc
                        .row(channel)
                        .iter()
                        .skip(generated.len_gc[channel])
                        .all(|&value| value == 0.0)
                );
            }
        }
    }

    #[test]
    fn envelope_peak_normalization_has_unit_dc_gain() {
        let generated = gammachirp(
            &[100.0, 1_000.0, 8_000.0],
            48_000.0,
            4.0,
            1.019,
            -2.0,
            0.0,
            Carrier::Envelope,
            Normalization::Peak,
        )
        .unwrap();
        for channel in 0..3 {
            assert_relative_eq!(generated.gc.row(channel).sum(), 1.0, epsilon = 3e-15);
        }
    }

    #[test]
    fn realized_peak_search_handles_dc_and_nyquist_boundaries() {
        assert_relative_eq!(
            realized_peak_gain(&[1.0, 1.0], 48_000.0).unwrap(),
            2.0,
            epsilon = 2e-15
        );
        assert_relative_eq!(
            realized_peak_gain(&[1.0, -1.0], 48_000.0).unwrap(),
            2.0,
            epsilon = 2e-15
        );
        assert!(realized_peak_gain(&[0.0, 0.0], 48_000.0).is_err());
        assert!(realized_peak_gain(&[f64::NAN], 48_000.0).is_err());
    }

    #[test]
    fn reference_peak_reproduces_the_nearest_theoretical_bin_gain() {
        let arguments = (8_000.0, 4.0, 1.019, -2.0, 0.31, Carrier::Cosine);
        let raw = gammachirp(
            &[1_000.0],
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            arguments.5,
            Normalization::None,
        )
        .unwrap();
        let reference = gammachirp_reference_peak(
            &[1_000.0],
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            arguments.5,
        )
        .unwrap();
        let len = raw.len_gc[0];
        let points = 1usize << nextpow2(len);
        let peak_bin =
            ((raw.fps[0] / arguments.0 * 2.0 * points as f64).round() as usize).min(points - 1);
        let frequency = peak_bin as f64 / points as f64 * arguments.0 / 2.0;
        let gain = raw
            .gc
            .row(0)
            .iter()
            .take(len)
            .enumerate()
            .fold(Complex64::new(0.0, 0.0), |sum, (sample, &value)| {
                sum + Complex64::from_polar(
                    value,
                    -2.0 * PI * frequency * sample as f64 / arguments.0,
                )
            })
            .norm();
        for sample in 0..raw.gc.ncols() {
            assert_relative_eq!(
                reference.gc[[0, sample]],
                raw.gc[[0, sample]] / gain,
                epsilon = 2e-15,
                max_relative = 2e-15
            );
        }
        assert_eq!(reference.len_gc, raw.len_gc);
        assert_eq!(reference.fps, raw.fps);
        assert_eq!(reference.inst_freq, raw.inst_freq);
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
