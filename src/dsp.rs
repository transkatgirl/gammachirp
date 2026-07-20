use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::{Error, Result};

/// Fixed-memory state for a causal FIR filter.
#[derive(Clone, Debug)]
pub(crate) struct CausalFir {
    coefficients: Vec<f64>,
    history: Vec<f64>,
    next: usize,
}

impl CausalFir {
    pub(crate) fn new(coefficients: Vec<f64>) -> Self {
        debug_assert!(!coefficients.is_empty());
        Self {
            history: vec![0.0; coefficients.len()],
            coefficients,
            next: 0,
        }
    }

    pub(crate) fn process_sample(&mut self, sample: f64) -> f64 {
        self.history[self.next] = sample;
        let mut history_index = self.next;
        let mut output = 0.0;
        for &coefficient in &self.coefficients {
            output += coefficient * self.history[history_index];
            history_index = if history_index == 0 {
                self.history.len() - 1
            } else {
                history_index - 1
            };
        }
        self.next = (self.next + 1) % self.history.len();
        output
    }
}

/// A bank of causal FIR filters driven by the same scalar input.
#[derive(Clone, Debug)]
pub(crate) struct CausalFirBank {
    coefficients: Vec<Vec<f64>>,
    history: Vec<f64>,
    next: usize,
}

impl CausalFirBank {
    pub(crate) fn new(mut coefficients: Vec<Vec<f64>>) -> Self {
        debug_assert!(!coefficients.is_empty());
        // An empty FIR is the zero operator in the batch convolution path.
        // Retain that meaning while giving the circular buffer one addressable
        // slot for sample-by-sample processing.
        for row in &mut coefficients {
            if row.is_empty() {
                row.push(0.0);
            }
        }
        let history_len = coefficients.iter().map(Vec::len).max().unwrap();
        Self {
            coefficients,
            history: vec![0.0; history_len],
            next: 0,
        }
    }

    pub(crate) fn process_sample(&mut self, sample: f64) -> Array1<f64> {
        self.history[self.next] = sample;
        let mut output = Array1::zeros(self.coefficients.len());
        for (channel, coefficients) in self.coefficients.iter().enumerate() {
            let mut history_index = self.next;
            for &coefficient in coefficients {
                output[channel] += coefficient * self.history[history_index];
                history_index = if history_index == 0 {
                    self.history.len() - 1
                } else {
                    history_index - 1
                };
            }
        }
        self.next = (self.next + 1) % self.history.len();
        output
    }
}

pub(crate) fn fft(values: &mut [Complex64], inverse: bool) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = if inverse { 2.0 } else { -2.0 } * std::f64::consts::PI / len as f64;
        let w_len = Complex64::from_polar(1.0, angle);
        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = values[start + k];
                let v = values[start + k + len / 2] * w;
                values[start + k] = u + v;
                values[start + k + len / 2] = u - v;
                w *= w_len;
            }
        }
        len <<= 1;
    }
    if inverse {
        for value in values {
            *value /= n as f64;
        }
    }
}

/// Transform arbitrary-length data, falling back to a direct DFT when the
/// radix-2 FFT cannot preserve the requested length.
pub(crate) fn transform(values: &mut [Complex64], inverse: bool) {
    if values.len().is_power_of_two() {
        fft(values, inverse);
        return;
    }
    let input = values.to_vec();
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    for (k, output) in values.iter_mut().enumerate() {
        *output = input
            .iter()
            .enumerate()
            .map(|(sample, value)| {
                *value
                    * Complex64::from_polar(
                        1.0,
                        sign * 2.0 * std::f64::consts::PI * k as f64 * sample as f64 / n as f64,
                    )
            })
            .sum();
        if inverse {
            *output /= n as f64;
        }
    }
}

pub(crate) fn fft_convolve_truncated(b: &[f64], x: &[f64]) -> Vec<f64> {
    if b.is_empty() || x.is_empty() {
        return vec![0.0; x.len()];
    }
    let n = (b.len() + x.len() - 1).next_power_of_two();
    let mut bf = vec![Complex64::new(0.0, 0.0); n];
    let mut xf = vec![Complex64::new(0.0, 0.0); n];
    for (dst, &src) in bf.iter_mut().zip(b) {
        dst.re = src;
    }
    for (dst, &src) in xf.iter_mut().zip(x) {
        dst.re = src;
    }
    fft(&mut bf, false);
    fft(&mut xf, false);
    for (a, b) in xf.iter_mut().zip(&bf) {
        *a *= *b;
    }
    fft(&mut xf, true);
    xf[..x.len()].iter().map(|v| v.re).collect()
}

pub(crate) fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    if a.is_empty() || a[0] == 0.0 {
        return Err(Error::InvalidParameter(
            "IIR denominator a[0] must be non-zero".into(),
        ));
    }
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() {
        let mut value = 0.0;
        for (k, &coef) in b.iter().enumerate().take(n + 1) {
            value += coef * x[n - k];
        }
        for (k, &coef) in a.iter().enumerate().skip(1).take(n) {
            value -= coef * y[n - k];
        }
        y[n] = value / a[0];
    }
    Ok(y)
}

pub(crate) fn interp1(x: &[f64], y: &[f64], x_new: &[f64], extrapolate: bool) -> Result<Vec<f64>> {
    if x.len() != y.len() || x.len() < 2 || x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::InvalidParameter(
            "interpolation inputs must have equal length, at least two points, and increasing x"
                .into(),
        ));
    }
    let mut out = Vec::with_capacity(x_new.len());
    for &query in x_new {
        if !extrapolate && (query < x[0] || query > x[x.len() - 1]) {
            out.push(f64::NAN);
            continue;
        }
        let upper = match x.binary_search_by(|v| v.total_cmp(&query)) {
            Ok(i) => {
                out.push(y[i]);
                continue;
            }
            Err(i) => i.clamp(1, x.len() - 1),
        };
        let lower = upper - 1;
        let fraction = (query - x[lower]) / (x[upper] - x[lower]);
        out.push(y[lower] + fraction * (y[upper] - y[lower]));
    }
    Ok(out)
}

pub(crate) fn hanning(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    // Equivalent to np.hanning(n + 2)[1..-1], as used by GammachirPy.
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64).cos())
        .collect()
}

pub(crate) fn hamming(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

pub(crate) fn frame_sequence(
    snd: &[f64],
    len_win: usize,
    len_shift: usize,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if len_win == 0
        || len_shift == 0
        || !len_win.is_multiple_of(2)
        || !len_win.is_multiple_of(len_shift)
    {
        return Err(Error::InvalidParameter(
            "frame length must be positive and even; shift must be a positive divisor of it".into(),
        ));
    }
    let half = len_win / 2;
    let frames = snd.len() / len_shift + 1;
    let mut output = Array2::zeros((len_win, frames));
    let positions = Array1::from_iter((0..frames).map(|i| i * len_shift));
    for (frame, &center) in positions.iter().enumerate() {
        for offset in 0..len_win {
            let source = center as isize + offset as isize - half as isize;
            if source >= 0 && (source as usize) < snd.len() {
                output[[offset, frame]] = snd[source as usize];
            }
        }
    }
    Ok((output, positions))
}

pub(crate) fn polynomial_real_roots(coefficients: &[f64]) -> Vec<f64> {
    let first = match coefficients.iter().position(|v| v.abs() > 1e-15) {
        Some(i) => i,
        None => return Vec::new(),
    };
    let c = &coefficients[first..];
    let degree = c.len() - 1;
    if degree == 0 {
        return Vec::new();
    }
    if degree == 1 {
        return vec![-c[1] / c[0]];
    }
    let lead = c[0];
    let normalized: Vec<f64> = c.iter().map(|v| v / lead).collect();
    let radius = 1.0 + normalized[1..].iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut roots: Vec<Complex64> = (0..degree)
        .map(|i| {
            Complex64::from_polar(
                radius,
                2.0 * std::f64::consts::PI * i as f64 / degree as f64 + 0.37,
            )
        })
        .collect();
    for _ in 0..200 {
        let old = roots.clone();
        let mut max_delta: f64 = 0.0;
        for i in 0..degree {
            let mut value = Complex64::new(normalized[0], 0.0);
            for &coef in &normalized[1..] {
                value = value * old[i] + coef;
            }
            let mut denominator = Complex64::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denominator *= old[i] - old[j];
                }
            }
            if denominator.norm() > 1e-20 {
                roots[i] = old[i] - value / denominator;
                max_delta = max_delta.max((roots[i] - old[i]).norm());
            }
        }
        if max_delta < 1e-12 {
            break;
        }
    }
    roots
        .into_iter()
        .filter(|r| r.im.abs() < 1e-7)
        .map(|r| r.re)
        .collect()
}

pub(crate) fn first_order_lowpass(cutoff: f64, fs: f64) -> ([f64; 2], [f64; 2]) {
    let k = (std::f64::consts::PI * cutoff / fs).tan();
    let norm = 1.0 / (1.0 + k);
    ([k * norm, k * norm], [1.0, (k - 1.0) * norm])
}

pub(crate) fn third_order_butterworth_lowpass(cutoff: f64, fs: f64) -> ([f64; 4], [f64; 4]) {
    let k = (std::f64::consts::PI * cutoff / fs).tan();
    let analog_poles = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0),
        Complex64::new(-0.5, -3.0_f64.sqrt() / 2.0),
    ];
    let poles = analog_poles
        .map(|pole| (Complex64::new(1.0, 0.0) + k * pole) / (Complex64::new(1.0, 0.0) - k * pole));
    let sum = poles[0] + poles[1] + poles[2];
    let pairs = poles[0] * poles[1] + poles[0] * poles[2] + poles[1] * poles[2];
    let product = poles[0] * poles[1] * poles[2];
    let a = [1.0, -sum.re, pairs.re, -product.re];
    let gain = a.iter().sum::<f64>() / 8.0;
    ([gain, 3.0 * gain, 3.0 * gain, gain], a)
}
