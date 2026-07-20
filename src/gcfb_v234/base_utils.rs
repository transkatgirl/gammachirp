//! Shared auditory scales, signal-level calibration, framing, and basic DSP utilities.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::{Error, Result, dsp};

pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

pub fn nextpow2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

/// Read a mono, uncompressed 16-bit PCM WAV file, normalized to `[-1, 1)`.
pub fn audioread(path: impl AsRef<Path>) -> Result<(Array1<f64>, u32)> {
    let mut file = File::open(path)?;
    let mut header = [0_u8; 12];
    file.read_exact(&mut header)
        .map_err(|_| Error::Wav("truncated RIFF header".into()))?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(Error::Wav("missing RIFF/WAVE signature".into()));
    }
    let riff_size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
    if riff_size < 4 {
        return Err(Error::Wav(
            "RIFF container is shorter than its WAVE header".into(),
        ));
    }
    let riff_end = 8 + riff_size;
    if file.metadata()?.len() < riff_end {
        return Err(Error::Wav(
            "file is shorter than the declared RIFF container".into(),
        ));
    }
    let mut sample_rate = None;
    let mut format = None;
    let mut channels = None;
    let mut bits = None;
    let mut data = Vec::new();
    let mut position = 12_u64;
    while position < riff_end {
        if riff_end - position < 8 {
            return Err(Error::Wav("partial chunk header".into()));
        }
        let mut chunk_header = [0_u8; 8];
        file.read_exact(&mut chunk_header)
            .map_err(|_| Error::Wav("partial chunk header".into()))?;
        position += 8;
        let size = u32::from_le_bytes(chunk_header[4..8].try_into().unwrap()) as usize;
        let size_u64 = size as u64;
        if size_u64 > riff_end - position {
            return Err(Error::Wav("partial chunk payload".into()));
        }
        if size % 2 == 1 && size_u64 + 1 > riff_end - position {
            return Err(Error::Wav("missing odd-byte chunk padding".into()));
        }
        match &chunk_header[0..4] {
            b"fmt " => {
                let mut chunk = vec![0; size];
                file.read_exact(&mut chunk)
                    .map_err(|_| Error::Wav("partial fmt chunk payload".into()))?;
                if size < 16 {
                    return Err(Error::Wav("short fmt chunk".into()));
                }
                format = Some(u16::from_le_bytes(chunk[0..2].try_into().unwrap()));
                channels = Some(u16::from_le_bytes(chunk[2..4].try_into().unwrap()));
                sample_rate = Some(u32::from_le_bytes(chunk[4..8].try_into().unwrap()));
                bits = Some(u16::from_le_bytes(chunk[14..16].try_into().unwrap()));
            }
            b"data" => {
                data.resize(size, 0);
                file.read_exact(&mut data)
                    .map_err(|_| Error::Wav("partial data chunk payload".into()))?;
            }
            _ => {
                file.seek(SeekFrom::Current(size as i64))
                    .map_err(|_| Error::Wav("could not skip unknown chunk payload".into()))?;
            }
        }
        position += size_u64;
        if size % 2 == 1 {
            let mut padding = [0_u8; 1];
            file.read_exact(&mut padding)
                .map_err(|_| Error::Wav("missing odd-byte chunk padding".into()))?;
            position += 1;
        }
    }
    if format != Some(1)
        || channels != Some(1)
        || bits != Some(16)
        || data.is_empty()
        || !data.len().is_multiple_of(2)
        || sample_rate == Some(0)
    {
        return Err(Error::Wav(
            "only valid non-empty mono 16-bit PCM WAV files are supported".into(),
        ));
    }
    let samples = data
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f64 / 32768.0)
        .collect();
    Ok((
        samples,
        sample_rate.ok_or_else(|| Error::Wav("missing sample rate".into()))?,
    ))
}

/// Equalize an input to a requested Meddis hair-cell level.
pub fn eqlz2meddis_hc_level(snd: &[f64], out_level_db: f64) -> Result<(Array1<f64>, [f64; 3])> {
    if !out_level_db.is_finite() {
        return Err(Error::InvalidParameter(
            "requested calibration level must be finite".into(),
        ));
    }
    let source_level = rms(snd) * 10_f64.powf(30.0 / 20.0);
    if !source_level.is_finite() || source_level <= 0.0 {
        return Err(Error::InvalidParameter(
            "cannot equalize an empty or silent signal".into(),
        ));
    }
    let amp = 10_f64.powf(out_level_db / 20.0) / source_level;
    Ok((
        Array1::from_iter(snd.iter().map(|v| amp * v)),
        [
            out_level_db,
            20.0 * amp.log10(),
            20.0 * source_level.log10(),
        ],
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrequencyScale {
    Erb,
    Mel,
    Log,
    Linear,
}

pub fn equal_freq_scale(
    scale: FrequencyScale,
    num_ch: usize,
    range_freq: [f64; 2],
) -> Result<(Array1<f64>, Array1<f64>)> {
    if num_ch < 2
        || range_freq.iter().any(|value| !value.is_finite())
        || range_freq[0] <= 0.0
        || range_freq[1] <= range_freq[0]
    {
        return Err(Error::InvalidParameter(
            "frequency scale requires at least two channels and a positive increasing range".into(),
        ));
    }
    let wrapped_range = match scale {
        FrequencyScale::Linear => range_freq,
        FrequencyScale::Mel => [freq2mel(range_freq[0]), freq2mel(range_freq[1])],
        FrequencyScale::Erb => {
            let (v, _) = freq2erb(&range_freq);
            [v[0], v[1]]
        }
        FrequencyScale::Log => [range_freq[0].log10(), range_freq[1].log10()],
    };
    let wrapped = Array1::linspace(wrapped_range[0], wrapped_range[1], num_ch);
    let mut frequencies = match scale {
        FrequencyScale::Linear => wrapped.clone(),
        FrequencyScale::Mel => wrapped.mapv(mel2freq),
        FrequencyScale::Erb => erb2freq(wrapped.as_slice().unwrap()).0,
        FrequencyScale::Log => wrapped.mapv(|v| 10_f64.powf(v)),
    };
    // Inverse scale conversions can move either endpoint by an ulp.  The
    // requested frequency interval is the public contract, so retain it
    // exactly while leaving the scale-spaced interior channels unchanged.
    frequencies[0] = range_freq[0];
    frequencies[num_ch - 1] = range_freq[1];
    Ok((frequencies, wrapped))
}

pub fn freq2mel(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}
pub fn mel2freq(mel: f64) -> f64 {
    700.0 * (10_f64.powf(mel / 2595.0) - 1.0)
}

pub fn freq2erb(cf: &[f64]) -> (Array1<f64>, Array1<f64>) {
    (
        Array1::from_iter(cf.iter().map(|v| 21.4 * (4.37 * v / 1000.0 + 1.0).log10())),
        Array1::from_iter(cf.iter().map(|v| 24.7 * (4.37 * v / 1000.0 + 1.0))),
    )
}

pub fn erb2freq(erb_rate: &[f64]) -> (Array1<f64>, Array1<f64>) {
    let cf = Array1::from_iter(
        erb_rate
            .iter()
            .map(|v| (10_f64.powf(v / 21.4) - 1.0) / 4.37 * 1000.0),
    );
    let width = Array1::from_iter(cf.iter().map(|v| 24.7 * (4.37 * v / 1000.0 + 1.0)));
    (cf, width)
}

pub fn taper_window(
    len_win: usize,
    type_taper: &str,
    len_taper: Option<usize>,
    range_sigma: f64,
) -> Result<(Array1<f64>, &'static str)> {
    if len_win == 0 {
        return Err(Error::InvalidParameter(
            "window length must be positive".into(),
        ));
    }
    let taper_len = len_taper.unwrap_or(len_win / 2).min(len_win / 2);
    let key = type_taper.to_ascii_uppercase();
    let (full, name) = if key.starts_with("HAM") {
        (dsp::hamming(2 * taper_len + 1), "Hamming")
    } else if key.starts_with("HAN") || key.starts_with("COS") {
        (dsp::hanning(2 * taper_len + 1), "Hanning/Cosine")
    } else if key.starts_with("BLA") {
        let n = 2 * taper_len + 1;
        (
            (0..n)
                .map(|i| {
                    0.42 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                        + 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                })
                .collect(),
            "Blackman",
        )
    } else if key.starts_with("GAU") {
        (
            (0..=2 * taper_len)
                .map(|i| {
                    let n = i as f64 - taper_len as f64;
                    (-(range_sigma * n / taper_len.max(1) as f64).powi(2) / 2.0).exp()
                })
                .collect(),
            "Gauss",
        )
    } else {
        (
            (0..=2 * taper_len)
                .map(|i| ((i + 1).min(2 * taper_len + 1 - i)) as f64 / (taper_len + 1) as f64)
                .collect(),
            "Line",
        )
    };
    let mut output = vec![1.0; len_win];
    output[..taper_len].copy_from_slice(&full[..taper_len]);
    if taper_len > 0 {
        output[len_win - taper_len..].copy_from_slice(&full[taper_len + 1..2 * taper_len + 1]);
    }
    Ok((Array1::from(output), name))
}

pub fn fftfilt(b: &[f64], x: &[f64]) -> Array1<f64> {
    Array1::from(dsp::fft_convolve_truncated(b, x))
}

/// Compatibility equivalent of NumPy's row-vector check for a Rust slice.
pub fn isrow<T>(_x: &[T]) -> bool {
    true
}

/// Compatibility equivalent of NumPy's column-vector check.
pub fn iscolumn<T>(x: &Array2<T>) -> bool {
    x.ncols() == 1
}

pub fn rceps(x: &[f64]) -> Result<(Array1<f64>, Array1<f64>)> {
    if x.is_empty() {
        return Err(Error::InvalidParameter(
            "cepstrum input cannot be empty".into(),
        ));
    }
    let n = x.len();
    let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
    for (dst, &src) in spectrum.iter_mut().zip(x) {
        dst.re = src;
    }
    dsp::transform(&mut spectrum, false);
    for value in &mut spectrum {
        *value = Complex64::new(value.norm().max(f64::MIN_POSITIVE).ln(), 0.0);
    }
    dsp::transform(&mut spectrum, true);
    let cep_full: Vec<f64> = spectrum.iter().map(|v| v.re).collect();
    let mut folded = vec![0.0; n];
    folded[0] = cep_full[0];
    let positive_end = n.div_ceil(2);
    for i in 1..positive_end {
        folded[i] = 2.0 * cep_full[i];
    }
    if n.is_multiple_of(2) {
        folded[n / 2] = cep_full[n / 2];
    }
    let mut minimum: Vec<Complex64> = folded.into_iter().map(|v| Complex64::new(v, 0.0)).collect();
    dsp::transform(&mut minimum, false);
    for value in &mut minimum {
        *value = value.exp();
    }
    dsp::transform(&mut minimum, true);
    Ok((
        Array1::from(cep_full[..x.len()].to_vec()),
        Array1::from(minimum[..x.len()].iter().map(|v| v.re).collect::<Vec<_>>()),
    ))
}

pub fn set_frame4time_sequence(
    snd: &[f64],
    len_frame: usize,
    shift_frame: Option<usize>,
) -> Result<(Array2<f64>, Array1<isize>)> {
    let shift = shift_frame.unwrap_or(len_frame / 2);
    if len_frame == 0
        || shift == 0
        || !len_frame.is_multiple_of(2)
        || !len_frame.is_multiple_of(shift)
    {
        return Err(Error::InvalidParameter(
            "frame length must be positive and even; shift must be a positive divisor of it".into(),
        ));
    }

    // This legacy utility labels the first zero-padded frame at `-shift` and
    // keeps a trailing frame whose label is at or before the input length.
    // The v2.34 filterbank's frame routine uses its own zero-based convention.
    let division = len_frame / shift;
    let padded_blocks = (snd.len() + len_frame).div_ceil(len_frame);
    let available_frames = padded_blocks.saturating_sub(1) * division + 1;
    let valid_frames = available_frames.min(snd.len() / shift + 2);
    let half = len_frame / 2;
    let mut output = Array2::zeros((len_frame, valid_frames));
    for frame in 0..valid_frames {
        let center = frame * shift;
        for offset in 0..len_frame {
            let source = center as isize + offset as isize - half as isize;
            if source >= 0 && (source as usize) < snd.len() {
                output[[offset, frame]] = snd[source as usize];
            }
        }
    }
    let positions =
        Array1::from_iter((0..valid_frames).map(|frame| (frame as isize - 1) * shift as isize));
    Ok((output, positions))
}

/// Tabulated legacy outer/middle-ear correction, returned as linear power.
pub fn out_mid_crct(
    kind: &str,
    n_frq_rsl: usize,
    fs: f64,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    const F1: &[f64] = &[
        20., 25., 30., 35., 40., 45., 50., 55., 60., 70., 80., 90., 100., 125., 150., 177., 200.,
        250., 300., 350., 400., 450., 500., 550., 600., 700., 800., 900., 1000., 1500., 2000.,
        2500., 2828., 3000., 3500., 4000., 4500., 5000., 5500., 6000., 7000., 8000., 9000., 10000.,
        12748., 15000.,
    ];
    const ELC: &[f64] = &[
        31.8, 26., 21.7, 18.8, 17.2, 15.4, 14., 12.6, 11.6, 10.6, 9.2, 8.2, 7.7, 6.7, 5.3, 4.6,
        3.9, 2.9, 2.7, 2.3, 2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7, 0.9, -1.3, -2.5,
        -3.2, -4.4, -4.1, -2.5, -0.5, 2., 5., 10.2, 15., 17., 15.5, 11., 22.,
    ];
    const MAF: &[f64] = &[
        73.4, 65.2, 57.9, 52.7, 48., 45., 41.9, 39.3, 36.8, 33., 29.7, 27.1, 25., 22., 18.2, 16.,
        14., 11.4, 9.2, 8., 6.9, 6.2, 5.7, 5.1, 5., 5., 4.4, 4.3, 3.9, 2.7, 0.9, -1.3, -2.5, -3.2,
        -4.4, -4.1, -2.5, -0.5, 2., 5., 10.2, 15., 17., 15.5, 11., 22.,
    ];
    const F2: &[f64] = &[
        125., 250., 500., 1000., 1500., 2000., 3000., 4000., 6000., 8000., 10000., 12000., 14000.,
        16000.,
    ];
    const MAP: &[f64] = &[
        30., 19., 12., 9., 11., 16., 16., 14., 14., 9.9, 24.7, 32.7, 44.1, 63.7,
    ];
    let (source_f, source_db, value_at_nyquist) = match kind.to_ascii_uppercase().as_str() {
        "ELC" => (F1, ELC, 130.0),
        "MAF" => (F1, MAF, 130.0),
        "MAP" => (F2, MAP, 180.0),
        "NO" => {
            let freq = Array1::from_iter(
                (0..n_frq_rsl.max(1)).map(|i| i as f64 / n_frq_rsl.max(1) as f64 * fs / 2.0),
            );
            let db = Array1::zeros(freq.len());
            return Ok((Array1::ones(freq.len()), freq, db));
        }
        _ => {
            return Err(Error::InvalidParameter(
                "correction must be ELC, MAF, MAP, or NO".into(),
            ));
        }
    };
    let mut table_f = source_f.to_vec();
    let mut table_db = source_db.to_vec();
    if fs > 32_000.0 {
        table_f.push(fs / 2.0);
        table_db.push(value_at_nyquist);
    }
    let (freq, db) = if n_frq_rsl == 0 {
        (table_f, table_db)
    } else {
        let f: Vec<f64> = (0..n_frq_rsl)
            .map(|i| i as f64 / n_frq_rsl as f64 * fs / 2.0)
            .collect();
        (f.clone(), dsp::interp1(&table_f, &table_db, &f, true)?)
    };
    let power = db.iter().map(|v| 10_f64.powf(-v / 10.0)).collect();
    Ok((power, Array1::from(freq), Array1::from(db)))
}

/// Calculate a safe odd tap count that leaves at least one minimum-phase tap.
pub(crate) fn correction_filter_coefficient_count(sr: f64, fft_len: usize) -> Result<usize> {
    if !sr.is_finite() || sr <= 0.0 {
        return Err(Error::InvalidParameter(
            "correction-filter sample rate must be finite and positive".into(),
        ));
    }
    let half_count = (200.0 / 16000.0 * sr / 2.0).trunc() as usize;
    let max_half_count = fft_len.saturating_sub(2) / 2;
    if half_count == 0 || max_half_count == 0 {
        return Err(Error::InvalidParameter(
            "sample rate is too low to construct a non-empty correction filter".into(),
        ));
    }
    Ok(half_count.min(max_half_count) * 2 + 1)
}

/// Construct the legacy outer/middle-ear correction FIR.
///
/// `filter_type` follows the Python switch: 0 is forward linear phase, 1 is
/// inverse linear phase, and 2 is forward minimum phase. The Rust port uses a
/// frequency-sampling FIR design; its magnitude follows the same correction
/// table while avoiding a dependency on SciPy's Parks–McClellan implementation.
pub fn out_mid_crct_filt(kind: &str, sr: f64, filter_type: u8) -> Result<Array1<f64>> {
    if !matches!(filter_type, 0..=2) {
        return Err(Error::InvalidParameter(
            "filter type must be 0, 1, or 2".into(),
        ));
    }
    let bins = 2048;
    let fft_len = bins * 2;
    let coefficient_count = correction_filter_coefficient_count(sr, fft_len)?;
    let (power, _, _) = out_mid_crct(kind, bins, sr)?;
    let mut magnitude: Vec<f64> = power.iter().map(|v| v.sqrt()).collect();
    if filter_type == 1 {
        for value in &mut magnitude {
            *value = 1.0 / value.max(0.1);
        }
    }
    let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    spectrum[0].re = magnitude[0];
    for i in 1..bins {
        spectrum[i].re = magnitude[i];
        spectrum[fft_len - i].re = magnitude[i];
    }
    spectrum[bins].re = *magnitude.last().unwrap();
    dsp::fft(&mut spectrum, true);
    let center = coefficient_count / 2;
    let window = dsp::hanning(coefficient_count);
    let mut linear = vec![0.0; coefficient_count];
    for i in 0..coefficient_count {
        let circular_index = (fft_len + i - center) % fft_len;
        linear[i] = spectrum[circular_index].re * window[i];
    }
    if filter_type == 2 {
        let (_, minimum) = rceps(&linear)?;
        Ok(minimum
            .slice(ndarray::s![..coefficient_count / 2])
            .to_owned())
    } else {
        Ok(Array1::from(linear))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::fs;

    #[test]
    fn auditory_scale_reference_values() {
        let (erb, width) = freq2erb(&[1000.0]);
        assert_relative_eq!(erb[0], 15.621449713970488, epsilon = 1e-12);
        assert_relative_eq!(width[0], 132.639, epsilon = 1e-12);
        assert_relative_eq!(mel2freq(freq2mel(1000.0)), 1000.0, epsilon = 1e-10);
    }

    #[test]
    fn fft_filter_has_scipy_lfilter_prefix_semantics() {
        let got = fftfilt(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0, 8.0]);
        assert_eq!(got.len(), 4);
        for (a, b) in got.iter().zip([1.0, 4.0, 11.0, 22.0]) {
            assert_relative_eq!(*a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn correction_filter_rejects_rates_that_would_produce_no_minimum_phase_taps() {
        assert!(out_mid_crct_filt("ELC", 128.0, 2).is_err());
        assert!(out_mid_crct_filt("ELC", f64::NAN, 2).is_err());
        assert!(!out_mid_crct_filt("ELC", 160.0, 2).unwrap().is_empty());
    }

    #[test]
    fn legacy_framing_includes_leading_and_trailing_frames() {
        let (frames, centers) = set_frame4time_sequence(&[1.0, 2.0, 3.0], 4, Some(2)).unwrap();
        assert_eq!(centers.as_slice().unwrap(), &[-2, 0, 2]);
        assert_eq!(frames.column(0).to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
        assert_eq!(frames.column(2).to_vec(), vec![3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn audioread_rejects_stereo_pcm_instead_of_flattening_channels() {
        let samples = [1000_i16, -1000, 2000, -2000];
        let data_len = (samples.len() * size_of::<i16>()) as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&8_000_u32.to_le_bytes());
        wav.extend_from_slice(&32_000_u32.to_le_bytes());
        wav.extend_from_slice(&4_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        let path =
            std::env::temp_dir().join(format!("gammachirpy-stereo-{}.wav", std::process::id()));
        fs::write(&path, wav).unwrap();
        let result = audioread(&path);
        fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(Error::Wav(message)) if message.contains("mono")));
    }

    #[test]
    fn audioread_rejects_an_incomplete_16_bit_sample() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&38_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&8_000_u32.to_le_bytes());
        wav.extend_from_slice(&16_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&1_u32.to_le_bytes());
        wav.extend_from_slice(&[0_u8, 0_u8]);

        let path = std::env::temp_dir().join(format!(
            "gammachirpy-incomplete-sample-{}.wav",
            std::process::id()
        ));
        fs::write(&path, wav).unwrap();
        let result = audioread(&path);
        fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(Error::Wav(_))));
    }
}
