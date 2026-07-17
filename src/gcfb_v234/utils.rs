//! v2.34 auditory utilities and field-to-cochlea transfer functions.

use ndarray::{Array1, Array2};

use crate::{Error, Result, dsp};

pub use crate::gcfb_v211::utils::{
    FrequencyScale, audioread, erb2freq, fftfilt, freq2erb, freq2mel, iscolumn, isrow, mel2freq,
    nextpow2, out_mid_crct, out_mid_crct_filt, rceps, rms, set_frame4time_sequence, taper_window,
};

#[derive(Clone, Debug)]
pub struct ParamTransFunc {
    pub fs: f64,
    pub n_frq_rsl: usize,
    pub freq_calib: f64,
    pub type_field2eardrum: String,
    pub type_midear2cochlea: String,
    pub type_field2cochlea_db: String,
    pub name_filter: String,
}

impl Default for ParamTransFunc {
    fn default() -> Self {
        Self {
            fs: 48_000.0,
            n_frq_rsl: 2048,
            freq_calib: 1000.0,
            type_field2eardrum: String::new(),
            type_midear2cochlea: String::new(),
            type_field2cochlea_db: String::new(),
            name_filter: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OutTransFunc {
    pub fs: f64,
    pub freq_calib: f64,
    pub freq: Array1<f64>,
    pub field2eardrum_db: Array1<f64>,
    pub field2eardrum_db_at_freq_calib: f64,
    pub field2eardrum_db_cmpnst_db: f64,
    pub field2cochlea_db: Array1<f64>,
    pub field2cochlea_db_at_freq_calib: f64,
    pub midear2cochlea_db: Array1<f64>,
    pub midear2cochlea_db_at_freq_calib: f64,
    pub type_field2cochlea_db: String,
    pub type_field2eardrum: String,
    pub type_midear2cochlea: String,
}

#[derive(Clone, Debug)]
pub struct SplAtHl0Db {
    pub freq: Array1<f64>,
    pub spl_db_at_hl_0db: Array1<f64>,
    pub speech: f64,
    pub standard: &'static str,
    pub earphone: &'static str,
    pub artificial_ear: &'static str,
}

/// v2.34 signal calibration. Supplying `input_rms1_dbspl` selects the precise
/// digital-level calibration introduced in v2.33.
pub fn eqlz2meddis_hc_level(
    snd: &[f64],
    out_level_db: Option<f64>,
    input_rms1_dbspl: Option<f64>,
) -> Result<(Array1<f64>, [f64; 3])> {
    let signal_rms = rms(snd);
    if !signal_rms.is_finite() || signal_rms <= 0.0 {
        return Err(Error::InvalidParameter(
            "cannot equalize an empty or silent signal".into(),
        ));
    }
    if let Some(rms1_spl) = input_rms1_dbspl {
        let source_db = 20.0 * signal_rms.log10() + rms1_spl;
        let compensation_db = rms1_spl - 30.0;
        let output = Array1::from_iter(snd.iter().map(|v| v * 10_f64.powf(compensation_db / 20.0)));
        Ok((output, [source_db, compensation_db, source_db]))
    } else {
        crate::gcfb_v211::utils::eqlz2meddis_hc_level(
            snd,
            out_level_db.ok_or_else(|| {
                Error::InvalidParameter("out_level_db is required without input_rms1_dbspl".into())
            })?,
        )
    }
}

pub fn equal_freq_scale(
    scale: FrequencyScale,
    num_ch: usize,
    range_freq: [f64; 2],
) -> Result<(Array1<f64>, Array1<f64>)> {
    crate::gcfb_v211::utils::equal_freq_scale(scale, num_ch, range_freq)
}

pub fn interp1(x: &[f64], y: &[f64], x_new: &[f64], extrapolate: bool) -> Result<Array1<f64>> {
    Ok(Array1::from(dsp::interp1(x, y, x_new, extrapolate)?))
}

pub fn trans_func_free_field2eardrum_moore16(kind: &str) -> Result<(Array1<f64>, Array1<f64>)> {
    const FREQ: &[f64] = &[
        20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250., 315., 400., 500., 630.,
        750., 800., 1000., 1250., 1500., 1600., 2000., 2500., 3000., 3150., 4000., 5000., 6000.,
        6300., 8000., 9000., 10000., 11200., 12500., 14000., 15000., 16000.,
    ];
    const FREE: &[f64] = &[
        0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.3, 0.5, 0.9, 1.4, 1.6, 1.7, 2.5, 2.7, 2.6, 2.6, 3.2,
        5.2, 6.6, 12., 16.8, 15.3, 15.2, 14.2, 10.7, 7.1, 6.4, 1.8, -0.9, -1.6, 1.9, 4.9, 2., -2.,
        2.5,
    ];
    const DIFFUSE: &[f64] = &[
        0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.3, 0.4, 0.5, 1., 1.6, 1.7, 2.2, 2.7, 2.9, 3.8, 5.3,
        6.8, 7.2, 10.2, 14.9, 14.5, 14.4, 12.7, 10.8, 8.9, 8.7, 8.5, 6.2, 5., 4.5, 4., 3.3, 2.6,
        2.,
    ];
    let values = match kind {
        "FreeField" => FREE,
        "DiffuseField" => DIFFUSE,
        _ => {
            return Err(Error::InvalidParameter(
                "field type must be FreeField or DiffuseField".into(),
            ));
        }
    };
    Ok((Array1::from(FREQ.to_vec()), Array1::from(values.to_vec())))
}

pub fn trans_func_free_field2eardrum_itu(kind: &str) -> Result<(Array1<f64>, Array1<f64>)> {
    if kind != "ITU" {
        return Err(Error::InvalidParameter("field type must be ITU".into()));
    }
    Ok((
        Array1::from(vec![
            0., 100., 125., 160., 200., 250., 315., 400., 500., 630., 800., 1000., 1250., 1600.,
            2000., 2500., 3150., 4000., 5000., 6300., 8000., 10000.,
        ]),
        Array1::from(vec![
            0., 0., 0., 0., 0., 0.3, 0.2, 0.5, 0.6, 0.7, 1.1, 1.7, 2.6, 4.2, 6.5, 9.4, 10.3, 6.6,
            3.2, 3.3, 16., 14.4,
        ]),
    ))
}

pub fn trans_func_middle_ear_moore16() -> (Array1<f64>, Array1<f64>) {
    (
        Array1::from(vec![
            20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250., 315., 400., 500.,
            630., 750., 800., 1000., 1250., 1500., 1600., 2000., 2500., 3000., 3150., 4000., 5000.,
            6000., 6300., 8000., 9000., 10000., 11200., 12500., 14000., 15000., 16000., 18000.,
            20000.,
        ]),
        Array1::from(vec![
            -39.6, -32., -25.85, -21.4, -18.5, -15.9, -14.1, -12.4, -11., -9.6, -8.3, -7.4, -6.2,
            -4.8, -3.8, -3.3, -2.9, -2.6, -2.6, -4.5, -5.4, -6.1, -8.5, -10.4, -7.3, -7., -6.6,
            -7., -9.2, -10.2, -12.2, -10.8, -10.1, -12.7, -15., -18.2, -23.8, -32.3, -45.5, -50.,
        ]),
    )
}

pub fn trans_func_field2eardrum_set(
    kind: &str,
) -> Result<(Array1<f64>, Array1<f64>, &'static str)> {
    if kind.contains("FreeField") {
        let (f, v) = trans_func_free_field2eardrum_moore16("FreeField")?;
        Ok((f, v, "FreeField"))
    } else if kind.contains("DiffuseField") {
        let (f, v) = trans_func_free_field2eardrum_moore16("DiffuseField")?;
        Ok((f, v, "DiffuseField"))
    } else if kind.contains("ITU") {
        let (f, v) = trans_func_free_field2eardrum_itu("ITU")?;
        Ok((f, v, "ITU"))
    } else {
        Err(Error::InvalidParameter(
            "unknown field-to-eardrum transfer function".into(),
        ))
    }
}

pub fn trans_func_field2cochlea(param: &ParamTransFunc) -> Result<OutTransFunc> {
    if param.fs <= 0.0 || param.n_frq_rsl < 2 {
        return Err(Error::InvalidParameter(
            "transfer function requires positive fs and at least two bins".into(),
        ));
    }
    let freq = Array1::from_iter(
        (0..param.n_frq_rsl).map(|i| i as f64 / param.n_frq_rsl as f64 * param.fs / 2.0),
    );
    let field_type = if param.type_field2eardrum.contains("Diffuse") {
        "DiffuseField"
    } else if param.type_field2eardrum.contains("ITU") {
        "ITU"
    } else if param.type_field2eardrum.contains("NoField") {
        "NoField"
    } else {
        "FreeField"
    };
    let mut field_db = if field_type == "NoField" {
        Array1::zeros(freq.len())
    } else {
        let (table_f, table_v) = if field_type == "ITU" {
            trans_func_free_field2eardrum_itu("ITU")?
        } else {
            trans_func_free_field2eardrum_moore16(field_type)?
        };
        // The Python reference holds the final tabulated response constant up
        // to Nyquist when the sample rate extends beyond the table.  Adding
        // that endpoint before ERB-scale interpolation is observably different
        // from unconstrained linear extrapolation at high sample rates.
        let mut table_f = table_f.to_vec();
        let mut table_v = table_v.to_vec();
        let nyquist = param.fs / 2.0;
        if nyquist > *table_f.last().unwrap() {
            table_f.push(nyquist);
            table_v.push(*table_v.last().unwrap());
        }
        let (table_erb, _) = freq2erb(&table_f);
        let (query_erb, _) = freq2erb(freq.as_slice().unwrap());
        interp1(
            table_erb.as_slice().unwrap(),
            &table_v,
            query_erb.as_slice().unwrap(),
            true,
        )?
    };
    let calibration = freq
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - param.freq_calib)
                .abs()
                .total_cmp(&(b.1 - param.freq_calib).abs())
        })
        .unwrap()
        .0;
    let field_compensation = field_db[calibration];
    field_db.mapv_inplace(|v| v - field_compensation);
    let (middle_f, middle_v) = trans_func_middle_ear_moore16();
    let mut middle_f = middle_f.to_vec();
    let mut middle_v = middle_v.to_vec();
    let nyquist = param.fs / 2.0;
    if nyquist > *middle_f.last().unwrap() {
        middle_f.push(nyquist);
        middle_v.push(*middle_v.last().unwrap());
    }
    let (middle_erb, _) = freq2erb(&middle_f);
    let (query_erb, _) = freq2erb(freq.as_slice().unwrap());
    let middle_db = interp1(
        middle_erb.as_slice().unwrap(),
        &middle_v,
        query_erb.as_slice().unwrap(),
        true,
    )?;
    let total = &field_db + &middle_db;
    Ok(OutTransFunc {
        fs: param.fs,
        freq_calib: freq[calibration],
        freq,
        field2eardrum_db_at_freq_calib: field_db[calibration],
        field2eardrum_db_cmpnst_db: field_compensation,
        field2cochlea_db_at_freq_calib: total[calibration],
        midear2cochlea_db_at_freq_calib: middle_db[calibration],
        field2eardrum_db: field_db,
        midear2cochlea_db: middle_db,
        field2cochlea_db: total,
        type_field2cochlea_db: format!("{field_type} + MiddleEar_Moore16"),
        type_field2eardrum: field_type.into(),
        type_midear2cochlea: "MiddleEar_Moore16".into(),
    })
}

pub fn mk_filter_field2cochlea(
    kind: &str,
    fs: f64,
    forward: bool,
) -> Result<(Array1<f64>, ParamTransFunc)> {
    if kind.eq_ignore_ascii_case("ELC") {
        let coefficients = out_mid_crct_filt("ELC", fs, if forward { 2 } else { 1 })?;
        let param = ParamTransFunc {
            fs,
            type_field2cochlea_db: "ELC".into(),
            name_filter: if forward {
                "[ELC] forward minimum-phase FIR".into()
            } else {
                "[ELC] inverse FIR".into()
            },
            ..Default::default()
        };
        return Ok((coefficients, param));
    }
    let field = match kind {
        "FreeField" | "FF" => "FreeField",
        "DiffuseField" | "DF" => "DiffuseField",
        "ITU" => "ITU",
        "EarDrum" | "ED" => "NoField",
        _ => {
            return Err(Error::InvalidParameter(
                "correction must be FreeField, DiffuseField, ITU, EarDrum, or ELC".into(),
            ));
        }
    };
    let mut param = ParamTransFunc {
        fs,
        type_field2eardrum: field.into(),
        type_midear2cochlea: "MiddleEar".into(),
        ..Default::default()
    };
    let transfer = trans_func_field2cochlea(&param)?;
    let bins = transfer.freq.len();
    let fft_len = bins * 2;
    let mut spectrum = vec![num_complex::Complex64::new(0., 0.); fft_len];
    let mags: Vec<f64> = transfer
        .field2cochlea_db
        .iter()
        .map(|db| 10_f64.powf(db / 20.))
        .collect();
    for i in 0..bins {
        let m = if forward {
            mags[i]
        } else {
            1.0 / mags[i].max(0.1)
        };
        spectrum[i].re = m;
        if i > 0 {
            spectrum[fft_len - i].re = m;
        }
    }
    crate::dsp::fft(&mut spectrum, true);
    let count = ((200. / 16000. * fs / 2.).trunc() as usize * 2 + 1).min(fft_len - 1);
    let center = count / 2;
    let win = crate::dsp::hanning(count);
    let linear: Vec<f64> = (0..count)
        .map(|i| spectrum[(fft_len + i - center) % fft_len].re * win[i])
        .collect();
    let (_, minimum) = rceps(&linear)?;
    param.name_filter = format!("[{kind}] minimum-phase FIR");
    Ok((minimum.slice(ndarray::s![..count / 2]).to_owned(), param))
}

pub fn spl_at_hl_0db_table() -> SplAtHl0Db {
    SplAtHl0Db {
        freq: Array1::from(vec![
            125., 160., 200., 250., 315., 400., 500., 630., 750., 800., 1000., 1250., 1500., 1600.,
            2000., 2500., 3000., 3150., 4000., 5000., 6000., 6300., 8000.,
        ]),
        spl_db_at_hl_0db: Array1::from(vec![
            45., 38.5, 32.5, 27., 22., 17., 13.5, 10.5, 9., 8.5, 7.5, 7.5, 7.5, 8., 9., 10.5, 11.5,
            11.5, 12., 11., 16., 21., 15.5,
        ]),
        speech: 20.,
        standard: "ANSI-S3.6_2010",
        earphone: "Supra-aural earphone per ANSI clause 9.1.1 or ISO 389-1",
        artificial_ear: "IEC 60318-1",
    }
}

pub fn hl2spl(freq: f64, hl_db: f64) -> Result<f64> {
    let table = spl_at_hl_0db_table();
    let index = table
        .freq
        .iter()
        .position(|v| (*v - freq).abs() < 1e-9)
        .ok_or_else(|| {
            Error::InvalidParameter("frequency is not present in the ANSI HL/SPL table".into())
        })?;
    Ok(hl_db + table.spl_db_at_hl_0db[index])
}

pub fn hl2pin_cochlea(freq: f64, hl_db: f64) -> Result<f64> {
    let spl = hl2spl(freq, hl_db)?;
    let (f, v) = trans_func_middle_ear_moore16();
    let index = f
        .iter()
        .position(|x| (*x - freq).abs() < 1e-9)
        .ok_or_else(|| {
            Error::InvalidParameter("frequency is not present in the middle-ear table".into())
        })?;
    Ok(spl + v[index])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Floor {
    None,
    ZeroFloor,
    NoiseFloor,
}

pub fn eqlz_gcfb2rms1_at_0db(values: &Array2<f64>, floor: Floor) -> Array2<f64> {
    let mut out = values.mapv(|v| v * 10_f64.powf(30. / 20.));
    match floor {
        Floor::None => {}
        Floor::ZeroFloor => out.mapv_inplace(|v| (v - 1.).max(0.)),
        Floor::NoiseFloor => {
            // Deterministic Box–Muller noise makes analyses reproducible.
            let mut state: u64 = 0x9e3779b97f4a7c15;
            for value in &mut out {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
                *value += (-2. * u1.ln()).sqrt() * (2. * std::f64::consts::PI * u2).cos();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn hearing_level_reference() {
        assert_relative_eq!(hl2spl(1000., 0.).unwrap(), 7.5);
        assert_relative_eq!(hl2pin_cochlea(1000., 0.).unwrap(), 4.9);
    }
    #[test]
    fn precise_level_equalization() {
        let (x, db) = eqlz2meddis_hc_level(&[0.5, -0.5], None, Some(90.)).unwrap();
        assert_relative_eq!(x[0], 500., epsilon = 1e-12);
        assert_relative_eq!(db[1], 60.);
    }
}
