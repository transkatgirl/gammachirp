//! Generated differential tests against the checked-in Python implementation.
//!
//! The Python interpreter is kept alive and receives one JSON request per
//! generated case.  Set `GAMMACHIRPY_PYTHON` when Python is not on `PATH`.
//! Environments without NumPy/SciPy skip this suite unless
//! `GAMMACHIRPY_REQUIRE_PYTHON=1`, which is the recommended CI setting.

use std::{
    env,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Mutex, OnceLock},
};

use gammachirp_rs::gcfb_v234::{
    gammachirp::{self as gc, Carrier, Normalization},
    gcfb_v234::{
        self as fb234, AcfStatus, ControlMode as Control234, DynHpaf, EmParam, GainReference,
        GcParam as Param234,
    },
    utils::{self as utils234, Floor, FrequencyScale, ParamTransFunc},
};

use ndarray::{Array2, ArrayView2};
use proptest::{
    prelude::*,
    test_runner::{FileFailurePersistence, TestCaseError},
};
use serde_json::{Value, json};

struct PythonOracle {
    _child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PythonOracle {
    fn start() -> Result<Self, String> {
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/python_property_oracle.py");
        let candidates = match env::var("GAMMACHIRPY_PYTHON") {
            Ok(interpreter) => vec![interpreter],
            Err(_) => vec!["python3".into(), "python".into()],
        };
        let mut failures = Vec::new();
        for interpreter in candidates {
            let mut child = match Command::new(&interpreter)
                .arg(&script)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
            {
                Ok(child) => child,
                Err(error) => {
                    failures.push(format!("{interpreter}: {error}"));
                    continue;
                }
            };
            let stdin = child.stdin.take().expect("piped Python stdin");
            let stdout = child.stdout.take().expect("piped Python stdout");
            let mut stdout = BufReader::new(stdout);
            let mut line = String::new();
            if let Err(error) = stdout.read_line(&mut line) {
                failures.push(format!("{interpreter}: could not read readiness: {error}"));
                let _ = child.kill();
                continue;
            }
            let readiness: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(error) => {
                    failures.push(format!(
                        "{interpreter}: invalid readiness response {line:?}: {error}"
                    ));
                    let _ = child.kill();
                    continue;
                }
            };
            if readiness["ready"] == true {
                return Ok(Self {
                    _child: child,
                    stdin: BufWriter::new(stdin),
                    stdout,
                });
            }
            failures.push(format!(
                "{interpreter}: {}",
                readiness["error"]
                    .as_str()
                    .unwrap_or("oracle initialization failed")
            ));
            let _ = child.wait();
        }
        Err(format!(
            "Python reference unavailable ({}). Install NumPy and SciPy or set \
             GAMMACHIRPY_PYTHON to a suitable interpreter",
            failures.join("; ")
        ))
    }

    fn call(&mut self, request: &Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.stdin, request)
            .map_err(|error| format!("could not encode oracle request: {error}"))?;
        self.stdin
            .write_all(b"\n")
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("could not write oracle request: {error}"))?;
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .map_err(|error| format!("could not read oracle response: {error}"))?;
        if line.is_empty() {
            return Err("Python oracle exited before returning a response".into());
        }
        let response: Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid oracle response {line:?}: {error}"))?;
        if response["ok"] != true {
            return Err(format!(
                "Python operation {} failed: {}",
                request["op"], response["error"]
            ));
        }
        Ok(response["result"].clone())
    }
}

static ORACLE: OnceLock<Result<Mutex<PythonOracle>, String>> = OnceLock::new();

fn oracle(request: Value) -> Option<Value> {
    let state = ORACLE.get_or_init(|| {
        let result = PythonOracle::start().map(Mutex::new);
        if let Err(error) = &result {
            eprintln!("skipping live Rust/Python property comparisons: {error}");
        }
        result
    });
    match state {
        Ok(oracle) => {
            let mut oracle = oracle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Some(
                oracle
                    .call(&request)
                    .unwrap_or_else(|error| panic!("{error}; request={request}")),
            )
        }
        Err(error) => {
            let required = env::var("GAMMACHIRPY_REQUIRE_PYTHON")
                .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"));
            assert!(!required, "{error}");
            None
        }
    }
}

fn expected_values(node: &Value) -> Result<Vec<f64>, TestCaseError> {
    node["values"]
        .as_array()
        .ok_or_else(|| TestCaseError::fail(format!("oracle array has no values: {node}")))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| TestCaseError::fail(format!("non-numeric oracle value: {value}")))
        })
        .collect()
}

fn check_values(
    label: &str,
    actual: &[f64],
    expected: &Value,
    atol: f64,
    rtol: f64,
) -> Result<(), TestCaseError> {
    let expected = expected_values(expected)?;
    if actual.len() != expected.len() {
        return Err(TestCaseError::fail(format!(
            "{label}: Rust length {} != Python length {}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (&rust, &python)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = atol + rtol * python.abs();
        if !rust.is_finite() || !python.is_finite() || (rust - python).abs() > tolerance {
            return Err(TestCaseError::fail(format!(
                "{label}[{index}]: Rust {rust:.17e}, Python {python:.17e}, difference {:.3e}, tolerance {tolerance:.3e}",
                (rust - python).abs()
            )));
        }
    }
    Ok(())
}

fn check_array(
    label: &str,
    actual: &[f64],
    shape: &[usize],
    expected: &Value,
    atol: f64,
    rtol: f64,
) -> Result<(), TestCaseError> {
    let expected_shape: Vec<usize> = expected["shape"]
        .as_array()
        .ok_or_else(|| TestCaseError::fail(format!("oracle array has no shape: {expected}")))?
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    if shape != expected_shape {
        return Err(TestCaseError::fail(format!(
            "{label}: Rust shape {shape:?} != Python shape {expected_shape:?}"
        )));
    }
    check_values(label, actual, expected, atol, rtol)
}

fn check_scalar(
    label: &str,
    actual: f64,
    expected: &Value,
    atol: f64,
    rtol: f64,
) -> Result<(), TestCaseError> {
    let python = expected
        .as_f64()
        .ok_or_else(|| TestCaseError::fail(format!("{label}: expected scalar, got {expected}")))?;
    let tolerance = atol + rtol * python.abs();
    if !actual.is_finite() || !python.is_finite() || (actual - python).abs() > tolerance {
        return Err(TestCaseError::fail(format!(
            "{label}: Rust {actual:.17e}, Python {python:.17e}, difference {:.3e}, tolerance {tolerance:.3e}",
            (actual - python).abs()
        )));
    }
    Ok(())
}

fn flattened(view: ArrayView2<'_, f64>) -> Vec<f64> {
    view.rows()
        .into_iter()
        .flat_map(|row| row.to_vec())
        .collect()
}

fn non_silent(mut values: Vec<f64>) -> Vec<f64> {
    values[0] += if values[0] >= 0.0 { 1.0 } else { -1.0 };
    values
}

fn carrier(index: u8) -> (Carrier, &'static str) {
    match index {
        0 => (Carrier::Cosine, "cos"),
        1 => (Carrier::Sine, "sin"),
        _ => (Carrier::Envelope, "env"),
    }
}

fn normalization(peak: bool) -> (Normalization, &'static str) {
    if peak {
        (Normalization::Peak, "peak")
    } else {
        (Normalization::None, "no")
    }
}

fn property_config(cases: u32) -> ProptestConfig {
    let mut config = ProptestConfig::default();
    if env::var_os("PROPTEST_CASES").is_none() {
        config.cases = cases;
    }
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Direct(
        "tests/python_properties.proptest-regressions",
    )));
    config
}

proptest! {
    #![proptest_config(property_config(96))]

    #[test]
    fn auditory_scales_and_level_primitives_match_python(
        frequencies in prop::collection::vec(0.0f64..12_000.0, 1..16),
        mel in prop::collection::vec(0.0f64..4_000.0, 1..16),
        signal in prop::collection::vec(-4.0f64..4.0, 1..32),
        integer in 1usize..1_000_000,
        channels in 2usize..40,
        low in 20.0f64..1_000.0,
        span in 50.0f64..10_000.0,
    ) {
        let signal = non_silent(signal);
        let Some(expected) = oracle(json!({
            "op": "scales", "frequencies": frequencies, "mel": mel,
            "signal": signal, "integer": integer, "scale": "ERB",
            "channels": channels, "range": [low, low + span],
        })) else { return Ok(()) };
        check_scalar("rms", utils234::rms(&signal), &expected["rms"], 2e-14, 2e-14)?;
        prop_assert_eq!(utils234::nextpow2(integer) as u64, expected["nextpow2"].as_u64().unwrap());
        let mel_actual: Vec<f64> = frequencies.iter().map(|&value| utils234::freq2mel(value)).collect();
        check_array("freq2mel", &mel_actual, &[frequencies.len()], &expected["freq2mel"], 2e-12, 2e-13)?;
        let frequency_actual: Vec<f64> = mel.iter().map(|&value| utils234::mel2freq(value)).collect();
        check_array("mel2freq", &frequency_actual, &[mel.len()], &expected["mel2freq"], 2e-10, 2e-13)?;
        let (rate, width) = utils234::freq2erb(&frequencies);
        check_array("ERB rate", rate.as_slice().unwrap(), &[frequencies.len()], &expected["erb_rate"], 2e-12, 2e-13)?;
        check_array("ERB width", width.as_slice().unwrap(), &[frequencies.len()], &expected["erb_width"], 2e-11, 2e-13)?;
        let (inverse, inverse_width) = utils234::erb2freq(rate.as_slice().unwrap());
        check_array("ERB inverse", inverse.as_slice().unwrap(), &[frequencies.len()], &expected["erb_inverse"], 3e-10, 3e-13)?;
        check_array("ERB inverse width", inverse_width.as_slice().unwrap(), &[frequencies.len()], &expected["erb_inverse_width"], 3e-11, 3e-13)?;
        let (equal_frequency, equal_scale) = utils234::equal_freq_scale(FrequencyScale::Erb, channels, [low, low + span]).unwrap();
        check_values("equal ERB frequency", equal_frequency.as_slice().unwrap(), &expected["equal_frequency"], 3e-9, 3e-12)?;
        check_values("equal ERB scale", equal_scale.as_slice().unwrap(), &expected["equal_scale"], 3e-12, 3e-13)?;
    }

    #[test]
    fn signal_windows_filtering_cepstrum_and_framing_match_python(
        signal in prop::collection::vec(-2.0f64..2.0, 3..40),
        coefficients in prop::collection::vec(-2.0f64..2.0, 1..12),
        cepstrum_tail in prop::collection::vec(-0.4f64..0.4, 0..8),
        out_level in -10.0f64..120.0,
        window_selector in 0u8..4,
        window_length in 4usize..30,
        taper_fraction in 0usize..20,
        sigma in 0.5f64..5.0,
        frame_selector in 0usize..5,
        shift_selector in 0usize..8,
    ) {
        let signal = non_silent(signal);
        let mut cepstrum = vec![1.0];
        cepstrum.extend(cepstrum_tail.into_iter().take(6));
        if cepstrum.len().is_multiple_of(2) { cepstrum.push(0.125); }
        let kinds = ["HAM", "HAN", "BLA", "LINE"];
        let window_kind = kinds[window_selector as usize];
        let taper_length = taper_fraction.min(window_length / 2);
        let frame_lengths = [4usize, 6, 8, 10, 12];
        let frame_length = frame_lengths[frame_selector];
        let divisors: Vec<usize> = (1..=frame_length)
            .filter(|value| frame_length.is_multiple_of(*value))
            .collect();
        let frame_shift = divisors[shift_selector % divisors.len()];
        let Some(expected) = oracle(json!({
            "op": "signal_utils", "signal": signal, "coefficients": coefficients,
            "cepstrum": cepstrum, "out_level_db": out_level,
            "window_kind": window_kind, "window_length": window_length,
            "taper_length": taper_length, "range_sigma": sigma,
            "frame_length": frame_length, "frame_shift": frame_shift,
        })) else { return Ok(()) };

        let (equalized, level) =
            utils234::eqlz2meddis_hc_level(&signal, Some(out_level), None).unwrap();
        check_array("level-equalized signal", equalized.as_slice().unwrap(), &[signal.len()], &expected["equalized"], 2e-11, 3e-13)?;
        check_values("level metadata", &level, &expected["level"], 3e-12, 3e-13)?;
        let (window, name) = utils234::taper_window(window_length, window_kind, Some(taper_length), sigma).unwrap();
        prop_assert_eq!(name, expected["window_name"].as_str().unwrap());
        check_array("taper window", window.as_slice().unwrap(), &[window_length], &expected["window"], 3e-14, 3e-13)?;
        let filtered = utils234::fftfilt(&coefficients, &signal);
        check_array("FFT filtering", filtered.as_slice().unwrap(), &[signal.len()], &expected["filtered"], 2e-12, 4e-12)?;
        let (cep, minimum) = utils234::rceps(&cepstrum).unwrap();
        check_values("real cepstrum", cep.as_slice().unwrap(), &expected["cepstrum"], 5e-12, 5e-12)?;
        check_values("minimum-phase signal", minimum.as_slice().unwrap(), &expected["minimum_phase"], 8e-12, 8e-12)?;
        let (frames, centers) = fb234::set_frame4time_sequence(&signal, frame_length, Some(frame_shift)).unwrap();
        check_array("framed signal", &flattened(frames.view()), &[frames.nrows(), frames.ncols()], &expected["frames"], 3e-15, 3e-15)?;
        let centers: Vec<f64> = centers.iter().map(|&value| value as f64).collect();
        check_values("frame centers", &centers, &expected["centers"], 0.0, 0.0)?;
    }

    #[test]
    fn gammachirp_impulses_match_python(
        fs_index in 0usize..4,
        frequency_ratio in 0.015f64..0.35,
        order in 2.0f64..6.0,
        bandwidth in 0.65f64..2.2,
        chirp in -4.0f64..3.0,
        phase in -std::f64::consts::PI..std::f64::consts::PI,
        carrier_index in 0u8..3,
        normalize_peak in any::<bool>(),
    ) {
        let fs = [8_000.0, 16_000.0, 24_000.0, 48_000.0][fs_index];
        let frequency = frequency_ratio * fs;
        let (rust_carrier, python_carrier) = carrier(carrier_index);
        let (rust_normalization, python_normalization) = normalization(normalize_peak);
        let Some(expected) = oracle(json!({
            "op": "gammachirp_impulse", "frequency": frequency, "fs": fs,
            "order": order, "bandwidth": bandwidth, "chirp": chirp, "phase": phase,
            "carrier": python_carrier, "normalization": python_normalization,
        })) else { return Ok(()) };
        let output = gc::gammachirp(&[frequency], fs, order, bandwidth, chirp, phase, rust_carrier, rust_normalization).unwrap();
        // SciPy's FFT-based peak normalization and Rust's direct evaluation of
        // the same frequency bin accumulate floating-point error differently.
        check_array("gammachirp impulse", &flattened(output.gc.view()), &[output.gc.nrows(), output.gc.ncols()], &expected["gc"], 1e-10, 1e-8)?;
        check_values("impulse length", &[output.len_gc[0] as f64], &expected["length"], 0.0, 0.0)?;
        check_values("impulse peak", output.fps.as_slice().unwrap(), &expected["peak"], 3e-10, 3e-13)?;
        check_array("instantaneous frequency", &flattened(output.inst_freq.view()), &[output.inst_freq.nrows(), output.inst_freq.ncols()], &expected["instantaneous_frequency"], 3e-9, 5e-13)?;
    }

    #[test]
    fn analytic_gammachirp_responses_match_python(
        fs_index in 0usize..4,
        frequency_ratios in prop::collection::vec(0.01f64..0.42, 1..5),
        order in 2.0f64..6.0,
        bandwidth in 0.65f64..2.2,
        chirp in -4.0f64..3.0,
        phase in -std::f64::consts::PI..std::f64::consts::PI,
        bins_index in 0usize..2,
    ) {
        let fs = [8_000.0, 16_000.0, 24_000.0, 48_000.0][fs_index];
        let frequencies: Vec<f64> = frequency_ratios.iter().map(|ratio| ratio * fs).collect();
        let bins = [256usize, 512][bins_index];
        let Some(expected) = oracle(json!({
            "op": "gammachirp_response", "frequencies": frequencies, "fs": fs,
            "order": order, "bandwidth": bandwidth, "chirp": chirp,
            "phase": phase, "bins": bins,
        })) else { return Ok(()) };
        let output = gc::gammachirp_frsp(&frequencies, fs, order, &[bandwidth], &[chirp], phase, bins).unwrap();
        check_array("response amplitude", &flattened(output.amp_frsp.view()), &[frequencies.len(), bins], &expected["amplitude"], 8e-12, 8e-12)?;
        check_values("response frequency", output.freq.as_slice().unwrap(), &expected["frequency"], 2e-12, 2e-13)?;
        check_values("response peak", output.f_peak.as_slice().unwrap(), &expected["peak"], 3e-10, 3e-13)?;
        check_array("response group delay", &flattened(output.grp_dly.view()), &[frequencies.len(), bins], &expected["group_delay"], 2e-13, 8e-12)?;
        check_array("response phase", &flattened(output.phs_frsp.view()), &[frequencies.len(), bins], &expected["phase"], 2e-11, 8e-12)?;
    }

    #[test]
    fn asymmetric_coefficients_responses_and_state_match_python(
        fs_index in 0usize..3,
        channels in 1usize..4,
        raw in prop::collection::vec((0.02f64..0.35, 1.2f64..3.0, 0.2f64..3.5), 3),
        sequence in prop::collection::vec(prop::collection::vec(-2.0f64..2.0, 3), 1..16),
        reverse in any::<bool>(),
    ) {
        let fs = [8_000.0, 16_000.0, 48_000.0][fs_index];
        let frequencies: Vec<f64> = raw[..channels].iter().map(|value| value.0 * fs).collect();
        let bandwidth: Vec<f64> = raw[..channels].iter().map(|value| value.1).collect();
        let chirp: Vec<f64> = raw[..channels].iter().map(|value| value.2).collect();
        let samples: Vec<Vec<f64>> = sequence.iter().map(|row| row[..channels].to_vec()).collect();
        let Some(expected) = oracle(json!({
            "op": "asymmetric_filters", "frequencies": frequencies, "fs": fs,
            "bandwidth": bandwidth, "chirp": chirp, "bins": 256,
            "samples": samples, "reverse": reverse,
        })) else { return Ok(()) };
        let coefficients = fb234::make_asym_cmp_filters_v2(fs, &frequencies, &bandwidth, &chirp).unwrap();
        check_array("asymmetric poles", coefficients.ap.as_slice().unwrap(), &[channels, 3, 4], &expected["ap"], 8e-13, 8e-13)?;
        check_array("asymmetric zeros", coefficients.bz.as_slice().unwrap(), &[channels, 3, 4], &expected["bz"], 8e-13, 8e-13)?;
        let response = fb234::asym_cmp_frsp_v2(&frequencies, fs, &bandwidth, &chirp, 256, 4).unwrap();
        check_array("asymmetric response", &flattened(response.acf_frsp.view()), &[channels, 256], &expected["response"], 2e-9, 3e-9)?;
        check_values("asymmetric response frequency", response.freq.as_slice().unwrap(), &expected["response_frequency"], 2e-12, 2e-13)?;
        check_array("asymmetric function", &flattened(response.asym_func.view()), &[channels, 256], &expected["asymmetry"], 2e-11, 2e-11)?;
        let mut status = AcfStatus::new(&coefficients);
        let mut actual = Vec::new();
        for row in &samples { actual.extend(status.process(&coefficients, row, reverse).unwrap()); }
        check_array("stateful asymmetric filtering", &actual, &[samples.len(), channels], &expected["sequence"], 3e-10, 3e-9)?;
    }

    #[test]
    fn compressed_responses_match_python(
        fs_index in 0usize..3,
        channels in 1usize..4,
        raw in prop::collection::vec((0.025f64..0.30, 1.3f64..2.2, -3.8f64..-1.0, 0.65f64..1.25, 1.5f64..2.8, 1.0f64..3.0), 3),
        order in 3.0f64..5.0,
    ) {
        let fs = [8_000.0, 16_000.0, 48_000.0][fs_index];
        let rows = &raw[..channels];
        let frequencies: Vec<f64> = rows.iter().map(|v| v.0 * fs).collect();
        let b1: Vec<f64> = rows.iter().map(|v| v.1).collect();
        let c1: Vec<f64> = rows.iter().map(|v| v.2).collect();
        let ratio: Vec<f64> = rows.iter().map(|v| v.3).collect();
        let b2: Vec<f64> = rows.iter().map(|v| v.4).collect();
        let c2: Vec<f64> = rows.iter().map(|v| v.5).collect();
        let Some(expected) = oracle(json!({
            "op": "compressed_response", "frequencies": frequencies, "fs": fs,
            "order": order, "b1": b1, "c1": c1, "ratio": ratio,
            "b2": b2, "c2": c2, "bins": 256,
        })) else { return Ok(()) };
        let output = fb234::cmprs_gc_frsp(&frequencies, fs, order, &b1, &c1, &ratio, &b2, &c2, 256).unwrap();
        for (label, actual, key, atol, rtol) in [
            ("compressed PGC", output.pgc_frsp.as_slice().unwrap(), "pgc", 2e-11, 2e-11),
            ("compressed CGC", output.cgc_frsp.as_slice().unwrap(), "cgc", 3e-10, 3e-10),
            ("normalized compressed response", output.cgc_nrm_frsp.as_slice().unwrap(), "normalized", 3e-10, 3e-10),
            ("compressed ACF", output.acf_frsp.as_slice().unwrap(), "acf", 2e-9, 3e-9),
            ("compressed asymmetry", output.asym_func.as_slice().unwrap(), "asymmetry", 2e-11, 2e-11),
        ] { check_array(label, actual, &[channels, 256], &expected[key], atol, rtol)?; }
        for (label, actual, key) in [
            ("fp1", output.fp1.as_slice().unwrap(), "fp1"),
            ("fr2", output.fr2.as_slice().unwrap(), "fr2"),
            ("fp2", output.fp2.as_slice().unwrap(), "fp2"),
            ("peak value", output.val_fp2.as_slice().unwrap(), "peak_value"),
            ("normalization", output.norm_fct_fp2.as_slice().unwrap(), "normalization"),
        ] { check_values(label, actual, &expected[key], 3e-8, 3e-9)?; }
        check_values("compressed frequency", output.freq.as_slice().unwrap(), &expected["frequency"], 2e-12, 2e-13)?;
    }

    #[test]
    fn peak_frequency_conversions_match_python(
        order in 3.0f64..5.0,
        b1 in 1.4f64..2.1,
        c1 in -3.6f64..-1.5,
        b2 in 1.7f64..2.7,
        c2 in 1.3f64..3.0,
        ratio in 0.72f64..1.15,
        fr1 in 100.0f64..5_000.0,
    ) {
        let Some(expected) = oracle(json!({
            "op": "frequency_conversion", "order": order, "b1": b1, "c1": c1,
            "b2": b2, "c2": c2, "ratio": ratio, "fr1": fr1,
        })) else { return Ok(()) };
        let (peak, second_center) = fb234::fr1_to_fp2(order, b1, c1, b2, c2, ratio, fr1).unwrap();
        let (inverse_center, inverse_peak) = fb234::fp2_to_fr1(order, b1, c1, b2, c2, ratio, peak).unwrap();
        check_scalar("compressive peak", peak, &expected["peak"], 1e-4, 5e-8)?;
        check_scalar("second center", second_center, &expected["second_center"], 1e-8, 2e-11)?;
        check_scalar("inverse center", inverse_center, &expected["inverse_center"], 1e-4, 5e-8)?;
        check_scalar("inverse peak", inverse_peak, &expected["inverse_peak"], 1e-4, 5e-8)?;
    }

    #[test]
    fn v234_interpolation_calibration_floor_and_hearing_level_match_python(
        y in prop::collection::vec(-20.0f64..20.0, 4),
        query_positions in prop::collection::vec(-0.5f64..1.5, 1..12),
        signal in prop::collection::vec(-2.0f64..2.0, 1..30),
        out_level in -10.0f64..120.0,
        rms1_spl in 30.0f64..120.0,
        precise in any::<bool>(),
        floor_values in prop::collection::vec(-1.0f64..2.0, 6),
        zero_floor in any::<bool>(),
        hl_index in 0usize..7,
        hl_db in -20.0f64..100.0,
    ) {
        let signal = non_silent(signal);
        let x = [0.0, 0.5, 2.0, 5.0];
        let queries: Vec<f64> = query_positions.iter().map(|q| x[0] + q * (x[3] - x[0])).collect();
        let extrapolate = query_positions.iter().any(|q| !(0.0..=1.0).contains(q));
        let input_rms1_dbspl = precise.then_some(rms1_spl);
        let out_level_db = (!precise).then_some(out_level);
        let hl_frequency = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0][hl_index];
        let Some(expected) = oracle(json!({
            "op": "v234_utils", "x": x, "y": y, "queries": queries,
            "extrapolate": extrapolate, "signal": signal,
            "out_level_db": out_level_db, "input_rms1_dbspl": input_rms1_dbspl,
            "floor_values": floor_values, "floor_shape": [2, 3],
            "floor": if zero_floor { "zero" } else { "none" },
            "hl_frequency": hl_frequency, "hl_db": hl_db,
        })) else { return Ok(()) };
        let interpolated = utils234::interp1(&x, &y, &queries, extrapolate).unwrap();
        check_array("v234 interpolation", interpolated.as_slice().unwrap(), &[queries.len()], &expected["interpolated"], 3e-13, 3e-13)?;
        let (equalized, level) = utils234::eqlz2meddis_hc_level(&signal, out_level_db, input_rms1_dbspl).unwrap();
        check_array("v234 level equalization", equalized.as_slice().unwrap(), &[signal.len()], &expected["equalized"], 3e-11, 3e-13)?;
        check_values("v234 level metadata", &level, &expected["level"], 3e-11, 3e-13)?;
        let floor_input = Array2::from_shape_vec((2, 3), floor_values).unwrap();
        let floor = utils234::eqlz_gcfb2rms1_at_0db(&floor_input, if zero_floor { Floor::ZeroFloor } else { Floor::None });
        check_array("v234 absolute threshold scaling", floor.as_slice().unwrap(), &[2, 3], &expected["floor"], 3e-13, 3e-13)?;
        check_scalar("HL to SPL", utils234::hl2spl(hl_frequency, hl_db).unwrap(), &expected["spl"], 1e-13, 1e-13)?;
        check_scalar("HL to cochlear input", utils234::hl2pin_cochlea(hl_frequency, hl_db).unwrap(), &expected["cochlea"], 1e-12, 1e-13)?;
    }

    #[test]
    fn field_to_cochlea_transfer_functions_match_python(
        fs_index in 0usize..4,
        bins_index in 0usize..4,
        calibration_ratio in 0.02f64..0.45,
        field_index in 0usize..4,
    ) {
        let fs = [8_000.0, 16_000.0, 32_000.0, 48_000.0][fs_index];
        let bins = [32usize, 64, 128, 256][bins_index];
        let calibration = calibration_ratio * fs;
        let fields = ["FreeField", "DiffuseField", "ITU", "NoField"];
        let field = fields[field_index];
        let Some(expected) = oracle(json!({
            "op": "field_transfer", "fs": fs, "bins": bins,
            "calibration": calibration, "field": field,
        })) else { return Ok(()) };
        let output = utils234::trans_func_field2cochlea(&ParamTransFunc {
            fs, n_frq_rsl: bins, freq_calib: calibration,
            type_field2eardrum: field.into(), type_midear2cochlea: "MiddleEar".into(),
            ..ParamTransFunc::default()
        }).unwrap();
        for (label, actual, key) in [
            ("transfer frequency", output.freq.as_slice().unwrap(), "frequency"),
            ("field transfer", output.field2eardrum_db.as_slice().unwrap(), "field"),
            ("middle-ear transfer", output.midear2cochlea_db.as_slice().unwrap(), "middle"),
            ("total transfer", output.field2cochlea_db.as_slice().unwrap(), "total"),
        ] { check_values(label, actual, &expected[key], 8e-10, 8e-10)?; }
        for (label, actual, key) in [
            ("calibration frequency", output.freq_calib, "frequency_calibration"),
            ("field at calibration", output.field2eardrum_db_at_freq_calib, "field_at_calibration"),
            ("field compensation", output.field2eardrum_db_cmpnst_db, "field_compensation"),
            ("middle at calibration", output.midear2cochlea_db_at_freq_calib, "middle_at_calibration"),
            ("total at calibration", output.field2cochlea_db_at_freq_calib, "total_at_calibration"),
        ] { check_scalar(label, actual, &expected[key], 8e-10, 8e-10)?; }
    }

    #[test]
    fn v234_modulation_filterbank_matches_python(
        fs in 500.0f64..5_000.0,
        frequency_ratios in prop::collection::vec(0.001f64..0.20, 1..7),
        signal in prop::collection::vec(-2.0f64..2.0, 2..48),
    ) {
        let center_frequencies: Vec<f64> = frequency_ratios.iter().map(|ratio| ratio * fs).collect();
        let Some(expected) = oracle(json!({
            "op": "modulation_filterbank", "fs": fs,
            "center_frequencies": center_frequencies, "signal": signal,
        })) else { return Ok(()) };
        let parameters = EmParam {
            fs,
            fc_mod_list: ndarray::Array1::from(center_frequencies),
            ..EmParam::default()
        };
        let output = fb234::gcfb_v23_env_mod_fb(&signal, &parameters).unwrap();
        check_array(
            "v234 modulation filterbank",
            output.as_slice().unwrap(),
            &[frequency_ratios.len(), signal.len()],
            &expected,
            5e-11,
            5e-10,
        )?;
    }
}

proptest! {
    #![proptest_config(property_config(48))]

    #[test]
    fn envelope_modulation_loss_matches_python_for_generated_envelopes(
        fs_index in 0usize..2,
        raw_envelope in prop::collection::vec(-1.0f64..1.0, 8..32),
        reduce_db in prop::collection::vec(0.0f64..30.0, 7),
        f_cutoff in prop::collection::vec(8.0f64..400.0, 7),
        low in 150.0f64..450.0,
        high in 1_200.0f64..3_200.0,
    ) {
        let fs = [8_000.0, 16_000.0][fs_index];
        let frames = Array2::from_shape_fn((4, raw_envelope.len()), |(channel, frame)| {
            1.2 + 0.2 * channel as f64
                + 0.25 * raw_envelope[frame]
                + (0.04 + 0.01 * channel as f64)
                    * (2.0 * std::f64::consts::PI * (channel + 1) as f64 * frame as f64
                        / raw_envelope.len() as f64)
                        .sin()
        });
        let encoded_frames: Vec<Vec<f64>> =
            frames.rows().into_iter().map(|row| row.to_vec()).collect();
        let Some(expected) = oracle(json!({
            "op": "envelope_modulation_loss",
            "frames": encoded_frames,
            "fs": fs,
            "f_range": [low, high],
            "reduce_db": reduce_db,
            "f_cutoff": f_cutoff,
        })) else { return Ok(()) };
        let (parameters, _) = fb234::set_param(v234_param(
            fs,
            frames.nrows(),
            [low, high],
            "NH",
        )).unwrap();
        let (output, em) = fb234::gcfb_v23_env_mod_loss(
            &frames,
            &parameters,
            EmParam {
                reduce_db: ndarray::Array1::from(reduce_db),
                f_cutoff: ndarray::Array1::from(f_cutoff),
                ..EmParam::default()
            },
        ).unwrap();
        check_array(
            "envelope modulation loss",
            &flattened(output.view()),
            &[frames.nrows(), frames.ncols()],
            &expected["output"],
            8e-12,
            8e-12,
        )?;
        check_scalar("envelope modulation sampling rate", em.fs, &expected["fs"], 0.0, 0.0)?;
        check_values("envelope filterbank frequencies", em.fb_fr1.as_slice().unwrap(), &expected["fb_fr1"], 3e-9, 3e-12)?;
        check_values("interpolated envelope reductions", em.fb_reduce_db.as_slice().unwrap(), &expected["fb_reduce_db"], 3e-11, 3e-12)?;
        check_values("interpolated envelope cutoffs", em.fb_f_cutoff.as_slice().unwrap(), &expected["fb_f_cutoff"], 3e-10, 3e-12)?;
    }
}

fn v234_param(fs: f64, channels: usize, f_range: [f64; 2], hearing_loss: &str) -> Param234 {
    Param234 {
        fs,
        num_ch: channels,
        f_range,
        out_mid_crct: "No".into(),
        ctrl: Control234::Dynamic,
        dyn_hpaf: DynHpaf {
            str_prc: "frame-base".into(),
            ..DynHpaf::default()
        },
        hloss_type: hearing_loss.into(),
        hloss_compression_health: (hearing_loss == "HL3").then_some(0.5),
        gain_ref: GainReference::NormalizeIoFunction,
        ..Param234::default()
    }
}

proptest! {
    #![proptest_config(property_config(12))]

    #[test]
    fn v234_dynamic_frame_filterbank_matches_python_for_generated_signals(
        signal in prop::collection::vec(-1.5f64..1.5, 8..40),
        channels in 3usize..7,
        low in 100.0f64..500.0,
        high in 2_700.0f64..3_300.0,
        impaired in any::<bool>(),
    ) {
        let signal = non_silent(signal);
        let hearing_loss = if impaired { "HL3" } else { "NH" };
        let Some(expected) = oracle(json!({
            "op": "v234_filterbank", "signal": signal, "fs": 8_000.0,
            "channels": channels, "f_range": [low, high], "hearing_loss": hearing_loss,
        })) else { return Ok(()) };
        let output = fb234::gcfb_v234(&signal, v234_param(8_000.0, channels, [low, high], hearing_loss)).unwrap();
        check_array("v234 dynamic CGC", output.dcgc_out.as_slice().unwrap(), &[output.dcgc_out.nrows(), output.dcgc_out.ncols()], &expected["dcgc"], 3e-8, 8e-8)?;
        check_array("v234 static CGC", output.scgc_smpl.as_slice().unwrap(), &[channels, signal.len()], &expected["scgc"], 3e-9, 3e-8)?;
        for (label, actual, key, atol) in [
            ("v234 fr2", output.gc_resp.fr2.as_slice().unwrap(), "fr2", 2e-5),
            ("v234 ratio", output.gc_resp.frat_val.as_slice().unwrap(), "ratio", 3e-8),
            ("v234 level", output.gc_resp.lvl_db.as_slice().unwrap(), "level", 3e-7),
            ("v234 frame level", output.gc_resp.lvl_db_frame.as_slice().unwrap(), "level_frame", 3e-7),
            ("v234 PGC frames", output.gc_resp.pgc_frame.as_slice().unwrap(), "pgc_frame", 3e-9),
            ("v234 static CGC frames", output.gc_resp.scgc_frame.as_slice().unwrap(), "scgc_frame", 3e-9),
            ("v234 asymmetry gain", output.gc_resp.asym_func_gain.as_slice().unwrap(), "asymmetry_gain", 3e-8),
        ] { check_values(label, actual, &expected[key], atol, 8e-8)?; }
        if !expected_values(&expected["gain"])?.is_empty() {
            check_values("v234 gain", output.gc_resp.gain_factor.as_slice().unwrap(), &expected["gain"], 3e-8, 8e-8)?;
        }
    }

    #[test]
    fn v234_asymmetric_input_output_mapping_matches_python(
        channels in 3usize..7,
        low in 100.0f64..500.0,
        high in 1_000.0f64..2_600.0,
        query_position in 0.0f64..1.0,
        health in 0.05f64..1.0,
        pins in prop::collection::vec(-100.0f64..150.0, 1..12),
        impaired in any::<bool>(),
    ) {
        let hearing_loss = if impaired { "HL3" } else { "NH" };
        let query_frequency = low + query_position * (high - low);
        let Some(expected) = oracle(json!({
            "op": "v234_asymmetric_io", "fs": 8_000.0, "channels": channels,
            "f_range": [low, high], "hearing_loss": hearing_loss,
            "query_frequency": query_frequency, "health": health, "pins": pins,
        })) else { return Ok(()) };
        let (param, response) = fb234::set_param(v234_param(
            8_000.0, channels, [low, high], hearing_loss,
        )).unwrap();
        let (asymmetry, output) = fb234::gcfb_v23_asym_func_in_out(
            &param, &response, query_frequency, health, &pins,
        );
        check_array("v234 asymmetric I/O gain", asymmetry.as_slice().unwrap(), &[pins.len()], &expected["asymmetry"], 3e-9, 3e-9)?;
        check_array("v234 asymmetric I/O output", output.as_slice().unwrap(), &[pins.len()], &expected["output"], 3e-9, 3e-9)?;
        let inverse: Vec<f64> = output.iter().map(|&value| {
            fb234::gcfb_v23_asym_func_in_out_inv_io_func(
                &param, &response, query_frequency, health, value,
            ).unwrap()
        }).collect();
        check_array("v234 inverse asymmetric I/O", &inverse, &[pins.len()], &expected["inverse"], 8e-7, 8e-8)?;
    }
}

#[test]
fn python_oracle_dependency_check() {
    // This test gives strict CI runs an immediate, clearly named failure when
    // the reference dependencies are absent.  Other tests reuse the process.
    let _ = oracle(json!({
        "op": "scales", "frequencies": [1000.0], "mel": [1000.0],
        "signal": [1.0], "integer": 1, "scale": "ERB",
        "channels": 2, "range": [100.0, 1000.0],
    }));
}
