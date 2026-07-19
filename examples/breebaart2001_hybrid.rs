//! Run a deterministic waveform-to-EI Breebaart/GCFB hybrid analysis.

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

use gammachirp_rs::breebaart2001::{
    EiUnit, HybridBinauralConfig, HybridBinauralOutput, hybrid_binaural,
};
use gammachirp_rs::gcfb_v234::{ControlMode, GainReference, GcParam};
use ndarray::{Array3, s};
use plotters::coord::Shift;
use plotters::prelude::*;

const SAMPLE_RATE_HZ: f64 = 16_000.0;
const STIMULUS_SAMPLES: usize = 8_192;
const STIMULUS_DELAY_SECONDS: f64 = 0.5e-3;
const EXPECTED_ITD_SECONDS: f64 = -STIMULUS_DELAY_SECONDS;
const EXPECTED_IID_DB: f64 = 0.0;
const INTERIOR_MARGIN_SAMPLES: usize = 2_400;
const DEFAULT_OUTPUT: &str = "target/breebaart2001_hybrid.png";
const IMAGE_SIZE: (u32, u32) = (1_200, 800);
const ITD_MS: [f64; 9] = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
const IID_DB: [f64; 5] = [-6.0, -3.0, 0.0, 3.0, 6.0];

struct Analysis {
    output: HybridBinauralOutput,
    mean_activity: Vec<f64>,
    averaging_window: Range<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let output_path = output_path(std::env::args_os().skip(1))?;
    ensure_png_path(&output_path)?;

    let analysis = run_analysis()?;
    let ranking = ranked_units(&analysis.mean_activity);
    let observed_index = ranking[0];
    let observed = analysis.output.units[observed_index];

    println!(
        "stimulus: deterministic broadband noise; right ear delayed by +{:.3} ms; IID {:+.1} dB",
        STIMULUS_DELAY_SECONDS * 1e3,
        EXPECTED_IID_DB,
    );
    println!(
        "expected EI minimum: characteristic ITD {:+.3} ms and IID {:+.1} dB (paper-symmetric EI convention)",
        EXPECTED_ITD_SECONDS * 1e3,
        EXPECTED_IID_DB,
    );
    println!(
        "EI output: {} units x {} frequency channels x {} samples",
        analysis.output.ei_map.len_of(ndarray::Axis(0)),
        analysis.output.ei_map.len_of(ndarray::Axis(1)),
        analysis.output.ei_map.len_of(ndarray::Axis(2)),
    );
    println!(
        "frequency range: {:.1}-{:.1} Hz",
        analysis.output.center_frequencies_hz[0],
        analysis.output.center_frequencies_hz[analysis.output.center_frequencies_hz.len() - 1],
    );
    println!(
        "averaging window: {:.3}-{:.3} s (all frequency channels)",
        analysis.averaging_window.start as f64 / SAMPLE_RATE_HZ,
        analysis.averaging_window.end as f64 / SAMPLE_RATE_HZ,
    );
    println!(
        "observed minimum: characteristic ITD {:+.3} ms, IID {:+.1} dB, mean activity {:.6} MU",
        observed.delay_seconds * 1e3,
        observed.iid_db,
        analysis.mean_activity[observed_index],
    );
    println!("five lowest mean responses:");
    for &index in ranking.iter().take(5) {
        let unit = analysis.output.units[index];
        println!(
            "  ITD {:+.3} ms, IID {:+.1} dB: {:.6} MU",
            unit.delay_seconds * 1e3,
            unit.iid_db,
            analysis.mean_activity[index],
        );
    }

    render_heatmap(
        &output_path,
        &analysis.mean_activity,
        (EXPECTED_ITD_SECONDS * 1e3, EXPECTED_IID_DB),
        (observed.delay_seconds * 1e3, observed.iid_db),
    )?;
    println!("wrote {}", output_path.display());
    Ok(())
}

fn output_path(mut arguments: impl Iterator<Item = OsString>) -> io::Result<PathBuf> {
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: breebaart2001_hybrid [output.png]",
        ));
    }
    Ok(output)
}

fn ensure_png_path(path: &Path) -> io::Result<()> {
    let is_png = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"));
    if is_png {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the output path must have a .png extension",
        ))
    }
}

fn delayed_broadband_stimulus() -> (Vec<f64>, Vec<f64>) {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let left: Vec<f64> = (0..STIMULUS_SAMPLES)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
            2.0 * bits as f64 / ((1_u64 << 53) - 1) as f64 - 1.0
        })
        .collect();
    let delay_samples = (STIMULUS_DELAY_SECONDS * SAMPLE_RATE_HZ).round() as usize;
    let mut right = vec![0.0; STIMULUS_SAMPLES];
    right[delay_samples..].copy_from_slice(&left[..STIMULUS_SAMPLES - delay_samples]);
    (left, right)
}

fn ei_population() -> Vec<EiUnit> {
    IID_DB
        .iter()
        .flat_map(|&iid_db| {
            ITD_MS
                .iter()
                .map(move |&itd_ms| EiUnit::new(itd_ms * 1e-3, iid_db))
        })
        .collect()
}

fn analysis_config() -> HybridBinauralConfig {
    let mut config = HybridBinauralConfig {
        filterbank: GcParam {
            fs: SAMPLE_RATE_HZ,
            num_ch: 24,
            f_range: [100.0, 6_000.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            gain_ref: GainReference::Db(50.0),
            ..GcParam::default()
        },
        ..HybridBinauralConfig::default()
    };
    config.peripheral.absolute_threshold_noise_level_db_spl = None;
    config.ei.internal_noise_std_mu = 0.0;
    config
}

fn run_analysis() -> gammachirp_rs::Result<Analysis> {
    let (left, right) = delayed_broadband_stimulus();
    let units = ei_population();
    let output = hybrid_binaural(&left, &right, &units, analysis_config())?;
    let averaging_window = INTERIOR_MARGIN_SAMPLES..STIMULUS_SAMPLES - INTERIOR_MARGIN_SAMPLES;
    let mean_activity = mean_activity(&output.ei_map, averaging_window.clone());
    Ok(Analysis {
        output,
        mean_activity,
        averaging_window,
    })
}

fn mean_activity(activity: &Array3<f64>, window: Range<usize>) -> Vec<f64> {
    let values_per_unit = activity.len_of(ndarray::Axis(1)) * window.len();
    (0..activity.len_of(ndarray::Axis(0)))
        .map(|unit| activity.slice(s![unit, .., window.clone()]).sum() / values_per_unit as f64)
        .collect()
}

fn ranked_units(mean_activity: &[f64]) -> Vec<usize> {
    let mut ranking: Vec<usize> = (0..mean_activity.len()).collect();
    ranking.sort_by(|&left, &right| mean_activity[left].total_cmp(&mean_activity[right]));
    ranking
}

fn render_heatmap(
    output_path: &Path,
    mean_activity: &[f64],
    expected: (f64, f64),
    observed: (f64, f64),
) -> Result<(), Box<dyn Error>> {
    if mean_activity.len() != ITD_MS.len() * IID_DB.len()
        || mean_activity.iter().any(|value| !value.is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the EI heatmap requires one finite value for every ITD-IID unit",
        )
        .into());
    }
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let minimum = mean_activity.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = mean_activity
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let root = BitMapBackend::new(output_path, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let body = root.titled(
        "Breebaart/GCFB EI cue tuning (white X: expected; cyan ring: observed)",
        ("sans-serif", 28),
    )?;
    let (heatmap_area, colorbar_area) = body.split_horizontally(1_040);
    draw_heatmap(
        &heatmap_area,
        mean_activity,
        minimum,
        maximum,
        expected,
        observed,
    )?;
    draw_colorbar(&colorbar_area, minimum, maximum)?;
    root.present()?;
    Ok(())
}

fn draw_heatmap(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    mean_activity: &[f64],
    minimum: f64,
    maximum: f64,
    expected: (f64, f64),
    observed: (f64, f64),
) -> Result<(), Box<dyn Error>> {
    const HALF_ITD_STEP_MS: f64 = 0.125;
    const HALF_IID_STEP_DB: f64 = 1.5;

    let mut chart = ChartBuilder::on(area)
        .caption(
            "Mean EI activity across frequency and the interior time window",
            ("sans-serif", 21),
        )
        .margin(18)
        .x_label_area_size(58)
        .y_label_area_size(66)
        .build_cartesian_2d(-1.125_f64..1.125_f64, -7.5_f64..7.5_f64)?;
    chart
        .configure_mesh()
        .x_desc("Characteristic ITD (ms)")
        .y_desc("Characteristic IID (dB)")
        .x_labels(9)
        .y_labels(5)
        .light_line_style(WHITE.mix(0.3))
        .draw()?;

    chart.draw_series(IID_DB.iter().enumerate().flat_map(|(iid_index, &iid)| {
        ITD_MS.iter().enumerate().map(move |(itd_index, &itd)| {
            let index = iid_index * ITD_MS.len() + itd_index;
            Rectangle::new(
                [
                    (itd - HALF_ITD_STEP_MS, iid - HALF_IID_STEP_DB),
                    (itd + HALF_ITD_STEP_MS, iid + HALF_IID_STEP_DB),
                ],
                activity_color(mean_activity[index], minimum, maximum).filled(),
            )
        })
    }))?;
    chart.draw_series(std::iter::once(Cross::new(
        expected,
        18,
        ShapeStyle::from(&WHITE).stroke_width(4),
    )))?;
    chart.draw_series(std::iter::once(Circle::new(
        observed,
        14,
        ShapeStyle::from(&CYAN).stroke_width(4),
    )))?;
    Ok(())
}

fn draw_colorbar(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    minimum: f64,
    maximum: f64,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption("Mean EI\nactivity (MU)", ("sans-serif", 17))
        .margin_top(70)
        .margin_bottom(60)
        .margin_left(18)
        .margin_right(24)
        .y_label_area_size(82)
        .build_cartesian_2d(0.0..1.0, minimum..maximum)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(7)
        .y_label_style(("sans-serif", 22))
        .y_label_formatter(&|value| format!("{value:.3}"))
        .draw()?;
    chart.draw_series((0..120).map(|step| {
        let lower = minimum + (maximum - minimum) * step as f64 / 120.0;
        let upper = minimum + (maximum - minimum) * (step + 1) as f64 / 120.0;
        Rectangle::new(
            [(0.0, lower), (1.0, upper)],
            activity_color(lower, minimum, maximum).filled(),
        )
    }))?;
    Ok(())
}

fn activity_color(value: f64, minimum: f64, maximum: f64) -> HSLColor {
    let fraction = if maximum > minimum {
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    HSLColor(0.72 - 0.72 * fraction, 0.85, 0.16 + 0.48 * fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn stimulus_delays_the_right_ear_by_exactly_half_a_millisecond() {
        let (left, right) = delayed_broadband_stimulus();
        let delay_samples = (STIMULUS_DELAY_SECONDS * SAMPLE_RATE_HZ).round() as usize;
        assert_eq!(delay_samples, 8);
        assert_eq!(left.len(), STIMULUS_SAMPLES);
        assert_eq!(right.len(), STIMULUS_SAMPLES);
        assert!(right[..delay_samples].iter().all(|&sample| sample == 0.0));
        assert_eq!(right[delay_samples..], left[..left.len() - delay_samples]);
    }

    #[test]
    fn ei_grid_is_iid_major_and_itd_minor() {
        let units = ei_population();
        assert_eq!(units.len(), IID_DB.len() * ITD_MS.len());
        assert_eq!(units[0], EiUnit::new(-1.0e-3, -6.0));
        assert_eq!(units[ITD_MS.len() - 1], EiUnit::new(1.0e-3, -6.0));
        assert_eq!(units[ITD_MS.len()], EiUnit::new(-1.0e-3, -3.0));
        assert_eq!(units[units.len() - 1], EiUnit::new(1.0e-3, 6.0));
    }

    #[test]
    fn output_argument_is_optional_but_unique_and_png() {
        assert_eq!(
            output_path(std::iter::empty()).unwrap(),
            PathBuf::from(DEFAULT_OUTPUT)
        );
        let custom = output_path([OsString::from("map.PNG")].into_iter()).unwrap();
        ensure_png_path(&custom).unwrap();
        assert!(
            output_path([OsString::from("one.png"), OsString::from("two.png")].into_iter())
                .is_err()
        );
        assert!(ensure_png_path(Path::new("map.jpg")).is_err());
    }

    #[test]
    fn deterministic_hybrid_minimum_matches_the_stimulus_cues() {
        let analysis = run_analysis().unwrap();
        let best = ranked_units(&analysis.mean_activity)[0];
        assert_eq!(
            analysis.output.units[best],
            EiUnit::new(EXPECTED_ITD_SECONDS, EXPECTED_IID_DB)
        );
    }

    #[test]
    fn renderer_writes_a_png_and_creates_parent_directories() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "gammachirp-breebaart-example-{}-{nonce}",
            std::process::id()
        ));
        let output = directory.join("nested/heatmap.png");
        let values: Vec<f64> = (0..ITD_MS.len() * IID_DB.len())
            .map(|index| index as f64)
            .collect();

        render_heatmap(&output, &values, (-0.5, 0.0), (-0.5, 0.0)).unwrap();
        let bytes = fs::read(&output).unwrap();
        assert!(bytes.starts_with(&[137, b'P', b'N', b'G', 13, 10, 26, 10]));
        fs::remove_dir_all(directory).unwrap();
    }
}
