//! Render matched GCFB spectrograms before and after time-frequency reassignment.

use std::error::Error;
use std::f64::consts::PI;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use gammachirpy::gcfb_v234::{
    ControlMode, GainReference, GcParam, gcfb_v234_with_phase_reassignment,
};
use ndarray::{Array1, Array2};
use plotters::coord::Shift;
use plotters::prelude::*;

const IMAGE_SIZE: (u32, u32) = (1600, 720);
const DB_FLOOR: f64 = -60.0;
const DEFAULT_OUTPUT: &str = "target/v234_reassignment_spectrogram.png";

fn main() -> Result<(), Box<dyn Error>> {
    let output_path = output_path(std::env::args_os().skip(1))?;
    ensure_png_path(&output_path)?;
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let (input, sample_rate) = example_signal();
    let parameters = GcParam {
        fs: sample_rate,
        num_ch: 32,
        f_range: [180.0, 3000.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        gain_ref: GainReference::Db(50.0),
        ..GcParam::default()
    };
    let (_, phase) = gcfb_v234_with_phase_reassignment(&input, parameters)?;
    let comparison = phase.sparsity_comparison()?;

    render_comparison(
        &output_path,
        &phase.unreassigned_energy_map,
        &phase.reassignment.energy_map,
        &phase.reassignment.time_axis,
        &phase.reassignment.frequency_axis_erb,
    )?;

    println!("wrote {}", output_path.display());
    println!(
        "matched retained energy: unreassigned={:.6e}, reassigned={:.6e}",
        phase.unreassigned_energy_map.sum(),
        phase.reassignment.retained_energy(),
    );
    println!(
        "effective support: unreassigned={:.1} bins, reassigned={:.1} bins",
        comparison.unreassigned.effective_bins, comparison.reassigned.effective_bins,
    );
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
            "usage: v234_reassignment_spectrogram [output.png]",
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

fn example_signal() -> (Vec<f64>, f64) {
    let sample_rate = 16_000.0;
    let samples = 4096;
    let duration = samples as f64 / sample_rate;
    let chirp_start_hz = 350.0;
    let chirp_end_hz = 1800.0;
    let chirp_rate = (chirp_end_hz - chirp_start_hz) / duration;
    let mut noise_state = 0x243f_6a88_u32;

    let mut input: Vec<f64> = (0..samples)
        .map(|sample| {
            let time = sample as f64 / sample_rate;
            noise_state = noise_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise = (f64::from(noise_state) / f64::from(u32::MAX)) * 2.0 - 1.0;
            let chirp_phase = 2.0 * PI * (chirp_start_hz * time + 0.5 * chirp_rate * time.powi(2));
            0.20 * (2.0 * PI * 440.0 * time).cos()
                + 0.12 * (2.0 * PI * 1100.0 * time).cos()
                + 0.16 * chirp_phase.cos()
                + 0.015 * noise
        })
        .collect();
    input[960] += 1.0;
    input[2880] += 1.0;
    (input, sample_rate)
}

fn render_comparison(
    output_path: &Path,
    unreassigned: &Array2<f64>,
    reassigned: &Array2<f64>,
    time_axis: &Array1<f64>,
    frequency_axis_erb: &Array1<f64>,
) -> Result<(), Box<dyn Error>> {
    let expected_dimensions = (frequency_axis_erb.len(), time_axis.len());
    if unreassigned.dim() != expected_dimensions || reassigned.dim() != expected_dimensions {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram maps and axes have incompatible dimensions",
        )
        .into());
    }

    let maximum = unreassigned
        .iter()
        .chain(reassigned.iter())
        .copied()
        .fold(0.0_f64, f64::max);
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram maps must contain positive finite energy",
        )
        .into());
    }

    let time_edges = bin_edges(time_axis.as_slice().unwrap(), Some(0.0))?;
    let frequency_edges = bin_edges(frequency_axis_erb.as_slice().unwrap(), None)?;
    let root = BitMapBackend::new(output_path, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let body = root.titled("GCFB v2.34 Time-Frequency Reassignment", ("sans-serif", 30))?;
    let (panel_area, colorbar_area) = body.split_horizontally(1400);
    let panels = panel_area.split_evenly((1, 2));

    draw_spectrogram(
        &panels[0],
        "Without reassignment (matched energy)",
        unreassigned,
        &time_edges,
        &frequency_edges,
        maximum,
    )?;
    draw_spectrogram(
        &panels[1],
        "With time-frequency reassignment",
        reassigned,
        &time_edges,
        &frequency_edges,
        maximum,
    )?;
    draw_colorbar(&colorbar_area)?;
    root.present()?;
    Ok(())
}

fn bin_edges(centers: &[f64], lower_bound: Option<f64>) -> io::Result<Vec<f64>> {
    if centers.len() < 2
        || centers
            .windows(2)
            .any(|pair| !pair[0].is_finite() || pair[1] <= pair[0])
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram axes must have at least two finite increasing values",
        ));
    }
    let mut edges = Vec::with_capacity(centers.len() + 1);
    edges.push(centers[0] - (centers[1] - centers[0]) / 2.0);
    edges.extend(centers.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0));
    let last = centers.len() - 1;
    edges.push(centers[last] + (centers[last] - centers[last - 1]) / 2.0);
    if let Some(bound) = lower_bound {
        edges[0] = edges[0].max(bound);
    }
    Ok(edges)
}

fn draw_spectrogram(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    energy: &Array2<f64>,
    time_edges: &[f64],
    frequency_edges: &[f64],
    maximum: f64,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 22))
        .margin(12)
        .x_label_area_size(45)
        .y_label_area_size(75)
        .build_cartesian_2d(
            time_edges[0]..time_edges[time_edges.len() - 1],
            frequency_edges[0]..frequency_edges[frequency_edges.len() - 1],
        )?;
    chart.plotting_area().fill(&heat_color(DB_FLOOR))?;
    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Auditory frequency (Hz)")
        .x_labels(6)
        .y_labels(8)
        .y_label_formatter(&|erb_rate| format!("{:.0}", erb_to_frequency(*erb_rate)))
        .light_line_style(WHITE.mix(0.22))
        .draw()?;

    chart.draw_series((0..energy.nrows()).flat_map(|channel| {
        (0..energy.ncols()).map(move |time| {
            let db = energy_db(energy[[channel, time]], maximum);
            Rectangle::new(
                [
                    (time_edges[time], frequency_edges[channel]),
                    (time_edges[time + 1], frequency_edges[channel + 1]),
                ],
                heat_color(db).filled(),
            )
        })
    }))?;
    Ok(())
}

fn draw_colorbar(area: &DrawingArea<BitMapBackend<'_>, Shift>) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption("Energy (dB)", ("sans-serif", 17))
        .margin_top(65)
        .margin_bottom(57)
        .margin_left(20)
        .margin_right(45)
        .y_label_area_size(42)
        .build_cartesian_2d(0.0..1.0, DB_FLOOR..0.0)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(7)
        .y_label_formatter(&|db| format!("{db:.0}"))
        .draw()?;
    chart.draw_series((0..120).map(|step| {
        let lower = DB_FLOOR + -DB_FLOOR * step as f64 / 120.0;
        let upper = DB_FLOOR + -DB_FLOOR * (step + 1) as f64 / 120.0;
        Rectangle::new([(0.0, lower), (1.0, upper)], heat_color(lower).filled())
    }))?;
    Ok(())
}

fn energy_db(energy: f64, maximum: f64) -> f64 {
    if energy > 0.0 {
        (10.0 * (energy / maximum).log10()).clamp(DB_FLOOR, 0.0)
    } else {
        DB_FLOOR
    }
}

fn erb_to_frequency(erb_rate: f64) -> f64 {
    (10_f64.powf(erb_rate / 21.4) - 1.0) * 1000.0 / 4.37
}

fn heat_color(db: f64) -> RGBColor {
    const STOPS: [(f64, [u8; 3]); 5] = [
        (0.00, [0, 0, 4]),
        (0.25, [81, 18, 124]),
        (0.50, [183, 55, 121]),
        (0.75, [252, 137, 97]),
        (1.00, [252, 253, 191]),
    ];
    let position = ((db - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0);
    let upper = STOPS
        .iter()
        .position(|(stop, _)| *stop >= position)
        .unwrap_or(STOPS.len() - 1);
    if upper == 0 {
        return RGBColor(STOPS[0].1[0], STOPS[0].1[1], STOPS[0].1[2]);
    }
    let (lower_position, lower_color) = STOPS[upper - 1];
    let (upper_position, upper_color) = STOPS[upper];
    let fraction = (position - lower_position) / (upper_position - lower_position);
    let interpolate = |lower: u8, upper: u8| {
        (f64::from(lower) + fraction * (f64::from(upper) - f64::from(lower))).round() as u8
    };
    RGBColor(
        interpolate(lower_color[0], upper_color[0]),
        interpolate(lower_color[1], upper_color[1]),
        interpolate(lower_color[2], upper_color[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_decibel_scale_maps_the_joint_peak_and_floor() {
        assert_eq!(energy_db(4.0, 4.0), 0.0);
        assert_eq!(energy_db(0.0, 4.0), DB_FLOOR);
        assert_eq!(energy_db(4e-7, 4.0), DB_FLOOR);
        assert!((energy_db(0.4, 4.0) + 10.0).abs() < 1e-12);
    }

    #[test]
    fn output_argument_is_optional_but_unique_and_png() {
        let default = output_path(std::iter::empty()).unwrap();
        assert_eq!(default, PathBuf::from(DEFAULT_OUTPUT));

        let custom = output_path([OsString::from("comparison.png")].into_iter()).unwrap();
        ensure_png_path(&custom).unwrap();

        assert!(
            output_path([OsString::from("one.png"), OsString::from("two.png")].into_iter())
                .is_err()
        );
        assert!(ensure_png_path(Path::new("comparison.jpg")).is_err());
    }
}
