//! Render causal GCFB source and streaming-synchrosqueezed spectrograms.

use std::error::Error;
use std::f64::consts::PI;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use gammachirp_rs::gcfb_v234::{ControlMode, GainReference, GcParam, SynchrosqueezingStream};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use plotters::coord::Shift;
use plotters::prelude::*;

const IMAGE_SIZE: (u32, u32) = (2300, 720);
const DB_FLOOR: f64 = -60.0;
const DEFAULT_OUTPUT: &str = "target/v234_synchrosqueezing_spectrogram.png";

struct StreamingSynchrosqueezingResult {
    source_energy_map: Array2<f64>,
    squeezed_energy_map: Array2<f64>,
    complex_map: Array2<Complex64>,
    time_axis: Array1<f64>,
    frequency_axis_erb: Array1<f64>,
    frequency_unresolved_energy: f64,
    boundary_discarded_energy: f64,
    maximum_buffered_samples: usize,
}

impl StreamingSynchrosqueezingResult {
    fn source_energy(&self) -> f64 {
        self.source_energy_map.sum()
    }

    fn retained_energy(&self) -> f64 {
        self.squeezed_energy_map.sum()
    }

    fn discarded_energy(&self) -> f64 {
        self.frequency_unresolved_energy + self.boundary_discarded_energy
    }
}

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
    let result = collect_streaming_synchrosqueezing(&input, parameters)?;
    render_comparison(&output_path, &result)?;

    let accounting_residual =
        result.source_energy() - result.retained_energy() - result.discarded_energy();
    println!("wrote {}", output_path.display());
    println!(
        "causal energy: source={:.6e}, retained={:.6e}, unresolved={:.6e}, boundary-discarded={:.6e}, residual={accounting_residual:.3e}",
        result.source_energy(),
        result.retained_energy(),
        result.frequency_unresolved_energy,
        result.boundary_discarded_energy,
    );
    println!(
        "effective support: source={:.1} bins, synchrosqueezed={:.1} bins",
        effective_bins(&result.source_energy_map),
        effective_bins(&result.squeezed_energy_map),
    );
    println!(
        "stream: latency=0 samples, maximum buffered causal-atom history={} samples",
        result.maximum_buffered_samples,
    );
    Ok(())
}

fn collect_streaming_synchrosqueezing(
    input: &[f64],
    parameters: GcParam,
) -> gammachirp_rs::Result<StreamingSynchrosqueezingResult> {
    let mut stream = SynchrosqueezingStream::new(parameters)?;
    let channels = stream.gc_param().num_ch;
    let samples = input.len();
    let sample_rate = stream.gc_param().fs;
    let frequency_axis_erb = stream.frequency_axis_erb().clone();
    let time_axis = Array1::from_iter((0..samples).map(|sample| sample as f64 / sample_rate));
    let maximum_buffered_samples = stream.max_buffered_samples();
    let mut result = StreamingSynchrosqueezingResult {
        source_energy_map: Array2::zeros((channels, samples)),
        squeezed_energy_map: Array2::zeros((channels, samples)),
        complex_map: Array2::from_elem((channels, samples), Complex64::new(0.0, 0.0)),
        time_axis,
        frequency_axis_erb,
        frequency_unresolved_energy: 0.0,
        boundary_discarded_energy: 0.0,
        maximum_buffered_samples,
    };

    for (sample_index, &sample) in input.iter().enumerate() {
        let step = stream.process_sample(sample)?;
        debug_assert_eq!(step.filterbank.sample_index, sample_index);
        debug_assert_eq!(stream.latency_samples(), 0);
        debug_assert!(stream.buffered_samples() <= maximum_buffered_samples);
        result
            .source_energy_map
            .column_mut(sample_index)
            .assign(&step.source_energy);
        result
            .squeezed_energy_map
            .column_mut(sample_index)
            .assign(&step.energy_column);
        result
            .complex_map
            .column_mut(sample_index)
            .assign(&step.complex_column);
        result.frequency_unresolved_energy += step.frequency_unresolved_energy;
        result.boundary_discarded_energy += step.boundary_discarded_energy;
    }
    debug_assert_eq!(stream.samples_processed(), input.len());
    Ok(result)
}

fn effective_bins(map: &Array2<f64>) -> f64 {
    let sum = map.sum();
    let sum_of_squares = map.iter().map(|value| value * value).sum::<f64>();
    if sum_of_squares > 0.0 {
        sum * sum / sum_of_squares
    } else {
        0.0
    }
}

fn output_path(mut arguments: impl Iterator<Item = OsString>) -> io::Result<PathBuf> {
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: v234_synchrosqueezing_spectrogram [output.png]",
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
    result: &StreamingSynchrosqueezingResult,
) -> Result<(), Box<dyn Error>> {
    validate_render_dimensions(
        &result.source_energy_map,
        &result.squeezed_energy_map,
        &result.complex_map,
        &result.time_axis,
        &result.frequency_axis_erb,
    )?;
    if result
        .source_energy_map
        .iter()
        .chain(result.squeezed_energy_map.iter())
        .any(|&value| !value.is_finite() || value < 0.0)
        || result
            .complex_map
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram maps must contain finite values and nonnegative energy",
        )
        .into());
    }

    let complex_power = complex_power_map(&result.complex_map);
    let energy_maximum = result
        .source_energy_map
        .iter()
        .chain(result.squeezed_energy_map.iter())
        .copied()
        .fold(0.0_f64, f64::max);
    let complex_power_maximum = complex_power.iter().copied().fold(0.0_f64, f64::max);
    if energy_maximum <= 0.0 || complex_power_maximum <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram maps must contain positive energy",
        )
        .into());
    }

    let time_edges = bin_edges(result.time_axis.as_slice().unwrap(), Some(0.0))?;
    let frequency_edges = bin_edges(result.frequency_axis_erb.as_slice().unwrap(), None)?;
    let root = BitMapBackend::new(output_path, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let body = root.titled("GCFB v2.34 Streaming Synchrosqueezing", ("sans-serif", 30))?;
    let (panel_area, colorbar_area) = body.split_horizontally(2040);
    let panels = panel_area.split_evenly((1, 3));

    draw_spectrogram(
        &panels[0],
        "Causal source energy (before squeezing)",
        &result.source_energy_map,
        &time_edges,
        &frequency_edges,
        energy_maximum,
    )?;
    draw_spectrogram(
        &panels[1],
        "Streaming synchrosqueezed energy",
        &result.squeezed_energy_map,
        &time_edges,
        &frequency_edges,
        energy_maximum,
    )?;
    draw_spectrogram(
        &panels[2],
        "Complex-map power (phase interference)",
        &complex_power,
        &time_edges,
        &frequency_edges,
        complex_power_maximum,
    )?;
    let colorbars = colorbar_area.split_evenly((2, 1));
    draw_colorbar(&colorbars[0], "Analytic energy (dB)")?;
    draw_colorbar(&colorbars[1], "Complex-map power (dB)")?;
    root.present()?;
    Ok(())
}

fn validate_render_dimensions(
    source_energy: &Array2<f64>,
    squeezed_energy: &Array2<f64>,
    complex_map: &Array2<Complex64>,
    time_axis: &Array1<f64>,
    frequency_axis_erb: &Array1<f64>,
) -> io::Result<()> {
    let expected_dimensions = (frequency_axis_erb.len(), time_axis.len());
    if source_energy.dim() != expected_dimensions
        || squeezed_energy.dim() != expected_dimensions
        || complex_map.dim() != expected_dimensions
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spectrogram maps and axes have incompatible dimensions",
        ));
    }
    Ok(())
}

fn complex_power_map(complex_map: &Array2<Complex64>) -> Array2<f64> {
    complex_map.mapv(|coefficient| coefficient.norm_sqr() / 2.0)
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
        .caption(title, ("sans-serif", 21))
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

fn draw_colorbar(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 17))
        .margin_top(32)
        .margin_bottom(28)
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
    fn shared_decibel_scale_maps_the_peak_and_floor() {
        assert_eq!(energy_db(4.0, 4.0), 0.0);
        assert_eq!(energy_db(0.0, 4.0), DB_FLOOR);
        assert_eq!(energy_db(4e-7, 4.0), DB_FLOOR);
        assert!((energy_db(0.4, 4.0) + 10.0).abs() < 1e-12);
    }

    #[test]
    fn complex_power_uses_half_the_squared_magnitude() {
        let complex = Array2::from_shape_vec(
            (1, 2),
            vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 2.0)],
        )
        .unwrap();
        assert_eq!(
            complex_power_map(&complex),
            Array2::from_shape_vec((1, 2), vec![12.5, 2.0]).unwrap()
        );
    }

    #[test]
    fn rendering_validates_all_map_dimensions() {
        let time_axis = Array1::from_vec(vec![0.0, 0.1, 0.2]);
        let frequency_axis = Array1::from_vec(vec![10.0, 11.0]);
        let energy = Array2::zeros((2, 3));
        let complex = Array2::from_elem((2, 3), Complex64::new(0.0, 0.0));

        validate_render_dimensions(&energy, &energy, &complex, &time_axis, &frequency_axis)
            .unwrap();
        assert!(
            validate_render_dimensions(
                &energy,
                &Array2::zeros((2, 2)),
                &complex,
                &time_axis,
                &frequency_axis,
            )
            .is_err()
        );
        assert!(
            validate_render_dimensions(
                &energy,
                &energy,
                &Array2::from_elem((1, 3), Complex64::new(0.0, 0.0)),
                &time_axis,
                &frequency_axis,
            )
            .is_err()
        );
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
