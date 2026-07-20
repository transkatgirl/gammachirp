//! Causal phase reassignment and rolling bandwidth-consensus example.

use std::f64::consts::PI;

use gammachirp_rs::gcfb_v234::{
    BandwidthConsensusStream, BandwidthConsensusStreamConfig, ControlMode, GainReference, GcParam,
};

#[derive(Clone, Copy, Default)]
struct EnergyStats {
    sum: f64,
    sum_of_squares: f64,
    bins: usize,
}

impl EnergyStats {
    fn record(&mut self, values: impl IntoIterator<Item = f64>) {
        for value in values {
            self.sum += value;
            self.sum_of_squares += value * value;
            self.bins += 1;
        }
    }

    fn effective_bins(self) -> f64 {
        if self.sum_of_squares > 0.0 {
            self.sum * self.sum / self.sum_of_squares
        } else {
            0.0
        }
    }
}

fn main() -> gammachirp_rs::Result<()> {
    let sample_rate = 16_000.0;
    let samples = 4096;
    let click_samples = [960, 2880];
    let chirp_start_hz = 350.0;
    let chirp_end_hz = 1800.0;
    let duration = samples as f64 / sample_rate;
    let chirp_rate = (chirp_end_hz - chirp_start_hz) / duration;

    // A tiny LCG keeps the noise deterministic without adding a dependency.
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
    for &sample in &click_samples {
        input[sample] += 1.0;
    }

    let parameters = GcParam {
        fs: sample_rate,
        num_ch: 32,
        f_range: [180.0, 3000.0],
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        gain_ref: GainReference::Db(50.0),
        ..GcParam::default()
    };
    let config = BandwidthConsensusStreamConfig::default();
    let required_agreement = config.required_agreement;
    let mut analysis = BandwidthConsensusStream::new(parameters, config)?;
    let channels = analysis.gc_param().num_ch;
    let center_frequencies_hz = analysis.gc_param().fr1.to_vec();
    let scales = analysis.scales().to_vec();
    let baseline_index = analysis.baseline_index();
    let latency_samples = analysis.latency_samples();
    let mut source_stats = EnergyStats::default();
    let mut reassigned_stats = EnergyStats::default();
    let mut target_energy = vec![0.0; samples];
    let mut accepted_bins = 0;
    let mut strongest_coordinates = Vec::new();
    let mut strongest_consensus = Vec::new();

    for (sample_index, sample) in input.into_iter().enumerate() {
        let step = analysis.process_sample(sample)?;
        let baseline = step.baseline();
        source_stats.record(baseline.source_energy.iter().copied());
        for channel in 0..channels {
            if baseline.coordinate_mask[channel] {
                retain_strongest(
                    &mut strongest_coordinates,
                    (
                        baseline.source_energy[channel],
                        sample_index,
                        baseline.t_hat[channel],
                        baseline.f_hat[channel],
                    ),
                );
            }
        }
        if let Some(frame) = step.consensus {
            record_consensus_frame(
                frame,
                baseline_index,
                &mut reassigned_stats,
                &mut target_energy,
                &mut accepted_bins,
                &mut strongest_consensus,
            );
        }
    }
    for frame in analysis.finish()? {
        record_consensus_frame(
            frame,
            baseline_index,
            &mut reassigned_stats,
            &mut target_energy,
            &mut accepted_bins,
            &mut strongest_consensus,
        );
    }

    println!(
        "causal energy: source={:.6e}, retained on rolling target grid={:.6e}",
        source_stats.sum, reassigned_stats.sum,
    );
    println!(
        "effective support: source={:.1} bins ({:.3}%), reassigned={:.1} bins ({:.3}%)",
        source_stats.effective_bins(),
        100.0 * source_stats.effective_bins() / source_stats.bins as f64,
        reassigned_stats.effective_bins(),
        100.0 * reassigned_stats.effective_bins() / reassigned_stats.bins as f64,
    );

    println!("strongest immediate causal phase coordinates:");
    for &(energy, source_sample, target_time, target_frequency) in
        strongest_coordinates.iter().take(5)
    {
        println!(
            "  source t={:.5} s, reassigned t={target_time:.5} s, f={target_frequency:.1} Hz, energy={energy:.3e}",
            source_sample as f64 / sample_rate,
        );
    }

    for &click in &click_samples {
        let lower = click.saturating_sub(32);
        let upper = (click + 33).min(samples);
        let localized_sample = (lower..upper)
            .max_by(|&left, &right| target_energy[left].total_cmp(&target_energy[right]))
            .unwrap();
        println!(
            "click localization: expected={:.5} s, rolling-grid peak={:.5} s",
            click as f64 / sample_rate,
            localized_sample as f64 / sample_rate,
        );
    }

    println!("strongest rolling bandwidth-consensus bins:");
    for &(salience, time, channel, agreement) in strongest_consensus.iter().take(5) {
        println!(
            "  t={time:.5} s, f={:.1} Hz, agreement={agreement:.3}, salience={salience:.3e}",
            center_frequencies_hz[channel],
        );
    }
    let required_scales = (required_agreement * scales.len() as f64).ceil() as usize;
    println!(
        "rolling consensus: scales={scales:?}, latency={latency_samples} samples, accepted={accepted_bins}/{} bins, requirement={required_scales}/{} scales",
        channels * samples,
        scales.len(),
    );

    Ok(())
}

fn record_consensus_frame(
    frame: gammachirp_rs::gcfb_v234::BandwidthConsensusStreamFrame,
    baseline_index: usize,
    reassigned_stats: &mut EnergyStats,
    target_energy: &mut [f64],
    accepted_bins: &mut usize,
    strongest_consensus: &mut Vec<(f64, f64, usize, f64)>,
) {
    let baseline = &frame.scale_energy_columns[baseline_index];
    reassigned_stats.record(baseline.iter().copied());
    target_energy[frame.sample_index] = baseline.sum();
    for channel in 0..baseline.len() {
        if frame.consensus_mask[channel] {
            *accepted_bins += 1;
            retain_strongest(
                strongest_consensus,
                (
                    frame.salience[channel],
                    frame.time_seconds,
                    channel,
                    frame.agreement[channel],
                ),
            );
        }
    }
}

fn retain_strongest<T>(entries: &mut Vec<T>, entry: T)
where
    T: AsRefScore,
{
    entries.push(entry);
    entries.sort_by(|left, right| right.score().total_cmp(&left.score()));
    entries.truncate(5);
}

trait AsRefScore {
    fn score(&self) -> f64;
}

impl AsRefScore for (f64, usize, f64, f64) {
    fn score(&self) -> f64 {
        self.0
    }
}

impl AsRefScore for (f64, f64, usize, f64) {
    fn score(&self) -> f64 {
        self.0
    }
}
