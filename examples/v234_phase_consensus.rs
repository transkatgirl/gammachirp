//! Phase-aware reassignment, sparsity, and bandwidth-consensus example.

use std::f64::consts::PI;

use gammachirp_rs::gcfb_v234::{
    BandwidthConsensusConfig, ControlMode, GainReference, GcParam,
    gcfb_v234_with_bandwidth_consensus,
};

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
    let (filterbank, consensus) = gcfb_v234_with_bandwidth_consensus(
        &input,
        parameters,
        &BandwidthConsensusConfig::default(),
    )?;
    let phase = &consensus.analyses[consensus.baseline_index];

    let comparison = phase.sparsity_comparison()?;
    println!(
        "matched retained energy: source={:.6e}, reassigned={:.6e}",
        phase.unreassigned_energy_map.sum(),
        phase.reassignment.retained_energy()
    );
    println!(
        "effective support: source={:.1} bins ({:.3}%), reassigned={:.1} bins ({:.3}%)",
        comparison.unreassigned.effective_bins,
        100.0 * comparison.unreassigned.effective_bin_fraction,
        comparison.reassigned.effective_bins,
        100.0 * comparison.reassigned.effective_bin_fraction,
    );

    let mut energy_bins: Vec<(usize, usize)> = phase
        .reassignment
        .energy_map
        .indexed_iter()
        .map(|(index, _)| index)
        .collect();
    energy_bins.sort_by(|&left, &right| {
        phase.reassignment.energy_map[right].total_cmp(&phase.reassignment.energy_map[left])
    });
    println!("strongest reassigned localization peaks:");
    for &(channel, time) in energy_bins.iter().take(5) {
        println!(
            "  t={:.5} s, f={:.1} Hz, coherence={:.3}, energy={:.3e}",
            phase.reassignment.time_axis[time],
            phase.reassignment.frequency_axis_hz[channel],
            phase.phase_coherence_map[[channel, time]],
            phase.reassignment.energy_map[[channel, time]],
        );
    }

    for &click in &click_samples {
        let lower = click.saturating_sub(32);
        let upper = (click + 33).min(samples);
        let localized_time = (lower..upper)
            .max_by(|&left, &right| {
                phase
                    .reassignment
                    .energy_map
                    .column(left)
                    .sum()
                    .total_cmp(&phase.reassignment.energy_map.column(right).sum())
            })
            .unwrap();
        println!(
            "click localization: expected={:.5} s, peak={:.5} s",
            click as f64 / sample_rate,
            phase.reassignment.time_axis[localized_time],
        );
    }

    let mut salience_bins: Vec<(usize, usize)> = consensus
        .salience_map
        .indexed_iter()
        .filter_map(|(index, _)| consensus.consensus_mask[index].then_some(index))
        .collect();
    salience_bins.sort_by(|&left, &right| {
        consensus.salience_map[right].total_cmp(&consensus.salience_map[left])
    });
    println!("strongest full-bandwidth-consensus peaks:");
    for &(channel, time) in salience_bins.iter().take(5) {
        println!(
            "  t={:.5} s, f={:.1} Hz, agreement={:.3}, salience={:.3e}",
            phase.reassignment.time_axis[time],
            filterbank.gc_param.fr1[channel],
            consensus.agreement_map[[channel, time]],
            consensus.salience_map[[channel, time]],
        );
    }

    Ok(())
}
