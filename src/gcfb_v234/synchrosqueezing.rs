//! Frequency-only synchrosqueezing for GCFB v2.34.
//!
//! Unlike full time-frequency reassignment, synchrosqueezing preserves every
//! accepted coefficient's source time and moves only its frequency. Complex
//! GCFB coefficients are summed into the nearest prepared auditory channel,
//! following the [general filter-bank construction of Holighaus et al.](https://ltfat.org/notes/ltfatnote041.pdf).
//! The separate energy map sums
//! `|C|^2 / 2`, because complex coefficients that collide in a target bin can
//! interfere.
//!
//! This implementation is analysis-only. The nonlinear GCFB has no
//! established frame inverse, and this module provides no reconstruction API.
//! Batch analysis uses the same offline analytic projection as reassignment;
//! [`SynchrosqueezingStream`] uses its causal approximation. Dynamic frame-base
//! processing is rejected because it does not retain the full-rate time grid
//! required by this construction. Supported dynamic processing always uses the
//! continuous-DTFT peak-lock path.

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::gcfb_v234::{
    ControlMode, GcParam, GcResp, GcfbOutput, gcfb_v234, gcfb_v234_with_bandwidth_peak_lock,
    prepare_bandwidth_peak_grid,
};
use super::reassignment::{
    ReassignmentMode, ReassignmentStream, ReassignmentStreamStep, conditioned_frequency_analysis,
};
use super::stream::StreamStep;
use super::utils;
use crate::{Error, Result};

/// Options for GCFB synchrosqueezing.
#[derive(Clone, Debug)]
pub struct SynchrosqueezingConfig {
    /// Per-channel relative analytic-power floor. A coefficient is rejected
    /// when its power is below this fraction of the channel maximum.
    pub coefficient_floor: f64,
}

impl Default for SynchrosqueezingConfig {
    fn default() -> Self {
        Self {
            coefficient_floor: 1e-8,
        }
    }
}

/// Mathematical status of a synchrosqueezing result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynchrosqueezingMode {
    /// A fixed complex cGC, used by static and level-control analyses.
    Fixed,
    /// The frequency is derived from the realized sample-dynamic complex
    /// coefficient history without differentiating its nonlinear estimator.
    SampleConditional,
}

/// Analysis-only, frequency-synchrosqueezed GCFB representation.
#[derive(Clone, Debug)]
pub struct SynchrosqueezingResult {
    /// Gained complex coefficients summed into their nearest instantaneous-
    /// frequency channel on `[ERB channel, source sample]`.
    pub complex_map: Array2<Complex64>,
    /// Nonnegative analytic energy deposited on the same grid. This is the
    /// authoritative energy representation; it is not `|complex_map|^2 / 2`
    /// when multiple coefficients collide.
    pub energy_map: Array2<f64>,
    /// Instantaneous frequency, in hertz, for every source coefficient.
    pub f_hat: Array2<f64>,
    /// Coefficients that passed the per-channel relative power floor and have
    /// a finite instantaneous frequency. The mask does not include target-grid
    /// boundary rejection.
    pub validity_mask: Array2<bool>,
    /// Unchanged source-sample times, in seconds.
    pub time_axis: Array1<f64>,
    /// Centers of the target auditory-frequency bins, in hertz.
    pub frequency_axis_hz: Array1<f64>,
    /// Centers of the target auditory-frequency bins, in ERB rate.
    pub frequency_axis_erb: Array1<f64>,
    /// Analytic-representation energy before floor and boundary rejection.
    pub source_energy: f64,
    /// Analytic energy rejected by the relative coefficient floor.
    pub floor_discarded_energy: f64,
    /// Retained-coefficient energy with a nonpositive or out-of-band
    /// instantaneous frequency.
    pub boundary_discarded_energy: f64,
    /// Sum of floor- and boundary-discarded analytic energy.
    pub discarded_energy: f64,
    /// Type and guarantee of the frequency calculation.
    pub mode: SynchrosqueezingMode,
    /// Length of the zero-padded finite DFT domain used by the batch analysis.
    pub analysis_fft_len: usize,
}

impl SynchrosqueezingResult {
    /// Analytic energy successfully deposited into the map.
    pub fn retained_energy(&self) -> f64 {
        self.energy_map.sum()
    }
}

/// Run GCFB v2.34 and its synchrosqueezing analysis together.
///
/// Static and level-control outputs are identical to calling [`gcfb_v234`]
/// alone. Dynamic sample-control output is always reprocessed through the
/// continuous-DTFT peak-lock path used by bandwidth analysis.
pub fn gcfb_v234_with_synchrosqueezing(
    snd_in: &[f64],
    gc_param: GcParam,
) -> Result<(GcfbOutput, SynchrosqueezingResult)> {
    let output = synchrosqueezing_filterbank(snd_in, gc_param)?;
    let synchrosqueezing =
        synchrosqueeze_prepared(snd_in, &output, &SynchrosqueezingConfig::default())?;
    Ok((output, synchrosqueezing))
}

/// Synchrosqueeze an existing GCFB output with the default options.
///
/// `snd_in` must be the same input used to produce `output`.
pub fn synchrosqueeze_gcfb_v234(
    snd_in: &[f64],
    output: &GcfbOutput,
) -> Result<SynchrosqueezingResult> {
    synchrosqueeze_gcfb_v234_with_config(snd_in, output, &SynchrosqueezingConfig::default())
}

/// Synchrosqueeze an existing GCFB output with explicit options.
///
/// A dynamic output supplies the reference parameters and response; the input
/// is reprocessed through the peak-lock path before its coefficients are
/// squeezed.
pub fn synchrosqueeze_gcfb_v234_with_config(
    snd_in: &[f64],
    output: &GcfbOutput,
    config: &SynchrosqueezingConfig,
) -> Result<SynchrosqueezingResult> {
    validate_batch_input(snd_in, output, config)?;
    if output.gc_param.ctrl == ControlMode::Dynamic {
        let peak_locked = peak_locked_dynamic_filterbank(snd_in, output.gc_param.clone(), output)?;
        verify_matching_dynamic_output(output, &peak_locked)?;
        return synchrosqueeze_prepared(snd_in, &peak_locked, config);
    }
    synchrosqueeze_prepared(snd_in, output, config)
}

fn synchrosqueeze_prepared(
    snd_in: &[f64],
    output: &GcfbOutput,
    config: &SynchrosqueezingConfig,
) -> Result<SynchrosqueezingResult> {
    let analysis = conditioned_frequency_analysis(snd_in, output, config.coefficient_floor)?;
    let mode = match analysis.mode {
        ReassignmentMode::Fixed => SynchrosqueezingMode::Fixed,
        ReassignmentMode::SampleConditional => SynchrosqueezingMode::SampleConditional,
        ReassignmentMode::Frame => {
            return Err(Error::Unsupported(
                "synchrosqueezing does not support dynamic frame-base processing".into(),
            ));
        }
    };
    let channels = output.gc_param.num_ch;
    let samples = snd_in.len();
    let frequency_axis_hz = output.gc_param.fr1.clone();
    let (frequency_axis_erb, _) = utils::freq2erb(frequency_axis_hz.as_slice().unwrap());
    let time_axis =
        Array1::from_iter((0..samples).map(|sample| sample as f64 / output.gc_param.fs));
    let mut complex_map = Array2::from_elem((channels, samples), Complex64::new(0.0, 0.0));
    let mut energy_map = Array2::zeros((channels, samples));
    let mut source_energy = 0.0;
    let mut floor_discarded_energy = 0.0;
    let mut boundary_discarded_energy = 0.0;

    for source_channel in 0..channels {
        let gain = output.gc_resp.gain_factor[source_channel];
        if !gain.is_finite() {
            return Err(Error::Numerical(format!(
                "non-finite GCFB gain in synchrosqueezing channel {source_channel}"
            )));
        }
        for sample in 0..samples {
            let energy = gain.powi(2) * analysis.power[[source_channel, sample]];
            if !energy.is_finite() || energy < 0.0 {
                return Err(Error::Numerical(format!(
                    "non-finite analytic energy in synchrosqueezing channel {source_channel}, sample {sample}"
                )));
            }
            if energy == 0.0 {
                continue;
            }
            source_energy += energy;
            if !source_energy.is_finite() {
                return Err(Error::Numerical(
                    "non-finite total energy during synchrosqueezing".into(),
                ));
            }
            if !analysis.validity_mask[[source_channel, sample]] {
                floor_discarded_energy += energy;
                continue;
            }
            let frequency_hz = analysis.f_hat[[source_channel, sample]];
            let Some(target_channel) =
                nearest_frequency_channel(frequency_axis_hz.as_slice().unwrap(), frequency_hz)
            else {
                boundary_discarded_energy += energy;
                continue;
            };
            let contribution = analysis.coefficient[[source_channel, sample]] * gain;
            if !contribution.re.is_finite() || !contribution.im.is_finite() {
                return Err(Error::Numerical(format!(
                    "non-finite complex synchrosqueezing contribution in channel {source_channel}, sample {sample}"
                )));
            }
            complex_map[[target_channel, sample]] += contribution;
            energy_map[[target_channel, sample]] += energy;
        }
    }
    let discarded_energy = floor_discarded_energy + boundary_discarded_energy;
    Ok(SynchrosqueezingResult {
        complex_map,
        energy_map,
        f_hat: analysis.f_hat,
        validity_mask: analysis.validity_mask,
        time_axis,
        frequency_axis_hz,
        frequency_axis_erb,
        source_energy,
        floor_discarded_energy,
        boundary_discarded_energy,
        discarded_energy,
        mode,
        analysis_fft_len: analysis.analysis_fft_len,
    })
}

fn synchrosqueezing_filterbank(snd_in: &[f64], gc_param: GcParam) -> Result<GcfbOutput> {
    let reference = gcfb_v234(snd_in, gc_param.clone())?;
    if reference.gc_param.ctrl == ControlMode::Dynamic {
        peak_locked_dynamic_filterbank(snd_in, gc_param, &reference)
    } else {
        Ok(reference)
    }
}

fn peak_locked_dynamic_filterbank(
    snd_in: &[f64],
    gc_param: GcParam,
    reference: &GcfbOutput,
) -> Result<GcfbOutput> {
    let peak_grid =
        prepare_bandwidth_peak_grid(&gc_param, &[1.0], &reference.gc_param, &reference.gc_resp)?;
    gcfb_v234_with_bandwidth_peak_lock(snd_in, gc_param, &reference.gc_param.hloss, peak_grid)
}

fn verify_matching_dynamic_output(reference: &GcfbOutput, peak_locked: &GcfbOutput) -> Result<()> {
    if reference.scgc_smpl.dim() != peak_locked.scgc_smpl.dim()
        || reference.dcgc_out.dim() != peak_locked.dcgc_out.dim()
    {
        return Err(Error::InvalidParameter(
            "input does not match the supplied dynamic GCFB output".into(),
        ));
    }
    for (&expected, &actual) in reference
        .scgc_smpl
        .iter()
        .chain(reference.dcgc_out.iter())
        .zip(
            peak_locked
                .scgc_smpl
                .iter()
                .chain(peak_locked.dcgc_out.iter()),
        )
    {
        let tolerance = 2e-8 * actual.abs().max(expected.abs()).max(1.0);
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > tolerance {
            return Err(Error::InvalidParameter(format!(
                "input does not match the supplied dynamic GCFB output ({actual} versus {expected})"
            )));
        }
    }
    Ok(())
}

fn validate_batch_input(
    snd: &[f64],
    output: &GcfbOutput,
    config: &SynchrosqueezingConfig,
) -> Result<()> {
    if !config.coefficient_floor.is_finite()
        || config.coefficient_floor <= 0.0
        || config.coefficient_floor >= 1.0
    {
        return Err(Error::InvalidParameter(
            "synchrosqueezing coefficient floor must be in (0, 1)".into(),
        ));
    }
    if output.gc_param.ctrl == ControlMode::Dynamic
        && output.gc_param.dyn_hpaf.str_prc.contains("frame")
    {
        return Err(Error::Unsupported(
            "synchrosqueezing does not support dynamic frame-base processing".into(),
        ));
    }
    let channels = output.gc_param.num_ch;
    if snd.is_empty()
        || channels == 0
        || output.gc_param.fr1.len() != channels
        || output.scgc_smpl.dim() != (channels, snd.len())
        || output.gc_resp.gain_factor.len() != channels
    {
        return Err(Error::InvalidParameter(
            "synchrosqueezing requires a matching non-empty GCFB output".into(),
        ));
    }
    Ok(())
}

/// Return the closest prepared frequency channel, rejecting coordinates
/// outside the represented band. Exact midpoint ties select the lower channel.
fn nearest_frequency_channel(axis: &[f64], value: f64) -> Option<usize> {
    if axis.is_empty()
        || !value.is_finite()
        || value <= 0.0
        || value < axis[0]
        || value > axis[axis.len() - 1]
    {
        return None;
    }
    match axis.binary_search_by(|candidate| candidate.total_cmp(&value)) {
        Ok(index) => Some(index),
        Err(upper) if upper > 0 && upper < axis.len() => {
            let lower = upper - 1;
            if value - axis[lower] <= axis[upper] - value {
                Some(lower)
            } else {
                Some(upper)
            }
        }
        _ => None,
    }
}

/// Immediate causal synchrosqueezing products for one input sample.
#[derive(Clone, Debug)]
pub struct SynchrosqueezingStreamStep {
    /// Ordinary GCFB v2.34 streaming output for the accepted input sample.
    pub filterbank: StreamStep,
    /// Ungained causal complex analysis coefficient per source channel.
    pub coefficient: Array1<Complex64>,
    /// Gained analytic energy, `gain^2 * |C|^2 / 2`, per source channel.
    pub source_energy: Array1<f64>,
    /// Causal instantaneous frequency, in hertz, per source channel.
    pub f_hat: Array1<f64>,
    /// Nonzero coefficients with a finite frequency estimate. No relative
    /// coefficient floor or target-boundary test is included.
    pub validity_mask: Array1<bool>,
    /// Gained complex coefficients summed into the nearest target channel for
    /// the current sample.
    pub complex_column: Array1<Complex64>,
    /// Nonnegative analytic energy summed into the same target column.
    pub energy_column: Array1<f64>,
    /// Source energy without an available causal frequency estimate.
    pub frequency_unresolved_energy: f64,
    /// Valid-frequency energy rejected outside the target frequency band.
    pub boundary_discarded_energy: f64,
    /// Sum of unresolved and boundary-discarded energy.
    pub discarded_energy: f64,
}

impl SynchrosqueezingStreamStep {
    /// Analytic energy deposited into this step's target column.
    pub fn retained_energy(&self) -> f64 {
        self.energy_column.sum()
    }
}

/// Bounded-memory, zero-latency causal synchrosqueezing approximation.
///
/// This wraps [`ReassignmentStream`] so both streaming analyses use identical
/// causal analytic coefficients and instantaneous-frequency estimates. Static,
/// level, and dynamic sample-base modes are supported; dynamic frame-base mode
/// is rejected. Dynamic control always uses continuous-DTFT peak locking. No
/// whole-stream relative coefficient floor is applied.
#[derive(Clone, Debug)]
pub struct SynchrosqueezingStream {
    analysis: ReassignmentStream,
    frequency_axis_hz: Array1<f64>,
    frequency_axis_erb: Array1<f64>,
}

impl SynchrosqueezingStream {
    /// Prepare a causal synchrosqueezing stream and its GCFB processor.
    /// Dynamic control is prepared with peak locking.
    pub fn new(gc_param: GcParam) -> Result<Self> {
        let mut analysis = ReassignmentStream::new(gc_param.clone())?;
        if analysis.gc_param().ctrl == ControlMode::Dynamic {
            let hearing_loss = analysis.gc_param().hloss.clone();
            let peak_grid = prepare_bandwidth_peak_grid(
                &gc_param,
                &[1.0],
                analysis.gc_param(),
                analysis.gc_resp(),
            )?;
            analysis = ReassignmentStream::new_with_bandwidth_peak_lock(
                gc_param,
                &hearing_loss,
                peak_grid,
            )?;
        }
        let frequency_axis_hz = analysis.gc_param().fr1.clone();
        let (frequency_axis_erb, _) = utils::freq2erb(frequency_axis_hz.as_slice().unwrap());
        Ok(Self {
            analysis,
            frequency_axis_hz,
            frequency_axis_erb,
        })
    }

    /// Process one finite input sample and emit its squeezed target column.
    pub fn process_sample(&mut self, sample: f64) -> Result<SynchrosqueezingStreamStep> {
        let ReassignmentStreamStep {
            filterbank,
            coefficient,
            source_energy,
            f_hat,
            ..
        } = self.analysis.process_sample(sample)?;
        let channels = coefficient.len();
        let gains = &self.analysis.gc_resp().gain_factor;
        let validity_mask = stream_frequency_validity_mask(&coefficient, &f_hat);
        let mut complex_column = Array1::from_elem(channels, Complex64::new(0.0, 0.0));
        let mut energy_column = Array1::zeros(channels);
        let mut frequency_unresolved_energy = 0.0;
        let mut boundary_discarded_energy = 0.0;
        for source_channel in 0..channels {
            let energy = source_energy[source_channel];
            if energy == 0.0 {
                continue;
            }
            if !validity_mask[source_channel] {
                frequency_unresolved_energy += energy;
                continue;
            }
            let Some(target_channel) = nearest_frequency_channel(
                self.frequency_axis_hz.as_slice().unwrap(),
                f_hat[source_channel],
            ) else {
                boundary_discarded_energy += energy;
                continue;
            };
            complex_column[target_channel] += coefficient[source_channel] * gains[source_channel];
            energy_column[target_channel] += energy;
        }
        let discarded_energy = frequency_unresolved_energy + boundary_discarded_energy;
        Ok(SynchrosqueezingStreamStep {
            filterbank,
            coefficient,
            source_energy,
            f_hat,
            validity_mask,
            complex_column,
            energy_column,
            frequency_unresolved_energy,
            boundary_discarded_energy,
            discarded_energy,
        })
    }

    /// Prepared GCFB parameters used by both real and complex paths.
    pub fn gc_param(&self) -> &GcParam {
        self.analysis.gc_param()
    }

    /// Prepared time-invariant GCFB response metadata.
    pub fn gc_resp(&self) -> &GcResp {
        self.analysis.gc_resp()
    }

    /// Alias for [`Self::gc_param`].
    pub fn prepared_param(&self) -> &GcParam {
        self.gc_param()
    }

    /// Alias for [`Self::gc_resp`].
    pub fn prepared_response(&self) -> &GcResp {
        self.gc_resp()
    }

    /// Target auditory-frequency centers, in hertz.
    pub fn frequency_axis_hz(&self) -> &Array1<f64> {
        &self.frequency_axis_hz
    }

    /// Target auditory-frequency centers, in ERB rate.
    pub fn frequency_axis_erb(&self) -> &Array1<f64> {
        &self.frequency_axis_erb
    }

    /// This causal construction emits every accepted sample immediately.
    pub fn latency_samples(&self) -> usize {
        self.analysis.latency_samples()
    }

    /// Number of successfully accepted input samples.
    pub fn samples_processed(&self) -> usize {
        self.analysis.samples_processed()
    }

    /// Number of sample positions currently represented in FIR histories.
    pub fn buffered_samples(&self) -> usize {
        self.analysis.buffered_samples()
    }

    /// Fixed maximum number of sample positions retained by FIR histories.
    pub fn max_buffered_samples(&self) -> usize {
        self.analysis.max_buffered_samples()
    }
}

fn stream_frequency_validity_mask(
    coefficient: &Array1<Complex64>,
    f_hat: &Array1<f64>,
) -> Array1<bool> {
    debug_assert_eq!(coefficient.len(), f_hat.len());
    Array1::from_iter(
        coefficient
            .iter()
            .zip(f_hat)
            .map(|(coefficient, frequency)| {
                let norm = coefficient.norm_sqr();
                norm > 0.0 && norm.is_finite() && frequency.is_finite()
            }),
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::gcfb_v234::{DynHpaf, GainReference};

    fn compact_parameters() -> GcParam {
        GcParam {
            fs: 8_000.0,
            num_ch: 8,
            f_range: [200.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            gain_ref: GainReference::Db(50.0),
            ..GcParam::default()
        }
    }

    #[test]
    fn nearest_frequency_uses_hz_distance_and_lower_ties() {
        let axis = [100.0, 200.0, 500.0];
        assert_eq!(nearest_frequency_channel(&axis, 100.0), Some(0));
        assert_eq!(nearest_frequency_channel(&axis, 150.0), Some(0));
        assert_eq!(nearest_frequency_channel(&axis, 151.0), Some(1));
        assert_eq!(nearest_frequency_channel(&axis, 350.0), Some(1));
        assert_eq!(nearest_frequency_channel(&axis, 351.0), Some(2));
        assert_eq!(nearest_frequency_channel(&axis, 99.0), None);
        assert_eq!(nearest_frequency_channel(&axis, 501.0), None);
        assert_eq!(nearest_frequency_channel(&axis, f64::NAN), None);
    }

    #[test]
    fn streaming_frequency_validity_does_not_require_a_time_coordinate() {
        let coefficient = Array1::from_vec(vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.25, 0.5),
        ]);
        let frequency = Array1::from_vec(vec![700.0, 700.0, f64::NAN]);

        assert_eq!(
            stream_frequency_validity_mask(&coefficient, &frequency),
            Array1::from_vec(vec![true, false, false])
        );
    }

    #[test]
    fn batch_energy_is_fully_accounted_for() {
        let signal: Vec<f64> = (0..256)
            .map(|sample| (2.0 * std::f64::consts::PI * 700.0 * sample as f64 / 8_000.0).cos())
            .collect();
        let (_, squeezed) = gcfb_v234_with_synchrosqueezing(&signal, compact_parameters()).unwrap();
        assert_relative_eq!(
            squeezed.retained_energy() + squeezed.discarded_energy,
            squeezed.source_energy,
            epsilon = squeezed.source_energy.max(1.0) * 2e-12
        );
        assert_eq!(squeezed.energy_map.dim(), (8, signal.len()));
        assert_eq!(squeezed.complex_map.dim(), squeezed.energy_map.dim());
        assert_eq!(squeezed.mode, SynchrosqueezingMode::Fixed);
    }

    #[test]
    fn frame_mode_and_invalid_floor_are_rejected() {
        let signal = vec![0.0; 64];
        let frame = GcParam {
            ctrl: ControlMode::Dynamic,
            dyn_hpaf: DynHpaf {
                str_prc: "frame-base".into(),
                ..DynHpaf::default()
            },
            ..compact_parameters()
        };
        assert!(matches!(
            gcfb_v234_with_synchrosqueezing(&signal, frame),
            Err(Error::Unsupported(_))
        ));

        let output = gcfb_v234(&signal, compact_parameters()).unwrap();
        let config = SynchrosqueezingConfig {
            coefficient_floor: 0.0,
        };
        assert!(matches!(
            synchrosqueeze_gcfb_v234_with_config(&signal, &output, &config),
            Err(Error::InvalidParameter(_))
        ));
    }

    #[test]
    fn mismatched_and_malformed_outputs_are_rejected() {
        let signal: Vec<f64> = (0..64)
            .map(|sample| (2.0 * std::f64::consts::PI * 500.0 * sample as f64 / 8_000.0).cos())
            .collect();
        let output = gcfb_v234(&signal, compact_parameters()).unwrap();
        assert!(matches!(
            synchrosqueeze_gcfb_v234(&signal[..signal.len() - 1], &output),
            Err(Error::InvalidParameter(_))
        ));

        let mut different_input = signal.clone();
        different_input[0] += 0.5;
        assert!(matches!(
            synchrosqueeze_gcfb_v234(&different_input, &output),
            Err(Error::InvalidParameter(_))
        ));

        let mut malformed = output;
        malformed.gc_resp.gain_factor = Array1::zeros(7);
        assert!(matches!(
            synchrosqueeze_gcfb_v234(&signal, &malformed),
            Err(Error::InvalidParameter(_))
        ));
    }
}
