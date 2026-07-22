//! Zero-latency causal streaming approximation of dcGC reassignment.

use std::f64::consts::{PI, SQRT_2};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::{
    BandwidthScaleMetadata, ComplexCascadeState, scale_bandwidths, validate_consensus_parameters,
};
use crate::gcfb_v234::gammachirp::{self, Carrier, Normalization};
use crate::gcfb_v234::gcfb_v234::{
    AcfCoef, BandwidthPeakGrid, ControlMode, GcParam, GcResp, HLoss,
    initial_asymmetric_ratio_and_centers, make_asym_cmp_filters_v2, prepare_bandwidth_peak_grid,
    prepare_input_correction_fir,
};
use crate::gcfb_v234::stream::{DcgcEvent, GcfbStream, StreamStep};
use crate::gcfb_v234::utils;
use crate::{Error, Result};

/// Options for bounded-memory, rolling bandwidth consensus.
///
/// Unlike [`super::BandwidthConsensusConfig`], this configuration has no
/// coefficient floor. Such a floor is relative to a whole-input channel
/// maximum and cannot be applied retroactively to an indefinite stream.
#[derive(Clone, Debug)]
pub struct BandwidthConsensusStreamConfig {
    /// Multipliers applied to `b1`, `b2`, and `lvl_est.b2`.
    pub scales: Vec<f64>,
    /// A normalized scale map supports a bin only when it exceeds this value.
    pub relative_support_floor: f64,
    /// Minimum fraction of scales required by the consensus mask and salience
    /// order statistic.
    pub required_agreement: f64,
    /// Rolling normalization horizon. `None` selects the longest causal atom
    /// prepared across the configured bandwidth scales.
    pub window_samples: Option<usize>,
}

impl Default for BandwidthConsensusStreamConfig {
    fn default() -> Self {
        Self {
            scales: vec![0.8, 1.0, 1.2],
            relative_support_floor: 1e-6,
            required_agreement: 1.0,
            window_samples: None,
        }
    }
}

/// One immutable target-time column finalized by rolling consensus.
#[derive(Clone, Debug)]
pub struct BandwidthConsensusStreamFrame {
    /// Absolute target sample represented by this column.
    pub sample_index: usize,
    /// Target time in seconds.
    pub time_seconds: f64,
    /// Reassigned energy column for every scale, in configured scale order.
    pub scale_energy_columns: Vec<Array1<f64>>,
    /// Per-scale maxima used to normalize the active rolling maps.
    pub normalization_maxima: Array1<f64>,
    /// Fraction of normalized scale columns above the support floor.
    pub agreement: Array1<f64>,
    /// Channels meeting the configured required agreement.
    pub consensus_mask: Array1<bool>,
    /// Required-agreement order statistic of the normalized scale columns.
    pub salience: Array1<f64>,
}

/// Per-scale causal analyses and any rolling consensus column released for an
/// accepted input sample.
#[derive(Clone, Debug)]
pub struct BandwidthConsensusStreamStep {
    /// Causal reassignment result for every configured scale.
    pub scale_steps: Vec<ReassignmentStreamStep>,
    /// Index of the unique unscaled (`1.0`) analysis.
    pub baseline_index: usize,
    /// Oldest target column finalized by this input, once the window is full.
    pub consensus: Option<BandwidthConsensusStreamFrame>,
}

impl BandwidthConsensusStreamStep {
    /// The unscaled filterbank and reassignment step.
    pub fn baseline(&self) -> &ReassignmentStreamStep {
        &self.scale_steps[self.baseline_index]
    }
}

/// Immediate filterbank and causal-reassignment results for one input sample.
///
/// Unlike [`super::ReassignmentResult`], a stream step is not deposited into a
/// finite time grid. This permits processing an input with no known end. A
/// caller can threshold `source_energy` and deposit accepted contributions on
/// any finite, rolling, or sparse target representation.
#[derive(Clone, Debug)]
pub struct ReassignmentStreamStep {
    /// Ordinary GCFB v2.34 streaming output for the accepted input sample.
    pub filterbank: StreamStep,
    /// Ungained causal complex analysis coefficient before phase transport.
    pub coefficient: Array1<Complex64>,
    /// Gained analytic-representation energy, `gain^2 * |C|^2 / 2`, per
    /// source channel.
    pub source_energy: Array1<f64>,
    /// Causal reassigned time in seconds, per source channel.
    pub t_hat: Array1<f64>,
    /// Causal reassigned frequency in hertz, per source channel. Dynamic mode
    /// uses the wrapped backward phase increment of consecutive realized
    /// coefficients, so the first nonzero coefficient has no valid frequency.
    pub f_hat: Array1<f64>,
    /// Whether the corresponding coefficient is nonzero and both coordinates
    /// are finite. No relative coefficient floor is applied online.
    pub coordinate_mask: Array1<bool>,
    /// Phase-transported source contribution before target-grid interpolation.
    /// For a valid coordinate its squared magnitude equals `source_energy`.
    pub phase_contribution: Array1<Complex64>,
}

/// Bounded-memory, zero-latency causal approximation of GCFB reassignment.
///
/// The batch reassignment APIs use a whole-signal DFT to construct their
/// acausal imaginary and derivative branches. This stream instead pairs the
/// real passive gammachirp with its model-native sine quadrature. Fixed filters
/// use the derivative of that causal complex atom; sample-dynamic filters use
/// the realized coefficient's backward phase increment so changes in the
/// HP-AF are included. The ordinary real GCFB path remains the same, but
/// complex coefficients and reassigned coordinates are only expected to be
/// similar to batch results away from finite-signal boundaries.
///
/// Static, level, and dynamic sample-base control are supported. Dynamic
/// frame-base control is rejected because its transported result belongs to a
/// centered frame grid rather than to one immediate sample event.
#[derive(Clone, Debug)]
pub struct ReassignmentStream {
    filterbank: GcfbStream,
    atoms: CausalAtomBank,
    coefficients: AcfCoef,
    states: Vec<[ComplexCascadeState; 3]>,
    previous_coefficients: Array1<Complex64>,
}

impl ReassignmentStream {
    /// Prepare a causal reassignment stream and its ordinary GCFB processor.
    pub fn new(gc_param: GcParam) -> Result<Self> {
        Self::new_internal(gc_param, None, None)
    }

    pub(super) fn new_with_bandwidth_peak_lock(
        gc_param: GcParam,
        hearing_loss: &HLoss,
        peak_grid: Arc<BandwidthPeakGrid>,
    ) -> Result<Self> {
        Self::new_internal(gc_param, Some(hearing_loss), Some(peak_grid))
    }

    fn new_internal(
        gc_param: GcParam,
        hearing_loss: Option<&HLoss>,
        peak_grid: Option<Arc<BandwidthPeakGrid>>,
    ) -> Result<Self> {
        let filterbank = if let (Some(hearing_loss), Some(peak_grid)) = (hearing_loss, peak_grid) {
            GcfbStream::new_with_bandwidth_peak_lock(gc_param, hearing_loss, peak_grid)?
        } else if let Some(hearing_loss) = hearing_loss {
            GcfbStream::new_with_preserved_hearing_loss(gc_param, hearing_loss)?
        } else {
            GcfbStream::new(gc_param)?
        };
        let param = filterbank.gc_param();
        if param.ctrl == ControlMode::Dynamic && param.dyn_hpaf.str_prc.contains("frame") {
            return Err(Error::Unsupported(
                "causal reassignment streaming supports static, level, and dynamic sample-base control only"
                    .into(),
            ));
        }
        let response = filterbank.gc_resp();
        let correction = prepare_input_correction_fir(param)?;
        let prepared_atoms = causal_atoms(param, response, &correction)?;
        let atoms = CausalAtomBank::new(prepared_atoms.analytic, prepared_atoms.derivative)?;
        let coefficients = initial_hpaf_coefficients(param, response)?;
        let channels = param.num_ch;
        let states = (0..param.num_ch)
            .map(|_| std::array::from_fn(|_| ComplexCascadeState::default()))
            .collect();
        Ok(Self {
            filterbank,
            atoms,
            coefficients,
            states,
            previous_coefficients: Array1::from_elem(channels, Complex64::new(0.0, 0.0)),
        })
    }

    /// Process one finite input sample and immediately emit its stream step.
    ///
    /// Non-finite input is rejected before either the ordinary filterbank or
    /// causal reassignment histories advance.
    pub fn process_sample(&mut self, sample: f64) -> Result<ReassignmentStreamStep> {
        if !sample.is_finite() {
            return Err(Error::InvalidParameter(
                "input sound samples must be finite".into(),
            ));
        }
        let sample_index = self.filterbank.samples_processed();
        let sample_rate = self.filterbank.gc_param().fs;
        let weighted_sample = sample_index as f64 / sample_rate * sample;
        let filterbank = self.filterbank.process_sample(sample)?;
        debug_assert_eq!(filterbank.sample_index, sample_index);

        if self.filterbank.gc_param().ctrl == ControlMode::Dynamic {
            let Some(DcgcEvent::Sample { fr2: Some(fr2), .. }) = filterbank.event.as_ref() else {
                return Err(Error::Numerical(
                    "dynamic causal reassignment did not receive sample-domain filter centers"
                        .into(),
                ));
            };
            if sample_index.is_multiple_of(self.filterbank.gc_param().num_update_asym_cmp) {
                self.coefficients = make_asym_cmp_filters_v2(
                    sample_rate,
                    fr2.as_slice().unwrap(),
                    self.filterbank.gc_resp().b2_val.as_slice().unwrap(),
                    self.filterbank.gc_resp().c2_val.as_slice().unwrap(),
                )?;
            }
        }

        let atom_outputs = self.atoms.process_sample(sample, weighted_sample);
        let channels = self.filterbank.gc_param().num_ch;
        let mut coefficient = Array1::from_elem(channels, Complex64::new(0.0, 0.0));
        let mut time_weighted = coefficient.clone();
        let mut derivative = coefficient.clone();
        for ch in 0..channels {
            let inputs = [
                atom_outputs.coefficient[ch],
                atom_outputs.time_weighted[ch],
                atom_outputs.derivative[ch],
            ];
            let values: [Complex64; 3] = std::array::from_fn(|pass| {
                self.states[ch][pass].process(inputs[pass], &self.coefficients, ch)
            });
            coefficient[ch] = values[0];
            time_weighted[ch] = values[1];
            derivative[ch] = values[2];

            let expected = if self.filterbank.gc_param().ctrl == ControlMode::Dynamic {
                let Some(DcgcEvent::Sample { dcgc_out, .. }) = filterbank.event.as_ref() else {
                    unreachable!("dynamic sample event was validated above");
                };
                dcgc_out[ch] / self.filterbank.gc_resp().gain_factor[ch]
            } else {
                filterbank.scgc_smpl[ch]
            };
            let tolerance = 2e-8 * expected.abs().max(coefficient[ch].re.abs()).max(1.0);
            if !coefficient[ch].re.is_finite() || (coefficient[ch].re - expected).abs() > tolerance
            {
                return Err(Error::Numerical(format!(
                    "causal reassignment real branch diverged from the GCFB stream in channel {ch}: {} versus {expected}",
                    coefficient[ch].re,
                )));
            }
        }

        let gains = &self.filterbank.gc_resp().gain_factor;
        let centers = &self.filterbank.gc_resp().fr1;
        let source_time = sample_index as f64 / sample_rate;
        let mut source_energy = Array1::zeros(channels);
        let mut t_hat = Array1::from_elem(channels, f64::NAN);
        let mut f_hat = Array1::from_elem(channels, f64::NAN);
        let mut coordinate_mask = Array1::from_elem(channels, false);
        let mut phase_contribution = Array1::from_elem(channels, Complex64::new(0.0, 0.0));
        for ch in 0..channels {
            let norm = coefficient[ch].norm_sqr();
            source_energy[ch] = gains[ch].powi(2) * norm / 2.0;
            if !norm.is_finite() || !source_energy[ch].is_finite() {
                return Err(Error::Numerical(format!(
                    "non-finite causal reassignment energy in channel {ch}"
                )));
            }
            if norm == 0.0 {
                continue;
            }
            let time = (time_weighted[ch] / coefficient[ch]).re;
            let frequency = if self.filterbank.gc_param().ctrl == ControlMode::Dynamic {
                let previous = self.previous_coefficients[ch];
                if previous.norm_sqr() == 0.0 {
                    continue;
                }
                let phase_increment = coefficient[ch] * previous.conj();
                sample_rate * phase_increment.im.atan2(phase_increment.re) / (2.0 * PI)
            } else {
                (derivative[ch] / coefficient[ch]).im / (2.0 * PI)
            };
            if !time.is_finite() || !frequency.is_finite() {
                continue;
            }
            t_hat[ch] = time;
            f_hat[ch] = frequency;
            coordinate_mask[ch] = true;
            let phase = PI * (centers[ch] + frequency) * (time - source_time);
            phase_contribution[ch] =
                coefficient[ch] * (gains[ch].abs() / SQRT_2) * Complex64::from_polar(1.0, phase);
        }
        self.previous_coefficients.assign(&coefficient);

        Ok(ReassignmentStreamStep {
            filterbank,
            coefficient,
            source_energy,
            t_hat,
            f_hat,
            coordinate_mask,
            phase_contribution,
        })
    }

    /// Prepared GCFB parameters used by both real and complex paths.
    pub fn gc_param(&self) -> &GcParam {
        self.filterbank.gc_param()
    }

    /// Prepared time-invariant GCFB response metadata.
    pub fn gc_resp(&self) -> &GcResp {
        self.filterbank.gc_resp()
    }

    /// Alias for [`Self::gc_param`].
    pub fn prepared_param(&self) -> &GcParam {
        self.gc_param()
    }

    /// Alias for [`Self::gc_resp`].
    pub fn prepared_response(&self) -> &GcResp {
        self.gc_resp()
    }

    /// This causal construction emits every accepted sample immediately.
    pub fn latency_samples(&self) -> usize {
        0
    }

    /// Number of successfully accepted input samples.
    pub fn samples_processed(&self) -> usize {
        self.filterbank.samples_processed()
    }

    /// Number of sample positions currently represented in the FIR histories.
    pub fn buffered_samples(&self) -> usize {
        self.atoms.buffered_samples()
    }

    /// Fixed maximum number of sample positions retained by the FIR histories.
    pub fn max_buffered_samples(&self) -> usize {
        self.atoms.max_buffered_samples()
    }
}

/// Bounded-memory rolling consensus across causal bandwidth-scale analyses.
///
/// Each accepted input is processed immediately by one [`ReassignmentStream`]
/// per configured scale. Once the rolling window is full, the oldest target
/// column is normalized against the still-live target window and emitted.
/// Consequently, the stream has `window_samples - 1` samples of consensus
/// latency while its per-scale causal steps remain immediate.
///
/// Batch bandwidth consensus normalizes against an entire finite signal and
/// uses acausal analytic atoms. Rolling results are therefore expected to be
/// similar rather than identical to [`super::gcfb_v234_with_bandwidth_consensus`].
/// Scaled streams use their own causal level histories. At each update, their
/// HP-AF centers match the unscaled reference response evaluated at that
/// scale's own realized ratio at its continuous-DTFT main-lobe maximum. The
/// internal FFT only brackets that maximum. Because the realized baseline ratio
/// can differ, controller-induced peak drift remains part of rolling consensus.
/// Dynamic control also routes the unscaled baseline through this peak-lock
/// path; an unbracketed or unverifiable conditional solve permanently
/// terminates the stream.
#[derive(Clone, Debug)]
pub struct BandwidthConsensusStream {
    streams: Vec<ReassignmentStream>,
    scales: Vec<f64>,
    baseline_index: usize,
    relative_support_floor: f64,
    required_count: usize,
    window_samples: usize,
    next_output_sample: usize,
    frequency_axis_erb: Array1<f64>,
    rolling_maps: Vec<RollingScaleMap>,
    scale_metadata: Vec<BandwidthScaleMetadata>,
    failed: bool,
}

impl BandwidthConsensusStream {
    /// Prepare the causal scale ensemble and bounded rolling target maps.
    pub fn new(gc_param: GcParam, config: BandwidthConsensusStreamConfig) -> Result<Self> {
        let baseline_index = validate_consensus_parameters(
            &config.scales,
            config.relative_support_floor,
            config.required_agreement,
        )?;
        if config.window_samples == Some(0) {
            return Err(Error::InvalidParameter(
                "bandwidth consensus stream window must contain at least one sample".into(),
            ));
        }

        let mut baseline = ReassignmentStream::new(gc_param.clone())?;
        let hearing_loss = baseline.gc_param().hloss.clone();
        let reference_param = baseline.gc_param().clone();
        let reference_response = baseline.gc_resp().clone();
        let peak_grid = prepare_bandwidth_peak_grid(
            &gc_param,
            &config.scales,
            &reference_param,
            &reference_response,
        )?;
        if reference_param.ctrl == ControlMode::Dynamic {
            baseline = ReassignmentStream::new_with_bandwidth_peak_lock(
                gc_param.clone(),
                &hearing_loss,
                peak_grid.clone(),
            )?;
        }
        let (nominal_ratios, _) =
            initial_asymmetric_ratio_and_centers(&reference_param, &reference_response);
        let nominal_peaks = peak_grid.nominal_peak_frequencies_hz(&nominal_ratios)?;
        let channels = baseline.gc_param().num_ch;
        let sample_rate = baseline.gc_param().fs;
        let frequency_axis_hz = baseline.gc_param().fr1.clone();
        let mut streams: Vec<Option<ReassignmentStream>> =
            (0..config.scales.len()).map(|_| None).collect();
        streams[baseline_index] = Some(baseline);
        for (index, &scale) in config.scales.iter().enumerate() {
            if index == baseline_index {
                continue;
            }
            let stream = ReassignmentStream::new_with_bandwidth_peak_lock(
                scale_bandwidths(gc_param.clone(), scale),
                &hearing_loss,
                peak_grid.clone(),
            )?;
            if stream.gc_param().num_ch != channels
                || stream.gc_param().fs != sample_rate
                || stream.gc_param().fr1 != frequency_axis_hz
            {
                return Err(Error::Numerical(
                    "bandwidth consensus streams do not share a target grid".into(),
                ));
            }
            streams[index] = Some(stream);
        }
        let streams: Vec<ReassignmentStream> = streams
            .into_iter()
            .map(|stream| stream.expect("every validated scale was prepared"))
            .collect();
        let scale_metadata = streams
            .iter()
            .zip(&config.scales)
            .map(|(stream, &scale)| {
                Ok(BandwidthScaleMetadata {
                    scale,
                    carrier_frequencies_hz: stream.gc_resp().fr1.clone(),
                    nominal_peak_frequencies_hz: nominal_peaks.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let derived_window = streams
            .iter()
            .map(ReassignmentStream::max_buffered_samples)
            .max()
            .expect("bandwidth consensus requires at least two scales");
        let window_samples = config.window_samples.unwrap_or(derived_window);
        let capacity = window_samples.checked_add(1).ok_or_else(|| {
            Error::InvalidParameter("bandwidth consensus stream window is too large".into())
        })?;
        channels.checked_mul(capacity).ok_or_else(|| {
            Error::InvalidParameter("bandwidth consensus stream dimensions are too large".into())
        })?;
        capacity
            .checked_next_power_of_two()
            .and_then(|leaves| leaves.checked_mul(2))
            .ok_or_else(|| {
                Error::InvalidParameter("bandwidth consensus stream window is too large".into())
            })?;
        let rolling_maps = (0..streams.len())
            .map(|_| RollingScaleMap::new(channels, capacity))
            .collect();
        let (frequency_axis_erb, _) = utils::freq2erb(frequency_axis_hz.as_slice().unwrap());
        let required_count = (config.required_agreement * streams.len() as f64).ceil() as usize;
        Ok(Self {
            streams,
            scales: config.scales,
            baseline_index,
            relative_support_floor: config.relative_support_floor,
            required_count,
            window_samples,
            next_output_sample: 0,
            frequency_axis_erb,
            rolling_maps,
            scale_metadata,
            failed: false,
        })
    }

    /// Process one finite sample and possibly finalize one consensus column.
    ///
    /// Non-finite input is rejected before any scale advances. A processing
    /// failure after that validation makes the aggregate stream terminal.
    pub fn process_sample(&mut self, sample: f64) -> Result<BandwidthConsensusStreamStep> {
        if self.failed {
            return Err(Error::Numerical(
                "bandwidth consensus stream cannot continue after a previous processing error"
                    .into(),
            ));
        }
        if !sample.is_finite() {
            return Err(Error::InvalidParameter(
                "input sound samples must be finite".into(),
            ));
        }
        match self.process_finite_sample(sample) {
            Ok(step) => Ok(step),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn process_finite_sample(&mut self, sample: f64) -> Result<BandwidthConsensusStreamStep> {
        let sample_index = self.samples_processed();
        let mut scale_steps = Vec::with_capacity(self.streams.len());
        for stream in &mut self.streams {
            let step = stream.process_sample(sample)?;
            if step.filterbank.sample_index != sample_index {
                return Err(Error::Numerical(
                    "bandwidth consensus scale streams lost sample alignment".into(),
                ));
            }
            scale_steps.push(step);
        }

        let live_end = self
            .next_output_sample
            .checked_add(self.window_samples)
            .ok_or_else(|| Error::Numerical("stream sample index overflow".into()))?;
        let sample_rate = self.gc_param().fs;
        for (map, step) in self.rolling_maps.iter_mut().zip(&scale_steps) {
            deposit_stream_step(
                map,
                step,
                sample_rate,
                self.next_output_sample,
                live_end,
                self.frequency_axis_erb.as_slice().unwrap(),
            )?;
        }

        let completed_samples = sample_index
            .checked_add(1)
            .ok_or_else(|| Error::Numerical("stream sample index overflow".into()))?;
        let consensus = if completed_samples >= self.window_samples {
            Some(self.finalize_column(sample_index)?)
        } else {
            None
        };
        Ok(BandwidthConsensusStreamStep {
            scale_steps,
            baseline_index: self.baseline_index,
            consensus,
        })
    }

    fn finalize_column(&mut self, active_end: usize) -> Result<BandwidthConsensusStreamFrame> {
        let sample_index = self.next_output_sample;
        debug_assert!(sample_index <= active_end);
        let mut scale_energy_columns = Vec::with_capacity(self.rolling_maps.len());
        let mut normalization_maxima = Array1::zeros(self.rolling_maps.len());
        for (scale, map) in self.rolling_maps.iter().enumerate() {
            scale_energy_columns.push(map.column(sample_index));
            normalization_maxima[scale] = map.maximum(sample_index, active_end);
        }

        let channels = self.gc_param().num_ch;
        let mut agreement = Array1::zeros(channels);
        let mut consensus_mask = Array1::from_elem(channels, false);
        let mut salience = Array1::zeros(channels);
        for ch in 0..channels {
            let mut normalized: Vec<f64> = scale_energy_columns
                .iter()
                .zip(&normalization_maxima)
                .map(|(column, &maximum)| {
                    if maximum > 0.0 {
                        column[ch] / maximum
                    } else {
                        0.0
                    }
                })
                .collect();
            if normalized
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(Error::Numerical(
                    "non-finite rolling bandwidth-consensus energy".into(),
                ));
            }
            let support_count = normalized
                .iter()
                .filter(|&&value| value > self.relative_support_floor)
                .count();
            agreement[ch] = support_count as f64 / self.streams.len() as f64;
            consensus_mask[ch] = support_count >= self.required_count;
            normalized.sort_by(|left, right| right.total_cmp(left));
            salience[ch] = normalized[self.required_count - 1];
        }

        for map in &mut self.rolling_maps {
            map.clear_column(sample_index);
        }
        self.next_output_sample = self
            .next_output_sample
            .checked_add(1)
            .ok_or_else(|| Error::Numerical("stream sample index overflow".into()))?;
        Ok(BandwidthConsensusStreamFrame {
            sample_index,
            time_seconds: sample_index as f64 / self.gc_param().fs,
            scale_energy_columns,
            normalization_maxima,
            agreement,
            consensus_mask,
            salience,
        })
    }

    /// Consume a finite stream and flush its not-yet-finalized target columns.
    ///
    /// No causal filter tail is synthesized. Target columns beyond the last
    /// accepted input sample are discarded, matching the finite batch grid.
    pub fn finish(mut self) -> Result<Vec<BandwidthConsensusStreamFrame>> {
        if self.failed {
            return Err(Error::Numerical(
                "bandwidth consensus stream cannot finish after a processing error".into(),
            ));
        }
        let samples = self.samples_processed();
        if samples == 0 {
            return Err(Error::InvalidParameter(
                "input sound must be non-empty and finite".into(),
            ));
        }
        let last_sample = samples - 1;
        let live_end = self
            .next_output_sample
            .checked_add(self.window_samples)
            .ok_or_else(|| Error::Numerical("stream sample index overflow".into()))?;
        for target in samples..=live_end {
            for map in &mut self.rolling_maps {
                map.clear_column(target);
            }
        }
        let mut frames = Vec::with_capacity(samples - self.next_output_sample);
        while self.next_output_sample <= last_sample {
            frames.push(self.finalize_column(last_sample)?);
        }
        Ok(frames)
    }

    /// Validated bandwidth scales in processing order.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Index of the unique unscaled (`1.0`) stream.
    pub fn baseline_index(&self) -> usize {
        self.baseline_index
    }

    /// Retuned carriers and nominal continuous peak frequencies for every scale.
    pub fn scale_metadata(&self) -> &[BandwidthScaleMetadata] {
        &self.scale_metadata
    }

    /// Prepared parameters for the unscaled stream.
    pub fn gc_param(&self) -> &GcParam {
        self.streams[self.baseline_index].gc_param()
    }

    /// Prepared response metadata for the unscaled stream.
    pub fn gc_resp(&self) -> &GcResp {
        self.streams[self.baseline_index].gc_resp()
    }

    /// Resolved rolling normalization horizon.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Number of samples before a target column is finalized.
    pub fn latency_samples(&self) -> usize {
        self.window_samples - 1
    }

    /// Number of successfully accepted input samples.
    pub fn samples_processed(&self) -> usize {
        self.streams[self.baseline_index].samples_processed()
    }

    /// Number of live target columns contributing to rolling normalization.
    pub fn buffered_target_samples(&self) -> usize {
        self.samples_processed()
            .saturating_sub(self.next_output_sample)
            .min(self.window_samples)
    }

    /// Fixed number of target slots, including one interpolation lookahead.
    pub fn max_buffered_target_samples(&self) -> usize {
        self.window_samples + 1
    }

    /// Largest current causal-atom history among the scale streams.
    pub fn buffered_scale_samples(&self) -> usize {
        self.streams
            .iter()
            .map(ReassignmentStream::buffered_samples)
            .max()
            .unwrap_or(0)
    }

    /// Largest fixed causal-atom history among the scale streams.
    pub fn max_buffered_scale_samples(&self) -> usize {
        self.streams
            .iter()
            .map(ReassignmentStream::max_buffered_samples)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
struct RollingScaleMap {
    energy: Array2<f64>,
    column_maxima: Vec<f64>,
    maximum_tree: MaximumTree,
}

impl RollingScaleMap {
    fn new(channels: usize, capacity: usize) -> Self {
        Self {
            energy: Array2::zeros((channels, capacity)),
            column_maxima: vec![0.0; capacity],
            maximum_tree: MaximumTree::new(capacity),
        }
    }

    fn add(&mut self, channel: usize, target_sample: usize, energy: f64) -> Result<()> {
        if !energy.is_finite() || energy < 0.0 {
            return Err(Error::Numerical(
                "non-finite or negative rolling reassignment energy".into(),
            ));
        }
        if energy == 0.0 {
            return Ok(());
        }
        let slot = target_sample % self.energy.ncols();
        self.energy[[channel, slot]] += energy;
        let value = self.energy[[channel, slot]];
        if !value.is_finite() {
            return Err(Error::Numerical(
                "rolling reassignment energy overflowed".into(),
            ));
        }
        if value > self.column_maxima[slot] {
            self.column_maxima[slot] = value;
            self.maximum_tree.set(slot, value);
        }
        Ok(())
    }

    fn column(&self, target_sample: usize) -> Array1<f64> {
        self.energy
            .column(target_sample % self.energy.ncols())
            .to_owned()
    }

    fn maximum(&self, start_sample: usize, end_sample: usize) -> f64 {
        debug_assert!(start_sample <= end_sample);
        debug_assert!(end_sample - start_sample < self.energy.ncols());
        let capacity = self.energy.ncols();
        let start = start_sample % capacity;
        let end = end_sample % capacity;
        if start <= end {
            self.maximum_tree.range_max(start, end + 1)
        } else {
            self.maximum_tree
                .range_max(start, capacity)
                .max(self.maximum_tree.range_max(0, end + 1))
        }
    }

    fn clear_column(&mut self, target_sample: usize) {
        let slot = target_sample % self.energy.ncols();
        self.energy.column_mut(slot).fill(0.0);
        self.column_maxima[slot] = 0.0;
        self.maximum_tree.set(slot, 0.0);
    }
}

#[derive(Clone, Debug)]
struct MaximumTree {
    leaf_count: usize,
    values: Vec<f64>,
}

impl MaximumTree {
    fn new(len: usize) -> Self {
        let leaf_count = len
            .checked_next_power_of_two()
            .expect("rolling window capacity was validated");
        Self {
            leaf_count,
            values: vec![0.0; 2 * leaf_count],
        }
    }

    fn set(&mut self, index: usize, value: f64) {
        let mut node = self.leaf_count + index;
        self.values[node] = value;
        while node > 1 {
            node /= 2;
            self.values[node] = self.values[2 * node].max(self.values[2 * node + 1]);
        }
    }

    fn range_max(&self, mut start: usize, mut end: usize) -> f64 {
        let mut maximum: f64 = 0.0;
        start += self.leaf_count;
        end += self.leaf_count;
        while start < end {
            if !start.is_multiple_of(2) {
                maximum = maximum.max(self.values[start]);
                start += 1;
            }
            if !end.is_multiple_of(2) {
                end -= 1;
                maximum = maximum.max(self.values[end]);
            }
            start /= 2;
            end /= 2;
        }
        maximum
    }
}

fn deposit_stream_step(
    map: &mut RollingScaleMap,
    step: &ReassignmentStreamStep,
    sample_rate: f64,
    live_start: usize,
    live_end: usize,
    frequency_axis_erb: &[f64],
) -> Result<()> {
    for ch in 0..step.source_energy.len() {
        let energy = step.source_energy[ch];
        if !energy.is_finite() || energy < 0.0 {
            return Err(Error::Numerical(
                "non-finite causal energy during rolling consensus".into(),
            ));
        }
        if energy == 0.0 || !step.coordinate_mask[ch] || step.f_hat[ch] <= 0.0 {
            continue;
        }
        let Some(time_weights) =
            rolling_time_weights(step.t_hat[ch], sample_rate, live_start, live_end)
        else {
            continue;
        };
        let (frequency_erb, _) = utils::freq2erb(&[step.f_hat[ch]]);
        let Some(frequency_weights) = super::linear_weights(frequency_axis_erb, frequency_erb[0])
        else {
            continue;
        };
        for (target_sample, time_weight) in time_weights {
            for &(target_channel, frequency_weight) in &frequency_weights {
                map.add(
                    target_channel,
                    target_sample,
                    energy * time_weight * frequency_weight,
                )?;
            }
        }
    }
    Ok(())
}

fn rolling_time_weights(
    time_seconds: f64,
    sample_rate: f64,
    live_start: usize,
    live_end: usize,
) -> Option<[(usize, f64); 2]> {
    let sample = time_seconds * sample_rate;
    if !sample.is_finite() || sample < live_start as f64 || sample > live_end as f64 {
        return None;
    }
    let lower = sample.floor() as usize;
    if lower == live_end || sample == lower as f64 {
        return Some([(lower, 1.0), (lower, 0.0)]);
    }
    let upper = lower.checked_add(1)?;
    if upper > live_end {
        return None;
    }
    let upper_weight = sample - lower as f64;
    Some([(lower, 1.0 - upper_weight), (upper, upper_weight)])
}

fn initial_hpaf_coefficients(param: &GcParam, response: &GcResp) -> Result<AcfCoef> {
    let (_, centers) = initial_asymmetric_ratio_and_centers(param, response);
    match param.ctrl {
        ControlMode::Static | ControlMode::Dynamic => make_asym_cmp_filters_v2(
            param.fs,
            centers.as_slice().unwrap(),
            response.b2_val.as_slice().unwrap(),
            response.c2_val.as_slice().unwrap(),
        ),
        ControlMode::Level => {
            let b2 = Array1::from_elem(param.num_ch, param.lvl_est.b2);
            let c2 = &param.hloss.fb_compression_health * param.lvl_est.c2;
            make_asym_cmp_filters_v2(
                param.fs,
                centers.as_slice().unwrap(),
                b2.as_slice().unwrap(),
                c2.as_slice().unwrap(),
            )
        }
    }
}

struct PreparedCausalAtoms {
    analytic: Vec<Vec<Complex64>>,
    derivative: Vec<Vec<Complex64>>,
}

fn causal_atoms(
    param: &GcParam,
    response: &GcResp,
    correction: &[f64],
) -> Result<PreparedCausalAtoms> {
    let (_, erb_widths) = utils::freq2erb(response.fr1.as_slice().unwrap());
    let mut analytic_atoms = Vec::with_capacity(param.num_ch);
    let mut derivative_atoms = Vec::with_capacity(param.num_ch);
    for ch in 0..param.num_ch {
        let arguments = (
            &[response.fr1[ch]][..],
            param.fs,
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
        );
        let cosine_peak = gammachirp::gammachirp_reference_peak(
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            0.0,
            Carrier::Cosine,
        )?;
        let cosine_raw = gammachirp::gammachirp(
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            0.0,
            Carrier::Cosine,
            Normalization::None,
        )?;
        let sine_raw = gammachirp::gammachirp(
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            0.0,
            Carrier::Sine,
            Normalization::None,
        )?;
        let length = cosine_peak.len_gc[0];
        let pivot = (0..length)
            .max_by(|&left, &right| {
                cosine_raw.gc[[0, left]]
                    .abs()
                    .total_cmp(&cosine_raw.gc[[0, right]].abs())
            })
            .ok_or_else(|| Error::Numerical("causal gammachirp atom is empty".into()))?;
        let raw_pivot = cosine_raw.gc[[0, pivot]];
        if raw_pivot == 0.0 {
            return Err(Error::Numerical(
                "causal gammachirp atom has zero normalization reference".into(),
            ));
        }
        let scale = cosine_peak.gc[[0, pivot]] / raw_pivot;
        let mut atom = Vec::with_capacity(length);
        let mut derivative = Vec::with_capacity(length);
        for sample in 0..length {
            let value = Complex64::new(
                cosine_peak.gc[[0, sample]],
                sine_raw.gc[[0, sample]] * scale,
            );
            atom.push(value);
            if sample == 0 {
                derivative.push(Complex64::new(0.0, 0.0));
                continue;
            }
            let time = sample as f64 / param.fs;
            let envelope_rate =
                (param.n - 1.0) / time - 2.0 * PI * response.b1_val[ch] * erb_widths[ch];
            let angular_frequency = 2.0 * PI * cosine_raw.inst_freq[[0, sample]];
            derivative.push(value * Complex64::new(envelope_rate, angular_frequency));
        }
        analytic_atoms.push(convolve_real_complex(correction, &atom));
        derivative_atoms.push(convolve_real_complex(correction, &derivative));
    }
    Ok(PreparedCausalAtoms {
        analytic: analytic_atoms,
        derivative: derivative_atoms,
    })
}

fn convolve_real_complex(real: &[f64], complex: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); real.len() + complex.len() - 1];
    for (real_index, &real_value) in real.iter().enumerate() {
        for (complex_index, &complex_value) in complex.iter().enumerate() {
            output[real_index + complex_index] += real_value * complex_value;
        }
    }
    output
}

#[derive(Clone, Debug)]
struct AtomOutputs {
    coefficient: Array1<Complex64>,
    time_weighted: Array1<Complex64>,
    derivative: Array1<Complex64>,
}

#[derive(Clone, Debug)]
struct CausalAtomBank {
    analytic: Vec<Vec<Complex64>>,
    derivative: Vec<Vec<Complex64>>,
    sample_history: Vec<f64>,
    time_history: Vec<f64>,
    next: usize,
    samples_processed: usize,
}

impl CausalAtomBank {
    fn new(analytic: Vec<Vec<Complex64>>, derivative: Vec<Vec<Complex64>>) -> Result<Self> {
        if analytic.is_empty()
            || analytic.len() != derivative.len()
            || analytic.iter().any(Vec::is_empty)
            || analytic
                .iter()
                .zip(&derivative)
                .any(|(left, right)| left.len() != right.len())
        {
            return Err(Error::InvalidParameter(
                "causal reassignment atoms must be non-empty and dimensionally matched".into(),
            ));
        }
        let history_len = analytic.iter().map(Vec::len).max().unwrap();
        Ok(Self {
            analytic,
            derivative,
            sample_history: vec![0.0; history_len],
            time_history: vec![0.0; history_len],
            next: 0,
            samples_processed: 0,
        })
    }

    fn process_sample(&mut self, sample: f64, time_weighted: f64) -> AtomOutputs {
        self.sample_history[self.next] = sample;
        self.time_history[self.next] = time_weighted;
        let channels = self.analytic.len();
        let mut coefficient = Array1::from_elem(channels, Complex64::new(0.0, 0.0));
        let mut time_output = coefficient.clone();
        let mut derivative = coefficient.clone();
        for ch in 0..channels {
            let mut history_index = self.next;
            for tap in 0..self.analytic[ch].len() {
                coefficient[ch] += self.analytic[ch][tap] * self.sample_history[history_index];
                time_output[ch] += self.analytic[ch][tap] * self.time_history[history_index];
                derivative[ch] += self.derivative[ch][tap] * self.sample_history[history_index];
                history_index = if history_index == 0 {
                    self.sample_history.len() - 1
                } else {
                    history_index - 1
                };
            }
        }
        self.next = (self.next + 1) % self.sample_history.len();
        self.samples_processed += 1;
        AtomOutputs {
            coefficient,
            time_weighted: time_output,
            derivative,
        }
    }

    fn buffered_samples(&self) -> usize {
        self.samples_processed.min(self.sample_history.len())
    }

    fn max_buffered_samples(&self) -> usize {
        self.sample_history.len()
    }
}
