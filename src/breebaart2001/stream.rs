//! Bounded-memory sample streaming for the Breebaart processing stages.

use std::collections::VecDeque;

use ndarray::{Array1, Array2};

use super::{
    EiConfig, EiDelayConvention, EiIntegrationBoundary, EiUnit, GaussianNoise,
    HybridBinauralConfig, LANCZOS_RADIUS, MonauralConfig, MonauralTemporalMode, PeripheralConfig,
    absolute_threshold_noise_std_amplitude, ihc_section_cutoff_hz, lanczos_8,
};
use crate::gcfb_v234::{
    ControlMode, DcgcEvent, GcParam, GcResp, GcfbStream, StreamStep as GcfbStreamStep,
};
use crate::{Error, Result, dsp};

/// One processed monaural channel vector.
#[derive(Clone, Debug)]
pub struct MonauralStreamSample {
    /// Zero-based index of this sample.
    pub sample_index: usize,
    /// Central-detector monaural values, one per frequency channel.
    pub output: Array1<f64>,
}

/// Bounded-memory causal monaural processor.
///
/// Construct this processor with [`MonauralConfig::streaming`] or another
/// configuration whose temporal mode is [`MonauralTemporalMode::AmtCausal`].
#[derive(Clone, Debug)]
pub struct MonauralStream {
    channels: usize,
    sample_rate_hz: f64,
    config: MonauralConfig,
    pole: f64,
    previous: Array1<f64>,
    samples_processed: usize,
}

impl MonauralStream {
    /// Prepare a causal monaural stream with a fixed channel count.
    pub fn new(channels: usize, sample_rate_hz: f64, config: MonauralConfig) -> Result<Self> {
        validate_monaural_stream_config(channels, sample_rate_hz, &config)?;
        let pole = (-1.0 / (sample_rate_hz * config.integration_time_constant_seconds)).exp();
        Ok(Self {
            channels,
            sample_rate_hz,
            config,
            pole,
            previous: Array1::zeros(channels),
            samples_processed: 0,
        })
    }

    /// Process one finite channel vector.
    ///
    /// Invalid input is rejected before any filter or counter state advances.
    pub fn process_sample(&mut self, input: &[f64]) -> Result<MonauralStreamSample> {
        if input.len() != self.channels || input.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidParameter(format!(
                "monaural stream samples must contain exactly {} finite channel values",
                self.channels
            )));
        }
        let feedforward = 1.0 - self.pole;
        let mut output = Array1::zeros(self.channels);
        for channel in 0..self.channels {
            self.previous[channel] =
                feedforward * input[channel] + self.pole * self.previous[channel];
            output[channel] = self.config.sensitivity * self.previous[channel];
        }
        let sample_index = self.samples_processed;
        self.samples_processed += 1;
        Ok(MonauralStreamSample {
            sample_index,
            output,
        })
    }

    /// Number of frequency channels expected by [`Self::process_sample`].
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Prepared sample rate in hertz.
    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    /// Causal configuration used by this processor.
    pub fn config(&self) -> &MonauralConfig {
        &self.config
    }

    /// Number of successfully accepted samples.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }
}

fn validate_monaural_stream_config(
    channels: usize,
    sample_rate_hz: f64,
    config: &MonauralConfig,
) -> Result<()> {
    if channels == 0 {
        return Err(Error::InvalidParameter(
            "monaural stream channel count must be positive".into(),
        ));
    }
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !config.integration_time_constant_seconds.is_finite()
        || config.integration_time_constant_seconds <= 0.0
        || !config.sensitivity.is_finite()
        || config.sensitivity < 0.0
    {
        return Err(Error::InvalidParameter(
            "monaural sample rate and time constant must be finite and positive, and sensitivity must be finite and non-negative"
                .into(),
        ));
    }
    if config.temporal_mode != MonauralTemporalMode::AmtCausal {
        return Err(Error::InvalidParameter(
            "monaural streaming requires the causal temporal mode; use MonauralConfig::streaming()"
                .into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct DelayShift {
    shift: f64,
    integer_offset: isize,
    weights: Option<[f64; (2 * LANCZOS_RADIUS) as usize]>,
}

impl DelayShift {
    fn new(shift: f64) -> Result<Self> {
        if !shift.is_finite() || shift.abs() > (isize::MAX as f64 - 2.0 * LANCZOS_RADIUS as f64) {
            return Err(Error::InvalidParameter(
                "EI delay is too large to represent in sample indices".into(),
            ));
        }
        let integer_offset = shift.floor() as isize;
        let fraction = shift - shift.floor();
        let weights = (fraction != 0.0).then(|| {
            std::array::from_fn(|tap_index| {
                let tap_offset = tap_index as isize - (LANCZOS_RADIUS - 1);
                lanczos_8(fraction - tap_offset as f64)
            })
        });
        Ok(Self {
            shift,
            integer_offset,
            weights,
        })
    }

    fn required_offsets(&self) -> (isize, isize) {
        if self.weights.is_some() {
            (
                self.integer_offset - (LANCZOS_RADIUS - 1),
                self.integer_offset + LANCZOS_RADIUS,
            )
        } else {
            (self.integer_offset, self.integer_offset)
        }
    }

    fn sample(
        &self,
        buffer: &VecDeque<Array1<f64>>,
        buffer_start: usize,
        final_samples: usize,
        channel: usize,
        output_sample: usize,
    ) -> f64 {
        let source = output_sample as f64 + self.shift;
        if source < 0.0 || source > (final_samples - 1) as f64 {
            return 0.0;
        }
        if self.weights.is_none() {
            return buffered_value(
                buffer,
                buffer_start,
                final_samples,
                output_sample as isize + self.integer_offset,
                channel,
            );
        }
        self.weights
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(tap_index, weight)| {
                let tap_offset = tap_index as isize - (LANCZOS_RADIUS - 1);
                weight
                    * buffered_value(
                        buffer,
                        buffer_start,
                        final_samples,
                        output_sample as isize + self.integer_offset + tap_offset,
                        channel,
                    )
            })
            .sum()
    }
}

fn buffered_value(
    buffer: &VecDeque<Array1<f64>>,
    buffer_start: usize,
    final_samples: usize,
    sample: isize,
    channel: usize,
) -> f64 {
    if sample < 0 || sample as usize >= final_samples {
        return 0.0;
    }
    let sample = sample as usize;
    if sample < buffer_start {
        debug_assert!(false, "required EI delay history was discarded");
        return 0.0;
    }
    buffer
        .get(sample - buffer_start)
        .map_or(0.0, |values| values[channel])
}

#[derive(Clone, Debug)]
struct UnitKernel {
    left_shift: DelayShift,
    right_shift: DelayShift,
    left_gain: f64,
    right_gain: f64,
    delay_weight: f64,
}

/// One finalized EI population sample.
#[derive(Clone, Debug)]
pub struct EiStreamSample {
    /// Zero-based index in the peripheral input stream.
    pub sample_index: usize,
    /// EI activity with shape `(unit, frequency channel)`.
    pub activity: Array2<f64>,
}

/// Bounded-memory EI population processor.
///
/// The causal temporal integrator has fixed state. Paper-symmetric fractional
/// delays may add bounded latency; [`Self::process_sample`] returns `None`
/// until an output is final, and [`Self::finish`] emits the zero-extended tail.
#[derive(Clone, Debug)]
pub struct EiStream {
    channels: usize,
    sample_rate_hz: f64,
    units: Vec<EiUnit>,
    config: EiConfig,
    kernels: Vec<UnitKernel>,
    integration_pole: f64,
    integration_states: Array2<f64>,
    noise: GaussianNoise,
    left_buffer: VecDeque<Array1<f64>>,
    right_buffer: VecDeque<Array1<f64>>,
    buffer_start: usize,
    lookahead_samples: usize,
    minimum_offset: isize,
    max_buffered_samples: usize,
    samples_processed: usize,
    next_output_sample: usize,
}

impl EiStream {
    /// Prepare an EI stream with fixed channels, units, and causal settings.
    pub fn new(
        channels: usize,
        sample_rate_hz: f64,
        units: &[EiUnit],
        config: EiConfig,
    ) -> Result<Self> {
        validate_ei_stream_config(channels, sample_rate_hz, units, &config)?;
        let mut kernels = Vec::with_capacity(units.len());
        let mut minimum_offset = 0_isize;
        let mut maximum_offset = 0_isize;
        for unit in units {
            let (left_shift_samples, right_shift_samples) = match config.delay_convention {
                EiDelayConvention::PaperSymmetric => {
                    let half_delay_samples = unit.delay_seconds * sample_rate_hz / 2.0;
                    (half_delay_samples, -half_delay_samples)
                }
                EiDelayConvention::AmtOneSidedInteger => {
                    let delay_samples = (unit.delay_seconds.abs() * sample_rate_hz).round();
                    if unit.delay_seconds > 0.0 {
                        (-delay_samples, 0.0)
                    } else {
                        (0.0, -delay_samples)
                    }
                }
            };
            let left_shift = DelayShift::new(left_shift_samples)?;
            let right_shift = DelayShift::new(right_shift_samples)?;
            for shift in [&left_shift, &right_shift] {
                let (low, high) = shift.required_offsets();
                minimum_offset = minimum_offset.min(low);
                maximum_offset = maximum_offset.max(high);
            }
            kernels.push(UnitKernel {
                left_shift,
                right_shift,
                left_gain: 10_f64.powf(unit.iid_db / 40.0),
                right_gain: 10_f64.powf(-unit.iid_db / 40.0),
                delay_weight: (-unit.delay_seconds.abs()
                    / config.delay_weight_time_constant_seconds)
                    .exp(),
            });
        }
        let lookahead_samples = maximum_offset.max(0) as usize;
        let retained_span = maximum_offset.max(0) as i128 - minimum_offset.min(0) as i128 + 1;
        let max_buffered_samples = usize::try_from(retained_span).map_err(|_| {
            Error::InvalidParameter("EI delay buffer exceeds addressable memory".into())
        })?;
        let integration_pole =
            (-1.0 / (sample_rate_hz * config.integration_time_constant_seconds)).exp();
        Ok(Self {
            channels,
            sample_rate_hz,
            units: units.to_vec(),
            integration_states: Array2::zeros((units.len(), channels)),
            noise: GaussianNoise::new(config.noise_seed),
            config,
            kernels,
            integration_pole,
            left_buffer: VecDeque::new(),
            right_buffer: VecDeque::new(),
            buffer_start: 0,
            lookahead_samples,
            minimum_offset,
            max_buffered_samples,
            samples_processed: 0,
            next_output_sample: 0,
        })
    }

    /// Accept paired finite peripheral channel vectors.
    ///
    /// An indexed event is returned when the sample at the head of the bounded
    /// delay window is final. Invalid input does not advance any state.
    pub fn process_sample(
        &mut self,
        left: &[f64],
        right: &[f64],
    ) -> Result<Option<EiStreamSample>> {
        if left.len() != self.channels
            || right.len() != self.channels
            || left
                .iter()
                .chain(right.iter())
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter(format!(
                "EI stream samples must contain exactly {} finite values per ear",
                self.channels
            )));
        }
        if self.left_buffer.is_empty() {
            self.buffer_start = self.samples_processed;
        }
        self.left_buffer.push_back(Array1::from(left.to_vec()));
        self.right_buffer.push_back(Array1::from(right.to_vec()));
        self.samples_processed += 1;

        let ready = self
            .next_output_sample
            .checked_add(self.lookahead_samples)
            .is_some_and(|required| required < self.samples_processed);
        let event = ready.then(|| self.emit_next(self.samples_processed));
        debug_assert!(self.left_buffer.len() <= self.max_buffered_samples);
        Ok(event)
    }

    fn emit_next(&mut self, final_samples: usize) -> EiStreamSample {
        let sample_index = self.next_output_sample;
        let feedforward = 1.0 - self.integration_pole;
        let mut noise = Array1::zeros(self.channels);
        for channel in 0..self.channels {
            noise[channel] = self.config.internal_noise_std_mu * self.noise.sample();
        }
        let mut activity = Array2::zeros((self.units.len(), self.channels));
        for (unit_index, kernel) in self.kernels.iter().enumerate() {
            for channel in 0..self.channels {
                let left_value = kernel.left_shift.sample(
                    &self.left_buffer,
                    self.buffer_start,
                    final_samples,
                    channel,
                    sample_index,
                ) * kernel.left_gain;
                let right_value = kernel.right_shift.sample(
                    &self.right_buffer,
                    self.buffer_start,
                    final_samples,
                    channel,
                    sample_index,
                ) * kernel.right_gain;
                let instantaneous = (left_value - right_value).powi(2);
                self.integration_states[[unit_index, channel]] = feedforward * instantaneous
                    + self.integration_pole * self.integration_states[[unit_index, channel]];
                let deterministic = self.config.compression_a
                    * kernel.delay_weight
                    * (self.config.compression_b * self.integration_states[[unit_index, channel]]
                        + 1.0)
                        .ln();
                activity[[unit_index, channel]] = deterministic + noise[channel];
            }
        }
        self.next_output_sample += 1;
        self.discard_unneeded_history();
        EiStreamSample {
            sample_index,
            activity,
        }
    }

    fn discard_unneeded_history(&mut self) {
        let keep_from = (self.next_output_sample as isize + self.minimum_offset).max(0) as usize;
        while self.buffer_start < keep_from {
            if self.left_buffer.pop_front().is_none() {
                break;
            }
            self.right_buffer.pop_front();
            self.buffer_start += 1;
        }
    }

    /// Consume a non-empty finite stream and emit its zero-extended delay tail.
    pub fn finish(mut self) -> Result<Vec<EiStreamSample>> {
        if self.samples_processed == 0 {
            return Err(Error::InvalidParameter(
                "EI stream input must be non-empty".into(),
            ));
        }
        let final_samples = self.samples_processed;
        let mut tail = Vec::with_capacity(final_samples - self.next_output_sample);
        while self.next_output_sample < final_samples {
            tail.push(self.emit_next(final_samples));
        }
        Ok(tail)
    }

    /// Fixed frequency-channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Prepared sample rate in hertz.
    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    /// EI units in population-axis order.
    pub fn units(&self) -> &[EiUnit] {
        &self.units
    }

    /// Causal EI configuration used by this stream.
    pub fn config(&self) -> &EiConfig {
        &self.config
    }

    /// Fixed event latency caused by the largest future delay tap.
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Current number of paired channel vectors retained for delay processing.
    pub fn buffered_samples(&self) -> usize {
        self.left_buffer.len()
    }

    /// Maximum number of paired channel vectors this configuration can retain.
    pub fn max_buffered_samples(&self) -> usize {
        self.max_buffered_samples
    }

    /// Number of successfully accepted peripheral samples.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }
}

fn validate_ei_stream_config(
    channels: usize,
    sample_rate_hz: f64,
    units: &[EiUnit],
    config: &EiConfig,
) -> Result<()> {
    if channels == 0 {
        return Err(Error::InvalidParameter(
            "EI stream channel count must be positive".into(),
        ));
    }
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(Error::InvalidParameter(
            "EI sample rate must be finite and positive".into(),
        ));
    }
    if units.is_empty() {
        return Err(Error::InvalidParameter(
            "at least one EI unit is required".into(),
        ));
    }
    if !config.integration_time_constant_seconds.is_finite()
        || config.integration_time_constant_seconds <= 0.0
        || !config.compression_a.is_finite()
        || config.compression_a <= 0.0
        || !config.compression_b.is_finite()
        || config.compression_b <= 0.0
        || !config.delay_weight_time_constant_seconds.is_finite()
        || config.delay_weight_time_constant_seconds <= 0.0
        || !config.internal_noise_std_mu.is_finite()
        || config.internal_noise_std_mu < 0.0
        || config.max_abs_delay_seconds.is_nan()
        || config.max_abs_delay_seconds < 0.0
        || config.max_abs_iid_db.is_nan()
        || config.max_abs_iid_db < 0.0
    {
        return Err(Error::InvalidParameter(
            "EI time constants, compression, and noise must be finite and in their positive ranges; population limits must be non-negative and not NaN"
                .into(),
        ));
    }
    if config.integration_boundary != EiIntegrationBoundary::CausalZeroState {
        return Err(Error::InvalidParameter(
            "EI streaming requires causal zero-state integration; use EiConfig::streaming()".into(),
        ));
    }
    if units.iter().any(|unit| {
        !unit.delay_seconds.is_finite()
            || !unit.iid_db.is_finite()
            || unit.delay_seconds.abs() > config.max_abs_delay_seconds
            || unit.iid_db.abs() > config.max_abs_iid_db
    }) {
        return Err(Error::InvalidParameter(format!(
            "EI units must be finite and lie within +/-{} s and +/-{} dB",
            config.max_abs_delay_seconds, config.max_abs_iid_db
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct IirMemory {
    previous_input: f64,
    previous_output: f64,
}

#[derive(Clone, Debug)]
struct IhcBank {
    channels: usize,
    b: [f64; 2],
    a: [f64; 2],
    state: Vec<IirMemory>,
}

impl IhcBank {
    fn new(channels: usize, sample_rate_hz: f64, cutoff_hz: f64) -> Result<Self> {
        if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 || cutoff_hz >= sample_rate_hz / 2.0 {
            return Err(Error::InvalidParameter(
                "inner-hair-cell cutoff must be finite, positive, and below Nyquist".into(),
            ));
        }
        let section_cutoff = ihc_section_cutoff_hz(cutoff_hz, sample_rate_hz);
        let (b, a) = dsp::first_order_lowpass(section_cutoff, sample_rate_hz);
        Ok(Self {
            channels,
            b,
            a,
            state: vec![IirMemory::default(); channels * 5],
        })
    }

    fn process(&mut self, input: &[f64]) -> Array1<f64> {
        let mut output = Array1::zeros(self.channels);
        for channel in 0..self.channels {
            let mut value = input[channel].max(0.0);
            for stage in 0..5 {
                let memory = &mut self.state[channel * 5 + stage];
                let filtered = self.b[0] * value + self.b[1] * memory.previous_input
                    - self.a[1] * memory.previous_output;
                memory.previous_input = value;
                memory.previous_output = filtered;
                value = filtered;
            }
            output[channel] = value;
        }
        output
    }
}

#[derive(Clone, Debug)]
struct AdaptationBank {
    channels: usize,
    coefficients: [f64; 5],
    states: Vec<[f64; 5]>,
    limiter_parameters: Option<[(f64, f64, f64); 5]>,
    hundred_db_amplitude: f64,
    minimum_normalized: f64,
    minimum_steady: f64,
    model_unit_scale: f64,
}

impl AdaptationBank {
    fn new(
        channels: usize,
        sample_rate_hz: f64,
        amplitude_one_db_spl: f64,
        config: &PeripheralConfig,
    ) -> Result<Self> {
        if config
            .adaptation_time_constants_seconds
            .iter()
            .any(|time| !time.is_finite() || *time <= 0.0)
            || !config.minimum_level_db_spl.is_finite()
            || config.minimum_level_db_spl >= 100.0
            || !amplitude_one_db_spl.is_finite()
            || config
                .overshoot_limit
                .is_some_and(|limit| !limit.is_finite() || limit < 1.0)
        {
            return Err(Error::InvalidParameter(
                "adaptation constants must be positive, the minimum level must be below 100 dB SPL, and the optional AMT overshoot parameter must be at least one"
                    .into(),
            ));
        }
        let hundred_db_amplitude = 10_f64.powf((100.0 - amplitude_one_db_spl) / 20.0);
        let minimum_normalized = 10_f64.powf((config.minimum_level_db_spl - 100.0) / 20.0);
        if !hundred_db_amplitude.is_finite()
            || hundred_db_amplitude <= 0.0
            || !minimum_normalized.is_finite()
            || minimum_normalized <= 0.0
        {
            return Err(Error::InvalidParameter(
                "adaptation level calibration exceeds the finite floating-point range".into(),
            ));
        }
        let root_power = 2_f64.powi(-(config.adaptation_time_constants_seconds.len() as i32));
        let minimum_steady = minimum_normalized.powf(root_power);
        let model_unit_scale = 100.0 / (1.0 - minimum_steady);
        let coefficients = config
            .adaptation_time_constants_seconds
            .map(|time| (-1.0 / (sample_rate_hz * time)).exp());
        let mut initial_states = [0.0; 5];
        let mut stage_minimum = minimum_normalized;
        for state in &mut initial_states {
            stage_minimum = stage_minimum.sqrt();
            *state = stage_minimum;
        }
        let limiter_parameters = if let Some(limit) = config.overshoot_limit
            && limit > 1.0
        {
            let parameters = initial_states.map(|state| {
                let maximum = (1.0 - state * state) * limit - 1.0;
                (2.0 * maximum, -2.0 / maximum, maximum - 1.0)
            });
            if parameters.iter().any(|(factor, exponential, offset)| {
                !factor.is_finite()
                    || *factor <= 0.0
                    || !exponential.is_finite()
                    || !offset.is_finite()
            }) {
                return Err(Error::InvalidParameter(
                    "AMT overshoot parameter is too small for the configured minimum level".into(),
                ));
            }
            Some(parameters)
        } else {
            None
        };
        Ok(Self {
            channels,
            coefficients,
            states: vec![initial_states; channels],
            limiter_parameters,
            hundred_db_amplitude,
            minimum_normalized,
            minimum_steady,
            model_unit_scale,
        })
    }

    fn process(&mut self, input: &[f64]) -> Array1<f64> {
        let mut output = Array1::zeros(self.channels);
        for channel in 0..self.channels {
            let mut value =
                (input[channel] / self.hundred_db_amplitude).max(self.minimum_normalized);
            for stage in 0..5 {
                let mut stage_output = value / self.states[channel][stage].max(f64::MIN_POSITIVE);
                if stage_output > 1.0
                    && let Some(parameters) = self.limiter_parameters
                {
                    let (factor, exponential, offset) = parameters[stage];
                    stage_output =
                        factor / (1.0 + (exponential * (stage_output - 1.0)).exp()) - offset;
                }
                self.states[channel][stage] = self.coefficients[stage]
                    * self.states[channel][stage]
                    + (1.0 - self.coefficients[stage]) * stage_output;
                value = stage_output;
            }
            output[channel] = (value - self.minimum_steady) * self.model_unit_scale;
        }
        output
    }
}

#[derive(Clone, Debug)]
pub(super) struct PeripheralPairStream {
    channels: usize,
    threshold_noise: Option<(f64, GaussianNoise)>,
    left_ihc: IhcBank,
    right_ihc: IhcBank,
    left_adaptation: AdaptationBank,
    right_adaptation: AdaptationBank,
}

impl PeripheralPairStream {
    pub(super) fn new(
        channels: usize,
        sample_rate_hz: f64,
        amplitude_one_db_spl: f64,
        config: &PeripheralConfig,
    ) -> Result<Self> {
        let threshold_noise = absolute_threshold_noise_std_amplitude(
            config.absolute_threshold_noise_level_db_spl,
            amplitude_one_db_spl,
        )?
        .map(|standard_deviation| {
            (
                standard_deviation,
                GaussianNoise::new(config.absolute_threshold_noise_seed),
            )
        });
        Ok(Self {
            channels,
            threshold_noise,
            left_ihc: IhcBank::new(channels, sample_rate_hz, config.ihc_cutoff_hz)?,
            right_ihc: IhcBank::new(channels, sample_rate_hz, config.ihc_cutoff_hz)?,
            left_adaptation: AdaptationBank::new(
                channels,
                sample_rate_hz,
                amplitude_one_db_spl,
                config,
            )?,
            right_adaptation: AdaptationBank::new(
                channels,
                sample_rate_hz,
                amplitude_one_db_spl,
                config,
            )?,
        })
    }

    pub(super) fn process(
        &mut self,
        left: &[f64],
        right: &[f64],
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if left.len() != self.channels
            || right.len() != self.channels
            || left
                .iter()
                .chain(right.iter())
                .any(|value| !value.is_finite())
        {
            return Err(Error::Numerical(
                "paired filterbanks produced invalid sample-domain values".into(),
            ));
        }
        let mut left = left.to_vec();
        let mut right = right.to_vec();
        if let Some((standard_deviation, noise)) = &mut self.threshold_noise {
            for value in &mut left {
                *value += *standard_deviation * noise.sample();
            }
            for value in &mut right {
                *value += *standard_deviation * noise.sample();
            }
        }
        let left_ihc = self.left_ihc.process(&left);
        let right_ihc = self.right_ihc.process(&right);
        Ok((
            self.left_adaptation.process(left_ihc.as_slice().unwrap()),
            self.right_adaptation.process(right_ihc.as_slice().unwrap()),
        ))
    }
}

/// Immediate hybrid outputs and any delayed EI event released by one waveform
/// sample pair.
#[derive(Clone, Debug)]
pub struct HybridBinauralStreamStep {
    /// Zero-based waveform sample index.
    pub sample_index: usize,
    /// Complete left-ear GCFB streaming step.
    pub left_filterbank: GcfbStreamStep,
    /// Complete right-ear GCFB streaming step.
    pub right_filterbank: GcfbStreamStep,
    /// Left-ear adaptation-loop output for this input sample.
    pub left_internal: Array1<f64>,
    /// Right-ear adaptation-loop output for this input sample.
    pub right_internal: Array1<f64>,
    /// Finalized EI activity, delayed only when symmetric fractional shifts
    /// require future samples.
    pub ei_event: Option<EiStreamSample>,
}

/// Bounded-memory waveform-to-EI Breebaart/GCFB hybrid.
#[derive(Clone, Debug)]
pub struct HybridBinauralStream {
    left_filterbank: GcfbStream,
    right_filterbank: GcfbStream,
    peripheral: PeripheralPairStream,
    ei: EiStream,
    center_frequencies_hz: Array1<f64>,
    failed: bool,
}

impl HybridBinauralStream {
    /// Prepare paired sample-mode GCFBs and all causal Breebaart stages.
    pub fn new(units: &[EiUnit], mut config: HybridBinauralConfig) -> Result<Self> {
        if config.filterbank.ctrl == ControlMode::Dynamic {
            config.filterbank.dyn_hpaf.str_prc = "sample-base".into();
        }
        let amplitude_one_db_spl = config
            .peripheral
            .amplitude_one_db_spl
            .unwrap_or(config.filterbank.lvl_est.rms2spldb);
        config.filterbank.lvl_est.rms2spldb = amplitude_one_db_spl;
        let left_filterbank = GcfbStream::new(config.filterbank.clone())?;
        let right_filterbank = GcfbStream::new(config.filterbank)?;
        let channels = left_filterbank.gc_param().num_ch;
        let sample_rate_hz = left_filterbank.gc_param().fs;
        let peripheral = PeripheralPairStream::new(
            channels,
            sample_rate_hz,
            amplitude_one_db_spl,
            &config.peripheral,
        )?;
        let ei = EiStream::new(channels, sample_rate_hz, units, config.ei)?;
        let center_frequencies_hz = left_filterbank.gc_resp().fr1.clone();
        Ok(Self {
            left_filterbank,
            right_filterbank,
            peripheral,
            ei,
            center_frequencies_hz,
            failed: false,
        })
    }

    /// Process one finite left/right waveform sample pair.
    ///
    /// Non-finite inputs are rejected before either ear advances.
    /// Any later processing error makes the paired stream terminal so the two
    /// ears cannot be resumed with different causal histories.
    pub fn process_sample(&mut self, left: f64, right: f64) -> Result<HybridBinauralStreamStep> {
        if self.failed {
            return Err(Error::Numerical(
                "hybrid stream cannot continue after a previous processing error".into(),
            ));
        }
        if !left.is_finite() || !right.is_finite() {
            return Err(Error::InvalidParameter(
                "left and right waveform samples must be finite".into(),
            ));
        }
        match self.process_finite_sample_pair(left, right) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn process_finite_sample_pair(
        &mut self,
        left: f64,
        right: f64,
    ) -> Result<HybridBinauralStreamStep> {
        let left_filterbank = self.left_filterbank.process_sample(left)?;
        let right_filterbank = self.right_filterbank.process_sample(right)?;
        let left_dcgc = sample_dcgc(&left_filterbank)?;
        let right_dcgc = sample_dcgc(&right_filterbank)?;
        let (left_internal, right_internal) = self.peripheral.process(left_dcgc, right_dcgc)?;
        let ei_event = self.ei.process_sample(
            left_internal.as_slice().unwrap(),
            right_internal.as_slice().unwrap(),
        )?;
        let sample_index = left_filterbank.sample_index;
        debug_assert_eq!(right_filterbank.sample_index, sample_index);
        Ok(HybridBinauralStreamStep {
            sample_index,
            left_filterbank,
            right_filterbank,
            left_internal,
            right_internal,
            ei_event,
        })
    }

    /// Consume a non-empty finite stream and return its delayed EI tail.
    pub fn finish(self) -> Result<Vec<EiStreamSample>> {
        if self.failed {
            return Err(Error::Numerical(
                "hybrid stream cannot finish after a processing error".into(),
            ));
        }
        self.ei.finish()
    }

    /// Effective sample-mode GCFB parameters shared by both ears.
    pub fn gc_param(&self) -> &GcParam {
        self.left_filterbank.gc_param()
    }

    /// Prepared time-invariant GCFB response metadata.
    pub fn gc_resp(&self) -> &GcResp {
        self.left_filterbank.gc_resp()
    }

    /// GCFB center frequencies in channel order.
    pub fn center_frequencies_hz(&self) -> &Array1<f64> {
        &self.center_frequencies_hz
    }

    /// EI units in population-axis order.
    pub fn units(&self) -> &[EiUnit] {
        self.ei.units()
    }

    /// EI event latency in input samples.
    pub fn latency_samples(&self) -> usize {
        self.ei.latency_samples()
    }

    /// Current number of peripheral samples retained by the EI delay stage.
    pub fn buffered_ei_samples(&self) -> usize {
        self.ei.buffered_samples()
    }

    /// Fixed maximum number of peripheral samples retained by the EI stage.
    pub fn max_buffered_ei_samples(&self) -> usize {
        self.ei.max_buffered_samples()
    }

    /// Number of successfully accepted waveform sample pairs.
    pub fn samples_processed(&self) -> usize {
        self.ei.samples_processed()
    }
}

fn sample_dcgc(step: &GcfbStreamStep) -> Result<&[f64]> {
    match step.event.as_ref() {
        Some(DcgcEvent::Sample { dcgc_out, .. }) => Ok(dcgc_out.as_slice().unwrap()),
        _ => Err(Error::Numerical(
            "hybrid filterbank did not produce a sample-domain dcGC event".into(),
        )),
    }
}
