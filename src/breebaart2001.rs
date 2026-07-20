//! Hybrid binaural processing based on Breebaart et al. (2001).
//!
//! The original model uses a linear gammatone filterbank before inner-hair-cell
//! transduction, adaptation, and excitation-inhibition (EI) processing. The
//! end-to-end function in this module is deliberately a hybrid: it replaces the
//! gammatone stage with this crate's dynamic compressive gammachirp filterbank,
//! while retaining the fifth-order 770 Hz hair-cell low-pass, five adaptation
//! loops, and the EI equations from the paper.
//!
//! # Choosing an entry point
//!
//! - [`breebaart2001_ei`] applies only the EI population. Use it when left and
//!   right peripheral representations are already available in model units.
//! - [`hybrid_binaural`] accepts paired waveforms and runs the GCFB,
//!   inner-hair-cell, adaptation-loop, and EI stages.
//! - [`breebaart2001_monaural`] prepares an adaptation-loop output for the
//!   monaural channels of the central detector.
//! - [`CentralTemplate`] fits and applies the paper's Appendix-B ideal-observer
//!   template to caller-supplied internal representations.
//! - [`MonauralStream`], [`EiStream`], and [`HybridBinauralStream`] provide
//!   bounded-memory causal processing for inputs whose final length is not
//!   known in advance.
//!
//! The EI defaults follow the continuous-time equations in the 2001 paper.
//! AMT 1.6 uses a different delay convention and finite-record integration
//! boundary treatment; use [`EiConfig::amt_1_6`] when reproducing its
//! `breebaart2001_eicell` output. This does not turn [`hybrid_binaural`] into
//! AMT's end-to-end model because the hybrid intentionally retains the GCFB.
//!
//! # Streaming and causality
//!
//! The paper's double-sided exponential and AMT's forward-backward EI filter
//! are acausal. Exact final samples from those modes require the complete
//! finite record, so stream constructors reject them. Use the explicit
//! [`MonauralConfig::streaming`], [`EiConfig::streaming`], and
//! [`HybridBinauralConfig::streaming`] presets to select zero-state causal
//! integration. The EI stream exactly reproduces the batch implementation's
//! Lanczos-8 approximation to the paper-symmetric fractional delays: it retains
//! a fixed interpolation window, returns an event once each output is final,
//! and flushes the bounded zero-extended tail from [`EiStream::finish`].
//!
//! # Processing chain
//!
//! [`hybrid_binaural`] processes each ear independently through a GCFB v2.34,
//! optional absolute-threshold noise, half-wave rectification, a fifth-order
//! low-pass, and five adaptation loops. It then evaluates every requested
//! [`EiUnit`] and returns the resulting population activity. Dynamic GCFB runs
//! are forced to sample mode so that the EI stage retains the waveform's fine
//! structure.
//!
//! # Array layout and units
//!
//! Peripheral matrices have shape `(frequency channel, sample)`. EI activity
//! has shape `(unit, frequency channel, sample)`, with units stored in the same
//! order as the first axis. Times are in seconds, frequencies are in hertz,
//! levels and characteristic IIDs are in decibels, and post-adaptation values
//! are in model units (MU).
//!
//! The absolute-threshold noise in [`PeripheralConfig`] is added before the
//! inner-hair-cell stage and is calibrated in dB SPL. The internal noise in
//! [`EiConfig`] is added after EI compression and is calibrated in MU. They are
//! distinct, independently seeded noise sources.
//!
//! # EI-stage example
//!
//! ```
//! use gammachirp_rs::breebaart2001::{
//!     EiConfig, EiUnit, breebaart2001_ei,
//! };
//! use ndarray::Array2;
//!
//! let left = Array2::from_elem((2, 64), 10.0);
//! let right = left.clone();
//! let units = [EiUnit::default(), EiUnit::new(0.0, 3.0)];
//! let config = EiConfig {
//!     internal_noise_std_mu: 0.0,
//!     ..EiConfig::default()
//! };
//!
//! let activity = breebaart2001_ei(&left, &right, 48_000.0, &units, &config)?;
//! assert_eq!(activity.dim(), (2, 2, 64));
//! // Equal inputs cancel at the unit with zero characteristic ITD and IID.
//! assert!(activity.index_axis(ndarray::Axis(0), 0).iter().all(|&x| x == 0.0));
//! # Ok::<(), gammachirp_rs::Error>(())
//! ```
//!
//! # Reference
//!
//! J. Breebaart, S. van de Par, and A. Kohlrausch, "Binaural processing
//! model based on contralateral inhibition. I. Model structure," JASA 110,
//! 1074--1088 (2001), <https://doi.org/10.1121/1.1383297>.

use ndarray::{Array1, Array2, Array3, Zip};

mod stream;

pub use stream::{
    EiStream, EiStreamSample, HybridBinauralStream, HybridBinauralStreamStep, MonauralStream,
    MonauralStreamSample,
};

#[cfg(test)]
use crate::dsp;
use crate::gcfb_v211::gcfb_v211::ControlMode;
use crate::gcfb_v234::{GcParam, GcfbOutput, gcfb_v234};
use crate::{Error, Result};

/// The characteristic interaural parameters of one EI unit.
///
/// `delay_seconds` is the characteristic ITD, `tau`. Its application is
/// selected by [`EiConfig::delay_convention`]; the default evaluates the left
/// representation at `t + tau / 2` and the right representation at
/// `t - tau / 2`, as in Eq. 3 of the paper. `iid_db` is the paper's
/// characteristic IID, `alpha`, and is applied symmetrically as gains of
/// `10^(alpha/40)` and `10^(-alpha/40)`.
///
/// The default unit has zero characteristic ITD and IID. Constructing a unit
/// does not validate it; [`breebaart2001_ei`] and [`hybrid_binaural`] enforce
/// the population limits in [`EiConfig`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EiUnit {
    /// Characteristic interaural time difference (ITD), in seconds.
    pub delay_seconds: f64,
    /// Characteristic interaural intensity difference (IID), in decibels.
    pub iid_db: f64,
}

impl EiUnit {
    /// Creates an EI unit with the given characteristic ITD and IID.
    ///
    /// Values are validated when the unit is passed to an EI population.
    pub const fn new(delay_seconds: f64, iid_db: f64) -> Self {
        Self {
            delay_seconds,
            iid_db,
        }
    }
}

impl Default for EiUnit {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Convention used to apply an EI unit's characteristic delay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EiDelayConvention {
    /// Approximate the continuous-time expression from Eq. 3 of the paper
    /// using symmetric Lanczos-8 fractional shifts of `+tau / 2` and
    /// `-tau / 2`.
    #[default]
    PaperSymmetric,
    /// Reproduce AMT 1.6's `breebaart2001_eicell`: round the complete delay to
    /// an integer number of samples and delay the left ear for positive `tau`
    /// or the right ear for negative `tau`.
    AmtOneSidedInteger,
}

/// Temporal integration and finite-record boundary treatment used by the EI
/// stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EiIntegrationBoundary {
    /// Treat samples outside the supplied representation as zero when applying
    /// the normalized double-sided exponential from Eqs. 4 and 5.
    #[default]
    ZeroExtension,
    /// Reproduce the odd-reflection padding and steady-state initialization of
    /// the first-order forward-backward filter used by AMT 1.6.
    AmtFiltfilt,
    /// Apply a causal, unity-gain first-order low-pass with zero initial state.
    ///
    /// This mode is suitable for [`EiStream`] and [`HybridBinauralStream`].
    /// The two offline modes above are acausal and cannot produce final results
    /// for an input of indefinite length.
    CausalZeroState,
}

/// Temporal processing used to prepare a monaural central-detector channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MonauralTemporalMode {
    /// Apply the paper's normalized, double-sided 10 ms exponential with zero
    /// extension outside the supplied representation.
    #[default]
    PaperDoubleSided,
    /// Reproduce AMT 1.6's causal first-order 10 ms low-pass with zero initial
    /// state.
    AmtCausal,
}

/// Parameters for the monaural channels of the Breebaart central detector.
#[derive(Clone, Debug)]
pub struct MonauralConfig {
    /// Time constant of the temporal low-pass, in seconds.
    ///
    /// Must be finite and positive. The default is 10 ms.
    pub integration_time_constant_seconds: f64,
    /// Constant sensitivity multiplier applied after temporal integration.
    ///
    /// Must be finite and non-negative. The default is 0.0003, as used by the
    /// reference simulations and AMT 1.6.
    pub sensitivity: f64,
    /// Finite-record temporal-processing convention.
    pub temporal_mode: MonauralTemporalMode,
}

impl Default for MonauralConfig {
    fn default() -> Self {
        Self {
            integration_time_constant_seconds: 10e-3,
            sensitivity: 0.0003,
            temporal_mode: MonauralTemporalMode::PaperDoubleSided,
        }
    }
}

impl MonauralConfig {
    /// Returns the monaural-channel defaults used by AMT 1.6.
    pub fn amt_1_6() -> Self {
        Self {
            temporal_mode: MonauralTemporalMode::AmtCausal,
            ..Self::default()
        }
    }

    /// Returns the causal configuration accepted by [`MonauralStream`].
    ///
    /// Its numerical settings are the same as [`Self::amt_1_6`]; the separate
    /// constructor makes the streaming intent explicit at call sites.
    pub fn streaming() -> Self {
        Self::amt_1_6()
    }
}

/// Parameters of the Breebaart EI stage.
#[derive(Clone, Debug)]
pub struct EiConfig {
    /// Time constant of the selected temporal integrator, in seconds.
    ///
    /// Must be finite and positive. The default is 30 ms.
    pub integration_time_constant_seconds: f64,
    /// Scale `a` of the compressive EI input-output function in Eq. 6.
    ///
    /// Must be finite and positive.
    pub compression_a: f64,
    /// Scale `b` of the compressive EI input-output function in Eq. 6.
    ///
    /// Must be finite and positive.
    pub compression_b: f64,
    /// Exponential time constant of the characteristic-delay weighting.
    ///
    /// Must be finite and positive. The paper's
    /// `10^(-abs(tau_ms) / 5)` weighting is equivalent to the default time
    /// constant of `5 ms / ln(10)`. AMT 1.6 uses the close approximation
    /// 2.2 ms.
    pub delay_weight_time_constant_seconds: f64,
    /// Convention used to apply each unit's characteristic delay.
    pub delay_convention: EiDelayConvention,
    /// Temporal integration and finite-record boundary convention.
    pub integration_boundary: EiIntegrationBoundary,
    /// RMS of the additive Gaussian internal noise, in model units.
    ///
    /// Must be finite and non-negative. The paper uses 1 MU. Set this to zero
    /// for a noise-free activity map.
    pub internal_noise_std_mu: f64,
    /// Seed used to make the internal-noise realization reproducible.
    ///
    /// One independent realization is generated in sample-major order for
    /// each frequency and time sample, then shared by every EI unit as
    /// specified by the model. This seed is independent of
    /// [`PeripheralConfig::absolute_threshold_noise_seed`].
    pub noise_seed: u64,
    /// Largest allowed absolute characteristic delay, in seconds.
    ///
    /// Must be non-negative and not NaN; positive infinity disables the limit.
    /// This validation limit defaults to 5 ms, the range used in the paper. It
    /// does not create population units automatically.
    pub max_abs_delay_seconds: f64,
    /// Largest allowed absolute characteristic IID, in decibels.
    ///
    /// Must be non-negative and not NaN; positive infinity disables the limit.
    /// This validation limit defaults to 10 dB, the range used in the paper. It
    /// does not create population units automatically.
    pub max_abs_iid_db: f64,
}

impl Default for EiConfig {
    fn default() -> Self {
        Self {
            integration_time_constant_seconds: 30e-3,
            compression_a: 0.1,
            compression_b: 0.000_02,
            delay_weight_time_constant_seconds: 5e-3 / std::f64::consts::LN_10,
            delay_convention: EiDelayConvention::PaperSymmetric,
            integration_boundary: EiIntegrationBoundary::ZeroExtension,
            internal_noise_std_mu: 1.0,
            noise_seed: 0x4252_4545_4241_4152,
            max_abs_delay_seconds: 5e-3,
            max_abs_iid_db: 10.0,
        }
    }
}

impl EiConfig {
    /// Returns the EI-cell defaults used by AMT 1.6.
    ///
    /// This selects AMT's one-sided integer delay, forward-backward filter
    /// boundaries, 2.2 ms delay-weight time constant, and noise-free EI-cell
    /// output. It also disables the paper-specific 5 ms and 10 dB population
    /// limits because `breebaart2001_eicell` does not impose them. AMT
    /// introduces its unit-variance internal noise later in
    /// `breebaart2001_centralproc`; callers reproducing that complete detector
    /// must model the decision noise separately.
    pub fn amt_1_6() -> Self {
        Self {
            delay_weight_time_constant_seconds: 2.2e-3,
            delay_convention: EiDelayConvention::AmtOneSidedInteger,
            integration_boundary: EiIntegrationBoundary::AmtFiltfilt,
            internal_noise_std_mu: 0.0,
            max_abs_delay_seconds: f64::INFINITY,
            max_abs_iid_db: f64::INFINITY,
            ..Self::default()
        }
    }

    /// Returns the paper defaults with causal, zero-state temporal
    /// integration for bounded-memory streaming.
    pub fn streaming() -> Self {
        Self {
            integration_boundary: EiIntegrationBoundary::CausalZeroState,
            ..Self::default()
        }
    }

    /// Returns a copy with a deterministic, trial-specific internal-noise seed.
    ///
    /// Use a distinct `trial_index` for every independently presented stimulus
    /// used to fit a [`CentralTemplate`]. Reusing an unchanged configuration
    /// deliberately reproduces the same noise realization, which is useful for
    /// exact reruns but would remove internal-noise variability from the fitted
    /// masker variance.
    pub fn for_trial(&self, trial_index: u64) -> Self {
        let mut config = self.clone();
        config.noise_seed = derive_trial_seed(self.noise_seed, trial_index);
        config
    }
}

/// Parameters of the inner-hair-cell and adaptation stages in the hybrid.
#[derive(Clone, Debug)]
pub struct PeripheralConfig {
    /// Aggregate inner-hair-cell low-pass cutoff (the -3 dB frequency), in
    /// hertz.
    ///
    /// The filter is the model's cascade of five identical first-order
    /// Butterworth sections. The cutoff of each section is chosen so the
    /// complete cascade has this cutoff; the default corresponds to sections
    /// near 2 kHz, as in the reference implementation.
    ///
    /// Must be finite, positive, and below half the GCFB sample rate.
    pub ihc_cutoff_hz: f64,
    /// Time constants of the five consecutive adaptation loops, in seconds.
    ///
    /// Every value must be finite and positive.
    pub adaptation_time_constants_seconds: [f64; 5],
    /// Lowest represented level in dB SPL.
    ///
    /// Must be finite and below 100 dB SPL.
    pub minimum_level_db_spl: f64,
    /// AMT adaptation-loop overshoot parameter.
    ///
    /// For values greater than one, AMT's smooth limiter is applied inside
    /// every loop before its state is updated. `None` and `Some(1.0)` select
    /// the paper-faithful unlimited response. The usual limited value in the
    /// wider Dau model family is 10; Breebaart's defaults are unlimited.
    /// Configured values must be finite and at least one, and large enough for
    /// the selected minimum level to give every limiter a positive range.
    pub overshoot_limit: Option<f64>,
    /// Level assigned to a waveform and filterbank amplitude of one. If
    /// omitted, the GCFB level estimator's `rms2spldb` value is used. This
    /// calibration is propagated into the GCFB level estimator and used for
    /// both adaptation and absolute-threshold noise.
    ///
    /// A configured value must be finite.
    pub amplitude_one_db_spl: Option<f64>,
    /// RMS level of the independent Gaussian noise added before inner-hair-cell
    /// half-wave rectification, in dB SPL.
    ///
    /// The default is the SPL of 60 µPa relative to 20 µPa. Set this to
    /// `None` to disable pre-IHC noise.
    ///
    /// A configured value must be finite and produce a finite, positive linear
    /// amplitude under the selected calibration.
    pub absolute_threshold_noise_level_db_spl: Option<f64>,
    /// Seed for reproducible pre-IHC absolute-threshold noise.
    ///
    /// This noise stream is independent of [`EiConfig::noise_seed`] and
    /// supplies a distinct Gaussian sample for every ear, frequency channel,
    /// and time sample. Samples are drawn in time-major order, left channels
    /// before right channels, so batch and streaming runs replay identically.
    pub absolute_threshold_noise_seed: u64,
}

impl Default for PeripheralConfig {
    fn default() -> Self {
        Self {
            ihc_cutoff_hz: 770.0,
            adaptation_time_constants_seconds: [0.005, 0.12875, 0.2525, 0.37625, 0.500],
            minimum_level_db_spl: 0.0,
            overshoot_limit: None,
            amplitude_one_db_spl: None,
            absolute_threshold_noise_level_db_spl: Some(9.542_425_094_393_248),
            absolute_threshold_noise_seed: 0x5045_5249_5048_4552,
        }
    }
}

impl PeripheralConfig {
    /// Returns a copy with a deterministic, trial-specific absolute-threshold
    /// noise seed.
    ///
    /// Use a distinct `trial_index` for every independently presented stimulus.
    /// The same base configuration and index always produce the same seed.
    pub fn for_trial(&self, trial_index: u64) -> Self {
        let mut config = self.clone();
        config.absolute_threshold_noise_seed =
            derive_trial_seed(self.absolute_threshold_noise_seed, trial_index);
        config
    }
}

/// Configuration for the end-to-end gammachirp/Breebaart hybrid.
#[derive(Clone, Debug, Default)]
pub struct HybridBinauralConfig {
    /// GCFB v2.34 configuration shared by the left and right ears.
    ///
    /// [`hybrid_binaural`] forces a dynamic configuration's processing mode to
    /// sample-based, but otherwise passes these settings to the filterbank.
    pub filterbank: GcParam,
    /// Inner-hair-cell, level-calibration, and adaptation-loop settings.
    pub peripheral: PeripheralConfig,
    /// EI population settings, including post-EI internal noise.
    pub ei: EiConfig,
}

impl HybridBinauralConfig {
    /// Returns the hybrid defaults with a streamable causal EI stage.
    pub fn streaming() -> Self {
        Self {
            ei: EiConfig::streaming(),
            ..Self::default()
        }
    }

    /// Returns a copy whose two noise sources are reproducible and independent
    /// for the specified trial.
    ///
    /// When generating the masker and target trials passed to
    /// [`CentralTemplate::fit`], call this method with a distinct index for each
    /// presentation. This derives both seeds while leaving the filterbank and
    /// all other model parameters unchanged.
    pub fn for_trial(&self, trial_index: u64) -> Self {
        Self {
            filterbank: self.filterbank.clone(),
            peripheral: self.peripheral.for_trial(trial_index),
            ei: self.ei.for_trial(trial_index),
        }
    }
}

/// Output of [`hybrid_binaural`].
#[derive(Clone, Debug)]
pub struct HybridBinauralOutput {
    /// EI activity with shape `(unit, frequency channel, sample)`.
    pub ei_map: Array3<f64>,
    /// EI units corresponding, in order, to axis 0 of [`Self::ei_map`].
    pub units: Vec<EiUnit>,
    /// Left-ear adaptation-loop output in MU, with shape `(channel, sample)`.
    ///
    /// This is the raw peripheral representation used by the EI stage. Apply
    /// [`breebaart2001_monaural`] before using it as a central-detector channel.
    pub left_internal: Array2<f64>,
    /// Right-ear adaptation-loop output in MU, with shape `(channel, sample)`.
    ///
    /// This is the raw peripheral representation used by the EI stage. Apply
    /// [`breebaart2001_monaural`] before using it as a central-detector channel.
    pub right_internal: Array2<f64>,
    /// GCFB center frequency for each channel, in hertz.
    pub center_frequencies_hz: Array1<f64>,
    /// Complete left-ear GCFB output and effective filterbank configuration.
    pub left_filterbank: GcfbOutput,
    /// Complete right-ear GCFB output and effective filterbank configuration.
    pub right_filterbank: GcfbOutput,
}

fn validate_ei_inputs(
    left: &Array2<f64>,
    right: &Array2<f64>,
    sample_rate_hz: f64,
    units: &[EiUnit],
    config: &EiConfig,
) -> Result<()> {
    if left.dim() != right.dim() || left.is_empty() {
        return Err(Error::InvalidParameter(
            "left and right internal representations must be non-empty and have equal channel-major shapes"
                .into(),
        ));
    }
    if left
        .iter()
        .chain(right.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter(
            "internal representations must contain only finite values".into(),
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
    if config.integration_boundary == EiIntegrationBoundary::AmtFiltfilt
        && left.ncols() <= AMT_FILTFILT_PADDING
    {
        return Err(Error::InvalidParameter(format!(
            "AMT forward-backward EI integration requires more than {AMT_FILTFILT_PADDING} samples"
        )));
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
    if config.delay_convention == EiDelayConvention::AmtOneSidedInteger
        && config.integration_boundary != EiIntegrationBoundary::CausalZeroState
        && units.iter().any(|unit| {
            let delay_samples = (unit.delay_seconds.abs() * sample_rate_hz).round();
            !delay_samples.is_finite() || delay_samples > left.ncols() as f64
        })
    {
        return Err(Error::InvalidParameter(
            "AMT one-sided delay must round to no more samples than the supplied representation"
                .into(),
        ));
    }
    Ok(())
}

const LANCZOS_RADIUS: isize = 8;
const LANCZOS_TAPS: usize = (2 * LANCZOS_RADIUS) as usize;

/// A fixed fractional shift whose Lanczos weights are shared by every sample.
///
/// Samples beyond the supplied representation are zero. The source-position
/// check preserves that convention even when part of the finite interpolation
/// kernel would otherwise overlap the representation.
#[derive(Clone, Debug)]
struct FractionalShift {
    shift: f64,
    integer_offset: isize,
    weights: Option<[f64; LANCZOS_TAPS]>,
}

impl FractionalShift {
    fn new(shift: f64) -> Self {
        let integer_offset = shift.floor() as isize;
        let fraction = shift - shift.floor();
        let weights = (fraction != 0.0).then(|| {
            std::array::from_fn(|tap_index| {
                let tap_offset = tap_index as isize - (LANCZOS_RADIUS - 1);
                let distance = fraction - tap_offset as f64;
                lanczos_8(distance)
            })
        });
        Self {
            shift,
            integer_offset,
            weights,
        }
    }

    fn sample(&self, row: ndarray::ArrayView1<'_, f64>, output_sample: usize) -> f64 {
        let source = output_sample as f64 + self.shift;
        if source < 0.0 || source > (row.len() - 1) as f64 {
            return 0.0;
        }
        if self.weights.is_none() {
            return row[(output_sample as isize + self.integer_offset) as usize];
        }

        self.weights
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(tap_index, weight)| {
                let tap_offset = tap_index as isize - (LANCZOS_RADIUS - 1);
                let input_sample = output_sample as isize + self.integer_offset + tap_offset;
                (input_sample >= 0 && input_sample < row.len() as isize)
                    .then(|| row[input_sample as usize] * weight)
            })
            .sum()
    }
}

fn lanczos_8(value: f64) -> f64 {
    if value.abs() >= LANCZOS_RADIUS as f64 {
        0.0
    } else {
        sinc_pi(value) * sinc_pi(value / LANCZOS_RADIUS as f64)
    }
}

fn sinc_pi(value: f64) -> f64 {
    if value == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * value).sin() / (std::f64::consts::PI * value)
    }
}

const AMT_FILTFILT_PADDING: usize = 3;

fn zero_extended_double_sided_exponential(
    values: &[f64],
    sample_rate_hz: f64,
    time_constant: f64,
) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let pole = (-1.0 / (sample_rate_hz * time_constant)).exp();
    let normalization = (1.0 - pole) / (1.0 + pole);
    let mut causal = vec![0.0; values.len()];
    let mut anticausal = vec![0.0; values.len()];
    causal[0] = values[0];
    for sample in 1..values.len() {
        causal[sample] = values[sample] + pole * causal[sample - 1];
    }
    let last = values.len() - 1;
    anticausal[last] = values[last];
    for sample in (0..last).rev() {
        anticausal[sample] = values[sample] + pole * anticausal[sample + 1];
    }
    (0..values.len())
        .map(|sample| normalization * (causal[sample] + anticausal[sample] - values[sample]))
        .collect()
}

fn steady_state_one_pole(values: &[f64], pole: f64) -> Vec<f64> {
    let feedforward = 1.0 - pole;
    let mut previous = values[0];
    values
        .iter()
        .map(|value| {
            let output = feedforward * value + pole * previous;
            previous = output;
            output
        })
        .collect()
}

fn amt_forward_backward_exponential(
    values: &[f64],
    sample_rate_hz: f64,
    time_constant: f64,
) -> Vec<f64> {
    debug_assert!(values.len() > AMT_FILTFILT_PADDING);
    let first = values[0];
    let last = values[values.len() - 1];
    let mut padded = Vec::with_capacity(values.len() + 2 * AMT_FILTFILT_PADDING);
    padded.extend(
        (1..=AMT_FILTFILT_PADDING)
            .rev()
            .map(|index| 2.0 * first - values[index]),
    );
    padded.extend_from_slice(values);
    padded.extend(
        (1..=AMT_FILTFILT_PADDING).map(|offset| 2.0 * last - values[values.len() - 1 - offset]),
    );

    let pole = (-1.0 / (sample_rate_hz * time_constant)).exp();
    let mut forward = steady_state_one_pole(&padded, pole);
    forward.reverse();
    let mut backward = steady_state_one_pole(&forward, pole);
    backward.reverse();
    backward[AMT_FILTFILT_PADDING..AMT_FILTFILT_PADDING + values.len()].to_vec()
}

fn double_sided_exponential(
    values: &[f64],
    sample_rate_hz: f64,
    time_constant: f64,
    boundary: EiIntegrationBoundary,
) -> Vec<f64> {
    match boundary {
        EiIntegrationBoundary::ZeroExtension => {
            zero_extended_double_sided_exponential(values, sample_rate_hz, time_constant)
        }
        EiIntegrationBoundary::AmtFiltfilt => {
            amt_forward_backward_exponential(values, sample_rate_hz, time_constant)
        }
        EiIntegrationBoundary::CausalZeroState => {
            let pole = (-1.0 / (sample_rate_hz * time_constant)).exp();
            let feedforward = 1.0 - pole;
            let mut previous = 0.0;
            values
                .iter()
                .map(|value| {
                    previous = feedforward * value + pole * previous;
                    previous
                })
                .collect()
        }
    }
}

/// Prepare one monaural adaptation-loop output for the central detector.
///
/// `input` is a channel-major `(frequency channel, sample)` array such as
/// [`HybridBinauralOutput::left_internal`] or
/// [`HybridBinauralOutput::right_internal`]. The returned array has the same
/// shape. The default applies the paper's normalized double-sided exponential
/// with a 10 ms time constant, then multiplies the result by the reference
/// monaural sensitivity. [`MonauralConfig::amt_1_6`] instead selects AMT's
/// causal one-pole finite-record convention.
///
/// This helper is deterministic and implements only the temporal-filter and
/// sensitivity stages. Any internal noise required by the chosen central
/// decision model must be represented in the trials or added at that later
/// stage.
///
/// The paper's Appendix-B detector uses at most one selected EI unit together
/// with any desired processed monaural channels. Do not treat all units in a
/// population returned by [`breebaart2001_ei`] as independent detector
/// channels; their internal noise is shared.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if `input` is empty or non-finite, the
/// sample rate or time constant is not finite and positive, or the sensitivity
/// is not finite and non-negative.
pub fn breebaart2001_monaural(
    input: &Array2<f64>,
    sample_rate_hz: f64,
    config: &MonauralConfig,
) -> Result<Array2<f64>> {
    if input.is_empty() || input.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidParameter(
            "monaural representation must be non-empty and contain only finite values".into(),
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

    if config.temporal_mode == MonauralTemporalMode::AmtCausal {
        let mut stream = MonauralStream::new(input.nrows(), sample_rate_hz, config.clone())?;
        let mut output = Array2::zeros(input.dim());
        for sample in 0..input.ncols() {
            let input_sample = input.column(sample).to_vec();
            let event = stream.process_sample(&input_sample)?;
            output.column_mut(sample).assign(&event.output);
        }
        return Ok(output);
    }

    let mut output = Array2::zeros(input.dim());
    for channel in 0..input.nrows() {
        let values = input.row(channel).to_vec();
        let mut integrated = zero_extended_double_sided_exponential(
            &values,
            sample_rate_hz,
            config.integration_time_constant_seconds,
        );
        for value in &mut integrated {
            *value *= config.sensitivity;
        }
        output.row_mut(channel).assign(&Array1::from(integrated));
    }
    Ok(output)
}

fn derive_trial_seed(base_seed: u64, trial_index: u64) -> u64 {
    // SplitMix64 finalization gives nearby trial indices well-separated streams
    // while retaining deterministic replay from the base seed and index.
    let mut value = base_seed
        .wrapping_add(trial_index.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug)]
struct GaussianNoise {
    state: u64,
    spare: Option<f64>,
}

impl GaussianNoise {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
            spare: None,
        }
    }

    fn uniform_open(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let bits = value.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        (bits as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn sample(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform_open();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// Apply the Breebaart EI population to paired peripheral representations.
///
/// Both inputs are channel-major `(frequency channel, sample)` arrays in the
/// model units expected by the paper's EI stage.  The returned array has shape
/// `(unit, frequency channel, sample)`. The default characteristic-delay and
/// integration-boundary behavior follows the continuous-time equations in the
/// paper; [`EiConfig::amt_1_6`] selects the corresponding AMT EI-cell behavior.
/// The default and AMT temporal integrations are offline and acausal.
/// [`EiIntegrationBoundary::CausalZeroState`] instead selects the same
/// unity-gain one-pole recursion used by [`EiStream`]. Additive internal noise
/// is independent across frequency and time but uses the same realization for
/// all EI units. Set [`EiConfig::internal_noise_std_mu`] to zero to disable it.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] if the inputs are empty, differently
/// shaped, or non-finite; if the sample rate is not positive; if no units are
/// supplied; if a unit exceeds the configured population limits; or if an EI
/// parameter is outside its documented range; if AMT forward-backward
/// integration is requested for a representation of three samples or fewer;
/// or if an AMT one-sided delay rounds to more samples than the representation.
pub fn breebaart2001_ei(
    left: &Array2<f64>,
    right: &Array2<f64>,
    sample_rate_hz: f64,
    units: &[EiUnit],
    config: &EiConfig,
) -> Result<Array3<f64>> {
    validate_ei_inputs(left, right, sample_rate_hz, units, config)?;
    let (channels, samples) = left.dim();

    if config.integration_boundary == EiIntegrationBoundary::CausalZeroState {
        let mut stream = EiStream::new(channels, sample_rate_hz, units, config.clone())?;
        let mut output = Array3::zeros((units.len(), channels, samples));
        for sample in 0..samples {
            let left_sample = left.column(sample).to_vec();
            let right_sample = right.column(sample).to_vec();
            if let Some(event) = stream.process_sample(&left_sample, &right_sample)? {
                output
                    .index_axis_mut(ndarray::Axis(2), event.sample_index)
                    .assign(&event.activity);
            }
        }
        for event in stream.finish()? {
            output
                .index_axis_mut(ndarray::Axis(2), event.sample_index)
                .assign(&event.activity);
        }
        return Ok(output);
    }

    let mut output = Array3::zeros((units.len(), channels, samples));
    let mut noise = GaussianNoise::new(config.noise_seed);
    let mut internal_noise = Array2::zeros((channels, samples));
    for sample in 0..samples {
        for channel in 0..channels {
            internal_noise[[channel, sample]] = config.internal_noise_std_mu * noise.sample();
        }
    }

    for (unit_index, unit) in units.iter().enumerate() {
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
        let left_shift = FractionalShift::new(left_shift_samples);
        let right_shift = FractionalShift::new(right_shift_samples);
        let left_gain = 10_f64.powf(unit.iid_db / 40.0);
        let right_gain = 10_f64.powf(-unit.iid_db / 40.0);
        let delay_weight =
            (-unit.delay_seconds.abs() / config.delay_weight_time_constant_seconds).exp();
        for channel in 0..channels {
            let left_row = left.row(channel);
            let right_row = right.row(channel);
            let instantaneous: Vec<f64> = (0..samples)
                .map(|sample| {
                    let left_value = left_shift.sample(left_row, sample) * left_gain;
                    let right_value = right_shift.sample(right_row, sample) * right_gain;
                    (left_value - right_value).powi(2)
                })
                .collect();
            let integrated = double_sided_exponential(
                &instantaneous,
                sample_rate_hz,
                config.integration_time_constant_seconds,
                config.integration_boundary,
            );
            for sample in 0..samples {
                let deterministic = config.compression_a
                    * delay_weight
                    * (config.compression_b * integrated[sample] + 1.0).ln();
                output[[unit_index, channel, sample]] =
                    deterministic + internal_noise[[channel, sample]];
            }
        }
    }
    Ok(output)
}

fn ihc_section_cutoff_hz(aggregate_cutoff_hz: f64, sample_rate_hz: f64) -> f64 {
    const FILTER_ORDER: f64 = 5.0;

    // A first-order Butterworth section has |H|^2 = 1 / (1 + r^2),
    // where r is the ratio of bilinear-prewarped frequencies. For five
    // identical sections, solve (1 + r^2)^-5 = 1/2 at the requested aggregate
    // cutoff. This retains the reference model's repeated one-pole topology
    // while making the public cutoff parameter sample-rate independent.
    let target = (std::f64::consts::PI * aggregate_cutoff_hz / sample_rate_hz).tan();
    let ratio = (2_f64.powf(1.0 / FILTER_ORDER) - 1.0).sqrt();
    sample_rate_hz / std::f64::consts::PI * (target / ratio).atan()
}

#[cfg(test)]
fn inner_hair_cell(
    filterbank: &Array2<f64>,
    sample_rate_hz: f64,
    cutoff_hz: f64,
) -> Result<Array2<f64>> {
    const FILTER_ORDER: usize = 5;

    let mut output = Array2::zeros(filterbank.dim());
    let section_cutoff_hz = ihc_section_cutoff_hz(cutoff_hz, sample_rate_hz);
    let (b, a) = dsp::first_order_lowpass(section_cutoff_hz, sample_rate_hz);
    for channel in 0..filterbank.nrows() {
        let mut filtered: Vec<f64> = filterbank
            .row(channel)
            .iter()
            .map(|value| value.max(0.0))
            .collect();
        for _ in 0..FILTER_ORDER {
            filtered = dsp::lfilter(&b, &a, &filtered)?;
        }
        output.row_mut(channel).assign(&Array1::from(filtered));
    }
    Ok(output)
}

fn absolute_threshold_noise_std_amplitude(
    level_db_spl: Option<f64>,
    amplitude_one_db_spl: f64,
) -> Result<Option<f64>> {
    if !amplitude_one_db_spl.is_finite() || level_db_spl.is_some_and(|level| !level.is_finite()) {
        return Err(Error::InvalidParameter(
            "absolute-threshold noise level and amplitude calibration must be finite".into(),
        ));
    }
    let Some(level_db_spl) = level_db_spl else {
        return Ok(None);
    };
    let noise_std = 10_f64.powf((level_db_spl - amplitude_one_db_spl) / 20.0);
    if !noise_std.is_finite() || noise_std <= 0.0 {
        return Err(Error::InvalidParameter(
            "absolute-threshold noise calibration exceeds the finite floating-point range".into(),
        ));
    }
    Ok(Some(noise_std))
}

#[cfg(test)]
fn add_absolute_threshold_noise(
    left: &Array2<f64>,
    right: &Array2<f64>,
    amplitude_one_db_spl: f64,
    config: &PeripheralConfig,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let mut left_noisy = left.clone();
    let mut right_noisy = right.clone();
    if let Some(noise_std) = absolute_threshold_noise_std_amplitude(
        config.absolute_threshold_noise_level_db_spl,
        amplitude_one_db_spl,
    )? {
        let mut noise = GaussianNoise::new(config.absolute_threshold_noise_seed);
        for sample in 0..left.ncols() {
            for channel in 0..left.nrows() {
                left_noisy[[channel, sample]] += noise_std * noise.sample();
            }
            for channel in 0..right.nrows() {
                right_noisy[[channel, sample]] += noise_std * noise.sample();
            }
        }
    }
    Ok((left_noisy, right_noisy))
}

#[cfg(test)]
fn adaptation_loops(
    input: &Array2<f64>,
    sample_rate_hz: f64,
    amplitude_one_db_spl: f64,
    config: &PeripheralConfig,
) -> Result<Array2<f64>> {
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
            !factor.is_finite() || *factor <= 0.0 || !exponential.is_finite() || !offset.is_finite()
        }) {
            return Err(Error::InvalidParameter(
                "AMT overshoot parameter is too small for the configured minimum level".into(),
            ));
        }
        Some(parameters)
    } else {
        None
    };
    let mut output = Array2::zeros(input.dim());

    for channel in 0..input.nrows() {
        let mut states = initial_states;
        for sample in 0..input.ncols() {
            let normalized =
                (input[[channel, sample]] / hundred_db_amplitude).max(minimum_normalized);
            let mut value = normalized;
            for stage in 0..states.len() {
                let mut stage_output = value / states[stage].max(f64::MIN_POSITIVE);
                if stage_output > 1.0
                    && let Some(parameters) = limiter_parameters
                {
                    let (factor, exponential, offset) = parameters[stage];
                    stage_output =
                        factor / (1.0 + (exponential * (stage_output - 1.0)).exp()) - offset;
                }
                states[stage] = coefficients[stage] * states[stage]
                    + (1.0 - coefficients[stage]) * stage_output;
                value = stage_output;
            }
            output[[channel, sample]] = (value - minimum_steady) * model_unit_scale;
        }
    }
    Ok(output)
}

fn validate_hybrid_inputs(
    left: &[f64],
    right: &[f64],
    config: &HybridBinauralConfig,
) -> Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(Error::InvalidParameter(
            "left and right waveforms must be non-empty and have equal lengths".into(),
        ));
    }
    if left
        .iter()
        .chain(right.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidParameter(
            "left and right waveforms must contain only finite values".into(),
        ));
    }
    let sample_rate = config.filterbank.fs;
    if !config.peripheral.ihc_cutoff_hz.is_finite()
        || config.peripheral.ihc_cutoff_hz <= 0.0
        || config.peripheral.ihc_cutoff_hz >= sample_rate / 2.0
    {
        return Err(Error::InvalidParameter(
            "inner-hair-cell cutoff must be finite, positive, and below Nyquist".into(),
        ));
    }
    if config
        .peripheral
        .amplitude_one_db_spl
        .is_some_and(|level| !level.is_finite())
        || config
            .peripheral
            .absolute_threshold_noise_level_db_spl
            .is_some_and(|level| !level.is_finite())
    {
        return Err(Error::InvalidParameter(
            "peripheral level calibration and absolute-threshold noise level must be finite".into(),
        ));
    }
    Ok(())
}

/// Run the end-to-end GCFB/Breebaart hybrid on a binaural waveform.
///
/// Dynamic GCFB processing is evaluated sample by sample because the EI stage
/// operates on fine structure. The returned `left_filterbank` and
/// `right_filterbank` therefore contain the effective sample-mode setting even
/// if the supplied GCFB configuration selected frame mode. Absolute-threshold
/// noise is calibrated through [`PeripheralConfig::amplitude_one_db_spl`] and
/// added independently to every ear/channel/sample after GCFB processing and
/// before inner-hair-cell rectification.  Set its configured level to `None`
/// and set [`EiConfig::internal_noise_std_mu`] to zero to disable both model
/// noise sources.
///
/// The sample rate is read from [`HybridBinauralConfig::filterbank`]. The input
/// samples therefore use whatever amplitude-to-SPL calibration is selected by
/// [`PeripheralConfig::amplitude_one_db_spl`] or, when that is `None`, by the
/// GCFB level estimator.
///
/// # Errors
///
/// Returns an error for empty, unequal-length, or non-finite waveforms; invalid
/// peripheral, filterbank, or EI settings; or a failure in either GCFB run. A
/// mismatched pair of sample-domain filterbank matrices is reported as
/// [`Error::Numerical`].
///
/// # Example
///
/// ```no_run
/// use gammachirp_rs::breebaart2001::{
///     EiUnit, HybridBinauralConfig, hybrid_binaural,
/// };
/// use gammachirp_rs::gcfb_v234::ControlMode;
///
/// let sample_rate = 48_000.0;
/// let left: Vec<f64> = (0..480)
///     .map(|n| (2.0 * std::f64::consts::PI * 500.0 * n as f64 / sample_rate).sin())
///     .collect();
/// let right = left.clone();
/// let mut config = HybridBinauralConfig::default();
/// config.filterbank.fs = sample_rate;
/// config.filterbank.num_ch = 16;
/// config.filterbank.out_mid_crct = "No".into();
/// config.filterbank.ctrl = ControlMode::Static;
/// config.peripheral.absolute_threshold_noise_level_db_spl = None;
/// config.ei.internal_noise_std_mu = 0.0;
///
/// let units = [EiUnit::default(), EiUnit::new(0.0, 3.0)];
/// let output = hybrid_binaural(&left, &right, &units, config)?;
/// assert_eq!(output.ei_map.dim(), (2, 16, 480));
/// # Ok::<(), gammachirp_rs::Error>(())
/// ```
pub fn hybrid_binaural(
    left: &[f64],
    right: &[f64],
    units: &[EiUnit],
    mut config: HybridBinauralConfig,
) -> Result<HybridBinauralOutput> {
    validate_hybrid_inputs(left, right, &config)?;
    if config.filterbank.ctrl == ControlMode::Dynamic {
        config.filterbank.dyn_hpaf.str_prc = "sample-base".into();
    }
    let amplitude_one_db_spl = config
        .peripheral
        .amplitude_one_db_spl
        .unwrap_or(config.filterbank.lvl_est.rms2spldb);
    config.filterbank.lvl_est.rms2spldb = amplitude_one_db_spl;
    let left_filterbank = gcfb_v234(left, config.filterbank.clone())?;
    let right_filterbank = gcfb_v234(right, config.filterbank)?;
    if left_filterbank.dcgc_out.dim() != right_filterbank.dcgc_out.dim()
        || left_filterbank.dcgc_out.ncols() != left.len()
    {
        return Err(Error::Numerical(
            "paired filterbanks did not produce matching sample-domain representations".into(),
        ));
    }
    let sample_rate = left_filterbank.gc_param.fs;
    let channels = left_filterbank.dcgc_out.nrows();
    let samples = left_filterbank.dcgc_out.ncols();
    let mut peripheral = stream::PeripheralPairStream::new(
        channels,
        sample_rate,
        amplitude_one_db_spl,
        &config.peripheral,
    )?;
    let mut left_internal = Array2::zeros((channels, samples));
    let mut right_internal = Array2::zeros((channels, samples));
    for sample in 0..samples {
        let left_sample = left_filterbank.dcgc_out.column(sample).to_vec();
        let right_sample = right_filterbank.dcgc_out.column(sample).to_vec();
        let (left_output, right_output) = peripheral.process(&left_sample, &right_sample)?;
        left_internal.column_mut(sample).assign(&left_output);
        right_internal.column_mut(sample).assign(&right_output);
    }
    let ei_map = breebaart2001_ei(
        &left_internal,
        &right_internal,
        sample_rate,
        units,
        &config.ei,
    )?;
    let center_frequencies_hz = left_filterbank.gc_resp.fr1.clone();
    Ok(HybridBinauralOutput {
        ei_map,
        units: units.to_vec(),
        left_internal,
        right_internal,
        center_frequencies_hz,
        left_filterbank,
        right_filterbank,
    })
}

/// Appendix-B template for an ideal-observer decision stage.
///
/// The numerical template accepts any consistent three-dimensional shape. For
/// the paper's detector, however, the axes are detector channel, frequency, and
/// time. [`hybrid_binaural`] returns a population, so callers must first select
/// the single EI unit appropriate to the experiment. The first template axis
/// then contains that one binaural channel plus any left and right monaural
/// channels prepared by [`breebaart2001_monaural`].
/// Passing a complete EI population instead produces a generalized detector,
/// not the detector in Appendix B, because the EI units share internal noise.
///
/// # Assembling a paper detector representation
///
/// ```
/// use gammachirp_rs::breebaart2001::{
///     MonauralConfig, breebaart2001_monaural,
/// };
/// use ndarray::{Array2, Array3, Axis, stack};
///
/// let sample_rate_hz = 48_000.0;
/// let ei_population = Array3::zeros((3, 2, 64));
/// let left_internal = Array2::zeros((2, 64));
/// let right_internal = Array2::zeros((2, 64));
/// let monaural = MonauralConfig::default();
/// let left = breebaart2001_monaural(&left_internal, sample_rate_hz, &monaural)?;
/// let right = breebaart2001_monaural(&right_internal, sample_rate_hz, &monaural)?;
/// let selected_ei = ei_population.index_axis(Axis(0), 1);
/// let representation = stack(Axis(0), &[selected_ei, left.view(), right.view()])?;
/// assert_eq!(representation.dim(), (3, 2, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct CentralTemplate {
    /// Element-wise mean of the masker-only training trials.
    pub masker_mean: Array3<f64>,
    /// Element-wise population variance of the masker-only training trials.
    ///
    /// Every element is clamped to at least [`Self::variance_floor`].
    pub masker_variance: Array3<f64>,
    /// Element-wise target mean minus masker mean.
    pub expected_difference: Array3<f64>,
    /// Lower bound applied to the estimated masker variance during fitting.
    pub variance_floor: f64,
}

impl CentralTemplate {
    /// Estimate the masker template, uncertainty, and target-minus-masker
    /// weighting function from labeled internal representations.
    ///
    /// All trials must have the same non-empty shape. The variance estimate
    /// divides by the number of masker trials (the population-variance
    /// convention), then clamps each value to `variance_floor`.
    ///
    /// When a representation is assembled from [`breebaart2001_ei`] or
    /// [`hybrid_binaural`], each trial must use a distinct trial-specific
    /// configuration; see [`EiConfig::for_trial`] and
    /// [`HybridBinauralConfig::for_trial`]. An unchanged seeded configuration
    /// intentionally repeats its noise and therefore cannot contribute
    /// internal-noise variability to this fit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if either trial set or its array
    /// shape is empty, the variance floor is not finite and positive, or trial
    /// shapes/values are inconsistent or non-finite.
    pub fn fit(
        masker_trials: &[Array3<f64>],
        target_trials: &[Array3<f64>],
        variance_floor: f64,
    ) -> Result<Self> {
        if masker_trials.is_empty()
            || target_trials.is_empty()
            || !variance_floor.is_finite()
            || variance_floor <= 0.0
        {
            return Err(Error::InvalidParameter(
                "central-template fitting requires masker and target trials and a positive variance floor"
                    .into(),
            ));
        }
        let shape = masker_trials[0].dim();
        if shape.0 == 0
            || shape.1 == 0
            || shape.2 == 0
            || masker_trials
                .iter()
                .chain(target_trials.iter())
                .any(|trial| trial.dim() != shape || trial.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidParameter(
                "all central-template trials must have the same non-empty shape and finite values"
                    .into(),
            ));
        }
        let average = |trials: &[Array3<f64>]| {
            let mut mean = Array3::<f64>::zeros(shape);
            for trial in trials {
                Zip::from(&mut mean)
                    .and(trial)
                    .for_each(|mean, value| *mean += *value);
            }
            mean.mapv_inplace(|value| value / trials.len() as f64);
            mean
        };
        let masker_mean = average(masker_trials);
        let target_mean = average(target_trials);
        let expected_difference = &target_mean - &masker_mean;
        let mut masker_variance = Array3::<f64>::zeros(shape);
        for trial in masker_trials {
            Zip::from(&mut masker_variance)
                .and(trial)
                .and(&masker_mean)
                .for_each(|variance, value, mean| *variance += (*value - *mean).powi(2));
        }
        masker_variance
            .mapv_inplace(|value| (value / masker_trials.len() as f64).max(variance_floor));
        Ok(Self {
            masker_mean,
            masker_variance,
            expected_difference,
            variance_floor,
        })
    }

    /// Computes the weighted distance in Eq. B2.
    ///
    /// Larger scores are more target-like. The representation must use the
    /// same shape and axis meanings as the trials supplied to [`Self::fit`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if the representation and template
    /// fields have inconsistent shapes, contain non-finite values, or contain
    /// non-positive variances.
    pub fn score(&self, representation: &Array3<f64>) -> Result<f64> {
        let shape = self.masker_mean.dim();
        if representation.dim() != shape
            || self.masker_variance.dim() != shape
            || self.expected_difference.dim() != shape
            || representation.iter().any(|value| !value.is_finite())
            || self.masker_mean.iter().any(|value| !value.is_finite())
            || self
                .masker_variance
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .expected_difference
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidParameter(
                "central-processor input and template fields must have matching shapes, finite values, and positive variances"
                    .into(),
            ));
        }
        Ok(Zip::from(representation)
            .and(&self.masker_mean)
            .and(&self.masker_variance)
            .and(&self.expected_difference)
            .fold(0.0, |sum, value, mean, variance, expected| {
                sum + expected / variance * (value - mean)
            }))
    }

    /// Chooses the most target-like interval in a forced-choice trial.
    ///
    /// Ties are resolved in favor of the earliest interval.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `intervals` is empty or if an
    /// interval cannot be scored; see [`Self::score`].
    pub fn choose_interval(&self, intervals: &[Array3<f64>]) -> Result<usize> {
        if intervals.is_empty() {
            return Err(Error::InvalidParameter(
                "forced-choice detection requires at least one interval".into(),
            ));
        }
        let mut best_index = 0;
        let mut best_score = self.score(&intervals[0])?;
        for (index, interval) in intervals.iter().enumerate().skip(1) {
            let score = self.score(interval)?;
            if score > best_score {
                best_index = index;
                best_score = score;
            }
        }
        Ok(best_index)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, Array3};

    use super::*;
    use crate::gcfb_v234::{ControlMode, GainReference};

    fn noise_free_config() -> EiConfig {
        EiConfig {
            internal_noise_std_mu: 0.0,
            ..EiConfig::default()
        }
    }

    #[test]
    fn identical_inputs_cancel_at_the_median_unit() {
        let left = Array2::from_shape_fn((2, 128), |(channel, sample)| {
            (sample as f64 * 0.1 + channel as f64).sin()
        });
        let output = breebaart2001_ei(
            &left,
            &left,
            8_000.0,
            &[EiUnit::default()],
            &noise_free_config(),
        )
        .unwrap();
        assert_eq!(output.dim(), (1, 2, 128));
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn fractional_shift_is_exact_for_integers_and_zero_extended_at_boundaries() {
        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let row = input.row(0);

        let forward = FractionalShift::new(1.0);
        assert_eq!(forward.sample(row, 0), 2.0);
        assert_eq!(forward.sample(row, 2), 4.0);
        assert_eq!(forward.sample(row, 3), 0.0);

        let backward = FractionalShift::new(-1.0);
        assert_eq!(backward.sample(row, 0), 0.0);
        assert_eq!(backward.sample(row, 1), 1.0);
        assert_eq!(backward.sample(row, 3), 3.0);

        assert_eq!(FractionalShift::new(-0.25).sample(row, 0), 0.0);
        assert_eq!(FractionalShift::new(0.25).sample(row, 3), 0.0);
    }

    #[test]
    fn lanczos_fractional_shifts_cancel_a_half_sample_sinusoidal_delay() {
        let sample_rate = 8_000.0;
        let frequency = 3_000.0;
        let samples = 2_048;
        let angular_frequency = 2.0 * std::f64::consts::PI * frequency / sample_rate;
        let left = Array2::from_shape_fn((1, samples), |(_, sample)| {
            (angular_frequency * sample as f64).sin()
        });
        let right = Array2::from_shape_fn((1, samples), |(_, sample)| {
            (angular_frequency * (sample as f64 - 0.5)).sin()
        });
        let left_shift = FractionalShift::new(-0.25);
        let right_shift = FractionalShift::new(0.25);
        let mut residual_energy = 0.0;
        let mut reference_energy = 0.0;

        for sample in 32..samples - 32 {
            let left_value = left_shift.sample(left.row(0), sample);
            let right_value = right_shift.sample(right.row(0), sample);
            residual_energy += (left_value - right_value).powi(2);
            reference_energy += left_value.powi(2);
        }

        let relative_rms = (residual_energy / reference_energy).sqrt();
        assert!(
            relative_rms < 0.02,
            "Lanczos-8 fractional-delay residual was {relative_rms}"
        );
    }

    #[test]
    fn characteristic_iid_finds_the_expected_cancellation_minimum() {
        let left = Array2::ones((1, 2_000));
        let right = Array2::from_elem((1, 2_000), 10_f64.powf(6.0 / 20.0));
        let units = [EiUnit::default(), EiUnit::new(0.0, 6.0)];
        let output =
            breebaart2001_ei(&left, &right, 8_000.0, &units, &noise_free_config()).unwrap();
        let center = 1_000;
        assert!(output[[1, 0, center]] < 1e-20);
        assert!(output[[0, 0, center]] > output[[1, 0, center]]);
    }

    #[test]
    fn delay_weight_follows_equation_seven() {
        let left = Array2::ones((1, 20_000));
        let right = Array2::zeros((1, 20_000));
        let units = [EiUnit::default(), EiUnit::new(0.005, 0.0)];
        let output =
            breebaart2001_ei(&left, &right, 8_000.0, &units, &noise_free_config()).unwrap();
        let center = 10_000;
        assert_abs_diff_eq!(
            output[[1, 0, center]] / output[[0, 0, center]],
            0.1,
            epsilon = 1e-10
        );
    }

    #[test]
    fn characteristic_delay_finds_the_compensating_minimum() {
        let sample_rate = 8_000.0;
        let delay_samples = 2;
        let samples = 8_000;
        let left = Array2::from_shape_fn((1, samples), |(_, sample)| {
            (2.0 * std::f64::consts::PI * 431.0 * sample as f64 / sample_rate).sin()
        });
        let right = Array2::from_shape_fn((1, samples), |(_, sample)| {
            sample
                .checked_sub(delay_samples)
                .map_or(0.0, |source| left[[0, source]])
        });
        let external_delay = delay_samples as f64 / sample_rate;
        let units = [EiUnit::default(), EiUnit::new(-external_delay, 0.0)];
        let output =
            breebaart2001_ei(&left, &right, sample_rate, &units, &noise_free_config()).unwrap();
        let center = samples / 2;
        assert!(output[[1, 0, center]] < output[[0, 0, center]] * 1e-6);
    }

    #[test]
    fn amt_delay_convention_uses_positive_tau_for_a_leading_left_ear() {
        let sample_rate = 8_000.0;
        let delay_samples = 2;
        let samples = 8_000;
        let left = Array2::from_shape_fn((1, samples), |(_, sample)| {
            (2.0 * std::f64::consts::PI * 431.0 * sample as f64 / sample_rate).sin()
        });
        let right = Array2::from_shape_fn((1, samples), |(_, sample)| {
            sample
                .checked_sub(delay_samples)
                .map_or(0.0, |source| left[[0, source]])
        });
        let external_delay = delay_samples as f64 / sample_rate;
        let units = [EiUnit::default(), EiUnit::new(external_delay, 0.0)];
        let output =
            breebaart2001_ei(&left, &right, sample_rate, &units, &EiConfig::amt_1_6()).unwrap();
        let center = samples / 2;
        assert!(output[[1, 0, center]] < output[[0, 0, center]] * 1e-6);
    }

    #[test]
    fn amt_forward_backward_boundaries_match_reference_values() {
        let values = [0.0, 1.0, 4.0, 2.0, -1.0, 3.0, 5.0, 0.0];
        let expected = [
            -1.888_230_212_551_386_1,
            -1.887_781_110_629_530_1,
            -1.887_382_143_868_934,
            -1.887_085_388_751_766_7,
            -1.886_856_117_853_565_2,
            -1.886_642_243_785_240_8,
            -1.886_513_207_378_610_4,
            -1.886_503_728_665_912_8,
        ];
        let actual =
            double_sided_exponential(&values, 8_000.0, 30e-3, EiIntegrationBoundary::AmtFiltfilt);
        for (actual, expected) in actual.iter().zip(expected) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
        }

        let constant = double_sided_exponential(
            &vec![1.0; 1_000],
            8_000.0,
            30e-3,
            EiIntegrationBoundary::AmtFiltfilt,
        );
        assert!(constant.iter().all(|value| (*value - 1.0).abs() < 1e-14));
    }

    #[test]
    fn amt_forward_backward_mode_rejects_too_short_inputs() {
        let input = Array2::zeros((1, AMT_FILTFILT_PADDING));
        let error = breebaart2001_ei(
            &input,
            &input,
            8_000.0,
            &[EiUnit::default()],
            &EiConfig::amt_1_6(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
    }

    #[test]
    fn seeded_internal_noise_is_reproducible_and_shared_across_units() {
        let input = Array2::zeros((2, 64));
        let config = EiConfig::default();
        let units = [EiUnit::default(), EiUnit::default()];
        let first = breebaart2001_ei(&input, &input, 8_000.0, &units, &config).unwrap();
        let second = breebaart2001_ei(&input, &input, 8_000.0, &units, &config).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.index_axis(ndarray::Axis(0), 0),
            first.index_axis(ndarray::Axis(0), 1)
        );
        let mut expected = GaussianNoise::new(config.noise_seed);
        assert_eq!(first[[0, 0, 0]], expected.sample());
        assert_eq!(first[[0, 1, 0]], expected.sample());
        assert_eq!(first[[0, 0, 1]], expected.sample());
        assert_ne!(first[[0, 0, 0]], first[[0, 0, 1]]);
        assert_ne!(first[[0, 0, 0]], first[[0, 1, 0]]);
    }

    #[test]
    fn trial_configs_produce_distinct_reproducible_noise_realizations() {
        let base = HybridBinauralConfig::default();
        let trial_seven = base.for_trial(7);
        let trial_seven_replay = base.for_trial(7);
        let trial_eight = base.for_trial(8);

        assert_eq!(trial_seven.ei.noise_seed, trial_seven_replay.ei.noise_seed);
        assert_eq!(
            trial_seven.peripheral.absolute_threshold_noise_seed,
            trial_seven_replay.peripheral.absolute_threshold_noise_seed
        );
        assert_ne!(trial_seven.ei.noise_seed, trial_eight.ei.noise_seed);
        assert_ne!(
            trial_seven.peripheral.absolute_threshold_noise_seed,
            trial_eight.peripheral.absolute_threshold_noise_seed
        );

        let input = Array2::zeros((1, 16));
        let units = [EiUnit::default()];
        let seven = breebaart2001_ei(&input, &input, 8_000.0, &units, &trial_seven.ei).unwrap();
        let seven_replay =
            breebaart2001_ei(&input, &input, 8_000.0, &units, &trial_seven_replay.ei).unwrap();
        let eight = breebaart2001_ei(&input, &input, 8_000.0, &units, &trial_eight.ei).unwrap();
        assert_eq!(seven, seven_replay);
        assert_ne!(seven, eight);
    }

    #[test]
    fn peripheral_defaults_match_breebaart_model() {
        let peripheral = PeripheralConfig::default();
        assert_eq!(
            peripheral.adaptation_time_constants_seconds,
            [0.005, 0.12875, 0.2525, 0.37625, 0.500]
        );
        assert_eq!(peripheral.overshoot_limit, None);
        assert_abs_diff_eq!(
            peripheral.absolute_threshold_noise_level_db_spl.unwrap(),
            20.0 * (60.0_f64 / 20.0).log10(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn amt_ei_config_selects_reference_specific_defaults() {
        let config = EiConfig::amt_1_6();
        assert_eq!(
            config.delay_convention,
            EiDelayConvention::AmtOneSidedInteger
        );
        assert_eq!(
            config.integration_boundary,
            EiIntegrationBoundary::AmtFiltfilt
        );
        assert_eq!(config.delay_weight_time_constant_seconds, 2.2e-3);
        assert_eq!(config.internal_noise_std_mu, 0.0);
        assert_eq!(config.max_abs_delay_seconds, f64::INFINITY);
        assert_eq!(config.max_abs_iid_db, f64::INFINITY);
    }

    #[test]
    fn amt_ei_config_accepts_units_outside_the_paper_population() {
        let input = Array2::ones((1, 256));
        let unit = EiUnit::new(6e-3, 12.0);

        assert!(matches!(
            breebaart2001_ei(&input, &input, 8_000.0, &[unit], &noise_free_config()),
            Err(Error::InvalidParameter(_))
        ));
        let output =
            breebaart2001_ei(&input, &input, 8_000.0, &[unit], &EiConfig::amt_1_6()).unwrap();
        assert_eq!(output.dim(), (1, 1, 256));
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn amt_ei_config_rejects_delays_longer_than_the_representation() {
        let input = Array2::ones((1, 32));
        let error = breebaart2001_ei(
            &input,
            &input,
            8_000.0,
            &[EiUnit::new(6e-3, 0.0)],
            &EiConfig::amt_1_6(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
    }

    #[test]
    fn monaural_defaults_match_the_paper_and_amt_reference() {
        let paper = MonauralConfig::default();
        assert_eq!(paper.temporal_mode, MonauralTemporalMode::PaperDoubleSided);
        assert_eq!(paper.integration_time_constant_seconds, 10e-3);
        assert_eq!(paper.sensitivity, 0.0003);

        let amt = MonauralConfig::amt_1_6();
        assert_eq!(amt.temporal_mode, MonauralTemporalMode::AmtCausal);
        assert_eq!(amt.integration_time_constant_seconds, 10e-3);
        assert_eq!(amt.sensitivity, 0.0003);
    }

    #[test]
    fn paper_monaural_channel_matches_the_normalized_double_sided_exponential() {
        let sample_rate_hz = 1_000.0;
        let mut input = Array2::zeros((2, 7));
        input[[0, 3]] = 1.0;
        input[[1, 3]] = 2.0;
        let config = MonauralConfig::default();
        let actual = breebaart2001_monaural(&input, sample_rate_hz, &config).unwrap();
        let pole = (-1.0 / (sample_rate_hz * config.integration_time_constant_seconds)).exp();
        let normalization = (1.0 - pole) / (1.0 + pole);

        for channel in 0..2 {
            for sample in 0..7 {
                let expected = (channel + 1) as f64
                    * config.sensitivity
                    * normalization
                    * pole.powi((sample as i32 - 3).abs());
                assert_abs_diff_eq!(actual[[channel, sample]], expected, epsilon = 1e-18);
            }
        }
    }

    #[test]
    fn amt_monaural_channel_matches_the_zero_state_causal_one_pole() {
        let sample_rate_hz = 1_000.0;
        let input = Array2::from_shape_vec((1, 5), vec![1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let config = MonauralConfig::amt_1_6();
        let actual = breebaart2001_monaural(&input, sample_rate_hz, &config).unwrap();
        let pole = (-1.0 / (sample_rate_hz * config.integration_time_constant_seconds)).exp();

        for sample in 0..5 {
            let expected = config.sensitivity * (1.0 - pole) * pole.powi(sample as i32);
            assert_abs_diff_eq!(actual[[0, sample]], expected, epsilon = 1e-18);
        }
    }

    #[test]
    fn monaural_channel_rejects_invalid_inputs_and_parameters() {
        let valid = Array2::ones((1, 8));
        let invalid_values = Array2::from_elem((1, 1), f64::NAN);
        assert!(
            breebaart2001_monaural(&Array2::zeros((0, 0)), 8_000.0, &MonauralConfig::default())
                .is_err()
        );
        assert!(
            breebaart2001_monaural(&invalid_values, 8_000.0, &MonauralConfig::default()).is_err()
        );
        assert!(breebaart2001_monaural(&valid, 0.0, &MonauralConfig::default()).is_err());

        let invalid_time = MonauralConfig {
            integration_time_constant_seconds: 0.0,
            ..MonauralConfig::default()
        };
        assert!(breebaart2001_monaural(&valid, 8_000.0, &invalid_time).is_err());
        let invalid_sensitivity = MonauralConfig {
            sensitivity: -1.0,
            ..MonauralConfig::default()
        };
        assert!(breebaart2001_monaural(&valid, 8_000.0, &invalid_sensitivity).is_err());
    }

    #[test]
    fn inner_hair_cell_matches_reference_one_pole_cascade() {
        const FILTER_ORDER: i32 = 5;

        let sample_rate_hz = 48_000.0;
        let aggregate_cutoff_hz = 770.0;
        let section_cutoff_hz = ihc_section_cutoff_hz(aggregate_cutoff_hz, sample_rate_hz);
        assert_abs_diff_eq!(section_cutoff_hz, 1_987.224_369_219_654, epsilon = 1e-9);

        let warped_ratio = (std::f64::consts::PI * aggregate_cutoff_hz / sample_rate_hz).tan()
            / (std::f64::consts::PI * section_cutoff_hz / sample_rate_hz).tan();
        let section_magnitude = 1.0 / (1.0 + warped_ratio.powi(2)).sqrt();
        assert_abs_diff_eq!(
            section_magnitude.powi(FILTER_ORDER),
            1.0 / 2.0_f64.sqrt(),
            epsilon = 1e-14
        );

        let mut impulse = Array2::zeros((1, 64));
        impulse[[0, 0]] = 1.0;
        let actual = inner_hair_cell(&impulse, sample_rate_hz, aggregate_cutoff_hz).unwrap();
        let (b, a) = dsp::first_order_lowpass(section_cutoff_hz, sample_rate_hz);
        let mut expected = impulse.row(0).to_vec();
        for _ in 0..FILTER_ORDER {
            expected = dsp::lfilter(&b, &a, &expected).unwrap();
        }
        for (actual, expected) in actual.row(0).iter().zip(expected) {
            assert_abs_diff_eq!(*actual, expected, epsilon = 1e-15);
        }
    }

    #[test]
    fn absolute_threshold_noise_is_calibrated_and_independent_per_ear() {
        let peripheral = PeripheralConfig::default();
        let amplitude_one_db_spl = 20.0 * (1.0_f64 / 20e-6).log10();
        let noise_std = absolute_threshold_noise_std_amplitude(
            peripheral.absolute_threshold_noise_level_db_spl,
            amplitude_one_db_spl,
        )
        .unwrap()
        .unwrap();
        assert_abs_diff_eq!(noise_std, 60e-6, epsilon = 1e-18);

        let input = Array2::zeros((2, 8));
        let (left, right) =
            add_absolute_threshold_noise(&input, &input, amplitude_one_db_spl, &peripheral)
                .unwrap();
        let mut expected = GaussianNoise::new(peripheral.absolute_threshold_noise_seed);
        assert_abs_diff_eq!(left[[0, 0]], 60e-6 * expected.sample(), epsilon = 1e-18);
        assert_abs_diff_eq!(left[[1, 0]], 60e-6 * expected.sample(), epsilon = 1e-18);
        assert_abs_diff_eq!(right[[0, 0]], 60e-6 * expected.sample(), epsilon = 1e-18);
        assert_abs_diff_eq!(right[[1, 0]], 60e-6 * expected.sample(), epsilon = 1e-18);
        assert_ne!(left.row(0), left.row(1));
        assert_ne!(left, right);

        let disabled = PeripheralConfig {
            absolute_threshold_noise_level_db_spl: None,
            ..peripheral
        };
        let (left, right) =
            add_absolute_threshold_noise(&input, &input, amplitude_one_db_spl, &disabled).unwrap();
        assert_eq!(left, input);
        assert_eq!(right, input);
    }

    #[test]
    fn adaptation_maps_its_steady_endpoints_to_model_units() {
        let peripheral = PeripheralConfig {
            overshoot_limit: None,
            ..PeripheralConfig::default()
        };
        let minimum_amplitude = 10_f64.powf((peripheral.minimum_level_db_spl - 100.0) / 20.0);
        let minimum = Array2::from_elem((1, 16), minimum_amplitude);
        let maximum = Array2::ones((1, 200_000));
        let minimum_output = adaptation_loops(&minimum, 8_000.0, 100.0, &peripheral).unwrap();
        let maximum_output = adaptation_loops(&maximum, 8_000.0, 100.0, &peripheral).unwrap();
        assert_abs_diff_eq!(minimum_output[[0, 15]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(maximum_output[[0, 199_999]], 100.0, epsilon = 5e-3);
    }

    #[test]
    fn adaptation_overshoot_limit_matches_amt_stagewise_recursion() {
        let minimum_normalized = 10_f64.powf((0.0 - 100.0) / 20.0);
        let input = Array2::from_shape_vec((1, 2), vec![1.0, minimum_normalized]).unwrap();
        let unlimited = PeripheralConfig::default();
        let limited = PeripheralConfig {
            overshoot_limit: Some(10.0),
            ..unlimited.clone()
        };

        let unlimited_output = adaptation_loops(&input, 8_000.0, 100.0, &unlimited).unwrap();
        let limited_output = adaptation_loops(&input, 8_000.0, 100.0, &limited).unwrap();
        assert!(unlimited_output[[0, 0]] > 1_000.0);
        assert_abs_diff_eq!(
            limited_output[[0, 0]],
            1_443.980_867_945_067,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            limited_output[[0, 1]],
            -228.508_076_344_068_9,
            epsilon = 1e-9
        );

        let limit_one = PeripheralConfig {
            overshoot_limit: Some(1.0),
            ..unlimited
        };
        assert_eq!(
            adaptation_loops(&input, 8_000.0, 100.0, &limit_one).unwrap(),
            unlimited_output
        );
    }

    #[test]
    fn central_template_selects_the_target_interval() {
        let masker_trials = [
            Array3::from_elem((1, 1, 2), 0.9),
            Array3::from_elem((1, 1, 2), 1.1),
        ];
        let target_trials = [
            Array3::from_elem((1, 1, 2), 2.9),
            Array3::from_elem((1, 1, 2), 3.1),
        ];
        let template = CentralTemplate::fit(&masker_trials, &target_trials, 1e-9).unwrap();
        let intervals = [
            Array3::from_elem((1, 1, 2), 1.0),
            Array3::from_elem((1, 1, 2), 3.0),
            Array3::from_elem((1, 1, 2), 1.05),
        ];
        assert_eq!(template.choose_interval(&intervals).unwrap(), 1);
    }

    #[test]
    fn central_template_rejects_zero_sized_trial_axes() {
        for shape in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            let masker_trials = [Array3::zeros(shape)];
            let target_trials = [Array3::zeros(shape)];
            assert!(matches!(
                CentralTemplate::fit(&masker_trials, &target_trials, 1e-9),
                Err(Error::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn end_to_end_hybrid_returns_sample_domain_population() {
        let samples = 256;
        let left: Vec<f64> = (0..samples)
            .map(|sample| (2.0 * std::f64::consts::PI * 500.0 * sample as f64 / 8_000.0).sin())
            .collect();
        let right = left.clone();
        let mut config = HybridBinauralConfig::default();
        config.filterbank.fs = 8_000.0;
        config.filterbank.num_ch = 4;
        config.filterbank.f_range = [100.0, 3_000.0];
        config.filterbank.out_mid_crct = "No".into();
        config.filterbank.ctrl = ControlMode::Static;
        config.filterbank.gain_ref = GainReference::Db(50.0);
        config.peripheral.absolute_threshold_noise_level_db_spl = None;
        config.ei.internal_noise_std_mu = 0.0;
        let output = hybrid_binaural(
            &left,
            &right,
            &[EiUnit::default(), EiUnit::new(0.0, 3.0)],
            config,
        )
        .unwrap();
        assert_eq!(output.ei_map.dim(), (2, 4, samples));
        assert_eq!(output.left_internal.dim(), (4, samples));
        assert_eq!(output.center_frequencies_hz.len(), 4);
        assert!(output.ei_map.iter().all(|value| value.is_finite()));
        assert!(
            output
                .ei_map
                .index_axis(ndarray::Axis(0), 0)
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
    }

    #[test]
    fn hybrid_uses_one_calibration_for_filterbank_noise_and_adaptation() {
        let samples = 384;
        let sample_rate = 8_000.0;
        let input: Vec<f64> = (0..samples)
            .map(|sample| {
                0.1 * (2.0 * std::f64::consts::PI * 500.0 * sample as f64 / sample_rate).sin()
            })
            .collect();
        let mut overridden = HybridBinauralConfig::default();
        overridden.filterbank.fs = sample_rate;
        overridden.filterbank.num_ch = 4;
        overridden.filterbank.f_range = [100.0, 3_000.0];
        overridden.filterbank.out_mid_crct = "No".into();
        overridden.filterbank.ctrl = ControlMode::Dynamic;
        overridden.filterbank.lvl_est.rms2spldb = 30.0;
        overridden.peripheral.amplitude_one_db_spl = Some(72.0);
        overridden.peripheral.absolute_threshold_noise_level_db_spl = None;
        overridden.ei.internal_noise_std_mu = 0.0;

        let mut explicit = overridden.clone();
        explicit.filterbank.lvl_est.rms2spldb = 72.0;
        explicit.peripheral.amplitude_one_db_spl = None;

        let units = [EiUnit::default()];
        let overridden_output = hybrid_binaural(&input, &input, &units, overridden).unwrap();
        let explicit_output = hybrid_binaural(&input, &input, &units, explicit).unwrap();

        assert_eq!(
            overridden_output.left_filterbank.gc_param.lvl_est.rms2spldb,
            72.0
        );
        assert_eq!(
            overridden_output
                .right_filterbank
                .gc_param
                .lvl_est
                .rms2spldb,
            72.0
        );
        assert_eq!(
            explicit_output.left_filterbank.gc_param.lvl_est.rms2spldb,
            72.0
        );
        assert_eq!(
            overridden_output.left_filterbank.dcgc_out,
            explicit_output.left_filterbank.dcgc_out
        );
        assert_eq!(
            overridden_output.right_filterbank.dcgc_out,
            explicit_output.right_filterbank.dcgc_out
        );
        assert_eq!(
            overridden_output.left_internal,
            explicit_output.left_internal
        );
        assert_eq!(overridden_output.ei_map, explicit_output.ei_map);
    }
}
