//! Time-frequency reassignment for GCFB v2.34.
//!
//! Reassignment is analysis-only: it does not modify the ordinary real GCFB
//! output and does not provide a synthesis representation.  On a shared,
//! zero-padded DFT domain, the original input and both auxiliary inputs pass
//! through the configured outer/middle-ear correction FIR.  Each real
//! passive-gammachirp (pGC) impulse is then projected exactly onto the
//! nonnegative-frequency bins, and the resulting complex pGC is followed by
//! the same HP-AF operator as the real filterbank.  Its imaginary branch is
//! therefore offline and acausal even though the real GCFB remains causal and
//! unchanged.
//!
//! For the conditioned linear operator `T`, the reported coordinates are
//! `Re(T(t*x) / T(x))` and `Im(T(D*x) / T(x)) / (2*pi)`, where `x` is the
//! original public input, `T` includes the correction FIR, and `D` is the
//! skew-adjoint derivative on that same finite DFT domain.  This is the
//! auxiliary-atom construction of
//! [Holighaus et al.](https://ltfat.org/notes/ltfatnote041.pdf) written as
//! operator applications.  Exactness refers to this implemented finite
//! discrete operator, up to floating-point error.
//!
//! The opt-in complex map applies the phase transport proposed by
//! [Gardner and Magnasco (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1431718/):
//! each analytic coefficient is rotated by the phase accumulated between its
//! source and reassigned coordinates.  This preserves useful absolute-phase
//! information for analysis and makes phase coherence measurable, but it is
//! **not** an invertible GCFB representation and no reconstruction API is
//! provided.  The paper's zero topology, white-noise sparsity theorem,
//! unlimited localization claims, and reconstruction observations concern a
//! Gaussian STFT and do not transfer unchanged to this nonlinear gammachirp
//! filterbank.
//!
//! Bandwidth consensus is likewise a model-specific analogue: a scale changes
//! all passive and asymmetric bandwidth coefficients (`b1`, `b2`, and
//! `lvl_est.b2`) while leaving chirp, compression, level-control,
//! hearing-loss, and frequency-grid parameters unchanged.  Every complex
//! analysis remains offline/acausal.  Sample-mode coordinates and phase are
//! conditional on the recorded nonlinear coefficient history.

use std::f64::consts::PI;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::gcfb_v234::{
    AcfCoef, ControlMode, GcParam, GcfbOutput, cmprs_gc_frsp, gcfb_v234,
    gcfb_v234_with_preserved_hearing_loss, make_asym_cmp_filters_v2,
};
use super::utils;
use crate::gcfb_v211::gammachirp::{self, Carrier, Normalization};
use crate::{Error, Result, dsp};

/// Options for dcGC reassignment.
#[derive(Clone, Debug)]
pub struct ReassignmentConfig {
    /// Per-channel relative analytic-power floor. A coefficient is rejected
    /// when its power is below this fraction of the channel maximum.
    pub coefficient_floor: f64,
}

impl Default for ReassignmentConfig {
    fn default() -> Self {
        Self {
            coefficient_floor: 1e-8,
        }
    }
}

/// Mathematical status of a reassignment result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReassignmentMode {
    /// A fixed complex cGC (static or level-control output).
    Fixed,
    /// Exact fixed-filter coordinates with positive frame-dependent energy
    /// gains applied only during transport.
    Frame,
    /// Exact reassignment of the correction-plus-HP-AF operator conditioned on
    /// the recorded coefficient history. This does not differentiate through
    /// the nonlinear coefficient estimator that produced that history.
    SampleConditional,
}

/// Separate, analysis-only result produced from an existing GCFB output.
#[derive(Clone, Debug)]
pub struct ReassignmentResult {
    /// Reassigned analytic-representation energy on `[ERB channel, time bin]`.
    /// The source measure is `|C|^2 / 2`, not pointwise or framewise
    /// `dcgc_out^2`.
    pub energy_map: Array2<f64>,
    /// Reassigned time, in seconds, for every complex source coefficient.
    pub t_hat: Array2<f64>,
    /// Reassigned frequency, in Hz, for every complex source coefficient.
    pub f_hat: Array2<f64>,
    /// Coefficients that passed the relative analytic-power floor.
    pub validity_mask: Array2<bool>,
    /// Centers of the target time bins, in seconds.
    pub time_axis: Array1<f64>,
    /// Centers of the target auditory-frequency bins, in Hz.
    pub frequency_axis_hz: Array1<f64>,
    /// Centers of the target auditory-frequency bins, in ERB rate.
    pub frequency_axis_erb: Array1<f64>,
    /// Analytic-representation energy before floor and map-boundary rejection.
    pub source_energy: f64,
    /// Analytic-representation energy rejected by the relative floor.
    pub floor_discarded_energy: f64,
    /// Retained-coefficient energy whose coordinates lie outside the map.
    pub boundary_discarded_energy: f64,
    /// Sum of floor- and boundary-discarded analytic energy.
    pub discarded_energy: f64,
    /// Type and guarantee of the reassignment calculation.
    pub mode: ReassignmentMode,
    /// Length of the zero-padded finite DFT domain used by the correction FIR,
    /// analytic projection, and derivative.
    pub analysis_fft_len: usize,
}

impl ReassignmentResult {
    /// Analytic-representation energy successfully deposited into the map.
    pub fn retained_energy(&self) -> f64 {
        self.energy_map.sum()
    }
}

/// Phase-preserving, analysis-only reassignment output.
///
/// `complex_map` contains the bilinearly transported, phase-corrected analytic
/// coefficients.  Its squared magnitude is not generally equal to
/// `reassignment.energy_map`, because multiple contributions may interfere.
#[derive(Clone, Debug)]
pub struct PhaseReassignmentResult {
    /// Coordinates, energy map, masks, axes, and energy accounting.
    pub reassignment: ReassignmentResult,
    /// Phase-corrected complex contributions on the reassigned grid.
    pub complex_map: Array2<Complex64>,
    /// Magnitude of the complex sum divided by the sum of contribution
    /// magnitudes. Empty bins are zero; all values are in `[0, 1]`.
    pub phase_coherence_map: Array2<f64>,
    /// Successfully transported energy accumulated on the reassigned map's
    /// channel/time grid before reassignment. Sample-based analyses use each
    /// coefficient's source sample; frame-based analyses use the originating
    /// frame. Its total matches the reassigned map, so floor and boundary
    /// rejection affect both sides of a sparsity comparison identically.
    pub unreassigned_energy_map: Array2<f64>,
}

impl PhaseReassignmentResult {
    /// A descriptive alias for [`Self::complex_map`].
    pub fn phase_corrected_map(&self) -> &Array2<Complex64> {
        &self.complex_map
    }

    /// Compare the effective support of the matched retained-energy maps.
    pub fn sparsity_comparison(&self) -> Result<SparsityComparison> {
        SparsityComparison::from_maps(&self.reassignment.energy_map, &self.unreassigned_energy_map)
    }
}

/// Entropy-based support statistics for a nonnegative energy map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparsityMetrics {
    /// Shannon entropy `-sum(p_i * ln(p_i))`, in nats.
    pub shannon_entropy: f64,
    /// Perplexity `exp(shannon_entropy)`, interpreted as effective bins.
    pub effective_bins: f64,
    /// Effective bins divided by the total number of bins.
    pub effective_bin_fraction: f64,
}

impl SparsityMetrics {
    /// Compute metrics from a finite, nonnegative energy map.
    ///
    /// An all-zero or empty map has zero entropy, zero effective bins, and a
    /// zero effective-bin fraction.
    pub fn from_energy_map(energy_map: &Array2<f64>) -> Result<Self> {
        if energy_map
            .iter()
            .any(|energy| !energy.is_finite() || *energy < 0.0)
        {
            return Err(Error::InvalidParameter(
                "sparsity metrics require a finite, nonnegative energy map".into(),
            ));
        }
        let total = energy_map.sum();
        if energy_map.is_empty() || total == 0.0 {
            return Ok(Self {
                shannon_entropy: 0.0,
                effective_bins: 0.0,
                effective_bin_fraction: 0.0,
            });
        }
        if !total.is_finite() {
            return Err(Error::Numerical(
                "non-finite total energy while computing sparsity metrics".into(),
            ));
        }
        let shannon_entropy = energy_map
            .iter()
            .filter(|&&energy| energy > 0.0)
            .map(|&energy| {
                let probability = energy / total;
                -probability * probability.ln()
            })
            .sum::<f64>()
            .max(0.0);
        let effective_bins = shannon_entropy.exp();
        Ok(Self {
            shannon_entropy,
            effective_bins,
            effective_bin_fraction: effective_bins / energy_map.len() as f64,
        })
    }
}

/// Matched-energy sparsity statistics before and after reassignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparsityComparison {
    /// Metrics for the reassigned energy map.
    pub reassigned: SparsityMetrics,
    /// Metrics for the same retained contributions at their source bins.
    pub unreassigned: SparsityMetrics,
}

impl SparsityComparison {
    /// Compare maps only when they contain the same retained energy.
    pub fn from_maps(
        reassigned_energy_map: &Array2<f64>,
        unreassigned_energy_map: &Array2<f64>,
    ) -> Result<Self> {
        if reassigned_energy_map.dim() != unreassigned_energy_map.dim() {
            return Err(Error::InvalidParameter(
                "sparsity comparison requires maps on the same grid".into(),
            ));
        }
        let reassigned = SparsityMetrics::from_energy_map(reassigned_energy_map)?;
        let unreassigned = SparsityMetrics::from_energy_map(unreassigned_energy_map)?;
        let reassigned_energy = reassigned_energy_map.sum();
        let unreassigned_energy = unreassigned_energy_map.sum();
        let tolerance = reassigned_energy.abs().max(unreassigned_energy.abs()) * 1e-10;
        if (reassigned_energy - unreassigned_energy).abs() > tolerance {
            return Err(Error::InvalidParameter(
                "sparsity comparison requires maps with identical retained energy".into(),
            ));
        }
        Ok(Self {
            reassigned,
            unreassigned,
        })
    }
}

/// Options for model-specific bandwidth consensus.
///
/// The default scales bracket the unscaled, psychophysically fitted GCFB with
/// typical normal-hearing variation rather than simulating hearing loss. At
/// 1 kHz and 50 dB, scales `0.8` and `1.2` produce composite-filter ERBs of
/// approximately `0.81` and `1.19` times the baseline ERB. This is consistent
/// with the roughly 10% to 18% between-listener variation reported by
/// [Moore et al. (1990)](https://doi.org/10.1121/1.399960) and
/// [Shen and Richards (2013)](https://doi.org/10.1121/1.4812856).
/// Listener-specific widening remains the responsibility of the configured
/// hearing-loss and compression-health parameters described by
/// [Irino (2023)](https://doi.org/10.1109/ACCESS.2023.3298673).
#[derive(Clone, Debug)]
pub struct BandwidthConsensusConfig {
    /// Multipliers applied to `b1`, `b2`, and `lvl_est.b2`.
    pub scales: Vec<f64>,
    /// A normalized scale map supports a bin only when it exceeds this value.
    pub relative_support_floor: f64,
    /// Minimum fraction of scales required by the consensus mask and salience
    /// order statistic.
    pub required_agreement: f64,
    /// Reassignment options shared by every bandwidth scale.
    pub reassignment_config: ReassignmentConfig,
}

impl Default for BandwidthConsensusConfig {
    fn default() -> Self {
        Self {
            scales: vec![0.8, 1.0, 1.2],
            relative_support_floor: 1e-6,
            required_agreement: 1.0,
            reassignment_config: ReassignmentConfig::default(),
        }
    }
}

/// Multi-bandwidth phase analyses and their scale-stability maps.
#[derive(Clone, Debug)]
pub struct BandwidthConsensusResult {
    /// Validated bandwidth scales in analysis order.
    pub scales: Vec<f64>,
    /// Phase-aware reassignment result for every scale.
    pub analyses: Vec<PhaseReassignmentResult>,
    /// Index of the unique unscaled (`1.0`) analysis.
    pub baseline_index: usize,
    /// Fraction of normalized scale maps above the support floor per bin.
    pub agreement_map: Array2<f64>,
    /// Bins meeting `required_agreement`.
    pub consensus_mask: Array2<bool>,
    /// Required-agreement order statistic of the normalized scale maps.
    pub salience_map: Array2<f64>,
}

struct CoefficientAnalysis {
    coefficient: Array2<Complex64>,
    power: Array2<f64>,
    t_hat: Array2<f64>,
    f_hat: Array2<f64>,
    validity_mask: Array2<bool>,
    mode: ReassignmentMode,
    analysis_fft_len: usize,
}

struct OperatorOutputs {
    coefficient: Array2<Complex64>,
    time_weighted: Array2<Complex64>,
    derivative: Array2<Complex64>,
    mode: ReassignmentMode,
    analysis_fft_len: usize,
}

struct PreparedInputSpectra {
    values: [Vec<Complex64>; 3],
}

struct AnalyticPgc {
    #[cfg(test)]
    impulse: Vec<Complex64>,
    spectrum: Vec<Complex64>,
}

/// Run GCFB v2.34 and its reassignment analysis together. The ordinary GCFB
/// output is identical to calling [`gcfb_v234`] alone.
pub fn gcfb_v234_with_reassignment(
    snd_in: &[f64],
    gc_param: GcParam,
) -> Result<(GcfbOutput, ReassignmentResult)> {
    let output = gcfb_v234(snd_in, gc_param)?;
    let reassignment = reassign_gcfb_v234(snd_in, &output)?;
    Ok((output, reassignment))
}

/// Reassign an existing GCFB v2.34 analysis with the default construction.
/// `snd_in` must be the same input used to produce `output`.
pub fn reassign_gcfb_v234(snd_in: &[f64], output: &GcfbOutput) -> Result<ReassignmentResult> {
    reassign_gcfb_v234_with_config(snd_in, output, &ReassignmentConfig::default())
}

/// Reassign an existing GCFB v2.34 analysis with explicit options.
pub fn reassign_gcfb_v234_with_config(
    snd_in: &[f64],
    output: &GcfbOutput,
    config: &ReassignmentConfig,
) -> Result<ReassignmentResult> {
    Ok(reassignment_products(snd_in, output, config, false)?.0)
}

/// Run GCFB v2.34 and its opt-in phase-aware reassignment together.
///
/// The ordinary output is identical to [`gcfb_v234`]. The complex result is
/// analysis-only and carries no reconstruction guarantee.
pub fn gcfb_v234_with_phase_reassignment(
    snd_in: &[f64],
    gc_param: GcParam,
) -> Result<(GcfbOutput, PhaseReassignmentResult)> {
    let output = gcfb_v234(snd_in, gc_param)?;
    let reassignment = phase_reassign_gcfb_v234(snd_in, &output)?;
    Ok((output, reassignment))
}

/// Phase-reassign an existing GCFB output with default options.
/// `snd_in` must be the input used to produce `output`.
pub fn phase_reassign_gcfb_v234(
    snd_in: &[f64],
    output: &GcfbOutput,
) -> Result<PhaseReassignmentResult> {
    phase_reassign_gcfb_v234_with_config(snd_in, output, &ReassignmentConfig::default())
}

/// Phase-reassign an existing GCFB output with explicit options.
pub fn phase_reassign_gcfb_v234_with_config(
    snd_in: &[f64],
    output: &GcfbOutput,
    config: &ReassignmentConfig,
) -> Result<PhaseReassignmentResult> {
    let (reassignment, phase) = reassignment_products(snd_in, output, config, true)?;
    let phase = phase.expect("phase transport was requested");
    let phase_coherence_map = phase.coherence_map();
    Ok(PhaseReassignmentResult {
        reassignment,
        complex_map: phase.complex_map,
        phase_coherence_map,
        unreassigned_energy_map: phase.unreassigned_energy_map,
    })
}

/// Run a phase analysis at every configured bandwidth and form a consensus.
///
/// Exactly one scale must be `1.0`; its phase analysis is computed from the
/// returned, unscaled [`GcfbOutput`]. Other scales uniformly multiply `b1`,
/// `b2`, and `lvl_est.b2` before running independent GCFB analyses.
pub fn gcfb_v234_with_bandwidth_consensus(
    snd_in: &[f64],
    gc_param: GcParam,
    config: &BandwidthConsensusConfig,
) -> Result<(GcfbOutput, BandwidthConsensusResult)> {
    let baseline_index = validate_consensus_config(config)?;
    let output = gcfb_v234(snd_in, gc_param.clone())?;
    let mut analyses = Vec::with_capacity(config.scales.len());
    for (index, &scale) in config.scales.iter().enumerate() {
        let analysis = if index == baseline_index {
            phase_reassign_gcfb_v234_with_config(snd_in, &output, &config.reassignment_config)?
        } else {
            let scaled_output = gcfb_v234_with_preserved_hearing_loss(
                snd_in,
                scale_bandwidths(gc_param.clone(), scale),
                &output.gc_param.hloss,
            )?;
            phase_reassign_gcfb_v234_with_config(
                snd_in,
                &scaled_output,
                &config.reassignment_config,
            )?
        };
        analyses.push(analysis);
    }
    let (agreement_map, consensus_mask, salience_map) = consensus_maps(
        &analyses,
        config.relative_support_floor,
        config.required_agreement,
    )?;
    Ok((
        output,
        BandwidthConsensusResult {
            scales: config.scales.clone(),
            analyses,
            baseline_index,
            agreement_map,
            consensus_mask,
            salience_map,
        },
    ))
}

fn reassignment_products(
    snd_in: &[f64],
    output: &GcfbOutput,
    config: &ReassignmentConfig,
    include_phase: bool,
) -> Result<(ReassignmentResult, Option<PhaseAccounting>)> {
    validate_analysis_input(snd_in, output, config)?;
    let operator = conditioned_operator_outputs(snd_in, output)?;
    let analysis = analyze_coefficients(operator, config.coefficient_floor)?;
    let frame_mode = output.gc_param.ctrl == ControlMode::Dynamic
        && output.gc_param.dyn_hpaf.str_prc.contains("frame");
    let (time_axis, mut transported) = if frame_mode {
        transport_frame_energy(output, &analysis, include_phase)?
    } else {
        transport_sample_energy(output, &analysis, include_phase)?
    };
    let frequency_axis_hz = output.gc_param.fr1.clone();
    let (frequency_axis_erb, _) = utils::freq2erb(frequency_axis_hz.as_slice().unwrap());
    transported.energy.discarded = transported.energy.floor + transported.energy.boundary;

    let reassignment = ReassignmentResult {
        energy_map: transported.energy.map,
        t_hat: analysis.t_hat,
        f_hat: analysis.f_hat,
        validity_mask: analysis.validity_mask,
        time_axis,
        frequency_axis_hz,
        frequency_axis_erb,
        source_energy: transported.energy.source,
        floor_discarded_energy: transported.energy.floor,
        boundary_discarded_energy: transported.energy.boundary,
        discarded_energy: transported.energy.discarded,
        mode: analysis.mode,
        analysis_fft_len: analysis.analysis_fft_len,
    };
    Ok((reassignment, transported.phase))
}

fn validate_consensus_config(config: &BandwidthConsensusConfig) -> Result<usize> {
    if config.scales.len() < 2
        || !config.relative_support_floor.is_finite()
        || config.relative_support_floor <= 0.0
        || config.relative_support_floor >= 1.0
        || !config.required_agreement.is_finite()
        || config.required_agreement <= 0.0
        || config.required_agreement > 1.0
    {
        return Err(Error::InvalidParameter(
            "bandwidth consensus requires at least two scales, a support floor in (0, 1), and required agreement in (0, 1]"
                .into(),
        ));
    }
    if config
        .scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(Error::InvalidParameter(
            "bandwidth consensus scales must be positive and finite".into(),
        ));
    }
    for (index, scale) in config.scales.iter().enumerate() {
        if config.scales[..index].contains(scale) {
            return Err(Error::InvalidParameter(
                "bandwidth consensus scales must be unique".into(),
            ));
        }
    }
    let baselines: Vec<usize> = config
        .scales
        .iter()
        .enumerate()
        .filter_map(|(index, &scale)| (scale == 1.0).then_some(index))
        .collect();
    if baselines.len() != 1 {
        return Err(Error::InvalidParameter(
            "bandwidth consensus requires exactly one 1.0 baseline scale".into(),
        ));
    }
    validate_reassignment_config(&config.reassignment_config)?;
    Ok(baselines[0])
}

fn validate_reassignment_config(config: &ReassignmentConfig) -> Result<()> {
    if !config.coefficient_floor.is_finite()
        || config.coefficient_floor <= 0.0
        || config.coefficient_floor >= 1.0
    {
        return Err(Error::InvalidParameter(
            "reassignment coefficient floor must be in (0, 1)".into(),
        ));
    }
    Ok(())
}

fn scale_bandwidths(mut parameters: GcParam, scale: f64) -> GcParam {
    for coefficient in &mut parameters.b1 {
        *coefficient *= scale;
    }
    for row in &mut parameters.b2 {
        for coefficient in row {
            *coefficient *= scale;
        }
    }
    parameters.lvl_est.b2 *= scale;
    parameters
}

fn consensus_maps(
    analyses: &[PhaseReassignmentResult],
    relative_support_floor: f64,
    required_agreement: f64,
) -> Result<(Array2<f64>, Array2<bool>, Array2<f64>)> {
    let dimensions = analyses
        .first()
        .ok_or_else(|| Error::InvalidParameter("bandwidth consensus has no analyses".into()))?
        .reassignment
        .energy_map
        .dim();
    if analyses
        .iter()
        .any(|analysis| analysis.reassignment.energy_map.dim() != dimensions)
    {
        return Err(Error::InvalidParameter(
            "bandwidth consensus scale maps do not share a target grid".into(),
        ));
    }
    let maxima: Vec<f64> = analyses
        .iter()
        .map(|analysis| {
            analysis
                .reassignment
                .energy_map
                .iter()
                .copied()
                .fold(0.0, f64::max)
        })
        .collect();
    let required_count = (required_agreement * analyses.len() as f64).ceil() as usize;
    let mut agreement_map = Array2::zeros(dimensions);
    let mut consensus_mask = Array2::from_elem(dimensions, false);
    let mut salience_map = Array2::zeros(dimensions);
    for ch in 0..dimensions.0 {
        for time in 0..dimensions.1 {
            let mut normalized: Vec<f64> = analyses
                .iter()
                .zip(&maxima)
                .map(|(analysis, &maximum)| {
                    if maximum > 0.0 {
                        analysis.reassignment.energy_map[[ch, time]] / maximum
                    } else {
                        0.0
                    }
                })
                .collect();
            let support_count = normalized
                .iter()
                .filter(|&&value| value > relative_support_floor)
                .count();
            agreement_map[[ch, time]] = support_count as f64 / analyses.len() as f64;
            consensus_mask[[ch, time]] = support_count >= required_count;
            normalized.sort_by(|left, right| right.total_cmp(left));
            salience_map[[ch, time]] = normalized[required_count - 1];
        }
    }
    Ok((agreement_map, consensus_mask, salience_map))
}

fn validate_analysis_input(
    snd: &[f64],
    output: &GcfbOutput,
    config: &ReassignmentConfig,
) -> Result<()> {
    validate_reassignment_config(config)?;
    let channels = output.gc_param.num_ch;
    if snd.is_empty()
        || channels == 0
        || output.gc_param.fr1.len() != channels
        || output.scgc_smpl.nrows() != channels
        || output.gc_resp.gain_factor.len() != channels
    {
        return Err(Error::InvalidParameter(
            "reassignment requires a matching non-empty GCFB output and a coefficient floor in (0, 1)"
                .into(),
        ));
    }
    if output.scgc_smpl.ncols() != snd.len() {
        return Err(Error::InvalidParameter(
            "input length does not match the supplied GCFB output".into(),
        ));
    }
    Ok(())
}

fn correction_fir(param: &GcParam) -> Result<Vec<f64>> {
    if param.out_mid_crct.eq_ignore_ascii_case("no") {
        Ok(vec![1.0])
    } else {
        let (fir, _) = utils::mk_filter_field2cochlea(&param.out_mid_crct, param.fs, true)?;
        Ok(fir.to_vec())
    }
}

fn conditioned_operator_outputs(snd_in: &[f64], output: &GcfbOutput) -> Result<OperatorOutputs> {
    let pgcs: Vec<Vec<f64>> = (0..output.gc_param.num_ch)
        .map(|ch| real_pgc(&output.gc_param, output, ch))
        .collect::<Result<_>>()?;
    let correction = correction_fir(&output.gc_param)?;
    let maximum_pgc_len = pgcs.iter().map(Vec::len).max().unwrap_or(0);
    let analysis_fft_len = analysis_fft_len(snd_in.len(), correction.len(), maximum_pgc_len)?;
    let mut inputs = prepare_input_spectra(snd_in, output.gc_param.fs, analysis_fft_len);
    apply_correction_fir(&mut inputs, &correction);
    let sample_mode = output.gc_param.ctrl == ControlMode::Dynamic
        && output.gc_param.dyn_hpaf.str_prc.contains("sample");
    if sample_mode {
        sample_operator_outputs(snd_in.len(), output, &pgcs, &inputs, analysis_fft_len)
    } else {
        fixed_operator_outputs(snd_in.len(), output, &pgcs, &inputs, analysis_fft_len)
    }
}

fn analysis_fft_len(
    signal_len: usize,
    correction_len: usize,
    maximum_pgc_len: usize,
) -> Result<usize> {
    let convolution_len = signal_len
        .checked_add(correction_len)
        .and_then(|value| value.checked_add(maximum_pgc_len))
        .and_then(|value| value.checked_sub(2))
        .ok_or_else(|| Error::Unsupported("reassignment DFT length overflow".into()))?;
    convolution_len
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| Error::Unsupported("reassignment DFT length overflow".into()))
}

fn real_pgc(param: &GcParam, output: &GcfbOutput, ch: usize) -> Result<Vec<f64>> {
    let impulse = gammachirp::gammachirp(
        &[output.gc_resp.fr1[ch]],
        param.fs,
        param.n,
        output.gc_resp.b1_val[ch],
        output.gc_resp.c1_val[ch],
        0.0,
        Carrier::Cosine,
        Normalization::Peak,
    )?;
    Ok(impulse
        .gc
        .row(0)
        .iter()
        .take(impulse.len_gc[0])
        .copied()
        .collect())
}

fn analytic_pgc(real: &[f64], fft_len: usize) -> AnalyticPgc {
    let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    for (destination, &source) in spectrum.iter_mut().zip(real) {
        destination.re = source;
    }
    dsp::fft(&mut spectrum, false);
    for value in &mut spectrum[1..fft_len / 2] {
        *value *= 2.0;
    }
    for value in &mut spectrum[fft_len / 2 + 1..] {
        *value = Complex64::new(0.0, 0.0);
    }
    #[cfg(test)]
    let impulse = {
        let mut impulse = spectrum.clone();
        dsp::fft(&mut impulse, true);
        impulse
    };
    AnalyticPgc {
        #[cfg(test)]
        impulse,
        spectrum,
    }
}

fn prepare_input_spectra(signal: &[f64], sample_rate: f64, fft_len: usize) -> PreparedInputSpectra {
    let mut input = vec![Complex64::new(0.0, 0.0); fft_len];
    let mut time_weighted = input.clone();
    for (sample, &value) in signal.iter().enumerate() {
        input[sample].re = value;
        time_weighted[sample].re = sample as f64 / sample_rate * value;
    }
    dsp::fft(&mut input, false);
    dsp::fft(&mut time_weighted, false);
    let mut derivative = input.clone();
    for (bin, value) in derivative.iter_mut().enumerate() {
        if bin == 0 || bin == fft_len / 2 {
            *value = Complex64::new(0.0, 0.0);
            continue;
        }
        let frequency_hz = if bin < fft_len / 2 {
            bin as f64 * sample_rate / fft_len as f64
        } else {
            (bin as f64 - fft_len as f64) * sample_rate / fft_len as f64
        };
        *value *= Complex64::new(0.0, 2.0 * PI * frequency_hz);
    }
    PreparedInputSpectra {
        values: [input, time_weighted, derivative],
    }
}

fn apply_correction_fir(inputs: &mut PreparedInputSpectra, correction: &[f64]) {
    if correction.len() == 1 && correction[0] == 1.0 {
        return;
    }
    let fft_len = inputs.values[0].len();
    let mut correction_spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    for (destination, &coefficient) in correction_spectrum.iter_mut().zip(correction) {
        destination.re = coefficient;
    }
    dsp::fft(&mut correction_spectrum, false);
    for values in &mut inputs.values {
        for (value, correction) in values.iter_mut().zip(&correction_spectrum) {
            *value *= correction;
        }
    }
}

fn apply_analytic_pgc(pgc: &AnalyticPgc, inputs: &PreparedInputSpectra) -> [Vec<Complex64>; 3] {
    std::array::from_fn(|pass| {
        let mut output: Vec<Complex64> = pgc
            .spectrum
            .iter()
            .zip(&inputs.values[pass])
            .map(|(filter, input)| filter * input)
            .collect();
        dsp::fft(&mut output, true);
        output
    })
}

fn fixed_operator_outputs(
    samples: usize,
    output: &GcfbOutput,
    pgcs: &[Vec<f64>],
    inputs: &PreparedInputSpectra,
    analysis_fft_len: usize,
) -> Result<OperatorOutputs> {
    let param = &output.gc_param;
    let response = &output.gc_resp;
    let channels = param.num_ch;
    let (centers, b2, c2) = if param.ctrl == ControlMode::Static {
        (
            response.fr2.column(0).to_owned(),
            response.b2_val.clone(),
            response.c2_val.clone(),
        )
    } else {
        (
            response.fp1.mapv(|value| param.lvl_est.frat * value),
            Array1::from_elem(channels, param.lvl_est.b2),
            &param.hloss.fb_compression_health * param.lvl_est.c2,
        )
    };
    let coefficients = make_asym_cmp_filters_v2(
        param.fs,
        centers.as_slice().unwrap(),
        b2.as_slice().unwrap(),
        c2.as_slice().unwrap(),
    )?;
    let mut result = empty_operator_outputs(
        channels,
        samples,
        if param.ctrl == ControlMode::Dynamic && param.dyn_hpaf.str_prc.contains("frame") {
            ReassignmentMode::Frame
        } else {
            ReassignmentMode::Fixed
        },
        analysis_fft_len,
    );
    for (ch, real_pgc) in pgcs.iter().enumerate() {
        let pgc = analytic_pgc(real_pgc, analysis_fft_len);
        let pgc_outputs = apply_analytic_pgc(&pgc, inputs);
        let filtered: [Vec<Complex64>; 3] = std::array::from_fn(|pass| {
            filter_fixed_cascade(&pgc_outputs[pass][..samples], &coefficients, ch)
        });
        assign_operator_row(&mut result, ch, &filtered);
        for sample in 0..samples {
            verify_real_branch(
                result.coefficient[[ch, sample]].re,
                output.scgc_smpl[[ch, sample]],
            )?;
        }
    }
    Ok(result)
}

fn sample_operator_outputs(
    samples: usize,
    output: &GcfbOutput,
    pgcs: &[Vec<f64>],
    inputs: &PreparedInputSpectra,
    analysis_fft_len: usize,
) -> Result<OperatorOutputs> {
    let param = &output.gc_param;
    let response = &output.gc_resp;
    let channels = param.num_ch;
    if response.fr2.dim() != (channels, samples) {
        return Err(Error::InvalidParameter(
            "sample reassignment requires the realized HP-AF center-frequency history".into(),
        ));
    }
    let mut result = empty_operator_outputs(
        channels,
        samples,
        ReassignmentMode::SampleConditional,
        analysis_fft_len,
    );
    for (ch, real_pgc) in pgcs.iter().enumerate() {
        let pgc = analytic_pgc(real_pgc, analysis_fft_len);
        let pgc_outputs = apply_analytic_pgc(&pgc, inputs);
        let mut states: [ComplexCascadeState; 3] = std::array::from_fn(|_| Default::default());
        let mut coefficients = make_asym_cmp_filters_v2(
            param.fs,
            &[response.fr2[[ch, 0]]],
            &[response.b2_val[ch]],
            &[response.c2_val[ch]],
        )?;
        let pass_inputs = pgc_outputs[0]
            .iter()
            .zip(&pgc_outputs[1])
            .zip(&pgc_outputs[2]);
        for (sample, ((&coefficient_input, &time_input), &derivative_input)) in
            pass_inputs.take(samples).enumerate()
        {
            if sample % param.num_update_asym_cmp == 0 {
                coefficients = make_asym_cmp_filters_v2(
                    param.fs,
                    &[response.fr2[[ch, sample]]],
                    &[response.b2_val[ch]],
                    &[response.c2_val[ch]],
                )?;
            }
            let inputs = [coefficient_input, time_input, derivative_input];
            let values: [Complex64; 3] =
                std::array::from_fn(|pass| states[pass].process(inputs[pass], &coefficients, 0));
            result.coefficient[[ch, sample]] = values[0];
            result.time_weighted[[ch, sample]] = values[1];
            result.derivative[[ch, sample]] = values[2];
            let gain = response.gain_factor[ch];
            if !gain.is_finite() || gain == 0.0 {
                return Err(Error::Numerical(
                    "sample reassignment encountered a non-finite or zero output gain".into(),
                ));
            }
            verify_real_branch(
                result.coefficient[[ch, sample]].re,
                output.dcgc_out[[ch, sample]] / gain,
            )?;
        }
    }
    Ok(result)
}

fn empty_operator_outputs(
    channels: usize,
    samples: usize,
    mode: ReassignmentMode,
    analysis_fft_len: usize,
) -> OperatorOutputs {
    let zeros = || Array2::from_elem((channels, samples), Complex64::new(0.0, 0.0));
    OperatorOutputs {
        coefficient: zeros(),
        time_weighted: zeros(),
        derivative: zeros(),
        mode,
        analysis_fft_len,
    }
}

fn assign_operator_row(result: &mut OperatorOutputs, ch: usize, values: &[Vec<Complex64>; 3]) {
    result
        .coefficient
        .row_mut(ch)
        .assign(&Array1::from(values[0].clone()));
    result
        .time_weighted
        .row_mut(ch)
        .assign(&Array1::from(values[1].clone()));
    result
        .derivative
        .row_mut(ch)
        .assign(&Array1::from(values[2].clone()));
}

#[derive(Clone, Copy, Default)]
struct BiquadState {
    input_previous: Complex64,
    input_before_previous: Complex64,
    output_previous: Complex64,
    output_before_previous: Complex64,
}

#[derive(Default)]
struct ComplexCascadeState {
    sections: [BiquadState; 4],
}

impl ComplexCascadeState {
    fn process(&mut self, input: Complex64, coefficients: &AcfCoef, ch: usize) -> Complex64 {
        let mut current = input;
        for (section, state) in self.sections.iter_mut().enumerate() {
            let output = (coefficients.bz[[ch, 0, section]] * current
                + coefficients.bz[[ch, 1, section]] * state.input_previous
                + coefficients.bz[[ch, 2, section]] * state.input_before_previous
                - coefficients.ap[[ch, 1, section]] * state.output_previous
                - coefficients.ap[[ch, 2, section]] * state.output_before_previous)
                / coefficients.ap[[ch, 0, section]];
            state.input_before_previous = state.input_previous;
            state.input_previous = current;
            state.output_before_previous = state.output_previous;
            state.output_previous = output;
            current = output;
        }
        current
    }
}

fn filter_fixed_cascade(input: &[Complex64], coefficients: &AcfCoef, ch: usize) -> Vec<Complex64> {
    let mut state = ComplexCascadeState::default();
    input
        .iter()
        .map(|&value| state.process(value, coefficients, ch))
        .collect()
}

fn verify_real_branch(actual: f64, expected: f64) -> Result<()> {
    let tolerance = 2e-8 * actual.abs().max(expected.abs()).max(1.0);
    if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > tolerance {
        return Err(Error::InvalidParameter(format!(
            "input does not match the supplied GCFB output or its real cGC branch ({actual} versus {expected}, difference {})",
            (actual - expected).abs()
        )));
    }
    Ok(())
}

fn analyze_coefficients(
    operator: OperatorOutputs,
    relative_floor: f64,
) -> Result<CoefficientAnalysis> {
    let dimensions = operator.coefficient.dim();
    let mut power = Array2::zeros(dimensions);
    let mut t_hat = Array2::from_elem(dimensions, f64::NAN);
    let mut f_hat = Array2::from_elem(dimensions, f64::NAN);
    for ch in 0..dimensions.0 {
        for sample in 0..dimensions.1 {
            let coefficient = operator.coefficient[[ch, sample]];
            let norm = coefficient.norm_sqr();
            power[[ch, sample]] = norm / 2.0;
            if norm > 0.0 && norm.is_finite() {
                t_hat[[ch, sample]] = (operator.time_weighted[[ch, sample]] / coefficient).re;
                f_hat[[ch, sample]] =
                    (operator.derivative[[ch, sample]] / coefficient).im / (2.0 * PI);
            }
        }
    }
    let validity_mask = apply_floor(&power, &t_hat, &f_hat, relative_floor)?;
    Ok(CoefficientAnalysis {
        coefficient: operator.coefficient,
        power,
        t_hat,
        f_hat,
        validity_mask,
        mode: operator.mode,
        analysis_fft_len: operator.analysis_fft_len,
    })
}

fn apply_floor(
    power: &Array2<f64>,
    t_hat: &Array2<f64>,
    f_hat: &Array2<f64>,
    relative_floor: f64,
) -> Result<Array2<bool>> {
    let mut mask = Array2::from_elem(power.dim(), false);
    for ch in 0..power.nrows() {
        if power.row(ch).iter().any(|value| !value.is_finite()) {
            return Err(Error::Numerical(format!(
                "non-finite analytic power in reassignment channel {ch}"
            )));
        }
        let maximum = power.row(ch).iter().copied().fold(0.0, f64::max);
        if maximum <= 0.0 {
            continue;
        }
        let threshold = relative_floor * maximum;
        for sample in 0..power.ncols() {
            if power[[ch, sample]] >= threshold {
                if !t_hat[[ch, sample]].is_finite() || !f_hat[[ch, sample]].is_finite() {
                    return Err(Error::Numerical(format!(
                        "non-finite reassignment coordinate above the coefficient floor at channel {ch}, sample {sample}"
                    )));
                }
                mask[[ch, sample]] = true;
            }
        }
    }
    Ok(mask)
}

#[derive(Default)]
struct EnergyAccounting {
    map: Array2<f64>,
    source: f64,
    floor: f64,
    boundary: f64,
    discarded: f64,
}

struct PhaseAccounting {
    complex_map: Array2<Complex64>,
    contribution_magnitude_map: Array2<f64>,
    unreassigned_energy_map: Array2<f64>,
}

impl PhaseAccounting {
    fn coherence_map(&self) -> Array2<f64> {
        Array2::from_shape_fn(self.complex_map.dim(), |index| {
            let magnitude_sum = self.contribution_magnitude_map[index];
            if magnitude_sum > 0.0 {
                (self.complex_map[index].norm() / magnitude_sum).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
    }
}

struct TransportAccounting {
    energy: EnergyAccounting,
    phase: Option<PhaseAccounting>,
}

impl TransportAccounting {
    fn new(target_dimensions: (usize, usize), include_phase: bool) -> Self {
        Self {
            energy: EnergyAccounting {
                map: Array2::zeros(target_dimensions),
                ..EnergyAccounting::default()
            },
            phase: include_phase.then(|| PhaseAccounting {
                complex_map: Array2::from_elem(target_dimensions, Complex64::new(0.0, 0.0)),
                contribution_magnitude_map: Array2::zeros(target_dimensions),
                unreassigned_energy_map: Array2::zeros(target_dimensions),
            }),
        }
    }
}

fn transport_frame_energy(
    output: &GcfbOutput,
    analysis: &CoefficientAnalysis,
    include_phase: bool,
) -> Result<(Array1<f64>, TransportAccounting)> {
    let param = &output.gc_param;
    let response = &output.gc_resp;
    let frames = response.asym_func_gain.ncols();
    if frames == 0
        || response.asym_func_gain.nrows() != param.num_ch
        || param.dyn_hpaf.val_win.len() != param.dyn_hpaf.len_frame
    {
        return Err(Error::InvalidParameter(
            "frame reassignment requires populated frame gains and window metadata".into(),
        ));
    }
    let time_axis = Array1::from_iter(
        (0..frames).map(|frame| frame as f64 * param.dyn_hpaf.len_shift as f64 / param.fs),
    );
    let (frequency_axis, _) = utils::freq2erb(param.fr1.as_slice().unwrap());
    let static_response = cmprs_gc_frsp(
        param.fr1.as_slice().unwrap(),
        param.fs,
        param.n,
        response.b1_val.as_slice().unwrap(),
        response.c1_val.as_slice().unwrap(),
        &[param.lvl_est.frat],
        &[param.lvl_est.b2],
        (&param.hloss.fb_compression_health * param.lvl_est.c2)
            .as_slice()
            .unwrap(),
        2048,
    )?;
    let mut accounting = TransportAccounting::new((param.num_ch, frames), include_phase);
    let half = param.dyn_hpaf.len_frame / 2;
    for ch in 0..param.num_ch {
        for frame in 0..frames {
            let gain = response.gain_factor[ch]
                * static_response.norm_fct_fp2[ch]
                * response.asym_func_gain[[ch, frame]];
            for offset in 0..param.dyn_hpaf.len_frame {
                let source = frame as isize * param.dyn_hpaf.len_shift as isize + offset as isize
                    - half as isize;
                if source < 0 || source as usize >= analysis.power.ncols() {
                    continue;
                }
                let sample = source as usize;
                let energy =
                    param.dyn_hpaf.val_win[offset] * gain.powi(2) * analysis.power[[ch, sample]];
                deposit_energy(
                    &mut accounting,
                    energy,
                    analysis.validity_mask[[ch, sample]],
                    analysis.t_hat[[ch, sample]],
                    analysis.f_hat[[ch, sample]],
                    ch,
                    frame,
                    sample as f64 / param.fs,
                    param.fr1[ch],
                    analysis.coefficient[[ch, sample]],
                    time_axis.as_slice().unwrap(),
                    frequency_axis.as_slice().unwrap(),
                )?;
            }
        }
    }
    Ok((time_axis, accounting))
}

fn transport_sample_energy(
    output: &GcfbOutput,
    analysis: &CoefficientAnalysis,
    include_phase: bool,
) -> Result<(Array1<f64>, TransportAccounting)> {
    let param = &output.gc_param;
    let samples = analysis.power.ncols();
    let time_axis = Array1::from_iter((0..samples).map(|sample| sample as f64 / param.fs));
    let (frequency_axis, _) = utils::freq2erb(param.fr1.as_slice().unwrap());
    let mut accounting = TransportAccounting::new((param.num_ch, samples), include_phase);
    for ch in 0..param.num_ch {
        let gain = output.gc_resp.gain_factor[ch];
        for sample in 0..samples {
            let energy = gain.powi(2) * analysis.power[[ch, sample]];
            deposit_energy(
                &mut accounting,
                energy,
                analysis.validity_mask[[ch, sample]],
                analysis.t_hat[[ch, sample]],
                analysis.f_hat[[ch, sample]],
                ch,
                sample,
                sample as f64 / param.fs,
                param.fr1[ch],
                analysis.coefficient[[ch, sample]],
                time_axis.as_slice().unwrap(),
                frequency_axis.as_slice().unwrap(),
            )?;
        }
    }
    Ok((time_axis, accounting))
}

#[allow(clippy::too_many_arguments)]
fn deposit_energy(
    accounting: &mut TransportAccounting,
    energy: f64,
    valid: bool,
    time: f64,
    frequency_hz: f64,
    source_channel: usize,
    source_time_bin: usize,
    source_time: f64,
    source_center_hz: f64,
    source_coefficient: Complex64,
    time_axis: &[f64],
    frequency_axis_erb: &[f64],
) -> Result<()> {
    if !energy.is_finite() || energy < 0.0 {
        return Err(Error::Numerical(
            "non-finite or negative analytic energy during reassignment transport".into(),
        ));
    }
    if energy == 0.0 {
        return Ok(());
    }
    accounting.energy.source += energy;
    if !valid {
        accounting.energy.floor += energy;
        return Ok(());
    }
    if frequency_hz <= 0.0 {
        accounting.energy.boundary += energy;
        return Ok(());
    }
    let (erb, _) = utils::freq2erb(&[frequency_hz]);
    let Some(time_weights) = linear_weights(time_axis, time) else {
        accounting.energy.boundary += energy;
        return Ok(());
    };
    let Some(frequency_weights) = linear_weights(frequency_axis_erb, erb[0]) else {
        accounting.energy.boundary += energy;
        return Ok(());
    };
    let phase_contribution = if accounting.phase.is_some() {
        let coefficient_power = source_coefficient.norm_sqr();
        if !coefficient_power.is_finite() || coefficient_power <= 0.0 {
            return Err(Error::Numerical(
                "retained phase contribution has zero or non-finite analytic magnitude".into(),
            ));
        }
        let scaled_coefficient = source_coefficient * (energy / coefficient_power).sqrt();
        let phase = PI * (source_center_hz + frequency_hz) * (time - source_time);
        Some(scaled_coefficient * Complex64::new(phase.cos(), phase.sin()))
    } else {
        None
    };
    for &(time_bin, time_weight) in &time_weights {
        for &(frequency_bin, frequency_weight) in &frequency_weights {
            let weight = time_weight * frequency_weight;
            accounting.energy.map[[frequency_bin, time_bin]] += energy * weight;
            if let (Some(phase), Some(contribution)) =
                (accounting.phase.as_mut(), phase_contribution)
            {
                phase.complex_map[[frequency_bin, time_bin]] += contribution * weight;
                phase.contribution_magnitude_map[[frequency_bin, time_bin]] +=
                    contribution.norm() * weight;
            }
        }
    }
    if let Some(phase) = accounting.phase.as_mut() {
        phase.unreassigned_energy_map[[source_channel, source_time_bin]] += energy;
    }
    Ok(())
}

fn linear_weights(axis: &[f64], value: f64) -> Option<Vec<(usize, f64)>> {
    if axis.is_empty() || !value.is_finite() || value < axis[0] || value > axis[axis.len() - 1] {
        return None;
    }
    if axis.len() == 1 {
        return ((value - axis[0]).abs() <= f64::EPSILON * axis[0].abs().max(1.0))
            .then(|| vec![(0, 1.0)]);
    }
    match axis.binary_search_by(|candidate| candidate.total_cmp(&value)) {
        Ok(index) => Some(vec![(index, 1.0)]),
        Err(upper) if upper > 0 && upper < axis.len() => {
            let lower = upper - 1;
            let upper_weight = (value - axis[lower]) / (axis[upper] - axis[lower]);
            Some(vec![(lower, 1.0 - upper_weight), (upper, upper_weight)])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;
    use crate::gcfb_v234::{DynHpaf, GainReference, GcParam};

    fn compact_frame_parameters() -> GcParam {
        GcParam {
            fs: 8000.0,
            num_ch: 8,
            f_range: [200.0, 1800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Dynamic,
            dyn_hpaf: DynHpaf {
                t_frame: 0.008,
                t_shift: 0.004,
                ..DynHpaf::default()
            },
            ..GcParam::default()
        }
    }

    fn static_parameters() -> GcParam {
        GcParam {
            ctrl: ControlMode::Static,
            ..compact_frame_parameters()
        }
    }

    fn deterministic_noise(samples: usize) -> Vec<f64> {
        let mut state = 0x9e37_79b9_u32;
        (0..samples)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (f64::from(state) / f64::from(u32::MAX)) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn analytic_projection_is_one_sided_and_preserves_the_real_pgc() {
        let signal = vec![0.0; 128];
        let output = gcfb_v234(&signal, static_parameters()).unwrap();
        for ch in 0..output.gc_param.num_ch {
            let real = real_pgc(&output.gc_param, &output, ch).unwrap();
            let fft_len = analysis_fft_len(signal.len(), 1, real.len()).unwrap();
            let analytic = analytic_pgc(&real, fft_len);
            for (sample, value) in analytic.impulse.iter().enumerate() {
                let expected = real.get(sample).copied().unwrap_or(0.0);
                assert_relative_eq!(value.re, expected, epsilon = 5e-13);
            }
            let mut recovered_spectrum = analytic.impulse.clone();
            dsp::fft(&mut recovered_spectrum, false);
            let total: f64 = recovered_spectrum
                .iter()
                .map(|value| value.norm_sqr())
                .sum();
            let negative: f64 = recovered_spectrum[fft_len / 2 + 1..]
                .iter()
                .map(|value| value.norm_sqr())
                .sum();
            let roundoff_bound = total.max(1.0)
                * (64.0 * f64::EPSILON * fft_len.ilog2() as f64).powi(2)
                * fft_len as f64;
            assert!(
                negative <= roundoff_bound,
                "negative-bin energy {negative} exceeded {roundoff_bound}"
            );
        }
    }

    #[test]
    fn fixed_and_frame_real_branches_match_scgc_before_gain() {
        let signal: Vec<f64> = (0..320)
            .map(|sample| (2.0 * PI * 700.0 * sample as f64 / 8000.0).sin())
            .collect();
        for parameters in [static_parameters(), compact_frame_parameters()] {
            for correction in ["No", "ELC", "EarDrum"] {
                let mut corrected = parameters.clone();
                corrected.out_mid_crct = correction.into();
                let output = gcfb_v234(&signal, corrected).unwrap();
                let operator = conditioned_operator_outputs(&signal, &output).unwrap();
                for (actual, expected) in operator.coefficient.iter().zip(output.scgc_smpl.iter()) {
                    assert_relative_eq!(actual.re, expected, epsilon = 2e-11, max_relative = 2e-10);
                }
            }
        }
    }

    #[test]
    fn sample_real_branch_matches_conditioned_dcgc_before_gain() {
        let mut parameters = compact_frame_parameters();
        parameters.num_ch = 3;
        parameters.f_range = [300.0, 1400.0];
        parameters.dyn_hpaf.str_prc = "sample-base".into();
        parameters.num_update_asym_cmp = 3;
        let signal: Vec<f64> = (0..48)
            .map(|sample| 0.1 * (2.0 * PI * 600.0 * sample as f64 / 8000.0).sin())
            .collect();
        for correction in ["No", "ELC", "EarDrum"] {
            let mut corrected = parameters.clone();
            corrected.out_mid_crct = correction.into();
            let output = gcfb_v234(&signal, corrected).unwrap();
            let operator = conditioned_operator_outputs(&signal, &output).unwrap();
            for ch in 0..output.gc_param.num_ch {
                for sample in 0..signal.len() {
                    assert_relative_eq!(
                        operator.coefficient[[ch, sample]].re,
                        output.dcgc_out[[ch, sample]] / output.gc_resp.gain_factor[ch],
                        epsilon = 2e-11,
                        max_relative = 2e-10
                    );
                }
            }
        }
    }

    #[test]
    fn corrected_frame_impulse_reassigns_to_the_exact_original_input_time() {
        let mut signal = vec![0.0; 512];
        let impulse_sample = 192;
        signal[impulse_sample] = 1.0;
        let expected = impulse_sample as f64 / 8000.0;
        for correction in ["No", "ELC", "EarDrum"] {
            let mut parameters = compact_frame_parameters();
            parameters.out_mid_crct = correction.into();
            let (_, reassigned) = gcfb_v234_with_reassignment(&signal, parameters).unwrap();
            let valid_coordinates = reassigned
                .validity_mask
                .iter()
                .filter(|&&valid| valid)
                .count();
            assert!(
                valid_coordinates > 0,
                "{correction} correction produced no valid reassignment coordinates"
            );
            assert!(
                reassigned.retained_energy() > 0.0,
                "{correction} correction retained no reassigned energy"
            );
            for ch in 0..reassigned.t_hat.nrows() {
                for sample in 0..reassigned.t_hat.ncols() {
                    if reassigned.validity_mask[[ch, sample]] {
                        assert_relative_eq!(
                            reassigned.t_hat[[ch, sample]],
                            expected,
                            epsilon = 2e-10
                        );
                    }
                }
            }
            assert_eq!(reassigned.mode, ReassignmentMode::Frame);
            assert!(reassigned.analysis_fft_len.is_power_of_two());
        }
    }

    #[test]
    fn tone_coordinates_are_stable_at_low_middle_and_high_frequencies() {
        let base_parameters = GcParam {
            num_ch: 12,
            f_range: [180.0, 2200.0],
            ..static_parameters()
        };
        for correction in ["No", "ELC"] {
            let mut parameters = base_parameters.clone();
            parameters.out_mid_crct = correction.into();
            for frequency in [275.0, 713.0, 1675.0] {
                let signal: Vec<f64> = (0..2048)
                    .map(|sample| (2.0 * PI * frequency * sample as f64 / 8000.0).cos())
                    .collect();
                let (_, reassigned) =
                    gcfb_v234_with_reassignment(&signal, parameters.clone()).unwrap();
                let ch = reassigned
                    .frequency_axis_hz
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - frequency).abs().total_cmp(&(*b - frequency).abs())
                    })
                    .unwrap()
                    .0;
                let estimates: Vec<f64> = (768..1536)
                    .filter(|&sample| reassigned.validity_mask[[ch, sample]])
                    .map(|sample| reassigned.f_hat[[ch, sample]])
                    .collect();
                assert!(estimates.len() > 500);
                let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
                let maximum = estimates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let minimum = estimates.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(
                    (mean - frequency).abs() < 1.0,
                    "{frequency} Hz tone with {correction} correction estimated at {mean} Hz"
                );
                assert!(
                    maximum - minimum < 3.0,
                    "{frequency} Hz tone with {correction} correction oscillated over {} Hz",
                    maximum - minimum
                );
            }
        }
    }

    #[test]
    fn analysis_dft_covers_the_correction_and_pgc_cascade() {
        let signal = vec![0.0; 128];
        for correction_name in ["No", "ELC", "EarDrum"] {
            let mut parameters = static_parameters();
            parameters.out_mid_crct = correction_name.into();
            let output = gcfb_v234(&signal, parameters).unwrap();
            let operator = conditioned_operator_outputs(&signal, &output).unwrap();
            let correction = correction_fir(&output.gc_param).unwrap();
            let maximum_pgc_len = (0..output.gc_param.num_ch)
                .map(|ch| real_pgc(&output.gc_param, &output, ch).unwrap().len())
                .max()
                .unwrap();
            let convolution_len = signal.len() + correction.len() + maximum_pgc_len - 2;
            assert_eq!(
                operator.analysis_fft_len,
                analysis_fft_len(signal.len(), correction.len(), maximum_pgc_len).unwrap()
            );
            assert!(operator.analysis_fft_len >= 2 * convolution_len);
            assert!(operator.analysis_fft_len.is_power_of_two());
        }
    }

    #[test]
    fn frame_coordinates_are_invariant_to_input_scaling() {
        let low: Vec<f64> = (0..320)
            .map(|sample| 0.05 * (2.0 * PI * 700.0 * sample as f64 / 8000.0).sin())
            .collect();
        let high: Vec<f64> = low.iter().map(|sample| sample * 10.0).collect();
        let (_, low_result) =
            gcfb_v234_with_reassignment(&low, compact_frame_parameters()).unwrap();
        let (_, high_result) =
            gcfb_v234_with_reassignment(&high, compact_frame_parameters()).unwrap();
        assert_eq!(low_result.validity_mask, high_result.validity_mask);
        for ((a, b), valid) in low_result
            .t_hat
            .iter()
            .zip(high_result.t_hat.iter())
            .zip(low_result.validity_mask.iter())
        {
            if *valid {
                assert_relative_eq!(a, b, epsilon = 2e-12);
            }
        }
        for ((a, b), valid) in low_result
            .f_hat
            .iter()
            .zip(high_result.f_hat.iter())
            .zip(low_result.validity_mask.iter())
        {
            if *valid {
                assert_relative_eq!(a, b, epsilon = 2e-8);
            }
        }
        assert!(high_result.source_energy > low_result.source_energy);
    }

    #[test]
    fn conditioned_sample_passes_match_an_explicit_kernel() {
        let mut parameters = compact_frame_parameters();
        parameters.num_ch = 2;
        parameters.f_range = [500.0, 1200.0];
        parameters.dyn_hpaf.str_prc = "sample-base".into();
        parameters.num_update_asym_cmp = 2;
        let signal: Vec<f64> = (0..10)
            .map(|sample| 0.1 * (2.0 * PI * 650.0 * sample as f64 / 8000.0).sin())
            .collect();
        let output = gcfb_v234(&signal, parameters).unwrap();
        let operator = conditioned_operator_outputs(&signal, &output).unwrap();
        let ch = 0;
        let pgc_real = real_pgc(&output.gc_param, &output, ch).unwrap();
        let pgc = analytic_pgc(&pgc_real, operator.analysis_fft_len);
        let inputs = prepare_input_spectra(&signal, output.gc_param.fs, operator.analysis_fft_len);
        let mut padded_inputs = inputs.values.clone();
        for values in &mut padded_inputs {
            dsp::fft(values, true);
        }
        let target_sample = signal.len() - 1;
        let mut kernel = vec![Complex64::new(0.0, 0.0); operator.analysis_fft_len];
        for (input_sample, kernel_value) in kernel.iter_mut().enumerate() {
            let mut state = ComplexCascadeState::default();
            let mut coefficients = make_asym_cmp_filters_v2(
                output.gc_param.fs,
                &[output.gc_resp.fr2[[ch, 0]]],
                &[output.gc_resp.b2_val[ch]],
                &[output.gc_resp.c2_val[ch]],
            )
            .unwrap();
            let mut value = Complex64::new(0.0, 0.0);
            for sample in 0..=target_sample {
                if sample % output.gc_param.num_update_asym_cmp == 0 {
                    coefficients = make_asym_cmp_filters_v2(
                        output.gc_param.fs,
                        &[output.gc_resp.fr2[[ch, sample]]],
                        &[output.gc_resp.b2_val[ch]],
                        &[output.gc_resp.c2_val[ch]],
                    )
                    .unwrap();
                }
                let lag =
                    (operator.analysis_fft_len + sample - input_sample) % operator.analysis_fft_len;
                value = state.process(pgc.impulse[lag], &coefficients, 0);
            }
            *kernel_value = value;
        }
        let explicit: [Complex64; 3] = std::array::from_fn(|pass| {
            kernel
                .iter()
                .zip(&padded_inputs[pass])
                .map(|(kernel, input)| kernel * input)
                .sum()
        });
        assert_relative_eq!(
            explicit[0].re,
            operator.coefficient[[ch, target_sample]].re,
            epsilon = 2e-10
        );
        assert_relative_eq!(
            explicit[0].im,
            operator.coefficient[[ch, target_sample]].im,
            epsilon = 2e-10
        );
        assert_relative_eq!(
            explicit[1].re,
            operator.time_weighted[[ch, target_sample]].re,
            epsilon = 2e-10
        );
        assert_relative_eq!(
            explicit[1].im,
            operator.time_weighted[[ch, target_sample]].im,
            epsilon = 2e-10
        );
        assert_relative_eq!(
            explicit[2].re,
            operator.derivative[[ch, target_sample]].re,
            epsilon = 2e-8
        );
        assert_relative_eq!(
            explicit[2].im,
            operator.derivative[[ch, target_sample]].im,
            epsilon = 2e-8
        );
    }

    #[test]
    fn analytic_energy_is_fully_accounted_for() {
        let signal: Vec<f64> = (0..320)
            .map(|sample| (2.0 * PI * 700.0 * sample as f64 / 8000.0).sin())
            .collect();
        for parameters in [static_parameters(), compact_frame_parameters()] {
            let (_, reassigned) = gcfb_v234_with_reassignment(&signal, parameters).unwrap();
            assert_relative_eq!(
                reassigned.retained_energy()
                    + reassigned.floor_discarded_energy
                    + reassigned.boundary_discarded_energy,
                reassigned.source_energy,
                epsilon = reassigned.source_energy.max(1.0) * 2e-12
            );
        }
    }

    #[test]
    fn invalid_reassignment_floor_is_rejected() {
        let signal = vec![0.0, 1.0, 0.0, 0.0];
        let parameters = GcParam {
            num_ch: 2,
            f_range: [300.0, 1200.0],
            ..static_parameters()
        };
        let output = gcfb_v234(&signal, parameters).unwrap();
        let config = ReassignmentConfig {
            coefficient_floor: 0.0,
        };
        assert!(reassign_gcfb_v234_with_config(&signal, &output, &config).is_err());
    }

    #[test]
    fn gain_reference_variants_are_supported() {
        let signal = vec![1.0; 64];
        let parameters = GcParam {
            gain_ref: GainReference::Db(50.0),
            ..compact_frame_parameters()
        };
        let (_, result) = gcfb_v234_with_reassignment(&signal, parameters).unwrap();
        assert!(result.source_energy.is_finite());
    }

    #[test]
    fn phase_deposition_matches_the_average_frequency_rotation_and_linear_weights() {
        let time_axis = [0.0, 0.0025];
        let (frequency_axis_erb, _) = utils::freq2erb(&[100.0, 300.0, 500.0]);
        let mut accounting = TransportAccounting::new((3, 2), true);

        // The destination lies halfway between the two time bins. In radians,
        // the expected transport is
        //   pi * (100 Hz + 300 Hz) * (0.00125 s - 0 s) = pi / 2.
        // Scaling (3 + 4i) to energy 9 gives 1.8 + 2.4i, and a +pi/2
        // rotation therefore gives -2.4 + 1.8i before interpolation.
        deposit_energy(
            &mut accounting,
            9.0,
            true,
            0.00125,
            300.0,
            0,
            0,
            0.0,
            100.0,
            Complex64::new(3.0, 4.0),
            &time_axis,
            frequency_axis_erb.as_slice().unwrap(),
        )
        .unwrap();

        assert_relative_eq!(accounting.energy.source, 9.0, epsilon = 1e-14);
        assert_relative_eq!(accounting.energy.map[[1, 0]], 4.5, epsilon = 1e-14);
        assert_relative_eq!(accounting.energy.map[[1, 1]], 4.5, epsilon = 1e-14);
        assert_relative_eq!(accounting.energy.map.sum(), 9.0, epsilon = 1e-14);

        let phase = accounting.phase.unwrap();
        for time_bin in 0..2 {
            assert_relative_eq!(phase.complex_map[[1, time_bin]].re, -1.2, epsilon = 2e-15);
            assert_relative_eq!(phase.complex_map[[1, time_bin]].im, 0.9, epsilon = 2e-15);
            assert_relative_eq!(
                phase.contribution_magnitude_map[[1, time_bin]],
                1.5,
                epsilon = 2e-15
            );
        }
        assert_relative_eq!(phase.unreassigned_energy_map[[0, 0]], 9.0, epsilon = 1e-14);
        assert_relative_eq!(phase.coherence_map()[[1, 0]], 1.0, epsilon = 2e-15);
        assert_relative_eq!(phase.coherence_map()[[1, 1]], 1.0, epsilon = 2e-15);
    }

    #[test]
    fn phase_transport_preserves_energy_results_and_matches_retained_source_energy() {
        let signal: Vec<f64> = (0..640)
            .map(|sample| (2.0 * PI * 713.0 * sample as f64 / 8000.0).cos())
            .collect();
        for parameters in [static_parameters(), compact_frame_parameters()] {
            let output = gcfb_v234(&signal, parameters).unwrap();
            let energy = reassign_gcfb_v234(&signal, &output).unwrap();
            let phase = phase_reassign_gcfb_v234(&signal, &output).unwrap();
            assert_eq!(energy.energy_map, phase.reassignment.energy_map);
            assert_eq!(energy.t_hat, phase.reassignment.t_hat);
            assert_eq!(energy.f_hat, phase.reassignment.f_hat);
            assert_eq!(energy.validity_mask, phase.reassignment.validity_mask);
            assert_eq!(
                phase.unreassigned_energy_map.dim(),
                phase.reassignment.energy_map.dim()
            );
            assert_relative_eq!(
                phase.unreassigned_energy_map.sum(),
                phase.reassignment.retained_energy(),
                epsilon = phase.reassignment.retained_energy().max(1.0) * 2e-12
            );
            phase.sparsity_comparison().unwrap();
            assert!(
                phase
                    .phase_coherence_map
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            );
        }
    }

    #[test]
    fn static_phase_map_scales_linearly_and_energy_quadratically() {
        let signal: Vec<f64> = (0..768)
            .map(|sample| 0.1 * (2.0 * PI * 700.0 * sample as f64 / 8000.0).cos())
            .collect();
        let scaled: Vec<f64> = signal.iter().map(|value| 3.0 * value).collect();
        let parameters = GcParam {
            num_ch: 2,
            f_range: [650.0, 750.0],
            ..static_parameters()
        };
        let (_, low) = gcfb_v234_with_phase_reassignment(&signal, parameters.clone()).unwrap();
        let (_, high) = gcfb_v234_with_phase_reassignment(&scaled, parameters).unwrap();
        assert_eq!(
            low.reassignment.validity_mask,
            high.reassignment.validity_mask
        );
        for (low, high) in low.complex_map.iter().zip(&high.complex_map) {
            assert_relative_eq!(high.re, 3.0 * low.re, epsilon = 2e-9, max_relative = 2e-9);
            assert_relative_eq!(high.im, 3.0 * low.im, epsilon = 2e-9, max_relative = 2e-9);
        }
        for (low, high) in low
            .reassignment
            .energy_map
            .iter()
            .zip(&high.reassignment.energy_map)
        {
            assert_relative_eq!(*high, 9.0 * *low, epsilon = 2e-8, max_relative = 2e-9);
        }
        let dominant = (0..low.reassignment.energy_map.nrows())
            .flat_map(|ch| (256..512).map(move |time| (ch, time)))
            .max_by(|&left, &right| {
                low.reassignment.energy_map[left].total_cmp(&low.reassignment.energy_map[right])
            })
            .unwrap();
        let dominant_energy = low.reassignment.energy_map[dominant];
        let dominant_support: Vec<(usize, usize)> = (0..low.reassignment.energy_map.nrows())
            .flat_map(|ch| (256..512).map(move |time| (ch, time)))
            .filter(|&index| low.reassignment.energy_map[index] > 0.1 * dominant_energy)
            .collect();
        let support_energy: f64 = dominant_support
            .iter()
            .map(|&index| low.reassignment.energy_map[index])
            .sum();
        let energy_weighted_coherence: f64 = dominant_support
            .iter()
            .map(|&index| low.reassignment.energy_map[index] * low.phase_coherence_map[index])
            .sum::<f64>()
            / support_energy;
        assert!(
            dominant_support.len() > 100 && energy_weighted_coherence > 0.8,
            "tone dominant support had {} bins and energy-weighted coherence {energy_weighted_coherence}; dominant-bin coherence was {} at {dominant:?}",
            dominant_support.len(),
            low.phase_coherence_map[dominant]
        );
    }

    #[test]
    fn linear_chirp_coordinates_follow_frequency_at_reassigned_time() {
        let samples = 2048;
        let duration = samples as f64 / 8000.0;
        let start_hz = 690.0;
        let chirp_rate = 20.0 / duration;
        let signal: Vec<f64> = (0..samples)
            .map(|sample| {
                let time = sample as f64 / 8000.0;
                (2.0 * PI * (start_hz * time + 0.5 * chirp_rate * time.powi(2))).cos()
            })
            .collect();
        let (_, phase) = gcfb_v234_with_phase_reassignment(
            &signal,
            GcParam {
                num_ch: 2,
                f_range: [680.0, 720.0],
                ..static_parameters()
            },
        )
        .unwrap();
        let mut errors = Vec::new();
        for ch in 0..phase.reassignment.t_hat.nrows() {
            let channel_maximum = phase
                .reassignment
                .energy_map
                .row(ch)
                .iter()
                .copied()
                .fold(0.0, f64::max);
            for sample in 0..samples {
                let time = phase.reassignment.t_hat[[ch, sample]];
                if phase.reassignment.validity_mask[[ch, sample]]
                    && (0.04..duration - 0.04).contains(&time)
                    && phase.reassignment.f_hat[[ch, sample]] > 0.0
                    && channel_maximum > 0.0
                {
                    errors.push(
                        (phase.reassignment.f_hat[[ch, sample]] - (start_hz + chirp_rate * time))
                            .abs(),
                    );
                }
            }
        }
        errors.sort_by(f64::total_cmp);
        assert!(errors.len() > 1000);
        assert!(errors[errors.len() / 2] < 2.0);
        let dominant = (0..phase.reassignment.energy_map.nrows())
            .flat_map(|ch| (400..samples - 400).map(move |time| (ch, time)))
            .max_by(|&left, &right| {
                phase.reassignment.energy_map[left].total_cmp(&phase.reassignment.energy_map[right])
            })
            .unwrap();
        let dominant_energy = phase.reassignment.energy_map[dominant];
        let dominant_support: Vec<(usize, usize)> = (0..phase.reassignment.energy_map.nrows())
            .flat_map(|ch| (400..samples - 400).map(move |time| (ch, time)))
            .filter(|&index| phase.reassignment.energy_map[index] > 0.1 * dominant_energy)
            .collect();
        let support_energy: f64 = dominant_support
            .iter()
            .map(|&index| phase.reassignment.energy_map[index])
            .sum();
        let energy_weighted_coherence: f64 = dominant_support
            .iter()
            .map(|&index| phase.reassignment.energy_map[index] * phase.phase_coherence_map[index])
            .sum::<f64>()
            / support_energy;
        assert!(
            dominant_support.len() > 500 && energy_weighted_coherence > 0.9,
            "chirp dominant support had {} bins and energy-weighted coherence {energy_weighted_coherence}; dominant-bin coherence was {} at {dominant:?}",
            dominant_support.len(),
            phase.phase_coherence_map[dominant]
        );
    }

    #[test]
    fn sparsity_metrics_match_single_bin_and_uniform_maps() {
        let single = SparsityMetrics::from_energy_map(&array![[4.0, 0.0], [0.0, 0.0]]).unwrap();
        assert_relative_eq!(single.shannon_entropy, 0.0, epsilon = 1e-15);
        assert_relative_eq!(single.effective_bins, 1.0, epsilon = 1e-15);
        assert_relative_eq!(single.effective_bin_fraction, 0.25, epsilon = 1e-15);

        let uniform = SparsityMetrics::from_energy_map(&array![[1.0, 1.0], [1.0, 1.0]]).unwrap();
        assert_relative_eq!(uniform.shannon_entropy, 4.0_f64.ln(), epsilon = 1e-15);
        assert_relative_eq!(uniform.effective_bins, 4.0, epsilon = 1e-14);
        assert_relative_eq!(uniform.effective_bin_fraction, 1.0, epsilon = 1e-14);
        assert!(SparsityComparison::from_maps(&array![[1.0, 0.0]], &array![[2.0, 0.0]],).is_err());
    }

    #[test]
    fn sparsity_comparison_rejects_different_grids_and_quiet_energy_mismatches() {
        assert!(SparsityComparison::from_maps(&array![[1.0]], &array![[1.0, 0.0]]).is_err());
        assert!(SparsityComparison::from_maps(&array![[0.0]], &array![[5e-11]]).is_err());
        assert!(SparsityComparison::from_maps(&array![[5e-11]], &array![[5e-11]]).is_ok());
    }

    #[test]
    fn deterministic_noise_has_smaller_empirical_reassigned_support() {
        let signal = deterministic_noise(1024);
        let (_, phase) = gcfb_v234_with_phase_reassignment(
            &signal,
            GcParam {
                num_ch: 12,
                f_range: [180.0, 2200.0],
                ..static_parameters()
            },
        )
        .unwrap();
        let comparison = phase.sparsity_comparison().unwrap();
        assert!(
            comparison.reassigned.effective_bin_fraction
                < comparison.unreassigned.effective_bin_fraction,
            "reassigned fraction {} did not beat source fraction {}",
            comparison.reassigned.effective_bin_fraction,
            comparison.unreassigned.effective_bin_fraction
        );
    }

    fn composite_erb_at_1khz(scale: f64) -> f64 {
        let parameters = scale_bandwidths(GcParam::default(), scale);
        let level = parameters.level_db_scgcfb;
        let frat = parameters.frat[0][0] + parameters.frat[1][0] * level;
        let response = cmprs_gc_frsp(
            &[1000.0],
            parameters.fs,
            parameters.n,
            &[parameters.b1[0]],
            &[parameters.c1[0]],
            &[frat],
            &[parameters.b2[0][0]],
            &[parameters.c2[0][0]],
            4096,
        )
        .unwrap();
        let bin_width = response.freq[1] - response.freq[0];
        response
            .cgc_nrm_frsp
            .row(0)
            .iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
            * bin_width
    }

    #[test]
    fn consensus_defaults_span_typical_normal_hearing_bandwidth_variation() {
        let config = BandwidthConsensusConfig::default();
        assert_eq!(config.scales, vec![0.8, 1.0, 1.2]);
        assert_eq!(validate_consensus_config(&config).unwrap(), 1);

        let baseline_erb = composite_erb_at_1khz(config.scales[1]);
        let narrow_ratio = composite_erb_at_1khz(config.scales[0]) / baseline_erb;
        let wide_ratio = composite_erb_at_1khz(config.scales[2]) / baseline_erb;
        assert!(
            (0.80..=0.82).contains(&narrow_ratio),
            "narrow default produced an ERB ratio of {narrow_ratio}"
        );
        assert!(
            (1.17..=1.20).contains(&wide_ratio),
            "wide default produced an ERB ratio of {wide_ratio}"
        );
    }

    #[test]
    fn consensus_validates_configuration_and_reuses_the_baseline_grid() {
        for scales in [
            vec![1.0],
            vec![0.75, 1.0, 1.0],
            vec![0.75, 1.5],
            vec![0.75, 1.0, f64::NAN],
        ] {
            let config = BandwidthConsensusConfig {
                scales,
                ..BandwidthConsensusConfig::default()
            };
            assert!(validate_consensus_config(&config).is_err());
        }
        for (support, agreement) in [(0.0, 1.0), (1.0, 1.0), (1e-6, 0.0), (1e-6, 1.1)] {
            let config = BandwidthConsensusConfig {
                relative_support_floor: support,
                required_agreement: agreement,
                ..BandwidthConsensusConfig::default()
            };
            assert!(validate_consensus_config(&config).is_err());
        }

        let sample_rate = 48_000.0;
        let samples = 2048;
        let tone_hz = 1000.0;
        let click_sample = 768;
        let mut signal: Vec<f64> = (0..samples)
            .map(|sample| 0.2 * (2.0 * PI * tone_hz * sample as f64 / sample_rate).cos())
            .collect();
        signal[click_sample] += 1.0;
        let parameters = GcParam {
            fs: sample_rate,
            num_ch: 8,
            f_range: [200.0, 6000.0],
            ..static_parameters()
        };
        let config = BandwidthConsensusConfig::default();
        let ordinary_output = gcfb_v234(&signal, parameters.clone()).unwrap();
        let (baseline_output, consensus) =
            gcfb_v234_with_bandwidth_consensus(&signal, parameters, &config).unwrap();
        assert_eq!(baseline_output.dcgc_out, ordinary_output.dcgc_out);
        assert_eq!(baseline_output.scgc_smpl, ordinary_output.scgc_smpl);
        assert_eq!(consensus.baseline_index, 1);
        assert_eq!(consensus.analyses.len(), 3);
        assert_eq!(
            consensus.analyses[consensus.baseline_index]
                .reassignment
                .frequency_axis_hz,
            baseline_output.gc_param.fr1
        );
        let direct_baseline = phase_reassign_gcfb_v234_with_config(
            &signal,
            &baseline_output,
            &config.reassignment_config,
        )
        .unwrap();
        assert_eq!(
            consensus.analyses[consensus.baseline_index]
                .reassignment
                .energy_map,
            direct_baseline.reassignment.energy_map
        );
        assert_eq!(
            consensus.analyses[consensus.baseline_index].complex_map,
            direct_baseline.complex_map
        );
        assert!(
            consensus
                .agreement_map
                .iter()
                .all(|agreement| (0.0..=1.0).contains(agreement))
        );
        let tone_channel = baseline_output
            .gc_param
            .fr1
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left - tone_hz).abs().total_cmp(&(*right - tone_hz).abs())
            })
            .unwrap()
            .0;
        let tone_agreement = consensus
            .agreement_map
            .row(tone_channel)
            .iter()
            .skip(samples / 3)
            .take(samples / 3)
            .copied()
            .fold(0.0, f64::max);
        let click_agreement = consensus
            .agreement_map
            .column(click_sample)
            .iter()
            .copied()
            .fold(0.0, f64::max);
        assert_relative_eq!(tone_agreement, 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(click_agreement, 1.0, epsilon = f64::EPSILON);
        let consensus_bins = consensus
            .consensus_mask
            .iter()
            .filter(|&&value| value)
            .count();
        assert!(consensus_bins > 0);
        assert!(consensus_bins < consensus.consensus_mask.len());
    }
}
