//! Frame-based GCFB v2.34 with hearing-loss characteristics.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis, s};
use num_complex::Complex64;

use super::{
    gammachirp::{self, Carrier, Normalization},
    utils::{self, FrequencyScale},
};
use crate::{Error, Result, dsp};

pub use super::common::{
    AcfCoef, AcfStatus, AsymCmpResponse, CgcResponse, ControlMode, LvlEst, SmoothSpecParam,
    acfilterbank, asym_cmp_frsp_v2, cal_smooth_spec, cmprs_gc_frsp, fp2_to_fr1, fr1_to_fp2,
    make_asym_cmp_filters_v2,
};

#[derive(Clone, Debug)]
pub struct DynHpaf {
    pub str_prc: String,
    pub t_frame: f64,
    pub t_shift: f64,
    pub len_frame: usize,
    pub len_shift: usize,
    pub fs: f64,
    pub name_win: String,
    pub val_win: Array1<f64>,
}

impl Default for DynHpaf {
    fn default() -> Self {
        Self {
            str_prc: "frame-base".into(),
            t_frame: 0.001,
            t_shift: 0.0005,
            len_frame: 0,
            len_shift: 0,
            fs: 0.0,
            name_win: "hanning".into(),
            val_win: Array1::zeros(0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HLoss {
    pub f_audgram_list: Array1<f64>,
    pub type_name: String,
    pub hearing_level_db: Array1<f64>,
    pub pin_loss_db_act: Array1<f64>,
    pub pin_loss_db_act_init: Array1<f64>,
    pub pin_loss_db_pas: Array1<f64>,
    pub compression_health: Array1<f64>,
    pub compression_health_initval: Array1<f64>,
    pub af_gain_cmpnst_db: Array1<f64>,
    pub hl_val_pin_cochlea_db: Array1<f64>,
    pub fb_fr1: Array1<f64>,
    pub fb_hearing_level_db: Array1<f64>,
    pub fb_pin_cochlea_db: Array1<f64>,
    pub fb_pin_loss_db_act: Array1<f64>,
    pub fb_pin_loss_db_pas: Array1<f64>,
    pub fb_compression_health: Array1<f64>,
    pub fb_af_gain_cmpnst_db: Array1<f64>,
}

impl Default for HLoss {
    fn default() -> Self {
        Self {
            f_audgram_list: Array1::from(vec![125., 250., 500., 1000., 2000., 4000., 8000.]),
            type_name: "NH_NormalHearing".into(),
            hearing_level_db: Array1::zeros(7),
            pin_loss_db_act: Array1::zeros(7),
            pin_loss_db_act_init: Array1::zeros(7),
            pin_loss_db_pas: Array1::zeros(7),
            compression_health: Array1::ones(7),
            compression_health_initval: Array1::ones(7),
            af_gain_cmpnst_db: Array1::zeros(7),
            hl_val_pin_cochlea_db: Array1::zeros(7),
            fb_fr1: Array1::zeros(0),
            fb_hearing_level_db: Array1::zeros(0),
            fb_pin_cochlea_db: Array1::zeros(0),
            fb_pin_loss_db_act: Array1::zeros(0),
            fb_pin_loss_db_pas: Array1::zeros(0),
            fb_compression_health: Array1::zeros(0),
            fb_af_gain_cmpnst_db: Array1::zeros(0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GainReference {
    Db(f64),
    NormalizeIoFunction,
}

#[derive(Clone, Debug)]
pub struct GcParam {
    pub fs: f64,
    pub num_ch: usize,
    /// Requested channel-frequency interval `[low, high]`, satisfying
    /// `0 < low < high < fs / 2`.
    pub f_range: [f64; 2],
    pub out_mid_crct: String,
    pub n: f64,
    pub b1: [f64; 2],
    pub c1: [f64; 2],
    pub frat: [[f64; 2]; 2],
    pub b2: [[f64; 2]; 2],
    pub c2: [[f64; 2]; 2],
    pub ctrl: ControlMode,
    pub gain_cmpnst_db: f64,
    pub gain_ref: GainReference,
    pub level_db_scgcfb: f64,
    pub lvl_est: LvlEst,
    pub num_update_asym_cmp: usize,
    pub dyn_hpaf: DynHpaf,
    pub hloss_type: String,
    /// Cochlear compression health, where `0.0` is fully impaired and `1.0`
    /// is healthy. Values must be finite and in `0.0..=1.0`.
    pub hloss_compression_health: Option<f64>,
    pub hloss_hearing_level_db: Option<Array1<f64>>,
    pub hloss: HLoss,
    pub fr1: Array1<f64>,
    pub meddis_hc_level_rms0db_spldb: f64,
}

impl Default for GcParam {
    fn default() -> Self {
        Self {
            fs: 48000.,
            num_ch: 100,
            f_range: [100., 6000.],
            out_mid_crct: "ELC".into(),
            n: 4.,
            b1: [1.81, 0.],
            c1: [-2.96, 0.],
            frat: [[0.466, 0.], [0.0109, 0.]],
            b2: [[2.17, 0.], [0., 0.]],
            c2: [[2.2, 0.], [0., 0.]],
            ctrl: ControlMode::Dynamic,
            gain_cmpnst_db: -1.,
            gain_ref: GainReference::NormalizeIoFunction,
            level_db_scgcfb: 50.,
            lvl_est: LvlEst::default(),
            num_update_asym_cmp: 1,
            dyn_hpaf: DynHpaf::default(),
            hloss_type: "NH".into(),
            hloss_compression_health: None,
            hloss_hearing_level_db: None,
            hloss: HLoss::default(),
            fr1: Array1::zeros(0),
            meddis_hc_level_rms0db_spldb: 30.,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GcResp {
    pub fr1: Array1<f64>,
    pub fr2: Array2<f64>,
    pub erb_space1: f64,
    pub ef: Array1<f64>,
    pub b1_val: Array1<f64>,
    pub c1_val: Array1<f64>,
    pub fp1: Array1<f64>,
    pub fp2: Array1<f64>,
    pub b2_val: Array1<f64>,
    pub c2_val: Array1<f64>,
    pub frat_val: Array2<f64>,
    pub frat0_val: Array1<f64>,
    pub frat1_val: Array1<f64>,
    pub pc_hpaf: Array1<f64>,
    pub frat0_pc: Array1<f64>,
    pub lvl_db: Array2<f64>,
    pub lvl_db_frame: Array2<f64>,
    pub pgc_frame: Array2<f64>,
    pub scgc_frame: Array2<f64>,
    pub asym_func_gain: Array2<f64>,
    pub gain_factor: Array1<f64>,
    pub cgc_ref: Option<CgcResponse>,
}

#[derive(Clone, Debug)]
pub struct GcfbOutput {
    pub dcgc_out: Array2<f64>,
    pub scgc_smpl: Array2<f64>,
    pub gc_param: GcParam,
    pub gc_resp: GcResp,
}

const MINIMUM_PEAK_GRID_FFT_LEN: usize = 65_536;
const MAXIMUM_PEAK_LOCK_ITERATIONS: usize = 128;

fn peak_lock_tolerance_hz(sample_rate: f64) -> f64 {
    4096.0 * f64::EPSILON * sample_rate.max(1.0)
}

fn peak_search_tolerance_hz(sample_rate: f64) -> f64 {
    32.0 * f64::EPSILON * sample_rate.max(1.0)
}

#[derive(Clone, Debug)]
struct PassiveSpectrum {
    impulse: Vec<f64>,
    power: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct CascadePeak {
    frequency_hz: f64,
    log_power: f64,
}

#[derive(Clone, Debug)]
pub(super) struct RealizedCascadePeaks {
    pub(super) frequency_hz: Array1<f64>,
    pub(super) value: Array1<f64>,
    pub(super) normalization: Array1<f64>,
}

pub(super) fn realized_cascade_peaks(
    param: &GcParam,
    response: &GcResp,
    passive_impulses: &[Vec<f64>],
    ratios: &Array1<f64>,
    b2: &Array1<f64>,
    c2: &Array1<f64>,
) -> Result<RealizedCascadePeaks> {
    let channels = param.num_ch;
    if passive_impulses.len() != channels
        || ratios.len() != channels
        || b2.len() != channels
        || c2.len() != channels
        || response.fr1.len() != channels
        || response.fp1.len() != channels
        || response.b1_val.len() != channels
        || response.c1_val.len() != channels
    {
        return Err(Error::InvalidParameter(
            "realized cascade normalization requires one passive impulse and parameter set per channel"
                .into(),
        ));
    }
    let fft_len = MINIMUM_PEAK_GRID_FFT_LEN;
    let mut frequency_hz = Array1::zeros(channels);
    let mut value = Array1::zeros(channels);
    for ch in 0..channels {
        let center = ratios[ch] * response.fp1[ch];
        let coefficients = make_asym_cmp_filters_v2(param.fs, &[center], &[b2[ch]], &[c2[ch]])?;
        let analytic_peak = fr1_to_fp2(
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            b2[ch],
            c2[ch],
            ratios[ch],
            response.fr1[ch],
        )?
        .0;
        let peak = continuous_peak_from_impulse(
            &passive_impulses[ch],
            &coefficients,
            analytic_peak,
            None,
            param.fs,
            fft_len,
        )?;
        let peak_value = (0.5 * peak.log_power).exp();
        if !peak_value.is_finite() || peak_value <= 0.0 {
            return Err(Error::Numerical(
                "implemented compressive-gammachirp peak is not finite and positive".into(),
            ));
        }
        frequency_hz[ch] = peak.frequency_hz;
        value[ch] = peak_value;
    }
    let normalization = value.mapv(|peak| 1.0 / peak);
    Ok(RealizedCascadePeaks {
        frequency_hz,
        value,
        normalization,
    })
}

/// Shared FFT bracketing grid and baseline pGC spectra for one
/// bandwidth-consensus ensemble.
#[derive(Clone, Debug)]
pub(crate) struct BandwidthPeakGrid {
    sample_rate: f64,
    fft_len: usize,
    reference_carriers: Array1<f64>,
    reference_fp1: Array1<f64>,
    reference_b1: Array1<f64>,
    reference_c1: Array1<f64>,
    reference_b2: Array1<f64>,
    reference_c2: Array1<f64>,
    reference_passive: Arc<Vec<PassiveSpectrum>>,
    order: f64,
}

impl BandwidthPeakGrid {
    #[cfg(test)]
    pub(crate) fn fft_len(&self) -> usize {
        self.fft_len
    }

    pub(crate) fn nominal_peak_frequencies_hz(&self, ratios: &Array1<f64>) -> Result<Array1<f64>> {
        self.reference_peak_frequencies_at_ratios(ratios, None)
    }

    /// Measure the unscaled reference cascade at the supplied ratio vector.
    ///
    /// For sample-dynamic consensus, callers deliberately supply each scale's
    /// independently realized ratios. This is a conditional reference curve,
    /// not the realized baseline scale's ratio history at the same sample.
    fn reference_peak_frequencies_at_ratios(
        &self,
        ratios: &Array1<f64>,
        previous: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>> {
        if ratios.len() != self.reference_carriers.len()
            || previous.is_some_and(|bins| bins.len() != ratios.len())
        {
            return Err(Error::InvalidParameter(
                "bandwidth peak search received a mismatched ratio vector".into(),
            ));
        }
        let mut peaks = Array1::zeros(ratios.len());
        for ch in 0..ratios.len() {
            let center = ratios[ch] * self.reference_fp1[ch];
            validate_lock_frequency(center, self.sample_rate, "baseline HP-AF center")?;
            let coefficients = make_asym_cmp_filters_v2(
                self.sample_rate,
                &[center],
                &[self.reference_b2[ch]],
                &[self.reference_c2[ch]],
            )?;
            let analytic_peak = fr1_to_fp2(
                self.order,
                self.reference_b1[ch],
                self.reference_c1[ch],
                self.reference_b2[ch],
                self.reference_c2[ch],
                ratios[ch],
                self.reference_carriers[ch],
            )?
            .0;
            peaks[ch] = continuous_peak_from_spectrum(
                &self.reference_passive[ch],
                &coefficients,
                analytic_peak,
                previous.map(|values| values[ch]),
                self.sample_rate,
                self.fft_len,
            )?
            .frequency_hz;
        }
        Ok(peaks)
    }
}

/// Internal state that removes implemented-cascade peak drift introduced by a
/// bandwidth scale while retaining that scale's independent level controller.
#[derive(Clone, Debug)]
pub(crate) struct BandwidthPeakLock {
    grid: Arc<BandwidthPeakGrid>,
    scaled_passive: Arc<Vec<PassiveSpectrum>>,
    scaled_carriers: Array1<f64>,
    scaled_fp1: Array1<f64>,
    scaled_b1: Array1<f64>,
    scaled_c1: Array1<f64>,
    scaled_b2: Array1<f64>,
    scaled_c2: Array1<f64>,
    previous_centers: Array1<f64>,
    previous_reference_peaks_hz: Array1<f64>,
    previous_scaled_peaks_hz: Array1<f64>,
}

impl BandwidthPeakLock {
    /// Retune the scaled HP-AF centers to the unscaled reference peaks
    /// evaluated at `ratios`.
    ///
    /// The ratios belong to the scale being processed. In sample-dynamic mode
    /// they need not equal the simultaneously realized baseline ratios.
    pub(crate) fn centers_for_reference_peaks_at_ratios(
        &mut self,
        ratios: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        if ratios.len() != self.previous_centers.len() {
            return Err(Error::InvalidParameter(
                "bandwidth peak lock received a mismatched ratio vector".into(),
            ));
        }
        let target_peaks = self.grid.reference_peak_frequencies_at_ratios(
            ratios,
            Some(&self.previous_reference_peaks_hz),
        )?;
        let mut centers = Array1::zeros(ratios.len());
        let mut actual_peaks = Array1::zeros(ratios.len());
        for ch in 0..ratios.len() {
            let analytic_target = fr1_to_fp2(
                self.grid.order,
                self.grid.reference_b1[ch],
                self.grid.reference_c1[ch],
                self.grid.reference_b2[ch],
                self.grid.reference_c2[ch],
                ratios[ch],
                self.grid.reference_carriers[ch],
            )?
            .0;
            let analytic_seed = center_for_composite_peak(
                self.grid.order,
                self.scaled_b1[ch],
                self.scaled_c1[ch],
                self.scaled_b2[ch],
                self.scaled_c2[ch],
                self.scaled_carriers[ch],
                self.scaled_fp1[ch],
                analytic_target,
                self.grid.sample_rate,
                self.previous_centers[ch],
            )
            .unwrap_or(self.previous_centers[ch]);
            let (center, peak) = center_for_continuous_peak(
                &self.scaled_passive[ch],
                self.scaled_b2[ch],
                self.scaled_c2[ch],
                self.scaled_b1[ch],
                self.scaled_c1[ch],
                self.scaled_carriers[ch],
                self.grid.order,
                analytic_seed,
                analytic_target,
                target_peaks[ch],
                self.previous_scaled_peaks_hz[ch],
                self.grid.sample_rate,
                self.grid.fft_len,
            )
            .or_else(|error| {
                // The analytic inverse can seed the root solve away from a
                // narrow main lobe. The previous center tracks the target
                // frequency (and was verified against it at preparation), so it
                // is the more reliable seed whenever it differs.
                if analytic_seed == self.previous_centers[ch] {
                    return Err(error);
                }
                center_for_continuous_peak(
                    &self.scaled_passive[ch],
                    self.scaled_b2[ch],
                    self.scaled_c2[ch],
                    self.scaled_b1[ch],
                    self.scaled_c1[ch],
                    self.scaled_carriers[ch],
                    self.grid.order,
                    self.previous_centers[ch],
                    analytic_target,
                    target_peaks[ch],
                    self.previous_scaled_peaks_hz[ch],
                    self.grid.sample_rate,
                    self.grid.fft_len,
                )
                .map_err(|_| error)
            })?;
            centers[ch] = center;
            actual_peaks[ch] = peak;
        }
        self.previous_centers.assign(&centers);
        self.previous_reference_peaks_hz.assign(&target_peaks);
        self.previous_scaled_peaks_hz.assign(&actual_peaks);
        Ok(centers)
    }

    pub(crate) fn current_centers(&self) -> &Array1<f64> {
        &self.previous_centers
    }
}

fn validate_prepared_frequencies(fs: f64, frequencies: &[f64], name: &str) -> Result<()> {
    if frequencies
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0 || *frequency >= fs / 2.0)
    {
        return Err(Error::InvalidParameter(format!(
            "{name} must be finite, positive, and below Nyquist"
        )));
    }
    Ok(())
}

fn validate_user_controlled_parameters(param: &GcParam) -> Result<()> {
    let coefficients_are_finite = param
        .b1
        .iter()
        .chain(&param.c1)
        .chain(param.frat.iter().flatten())
        .chain(param.b2.iter().flatten())
        .chain(param.c2.iter().flatten())
        .all(|value| value.is_finite());
    let level_parameters_are_finite = [
        param.gain_cmpnst_db,
        param.level_db_scgcfb,
        param.meddis_hc_level_rms0db_spldb,
        param.lvl_est.lct_erb,
        param.lvl_est.decay_hl,
        param.lvl_est.b2,
        param.lvl_est.c2,
        param.lvl_est.frat,
        param.lvl_est.rms2spldb,
        param.lvl_est.weight,
        param.lvl_est.ref_db,
        param.lvl_est.pwr[0],
        param.lvl_est.pwr[1],
    ]
    .iter()
    .all(|value| value.is_finite());
    let gain_reference_is_finite = match param.gain_ref {
        GainReference::Db(value) => value.is_finite(),
        GainReference::NormalizeIoFunction => true,
    };
    if !coefficients_are_finite
        || !level_parameters_are_finite
        || !gain_reference_is_finite
        || param.lvl_est.decay_hl <= 0.0
    {
        return Err(Error::InvalidParameter(
            "v2.34 filter coefficients, gain references, and level parameters must be finite, and level decay must be positive"
                .into(),
        ));
    }
    Ok(())
}

pub(super) fn initial_asymmetric_ratio_and_centers(
    param: &GcParam,
    response: &GcResp,
) -> (Array1<f64>, Array1<f64>) {
    let ratios = if param.ctrl == ControlMode::Static {
        let level = param.level_db_scgcfb;
        Array1::from_iter((0..param.num_ch).map(|ch| {
            response.frat0_pc[ch] + response.frat1_val[ch] * (level - response.pc_hpaf[ch])
        }))
    } else {
        Array1::from_elem(param.num_ch, param.lvl_est.frat)
    };
    let centers = &ratios * &response.fp1;
    (ratios, centers)
}

fn peak_path_coefficients(param: &GcParam, response: &GcResp) -> (Array1<f64>, Array1<f64>) {
    match param.ctrl {
        ControlMode::Static => (response.b2_val.clone(), response.c2_val.clone()),
        ControlMode::Dynamic if param.dyn_hpaf.str_prc.contains("sample") => {
            (response.b2_val.clone(), response.c2_val.clone())
        }
        ControlMode::Level | ControlMode::Dynamic => (
            Array1::from_elem(param.num_ch, param.lvl_est.b2),
            &param.hloss.fb_compression_health * param.lvl_est.c2,
        ),
    }
}

pub(crate) fn scale_bandwidths(mut parameters: GcParam, scale: f64) -> GcParam {
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

/// Prepare the common FFT bracketing grid and baseline passive spectra used by
/// every scale in one consensus analysis. If a numerically locked scaled
/// impulse is longer than the current grid, all locks are recomputed on the
/// next power of two. The FFT only locates the intended main lobe; all peak
/// measurements and locks use the continuous DTFT.
pub(crate) fn prepare_bandwidth_peak_grid(
    unprepared_reference: &GcParam,
    scales: &[f64],
    reference_param: &GcParam,
    reference_response: &GcResp,
) -> Result<Arc<BandwidthPeakGrid>> {
    let reference_impulses = prepare_passive_impulses(reference_param, reference_response)?;
    let maximum_reference_len = reference_impulses.iter().map(Vec::len).max().unwrap_or(0);
    let mut fft_len = MINIMUM_PEAK_GRID_FFT_LEN.max(
        maximum_reference_len
            .checked_next_power_of_two()
            .ok_or_else(|| Error::Unsupported("bandwidth peak grid is too large".into()))?,
    );
    for _ in 0..MAXIMUM_PEAK_LOCK_ITERATIONS {
        let grid = Arc::new(build_peak_grid(
            reference_param,
            reference_response,
            &reference_impulses,
            fft_len,
        )?);
        let mut maximum_len = maximum_reference_len;
        for &scale in scales {
            if scale == 1.0 {
                continue;
            }
            let (scaled_param, scaled_response) = set_param_with_preserved_hearing_loss(
                scale_bandwidths(unprepared_reference.clone(), scale),
                Some(&reference_param.hloss),
            )
            .map_err(|error| {
                Error::Unsupported(format!(
                    "bandwidth scale {scale} has no valid continuous peak-lock preparation: {error}"
                ))
            })?;
            let solutions = solve_scaled_passive_channels(&scaled_param, &scaled_response, &grid)?;
            maximum_len = maximum_len.max(
                solutions
                    .iter()
                    .map(|solution| solution.impulse.len())
                    .max()
                    .unwrap_or(0),
            );
        }
        if maximum_len <= fft_len {
            return Ok(grid);
        }
        fft_len = MINIMUM_PEAK_GRID_FFT_LEN.max(
            maximum_len
                .checked_next_power_of_two()
                .ok_or_else(|| Error::Unsupported("bandwidth peak grid is too large".into()))?,
        );
    }
    Err(Error::Unsupported(
        "bandwidth peak grid did not stabilize within 128 iterations".into(),
    ))
}

pub(crate) fn prepare_bandwidth_peak_lock(
    param: &GcParam,
    response: &mut GcResp,
    grid: Arc<BandwidthPeakGrid>,
) -> Result<BandwidthPeakLock> {
    if param.num_ch != grid.reference_carriers.len()
        || param.fs != grid.sample_rate
        || param.fr1.len() != grid.reference_carriers.len()
    {
        return Err(Error::InvalidParameter(
            "bandwidth peak lock requires matching baseline and scaled target grids".into(),
        ));
    }
    let (ratios, _) = initial_asymmetric_ratio_and_centers(param, response);
    let (scaled_b2, scaled_c2) = peak_path_coefficients(param, response);
    let target_peaks = grid.reference_peak_frequencies_at_ratios(&ratios, None)?;
    let solutions = solve_scaled_passive_channels(param, response, &grid)?;
    let scaled_carriers = Array1::from_iter(solutions.iter().map(|solution| solution.carrier));
    let scaled_fp1 = Array1::from_iter(solutions.iter().map(|solution| solution.fp1));
    validate_prepared_frequencies(
        param.fs,
        scaled_carriers.as_slice().unwrap(),
        "peak-locked passive carriers",
    )?;
    validate_prepared_frequencies(
        param.fs,
        scaled_fp1.as_slice().unwrap(),
        "peak-locked passive peaks",
    )?;
    if scaled_carriers
        .windows(2)
        .into_iter()
        .any(|window| window[0] >= window[1])
    {
        return Err(Error::Unsupported(
            "bandwidth scale cannot preserve ordered composite-filter peaks".into(),
        ));
    }
    response.fr1.assign(&scaled_carriers);
    response.fp1.assign(&scaled_fp1);
    let previous_centers = &scaled_fp1 * &ratios;
    let scaled_passive = Arc::new(
        solutions
            .iter()
            .map(|solution| passive_spectrum(&solution.impulse, grid.fft_len))
            .collect::<Result<Vec<_>>>()?,
    );
    let mut actual_peaks = Array1::zeros(param.num_ch);
    for ch in 0..param.num_ch {
        let coefficients = make_asym_cmp_filters_v2(
            param.fs,
            &[previous_centers[ch]],
            &[scaled_b2[ch]],
            &[scaled_c2[ch]],
        )?;
        let analytic_peak = fr1_to_fp2(
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            scaled_b2[ch],
            scaled_c2[ch],
            ratios[ch],
            scaled_carriers[ch],
        )?
        .0;
        actual_peaks[ch] = continuous_peak_from_spectrum(
            &scaled_passive[ch],
            &coefficients,
            analytic_peak,
            None,
            param.fs,
            grid.fft_len,
        )?
        .frequency_hz;
        verify_peak_frequency(actual_peaks[ch], target_peaks[ch], param.fs)?;
    }
    Ok(BandwidthPeakLock {
        grid,
        scaled_passive,
        scaled_carriers,
        scaled_fp1,
        scaled_b1: response.b1_val.clone(),
        scaled_c1: response.c1_val.clone(),
        scaled_b2,
        scaled_c2,
        previous_centers,
        previous_reference_peaks_hz: target_peaks,
        previous_scaled_peaks_hz: actual_peaks,
    })
}

#[derive(Debug)]
struct PassiveCarrierSolution {
    carrier: f64,
    fp1: f64,
    impulse: Vec<f64>,
}

fn build_peak_grid(
    param: &GcParam,
    response: &GcResp,
    impulses: &[Vec<f64>],
    fft_len: usize,
) -> Result<BandwidthPeakGrid> {
    if fft_len < MINIMUM_PEAK_GRID_FFT_LEN
        || !fft_len.is_power_of_two()
        || impulses.len() != param.num_ch
    {
        return Err(Error::InvalidParameter(
            "bandwidth peak grid requires a shared power-of-two DFT of at least 65,536 points"
                .into(),
        ));
    }
    let (reference_b2, reference_c2) = peak_path_coefficients(param, response);
    let reference_passive = impulses
        .iter()
        .map(|impulse| passive_spectrum(impulse, fft_len))
        .collect::<Result<Vec<_>>>()?;
    Ok(BandwidthPeakGrid {
        sample_rate: param.fs,
        fft_len,
        reference_carriers: response.fr1.clone(),
        reference_fp1: response.fp1.clone(),
        reference_b1: response.b1_val.clone(),
        reference_c1: response.c1_val.clone(),
        reference_b2,
        reference_c2,
        reference_passive: Arc::new(reference_passive),
        order: param.n,
    })
}

fn passive_spectrum(impulse: &[f64], fft_len: usize) -> Result<PassiveSpectrum> {
    if impulse.len() > fft_len {
        return Err(Error::Unsupported(format!(
            "a {}-sample passive impulse does not fit the {fft_len}-point bandwidth peak grid",
            impulse.len()
        )));
    }
    let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_len];
    for (destination, &source) in spectrum.iter_mut().zip(impulse) {
        destination.re = source;
    }
    dsp::fft(&mut spectrum, false);
    Ok(PassiveSpectrum {
        impulse: impulse.to_vec(),
        power: spectrum[..=fft_len / 2]
            .iter()
            .map(Complex64::norm_sqr)
            .collect(),
    })
}

#[allow(dead_code)]
pub(crate) fn continuous_cascade_peak_frequencies(
    param: &GcParam,
    response: &GcResp,
    centers: &Array1<f64>,
    b2: &Array1<f64>,
    c2: &Array1<f64>,
    fft_len: usize,
) -> Result<Array1<f64>> {
    if centers.len() != param.num_ch || b2.len() != param.num_ch || c2.len() != param.num_ch {
        return Err(Error::InvalidParameter(
            "continuous cascade measurement requires one center and HP-AF parameter per channel"
                .into(),
        ));
    }
    let impulses = prepare_passive_impulses(param, response)?;
    let mut peaks = Array1::zeros(param.num_ch);
    for ch in 0..param.num_ch {
        let coefficients =
            make_asym_cmp_filters_v2(param.fs, &[centers[ch]], &[b2[ch]], &[c2[ch]])?;
        let ratio = centers[ch] / response.fp1[ch];
        let analytic_peak = fr1_to_fp2(
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            b2[ch],
            c2[ch],
            ratio,
            response.fr1[ch],
        )?
        .0;
        peaks[ch] = continuous_peak_from_impulse(
            &impulses[ch],
            &coefficients,
            analytic_peak,
            None,
            param.fs,
            fft_len,
        )?
        .frequency_hz;
    }
    Ok(peaks)
}

fn passive_impulse(
    carrier: f64,
    sample_rate: f64,
    order: f64,
    b1: f64,
    c1: f64,
) -> Result<Vec<f64>> {
    Ok(gammachirp::gammachirp(
        &[carrier],
        sample_rate,
        order,
        b1,
        c1,
        0.0,
        Carrier::Cosine,
        Normalization::Peak,
    )?
    .gc
    .row(0)
    .to_vec())
}

fn passive_peak_frequency(carrier: f64, order: f64, b1: f64, c1: f64) -> f64 {
    let (_, width) = utils::freq2erb(&[carrier]);
    carrier + c1 * width[0] * b1 / order
}

fn solve_scaled_passive_channels(
    param: &GcParam,
    response: &GcResp,
    grid: &BandwidthPeakGrid,
) -> Result<Vec<PassiveCarrierSolution>> {
    let (ratios, _) = initial_asymmetric_ratio_and_centers(param, response);
    let (scaled_b2, scaled_c2) = peak_path_coefficients(param, response);
    let target_peaks = grid.reference_peak_frequencies_at_ratios(&ratios, None)?;
    let mut solutions = Vec::with_capacity(param.num_ch);
    for ch in 0..param.num_ch {
        let analytic_target = fr1_to_fp2(
            grid.order,
            grid.reference_b1[ch],
            grid.reference_c1[ch],
            grid.reference_b2[ch],
            grid.reference_c2[ch],
            ratios[ch],
            grid.reference_carriers[ch],
        )?
        .0;
        let analytic_seed = fp2_to_fr1(
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
            scaled_b2[ch],
            scaled_c2[ch],
            ratios[ch],
            analytic_target,
        )
        .map(|solution| solution.0)
        .unwrap_or(grid.reference_carriers[ch]);
        let evaluate = |carrier: f64| -> Result<f64> {
            validate_lock_frequency(carrier, param.fs, "peak-locked passive carrier")?;
            let fp1 =
                passive_peak_frequency(carrier, param.n, response.b1_val[ch], response.c1_val[ch]);
            validate_lock_frequency(fp1, param.fs, "peak-locked passive peak")?;
            let center = ratios[ch] * fp1;
            validate_lock_frequency(center, param.fs, "peak-locked HP-AF center")?;
            let impulse = passive_impulse(
                carrier,
                param.fs,
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
            )?;
            let coefficients =
                make_asym_cmp_filters_v2(param.fs, &[center], &[scaled_b2[ch]], &[scaled_c2[ch]])?;
            let analytic_peak = fr1_to_fp2(
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
                scaled_b2[ch],
                scaled_c2[ch],
                ratios[ch],
                carrier,
            )?
            .0;
            continuous_peak_from_impulse(
                &impulse,
                &coefficients,
                analytic_peak,
                None,
                param.fs,
                grid.fft_len,
            )
            .map(|peak| peak.frequency_hz)
        };
        let (carrier, peak) = continuous_frequency_root(
            analytic_seed,
            target_peaks[ch],
            param.fs,
            "passive carrier",
            evaluate,
        )?;
        verify_peak_frequency(peak, target_peaks[ch], param.fs)?;
        let fp1 =
            passive_peak_frequency(carrier, param.n, response.b1_val[ch], response.c1_val[ch]);
        validate_lock_frequency(fp1, param.fs, "peak-locked passive peak")?;
        let impulse = passive_impulse(
            carrier,
            param.fs,
            param.n,
            response.b1_val[ch],
            response.c1_val[ch],
        )?;
        solutions.push(PassiveCarrierSolution {
            carrier,
            fp1,
            impulse,
        });
    }
    Ok(solutions)
}

#[allow(clippy::too_many_arguments)]
fn center_for_continuous_peak(
    passive: &PassiveSpectrum,
    b2: f64,
    c2: f64,
    b1: f64,
    c1: f64,
    carrier: f64,
    order: f64,
    analytic_seed: f64,
    analytic_target: f64,
    target_peak_hz: f64,
    previous_peak_hz: f64,
    sample_rate: f64,
    fft_len: usize,
) -> Result<(f64, f64)> {
    let evaluate = |center: f64| -> Result<f64> {
        validate_lock_frequency(center, sample_rate, "peak-locked HP-AF center")?;
        let coefficients = make_asym_cmp_filters_v2(sample_rate, &[center], &[b2], &[c2])?;
        let fp1 = passive_peak_frequency(carrier, order, b1, c1);
        let analytic_peak = fr1_to_fp2(order, b1, c1, b2, c2, center / fp1, carrier)
            .map(|value| value.0)
            .unwrap_or(analytic_target);
        continuous_peak_from_spectrum(
            passive,
            &coefficients,
            analytic_peak,
            Some(previous_peak_hz),
            sample_rate,
            fft_len,
        )
        .map(|peak| peak.frequency_hz)
    };
    continuous_frequency_root(
        analytic_seed,
        target_peak_hz,
        sample_rate,
        "HP-AF center",
        evaluate,
    )
}

fn validate_lock_frequency(frequency: f64, sample_rate: f64, name: &str) -> Result<()> {
    if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate / 2.0 {
        return Err(Error::Unsupported(format!(
            "no finite, positive, sub-Nyquist {name} realizes the requested continuous peak"
        )));
    }
    Ok(())
}

fn optional_lock_evaluation(result: Result<f64>) -> Result<Option<f64>> {
    match result {
        Ok(value) if value.is_finite() => Ok(Some(value)),
        Ok(_) => Err(Error::Numerical(
            "continuous bandwidth peak evaluation returned a non-finite frequency".into(),
        )),
        Err(Error::Unsupported(_)) | Err(Error::InvalidParameter(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

type FrequencyResidual = (f64, f64, f64);

fn record_frequency_residual(
    evaluate: &mut impl FnMut(f64) -> Result<f64>,
    samples: &mut Vec<FrequencyResidual>,
    parameter: f64,
    target_peak_hz: f64,
) -> Result<Option<(f64, f64, f64)>> {
    if samples.iter().any(|sample| sample.0 == parameter) {
        return Ok(None);
    }
    if let Some(peak) = optional_lock_evaluation(evaluate(parameter))? {
        let sample = (parameter, peak - target_peak_hz, peak);
        samples.push(sample);
        Ok(Some(sample))
    } else {
        Ok(None)
    }
}

fn residual_bracket(
    samples: &mut [FrequencyResidual],
    seed: f64,
) -> Option<(FrequencyResidual, FrequencyResidual)> {
    samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    samples
        .windows(2)
        .filter(|pair| pair[0].1.is_sign_negative() != pair[1].1.is_sign_negative())
        .min_by(|left, right| {
            let left_distance = ((left[0].0 + left[1].0) * 0.5 - seed).abs();
            let right_distance = ((right[0].0 + right[1].0) * 0.5 - seed).abs();
            left_distance.total_cmp(&right_distance)
        })
        .map(|pair| (pair[0], pair[1]))
}

fn continuous_frequency_root(
    seed: f64,
    target_peak_hz: f64,
    sample_rate: f64,
    name: &str,
    mut evaluate: impl FnMut(f64) -> Result<f64>,
) -> Result<(f64, f64)> {
    validate_lock_frequency(target_peak_hz, sample_rate, "reference peak")?;
    let minimum = (sample_rate * 1e-12).max(f64::MIN_POSITIVE);
    let maximum = sample_rate / 2.0 * (1.0 - 1e-12);
    let seed = seed.clamp(minimum, maximum);
    let tolerance = peak_lock_tolerance_hz(sample_rate);
    let mut samples = Vec::new();
    if let Some(sample) =
        record_frequency_residual(&mut evaluate, &mut samples, seed, target_peak_hz)?
        && sample.1.abs() <= tolerance
    {
        return Ok((sample.0, sample.2));
    }

    let mut step = (seed.abs() * 0.02).max(sample_rate / MINIMUM_PEAK_GRID_FFT_LEN as f64);
    let mut bracket = None;
    for _ in 0..MAXIMUM_PEAK_LOCK_ITERATIONS {
        let lower = (seed - step).max(minimum);
        let upper = (seed + step).min(maximum);
        for parameter in [lower, upper] {
            if let Some(sample) =
                record_frequency_residual(&mut evaluate, &mut samples, parameter, target_peak_hz)?
                && sample.1.abs() <= tolerance
            {
                return Ok((sample.0, sample.2));
            }
        }
        bracket = residual_bracket(&mut samples, seed);
        if bracket.is_some() || (lower == minimum && upper == maximum) {
            break;
        }
        step *= 2.0;
    }
    let (mut lower, mut upper) = bracket.ok_or_else(|| {
        Error::Unsupported(format!(
            "no finite, positive, sub-Nyquist {name} brackets the requested continuous peak at {target_peak_hz} Hz"
        ))
    })?;
    for _ in 0..MAXIMUM_PEAK_LOCK_ITERATIONS {
        let midpoint = lower.0 + (upper.0 - lower.0) * 0.5;
        let width = upper.0 - lower.0;
        let secant = lower.0 - lower.1 * width / (upper.1 - lower.1);
        let guard = 0.01 * width;
        let parameter =
            if secant.is_finite() && secant > lower.0 + guard && secant < upper.0 - guard {
                secant
            } else {
                midpoint
            };
        if parameter == lower.0 || parameter == upper.0 {
            break;
        }
        let peak = optional_lock_evaluation(evaluate(parameter))?.ok_or_else(|| {
            Error::Unsupported(format!(
                "{name} became invalid inside its continuous peak-lock bracket"
            ))
        })?;
        let middle = (parameter, peak - target_peak_hz, peak);
        if middle.1.abs() <= tolerance {
            return Ok((middle.0, middle.2));
        }
        if lower.1.is_sign_negative() != middle.1.is_sign_negative() {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    let best = if lower.1.abs() <= upper.1.abs() {
        lower
    } else {
        upper
    };
    verify_peak_frequency(best.2, target_peak_hz, sample_rate)?;
    Ok((best.0, best.2))
}

fn continuous_peak_from_spectrum(
    passive: &PassiveSpectrum,
    coefficients: &AcfCoef,
    analytic_peak_hz: f64,
    previous_peak_hz: Option<f64>,
    sample_rate: f64,
    fft_len: usize,
) -> Result<CascadePeak> {
    let bin = local_main_lobe_peak_bin(
        analytic_peak_hz,
        previous_peak_hz,
        fft_len,
        sample_rate,
        |bin| {
            let passive_power = passive.power[bin];
            if !passive_power.is_finite() || passive_power <= 0.0 {
                return Err(Error::Numerical(
                    "non-positive FIR response on the bandwidth peak bracketing grid".into(),
                ));
            }
            let frequency = bin as f64 * sample_rate / fft_len as f64;
            Ok(passive_power.ln()
                + hpaf_log_power_and_derivative(coefficients, frequency, sample_rate)?.0)
        },
    )?;
    refine_main_lobe_peak(bin, fft_len, sample_rate, |frequency| {
        cascade_log_power_and_derivative(&passive.impulse, coefficients, frequency, sample_rate)
    })
}

fn continuous_peak_from_impulse(
    impulse: &[f64],
    coefficients: &AcfCoef,
    analytic_peak_hz: f64,
    previous_peak_hz: Option<f64>,
    sample_rate: f64,
    fft_len: usize,
) -> Result<CascadePeak> {
    let mut log_power_cache = HashMap::new();
    let bin = local_main_lobe_peak_bin(
        analytic_peak_hz,
        previous_peak_hz,
        fft_len,
        sample_rate,
        |bin| {
            if let Some(&log_power) = log_power_cache.get(&bin) {
                return Ok(log_power);
            }
            let frequency = bin as f64 * sample_rate / fft_len as f64;
            let log_power =
                cascade_log_power_and_derivative(impulse, coefficients, frequency, sample_rate)?.0;
            log_power_cache.insert(bin, log_power);
            Ok(log_power)
        },
    )?;
    refine_main_lobe_peak(bin, fft_len, sample_rate, |frequency| {
        cascade_log_power_and_derivative(impulse, coefficients, frequency, sample_rate)
    })
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let corrected = value - *compensation;
    let next = *sum + corrected;
    *compensation = (next - *sum) - corrected;
    *sum = next;
}

fn fir_log_power_and_derivative(
    impulse: &[f64],
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<(f64, f64)> {
    if impulse.is_empty()
        || !frequency_hz.is_finite()
        || frequency_hz < 0.0
        || frequency_hz > sample_rate / 2.0
    {
        return Err(Error::InvalidParameter(
            "FIR DTFT evaluation requires a non-empty impulse and a finite frequency from DC through Nyquist"
                .into(),
        ));
    }
    let radians_per_hz = std::f64::consts::TAU / sample_rate;
    let mut response_re = 0.0;
    let mut response_im = 0.0;
    let mut derivative_re = 0.0;
    let mut derivative_im = 0.0;
    let mut response_re_compensation = 0.0;
    let mut response_im_compensation = 0.0;
    let mut derivative_re_compensation = 0.0;
    let mut derivative_im_compensation = 0.0;
    for (sample, &coefficient) in impulse.iter().enumerate() {
        let phase = radians_per_hz * frequency_hz * sample as f64;
        let (sin, cos) = phase.sin_cos();
        kahan_add(
            &mut response_re,
            &mut response_re_compensation,
            coefficient * cos,
        );
        kahan_add(
            &mut response_im,
            &mut response_im_compensation,
            -coefficient * sin,
        );
        let scale = coefficient * radians_per_hz * sample as f64;
        kahan_add(
            &mut derivative_re,
            &mut derivative_re_compensation,
            -scale * sin,
        );
        kahan_add(
            &mut derivative_im,
            &mut derivative_im_compensation,
            -scale * cos,
        );
    }
    let power = response_re.mul_add(response_re, response_im * response_im);
    if !power.is_finite() || power <= 0.0 {
        return Err(Error::Numerical(
            "FIR DTFT power is not finite and positive near the selected main lobe".into(),
        ));
    }
    let derivative = 2.0 * response_re.mul_add(derivative_re, response_im * derivative_im) / power;
    if !derivative.is_finite() {
        return Err(Error::Numerical(
            "FIR DTFT log-power derivative is not finite near the selected main lobe".into(),
        ));
    }
    Ok((power.ln(), derivative))
}

fn hpaf_log_power_and_derivative(
    coefficients: &AcfCoef,
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<(f64, f64)> {
    if coefficients.bz.dim() != (1, 3, 4)
        || coefficients.ap.dim() != (1, 3, 4)
        || !frequency_hz.is_finite()
        || frequency_hz < 0.0
        || frequency_hz > sample_rate / 2.0
    {
        return Err(Error::InvalidParameter(
            "HP-AF DTFT evaluation requires one four-section filter and a finite frequency from DC through Nyquist"
                .into(),
        ));
    }
    let radians_per_hz = std::f64::consts::TAU / sample_rate;
    let z1 = Complex64::from_polar(1.0, -radians_per_hz * frequency_hz);
    let z2 = z1 * z1;
    let derivative_factor = Complex64::new(0.0, -radians_per_hz);
    let mut log_power = 0.0;
    let mut derivative = 0.0;
    for section in 0..4 {
        let numerator = coefficients.bz[[0, 0, section]]
            + coefficients.bz[[0, 1, section]] * z1
            + coefficients.bz[[0, 2, section]] * z2;
        let denominator = coefficients.ap[[0, 0, section]]
            + coefficients.ap[[0, 1, section]] * z1
            + coefficients.ap[[0, 2, section]] * z2;
        let numerator_derivative = derivative_factor
            * (coefficients.bz[[0, 1, section]] * z1 + 2.0 * coefficients.bz[[0, 2, section]] * z2);
        let denominator_derivative = derivative_factor
            * (coefficients.ap[[0, 1, section]] * z1 + 2.0 * coefficients.ap[[0, 2, section]] * z2);
        let numerator_power = numerator.norm_sqr();
        let denominator_power = denominator.norm_sqr();
        if !numerator_power.is_finite()
            || numerator_power <= 0.0
            || !denominator_power.is_finite()
            || denominator_power <= 0.0
        {
            return Err(Error::Numerical(
                "HP-AF DTFT contains a zero or non-finite section response".into(),
            ));
        }
        log_power += numerator_power.ln() - denominator_power.ln();
        derivative += 2.0
            * ((numerator_derivative / numerator).re - (denominator_derivative / denominator).re);
    }
    if !log_power.is_finite() || !derivative.is_finite() {
        return Err(Error::Numerical(
            "HP-AF DTFT log-power or its derivative is not finite".into(),
        ));
    }
    Ok((log_power, derivative))
}

fn cascade_log_power_and_derivative(
    impulse: &[f64],
    coefficients: &AcfCoef,
    frequency_hz: f64,
    sample_rate: f64,
) -> Result<(f64, f64)> {
    let fir = fir_log_power_and_derivative(impulse, frequency_hz, sample_rate)?;
    let hpaf = hpaf_log_power_and_derivative(coefficients, frequency_hz, sample_rate)?;
    let value = (fir.0 + hpaf.0, fir.1 + hpaf.1);
    if !value.0.is_finite() || !value.1.is_finite() {
        return Err(Error::Numerical(
            "implemented-cascade DTFT log-power or its derivative is not finite".into(),
        ));
    }
    Ok(value)
}

fn local_main_lobe_peak_bin(
    analytic_peak_hz: f64,
    previous_peak_hz: Option<f64>,
    fft_len: usize,
    sample_rate: f64,
    mut log_power_at_bin: impl FnMut(usize) -> Result<f64>,
) -> Result<usize> {
    if !analytic_peak_hz.is_finite() || previous_peak_hz.is_some_and(|value| !value.is_finite()) {
        return Err(Error::Unsupported(
            "the continuous main-lobe search received a non-finite frequency hint".into(),
        ));
    }
    let last_bin = fft_len / 2;
    let analytic_bin = analytic_peak_hz / sample_rate * fft_len as f64;
    let previous_bin = previous_peak_hz.map(|value| value / sample_rate * fft_len as f64);
    let mut current = analytic_bin.round().clamp(0.0, last_bin as f64) as usize;
    for _ in 0..=last_bin {
        let current_power = log_power_at_bin(current)?;
        if !current_power.is_finite() {
            return Err(Error::Numerical(
                "non-finite implemented-cascade response on the FFT bracketing grid".into(),
            ));
        }
        let mut best = (current, current_power);
        for neighbor in [
            current.checked_sub(1),
            (current < last_bin).then_some(current + 1),
        ]
        .into_iter()
        .flatten()
        {
            let power = log_power_at_bin(neighbor)?;
            if !power.is_finite() {
                return Err(Error::Numerical(
                    "non-finite implemented-cascade response on the FFT bracketing grid".into(),
                ));
            }
            let preferred = |bin: usize| {
                (
                    (bin as f64 - analytic_bin).abs(),
                    previous_bin.map_or(0.0, |previous| (bin as f64 - previous).abs()),
                    bin,
                )
            };
            if power > best.1 || (power == best.1 && preferred(neighbor) < preferred(best.0)) {
                best = (neighbor, power);
            }
        }
        if best.0 == current {
            if current == 0 || current == last_bin {
                return Err(Error::Unsupported(
                    "the intended implemented-cascade main lobe has no interior FFT-grid maximum"
                        .into(),
                ));
            }
            return Ok(current);
        }
        current = best.0;
    }
    Err(Error::Numerical(
        "FFT-grid main-lobe search did not converge".into(),
    ))
}

fn refine_main_lobe_peak(
    peak_bin: usize,
    fft_len: usize,
    sample_rate: f64,
    mut evaluate: impl FnMut(f64) -> Result<(f64, f64)>,
) -> Result<CascadePeak> {
    let spacing = sample_rate / fft_len as f64;
    let selected = peak_bin as f64 * spacing;
    let mut samples = Vec::new();
    let selected_value = evaluate(selected)?;
    if selected_value.1 == 0.0 {
        return Ok(CascadePeak {
            frequency_hz: selected,
            log_power: selected_value.0,
        });
    }
    samples.push((selected, selected_value.1));
    let mut bracket = None;
    for radius in 1..=MAXIMUM_PEAK_LOCK_ITERATIONS {
        for subdivision in 1..=8 {
            let offset = ((radius - 1) * 8 + subdivision) as f64 * spacing / 8.0;
            for frequency in [selected - offset, selected + offset] {
                if frequency <= 0.0 || frequency >= sample_rate / 2.0 {
                    continue;
                }
                let derivative = evaluate(frequency)?.1;
                if derivative == 0.0 {
                    return Ok(CascadePeak {
                        frequency_hz: frequency,
                        log_power: evaluate(frequency)?.0,
                    });
                }
                samples.push((frequency, derivative));
            }
        }
        samples.sort_by(|left, right| left.0.total_cmp(&right.0));
        samples.dedup_by(|left, right| left.0 == right.0);
        bracket = samples
            .windows(2)
            .filter(|pair| pair[0].1 > 0.0 && pair[1].1 < 0.0)
            .min_by(|left, right| {
                let left_distance = ((left[0].0 + left[1].0) * 0.5 - selected).abs();
                let right_distance = ((right[0].0 + right[1].0) * 0.5 - selected).abs();
                left_distance.total_cmp(&right_distance)
            })
            .map(|pair| (pair[0], pair[1]));
        if bracket.is_some() {
            break;
        }
    }
    let (mut lower, mut upper) = bracket.ok_or_else(|| {
        Error::Unsupported(
            "could not bracket the continuous-DTFT derivative zero for the intended main lobe"
                .into(),
        )
    })?;
    for _ in 0..MAXIMUM_PEAK_LOCK_ITERATIONS {
        let midpoint = lower.0 + (upper.0 - lower.0) * 0.5;
        if midpoint == lower.0
            || midpoint == upper.0
            || upper.0 - lower.0 <= peak_search_tolerance_hz(sample_rate)
        {
            let result = evaluate(midpoint)?;
            return Ok(CascadePeak {
                frequency_hz: midpoint,
                log_power: result.0,
            });
        }
        let middle = evaluate(midpoint)?;
        if middle.1 == 0.0 {
            return Ok(CascadePeak {
                frequency_hz: midpoint,
                log_power: middle.0,
            });
        }
        if middle.1 > 0.0 {
            lower = (midpoint, middle.1);
        } else {
            upper = (midpoint, middle.1);
        }
    }
    Err(Error::Numerical(
        "continuous-DTFT main-lobe maximum did not converge within 128 iterations".into(),
    ))
}

fn verify_peak_frequency(actual: f64, target: f64, sample_rate: f64) -> Result<()> {
    let tolerance = peak_lock_tolerance_hz(sample_rate);
    if !actual.is_finite() || !target.is_finite() || (actual - target).abs() > tolerance {
        return Err(Error::Unsupported(format!(
            "bandwidth peak lock residual {} Hz exceeds the {tolerance} Hz tolerance",
            actual - target
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn center_for_composite_peak(
    n: f64,
    b1: f64,
    c1: f64,
    b2: f64,
    c2: f64,
    fr1: f64,
    fp1: f64,
    target_peak: f64,
    sample_rate: f64,
    preferred_center: f64,
) -> Result<f64> {
    if !target_peak.is_finite()
        || target_peak <= 0.0
        || target_peak >= sample_rate / 2.0
        || !fp1.is_finite()
        || fp1 <= 0.0
    {
        return Err(Error::Unsupported(format!(
            "no valid HP-AF center realizes the requested composite peak {target_peak} Hz"
        )));
    }
    let (_, width1) = utils::freq2erb(&[fr1]);
    let (_, width_at_zero) = utils::freq2erb(&[0.0]);
    let (_, width_at_one) = utils::freq2erb(&[1.0]);
    let width_slope = width_at_one[0] - width_at_zero[0];
    let u = b2 * width_slope;
    let v = b2 * width_at_zero[0];
    let bw1 = b1 * width1[0];
    let k = c1 * bw1 + n * fr1;
    let l = bw1 * bw1 + fr1 * fr1;
    let p = target_peak;
    let quadratic = (u * u + 1.0) * (k - n * p);
    let linear = (c2 * u + 2.0 * n) * p * p
        + (-2.0 * k - 2.0 * n * u * v - 2.0 * c2 * u * fr1) * p
        + c2 * u * l
        + 2.0 * k * u * v;
    let constant = -n * p.powi(3)
        + (c1 * bw1 + c2 * v + n * fr1) * p * p
        + (-n * v * v - 2.0 * c2 * v * fr1) * p
        + c2 * v * l
        + k * v * v;
    let coefficient_scale = quadratic
        .abs()
        .max(linear.abs())
        .max(constant.abs())
        .max(1.0);
    let roots = if quadratic.abs() <= 64.0 * f64::EPSILON * coefficient_scale {
        if linear.abs() <= 64.0 * f64::EPSILON * coefficient_scale {
            Vec::new()
        } else {
            vec![-constant / linear]
        }
    } else {
        let discriminant = linear * linear - 4.0 * quadratic * constant;
        let tolerance = 128.0
            * f64::EPSILON
            * (linear * linear)
                .abs()
                .max((4.0 * quadratic * constant).abs());
        if discriminant < -tolerance {
            Vec::new()
        } else {
            let square_root = discriminant.max(0.0).sqrt();
            let stable_numerator = -0.5 * (linear + square_root.copysign(linear));
            if stable_numerator == 0.0 {
                vec![-linear / (2.0 * quadratic)]
            } else {
                vec![stable_numerator / quadratic, constant / stable_numerator]
            }
        }
    };
    let mut candidates = Vec::new();
    for center in roots {
        if !center.is_finite() || center <= 0.0 || center >= sample_rate / 2.0 {
            continue;
        }
        let ratio = center / fp1;
        let actual = fr1_to_fp2(n, b1, c1, b2, c2, ratio, fr1)?.0;
        if verify_peak_lock(actual, target_peak).is_ok() {
            candidates.push(center);
        }
    }
    candidates
        .into_iter()
        .min_by(|left, right| {
            (left - preferred_center)
                .abs()
                .total_cmp(&(right - preferred_center).abs())
        })
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "no valid HP-AF center realizes the requested composite peak {target_peak} Hz"
            ))
        })
}

fn verify_peak_lock(actual: f64, target: f64) -> Result<()> {
    let tolerance = 1e-7 * target.abs().max(1.0);
    if !actual.is_finite() || !target.is_finite() || (actual - target).abs() > tolerance {
        return Err(Error::Numerical(format!(
            "bandwidth peak lock missed its target: {actual} Hz versus {target} Hz"
        )));
    }
    Ok(())
}

pub(super) fn prepare_input_correction_fir(param: &GcParam) -> Result<Vec<f64>> {
    if param.out_mid_crct.eq_ignore_ascii_case("no") {
        Ok(vec![1.0])
    } else {
        Ok(
            utils::mk_filter_field2cochlea(&param.out_mid_crct, param.fs, true)?
                .0
                .to_vec(),
        )
    }
}

pub(super) fn prepare_passive_impulses(
    param: &GcParam,
    response: &GcResp,
) -> Result<Vec<Vec<f64>>> {
    (0..param.num_ch)
        .map(|ch| {
            let impulse = gammachirp::gammachirp(
                &[response.fr1[ch]],
                param.fs,
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
                0.0,
                Carrier::Cosine,
                Normalization::Peak,
            )?;
            Ok(impulse.gc.row(0).to_vec())
        })
        .collect()
}

/// Prepare the fixed asymmetric path and all time-invariant response metadata.
pub(super) fn prepare_time_invariant_response(
    param: &GcParam,
    response: &mut GcResp,
    passive_impulses: &[Vec<f64>],
) -> Result<AcfCoef> {
    let channels = param.num_ch;
    let (initial_ratios, fixed_centers) = initial_asymmetric_ratio_and_centers(param, response);
    let fixed_c2 = if param.ctrl == ControlMode::Static {
        let level = param.level_db_scgcfb;
        response.fr2 = Array2::zeros((channels, 1));
        response.fr2.column_mut(0).assign(&fixed_centers);
        response.frat_val = Array2::zeros((channels, 1));
        response.frat_val.column_mut(0).assign(&initial_ratios);
        response.lvl_db = Array2::from_elem((channels, 1), level);
        for ch in 0..channels {
            response.fp2[ch] = fr1_to_fp2(
                param.n,
                response.b1_val[ch],
                response.c1_val[ch],
                response.b2_val[ch],
                response.c2_val[ch],
                initial_ratios[ch],
                response.fr1[ch],
            )?
            .0;
        }
        response.c2_val.clone()
    } else {
        &param.hloss.fb_compression_health * param.lvl_est.c2
    };
    let level_b2 = [param.lvl_est.b2];
    let fixed_b2 = if param.ctrl == ControlMode::Static {
        response.b2_val.as_slice().unwrap()
    } else {
        &level_b2
    };
    let coefficients = make_asym_cmp_filters_v2(
        param.fs,
        fixed_centers.as_slice().unwrap(),
        fixed_b2,
        fixed_c2.as_slice().unwrap(),
    )?;
    match param.gain_ref {
        GainReference::NormalizeIoFunction => {
            for ch in 0..channels {
                response.gain_factor[ch] =
                    10_f64.powf(-param.hloss.fb_af_gain_cmpnst_db[ch] / 20.0);
            }
        }
        GainReference::Db(db) => {
            let ratios = Array1::from_iter((0..channels).map(|ch| {
                response.frat0_pc[ch] + response.frat1_val[ch] * (db - response.pc_hpaf[ch])
            }));
            let mut reference = cmprs_gc_frsp(
                response.fr1.as_slice().unwrap(),
                param.fs,
                param.n,
                response.b1_val.as_slice().unwrap(),
                response.c1_val.as_slice().unwrap(),
                ratios.as_slice().unwrap(),
                response.b2_val.as_slice().unwrap(),
                response.c2_val.as_slice().unwrap(),
                1024,
            )?;
            let peaks = realized_cascade_peaks(
                param,
                response,
                passive_impulses,
                &ratios,
                &response.b2_val,
                &response.c2_val,
            )?;
            reference.fp2 = peaks.frequency_hz;
            reference.val_fp2 = peaks.value;
            reference.norm_fct_fp2 = peaks.normalization;
            for ch in 0..channels {
                let peak = reference.val_fp2[ch];
                reference
                    .cgc_nrm_frsp
                    .row_mut(ch)
                    .assign(&reference.cgc_frsp.row(ch).mapv(|value| value / peak));
            }
            response.gain_factor = reference
                .norm_fct_fp2
                .mapv(|value| 10_f64.powf(param.gain_cmpnst_db / 20.0) * value);
            response.cgc_ref = Some(reference);
        }
    }
    Ok(coefficients)
}

pub fn set_param(param: GcParam) -> Result<(GcParam, GcResp)> {
    set_param_with_preserved_hearing_loss(param, None)
}

pub(crate) fn set_param_with_preserved_hearing_loss(
    mut param: GcParam,
    preserved_hearing_loss: Option<&HLoss>,
) -> Result<(GcParam, GcResp)> {
    if !param.fs.is_finite()
        || param.fs <= 0.
        || param.num_ch < 2
        || param.f_range.iter().any(|value| !value.is_finite())
        || param.f_range[0] <= 0.0
        || param.f_range[1] <= param.f_range[0]
        || param.f_range[1] >= param.fs / 2.0
        || param.num_update_asym_cmp == 0
        || !param.dyn_hpaf.t_frame.is_finite()
        || param.dyn_hpaf.t_frame <= 0.0
        || !param.dyn_hpaf.t_shift.is_finite()
        || param.dyn_hpaf.t_shift <= 0.0
    {
        return Err(Error::InvalidParameter(
            "v2.34 requires a finite positive sample rate, at least two channels, a frequency range satisfying 0 < low < high < fs / 2, and positive frame and update periods"
                .into(),
        ));
    }
    param.dyn_hpaf.len_frame = (param.dyn_hpaf.t_frame * param.fs).trunc() as usize;
    param.dyn_hpaf.len_shift = (param.dyn_hpaf.t_shift * param.fs).trunc() as usize;
    if param.dyn_hpaf.len_frame == 0 || param.dyn_hpaf.len_shift == 0 {
        return Err(Error::InvalidParameter(
            "dynamic frame and shift must contain at least one sample".into(),
        ));
    }
    param.dyn_hpaf.t_frame = param.dyn_hpaf.len_frame as f64 / param.fs;
    param.dyn_hpaf.t_shift = param.dyn_hpaf.len_shift as f64 / param.fs;
    param.dyn_hpaf.fs = 1. / param.dyn_hpaf.t_shift;
    let processing = param.dyn_hpaf.str_prc.to_ascii_lowercase();
    let frame_processing = processing.contains("frame");
    let sample_processing = processing.contains("sample");
    if frame_processing == sample_processing {
        return Err(Error::InvalidParameter(
            "dynamic processing mode must select exactly one of frame-base or sample-base".into(),
        ));
    }
    param.dyn_hpaf.str_prc = if frame_processing {
        "frame-base".into()
    } else {
        "sample-base".into()
    };
    if frame_processing {
        let window_name = param.dyn_hpaf.name_win.to_ascii_lowercase();
        let win = if window_name.contains("hann") {
            dsp::hanning(param.dyn_hpaf.len_frame)
        } else if window_name.contains("hamm") {
            dsp::hamming(param.dyn_hpaf.len_frame)
        } else {
            return Err(Error::InvalidParameter(
                "frame window must be hanning or hamming".into(),
            ));
        };
        let sum: f64 = win.iter().sum();
        param.dyn_hpaf.val_win = Array1::from_iter(win.into_iter().map(|v| v / sum));
    }
    let (fr1, erb_grid) =
        utils::equal_freq_scale(FrequencyScale::Erb, param.num_ch, param.f_range)?;
    validate_prepared_frequencies(
        param.fs,
        fr1.as_slice().unwrap(),
        "channel grid frequencies",
    )?;
    param.fr1 = fr1.clone();
    let erb_space = erb_grid
        .windows(2)
        .into_iter()
        .map(|w| w[1] - w[0])
        .sum::<f64>()
        / (param.num_ch - 1) as f64;
    let (erb, _) = utils::freq2erb(fr1.as_slice().unwrap());
    let (erb1k, _) = utils::freq2erb(&[1000.]);
    let ef = erb.mapv(|v| v / erb1k[0] - 1.);
    let b1 = ef.mapv(|v| param.b1[0] + param.b1[1] * v);
    let c1 = ef.mapv(|v| param.c1[0] + param.c1[1] * v);
    let (_, width) = utils::freq2erb(fr1.as_slice().unwrap());
    let fp1 =
        Array1::from_iter((0..param.num_ch).map(|i| fr1[i] + c1[i] * width[i] * b1[i] / param.n));
    validate_prepared_frequencies(param.fs, fp1.as_slice().unwrap(), "derived filter centers")?;
    validate_user_controlled_parameters(&param)?;
    let b2 = ef.mapv(|v| param.b2[0][0] + param.b2[0][1] * v);
    let c2 = ef.mapv(|v| param.c2[0][0] + param.c2[0][1] * v);
    let frat0 = ef.mapv(|v| param.frat[0][0] + param.frat[0][1] * v);
    let frat1 = ef.mapv(|v| param.frat[1][0] + param.frat[1][1] * v);
    let pc = Array1::from_iter((0..param.num_ch).map(|i| (1. - frat0[i]) / frat1[i]));
    let frat0_pc = Array1::from_iter((0..param.num_ch).map(|i| frat0[i] + frat1[i] * pc[i]));
    let mut response = GcResp {
        fr1: fr1.clone(),
        fr2: Array2::zeros((param.num_ch, 0)),
        erb_space1: erb_space,
        ef,
        b1_val: b1,
        c1_val: c1,
        fp1,
        fp2: Array1::zeros(param.num_ch),
        b2_val: b2,
        c2_val: c2,
        frat_val: Array2::zeros((param.num_ch, 0)),
        frat0_val: frat0,
        frat1_val: frat1,
        pc_hpaf: pc,
        frat0_pc,
        lvl_db: Array2::zeros((param.num_ch, 0)),
        lvl_db_frame: Array2::zeros((0, 0)),
        pgc_frame: Array2::zeros((0, 0)),
        scgc_frame: Array2::zeros((0, 0)),
        asym_func_gain: Array2::zeros((0, 0)),
        gain_factor: Array1::ones(param.num_ch),
        cgc_ref: None,
    };
    let (_, initial_centers) = initial_asymmetric_ratio_and_centers(&param, &response);
    validate_prepared_frequencies(
        param.fs,
        initial_centers.as_slice().unwrap(),
        "initial asymmetric filter centers",
    )?;
    if let Some(hearing_loss) = preserved_hearing_loss {
        param.hloss = hearing_loss.clone();
    } else {
        param = gcfb_v23_hearing_loss(param, &response)?;
    }
    let shift = (param.lvl_est.lct_erb / erb_space).round() as isize;
    param.lvl_est.exp_decay_val =
        (-1. / (param.lvl_est.decay_hl * param.fs / 1000.) * 2_f64.ln()).exp();
    param.lvl_est.erb_space1 = erb_space;
    param.lvl_est.n_ch_shift = shift;
    param.lvl_est.n_ch_lvl_est = Array1::from_iter(
        (0..param.num_ch)
            .map(|ch| (ch as isize + shift).clamp(0, param.num_ch as isize - 1) as usize),
    );
    param.lvl_est.lvl_lin_min_lim = 10_f64.powf(-param.lvl_est.rms2spldb / 20.);
    param.lvl_est.lvl_lin_ref = 10_f64.powf((param.lvl_est.ref_db - param.lvl_est.rms2spldb) / 20.);
    response.fr1 = param.fr1.clone();
    Ok((param, response))
}

fn hearing_pattern(
    kind: &str,
    manual: Option<&Array1<f64>>,
) -> Result<(&'static str, Array1<f64>)> {
    let key = kind.split('_').next().unwrap_or(kind).to_ascii_uppercase();
    let pair = match key.as_str() {
        "NH" => ("NH_NormalHearing", vec![0.; 7]),
        "HL0" => (
            "HLval_ManualSet",
            manual
                .ok_or_else(|| {
                    Error::InvalidParameter("HL0 requires hloss_hearing_level_db".into())
                })?
                .to_vec(),
        ),
        "HL1" => ("HL1_Example", vec![10., 4., 10., 13., 48., 58., 79.]),
        "HL2" => (
            "HL2_Tsuiki2002_80yr",
            vec![23.5, 24.3, 26.8, 27.9, 32.9, 48.3, 68.5],
        ),
        "HL3" => (
            "HL3_ISO7029_70yr_male",
            vec![8., 8., 9., 10., 19., 43., 59.],
        ),
        "HL4" => (
            "HL4_ISO7029_70yr_female",
            vec![8., 8., 9., 10., 16., 24., 41.],
        ),
        "HL5" => ("HL5_ISO7029_60yr_male", vec![5., 5., 6., 7., 12., 28., 39.]),
        "HL6" => (
            "HL6_ISO7029_60yr_female",
            vec![5., 5., 6., 7., 11., 16., 26.],
        ),
        "HL7" => (
            "HL7_Example_Otosclerosis",
            vec![50., 55., 50., 50., 40., 25., 20.],
        ),
        "HL8" => (
            "HL8_Example_NoiseInduced",
            vec![15., 10., 15., 10., 10., 40., 20.],
        ),
        _ => {
            return Err(Error::InvalidParameter(
                "hearing-loss type must be NH or HL0..HL8".into(),
            ));
        }
    };
    if pair.1.len() != 7 || pair.1.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err(Error::InvalidParameter(
            "audiogram must contain seven finite non-negative values".into(),
        ));
    }
    Ok((pair.0, Array1::from(pair.1)))
}

pub fn gcfb_v23_hearing_loss(mut param: GcParam, response: &GcResp) -> Result<GcParam> {
    if param
        .hloss_compression_health
        .is_some_and(|health| !health.is_finite() || !(0.0..=1.0).contains(&health))
    {
        return Err(Error::InvalidParameter(
            "hearing-loss compression health must be finite and in 0.0..=1.0".into(),
        ));
    }
    let (name, hearing) =
        hearing_pattern(&param.hloss_type, param.hloss_hearing_level_db.as_ref())?;
    let mut loss = HLoss {
        type_name: name.into(),
        hearing_level_db: hearing,
        ..HLoss::default()
    };
    let default_health = if name == "NH_NormalHearing" { 1. } else { 0.5 };
    loss.compression_health =
        Array1::from_elem(7, param.hloss_compression_health.unwrap_or(default_health));
    loss.compression_health_initval = loss.compression_health.clone();
    let mut act = Array1::zeros(7);
    let mut act_initial = Array1::zeros(7);
    let mut passive = Array1::zeros(7);
    let mut gain = Array1::zeros(7);
    let mut hl_pin = Array1::zeros(7);
    for i in 0..7 {
        let f = loss.f_audgram_list[i];
        let hl0 = utils::hl2pin_cochlea(f, 0.)?;
        let (_, io_normal) = asym_func_in_out_scalar(&param, response, f, 1., hl0);
        let mut health = loss.compression_health[i];
        let reduction =
            gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, health, io_normal)?;
        act[i] = reduction - hl0;
        act_initial[i] = act[i];
        passive[i] = (loss.hearing_level_db[i] - act[i]).max(0.);
        if passive[i] < f64::EPSILON * 1e4 {
            act[i] = loss.hearing_level_db[i];
            let health_values: Vec<f64> = (0..10).map(|j| 1. - j as f64 * 0.1).collect();
            let mut active_values = Vec::new();
            for &h in &health_values {
                active_values.push(
                    gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, h, io_normal)? - hl0,
                );
            }
            health =
                dsp::interp1(&active_values, &health_values, &[act[i]], true)?[0].clamp(0., 1.);
            act[i] = gcfb_v23_asym_func_in_out_inv_io_func(&param, response, f, health, io_normal)?
                - hl0;
            passive[i] = loss.hearing_level_db[i] - act[i];
        }
        loss.compression_health[i] = health;
        hl_pin[i] = utils::hl2pin_cochlea(f, 0.)? + loss.hearing_level_db[i];
        let (_, io) = asym_func_in_out_scalar(&param, response, f, health, hl_pin[i]);
        gain[i] = io;
    }
    loss.pin_loss_db_act = act.clone();
    loss.pin_loss_db_act_init = act_initial;
    loss.pin_loss_db_pas = passive.clone();
    loss.af_gain_cmpnst_db = gain.clone();
    loss.hl_val_pin_cochlea_db = hl_pin.clone();
    loss.fb_fr1 = response.fr1.clone();
    let (aud_erb, _) = utils::freq2erb(loss.f_audgram_list.as_slice().unwrap());
    let (fb_erb, _) = utils::freq2erb(response.fr1.as_slice().unwrap());
    let x = aud_erb.as_slice().unwrap();
    let q = fb_erb.as_slice().unwrap();
    loss.fb_hearing_level_db =
        utils::interp1(x, loss.hearing_level_db.as_slice().unwrap(), q, true)?;
    loss.fb_pin_cochlea_db = utils::interp1(x, hl_pin.as_slice().unwrap(), q, true)?;
    loss.fb_pin_loss_db_act = utils::interp1(x, act.as_slice().unwrap(), q, true)?;
    loss.fb_pin_loss_db_pas = utils::interp1(x, passive.as_slice().unwrap(), q, true)?;
    loss.fb_compression_health =
        utils::interp1(x, loss.compression_health.as_slice().unwrap(), q, true)?
            .mapv(|v| v.clamp(0., 1.));
    loss.fb_af_gain_cmpnst_db = utils::interp1(x, gain.as_slice().unwrap(), q, true)?;
    param.hloss = loss;
    Ok(param)
}

pub fn cal_asym_func(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    compression_health: f64,
    pin_db: f64,
) -> f64 {
    let ch = param
        .fr1
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - fr1query).abs().total_cmp(&(b.1 - fr1query).abs()))
        .unwrap()
        .0;
    let frat = response.frat0_pc[ch] + response.frat1_val[ch] * (pin_db - response.pc_hpaf[ch]);
    let fr2 = frat * response.fp1[ch];
    let (_, w) = utils::freq2erb(&[fr2]);
    (compression_health
        * response.c2_val[ch]
        * (response.fp1[ch] - fr2).atan2(response.b2_val[ch] * w[0]))
    .exp()
}

pub(super) fn asym_func_in_out_scalar(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    health: f64,
    pin: f64,
) -> (f64, f64) {
    let v = cal_asym_func(param, response, fr1query, health, pin);
    let norm = cal_asym_func(param, response, fr1query, health, 100.);
    let db = 20. * (v / norm).log10();
    (db, db + pin)
}

pub fn gcfb_v23_asym_func_in_out(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    compression_health: f64,
    pin_db: &[f64],
) -> (Array1<f64>, Array1<f64>) {
    let mut af = Array1::zeros(pin_db.len());
    let mut io = Array1::zeros(pin_db.len());
    for (i, &pin) in pin_db.iter().enumerate() {
        (af[i], io[i]) =
            asym_func_in_out_scalar(param, response, fr1query, compression_health, pin);
    }
    (af, io)
}

pub fn gcfb_v23_asym_func_in_out_inv_io_func(
    param: &GcParam,
    response: &GcResp,
    fr1query: f64,
    health: f64,
    io_db: f64,
) -> Result<f64> {
    let pins: Vec<f64> = (0..=2700).map(|i| -120. + i as f64 * 0.1).collect();
    let (_, ios) = gcfb_v23_asym_func_in_out(param, response, fr1query, health, &pins);
    Ok(dsp::interp1(ios.as_slice().unwrap(), &pins, &[io_db], true)?[0])
}

pub fn gcfb_v23_frame_base(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
) -> Result<Array2<f64>> {
    gcfb_v23_frame_base_internal(pgc, scgc, param, response, None)
}

fn gcfb_v23_frame_base_internal(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
    prepared_passive_impulses: Option<&[Vec<f64>]>,
) -> Result<Array2<f64>> {
    let (channels, samples) = pgc.dim();
    let channel_vectors_match = [
        param.fr1.len(),
        param.hloss.fb_compression_health.len(),
        param.lvl_est.n_ch_lvl_est.len(),
        response.b1_val.len(),
        response.c1_val.len(),
        response.fp1.len(),
        response.b2_val.len(),
        response.c2_val.len(),
        response.frat0_pc.len(),
        response.frat1_val.len(),
        response.pc_hpaf.len(),
    ]
    .into_iter()
    .all(|len| len == channels);
    if channels == 0
        || samples == 0
        || scgc.dim() != pgc.dim()
        || param.num_ch != channels
        || !channel_vectors_match
        || param.dyn_hpaf.val_win.len() != param.dyn_hpaf.len_frame
        || param
            .lvl_est
            .n_ch_lvl_est
            .iter()
            .any(|&source| source >= channels)
    {
        return Err(Error::InvalidParameter(
            "frame processing requires non-empty, equally shaped channel matrices and matching prepared parameters"
                .into(),
        ));
    }
    let decay = param
        .lvl_est
        .exp_decay_val
        .powf(param.dyn_hpaf.len_shift as f64);
    let c2: Array1<f64> = &param.hloss.fb_compression_health * param.lvl_est.c2;
    let owned_passive_impulses;
    let passive_impulses = if let Some(impulses) = prepared_passive_impulses {
        impulses
    } else {
        owned_passive_impulses = prepare_passive_impulses(param, response)?;
        &owned_passive_impulses
    };
    let static_peaks = realized_cascade_peaks(
        param,
        response,
        passive_impulses,
        &Array1::from_elem(channels, param.lvl_est.frat),
        &Array1::from_elem(channels, param.lvl_est.b2),
        &c2,
    )?;
    let first_frames = dsp::frame_sequence(
        pgc.row(0).as_slice().unwrap(),
        param.dyn_hpaf.len_frame,
        param.dyn_hpaf.len_shift,
    )?
    .0
    .ncols();
    let mut out = Array2::zeros((channels, first_frames));
    response.lvl_db_frame = Array2::zeros((channels, first_frames));
    response.pgc_frame = Array2::zeros((channels, first_frames));
    response.scgc_frame = Array2::zeros((channels, first_frames));
    response.asym_func_gain = Array2::zeros((channels, first_frames));
    for ch in 0..channels {
        let (pf, _) = dsp::frame_sequence(
            pgc.row(ch).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let (sf, _) = dsp::frame_sequence(
            scgc.row(ch).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let source = param.lvl_est.n_ch_lvl_est[ch];
        let (l1, _) = dsp::frame_sequence(
            pgc.row(source).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let (l2, _) = dsp::frame_sequence(
            scgc.row(source).as_slice().unwrap(),
            param.dyn_hpaf.len_frame,
            param.dyn_hpaf.len_shift,
        )?;
        let weighted = |m: &Array2<f64>, frame: usize| -> f64 {
            param
                .dyn_hpaf
                .val_win
                .iter()
                .enumerate()
                .map(|(i, w)| w * m[[i, frame]].powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut level1 = Array1::from_iter((0..first_frames).map(|f| weighted(&l1, f)));
        let mut level2 = Array1::from_iter((0..first_frames).map(|f| weighted(&l2, f)));
        for f in 1..first_frames {
            level1[f] = level1[f].max(level1[f - 1] * decay);
            level2[f] = level2[f].max(level2[f - 1] * decay);
        }
        for frame in 0..first_frames {
            response.pgc_frame[[ch, frame]] = weighted(&pf, frame);
            response.scgc_frame[[ch, frame]] = weighted(&sf, frame);
            let total = param.lvl_est.weight
                * param.lvl_est.lvl_lin_ref
                * (level1[frame] / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[0])
                + (1. - param.lvl_est.weight)
                    * param.lvl_est.lvl_lin_ref
                    * (level2[frame] / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[1]);
            let level_db = 20. * total.max(param.lvl_est.lvl_lin_min_lim).log10()
                + param.lvl_est.rms2spldb
                - 3.;
            response.lvl_db_frame[[ch, frame]] = level_db;
            let (af, _) = asym_func_in_out_scalar(
                param,
                response,
                param.fr1[ch],
                param.hloss.fb_compression_health[ch],
                level_db,
            );
            let gain = 10_f64.powf(af / 20.);
            response.asym_func_gain[[ch, frame]] = gain;
            out[[ch, frame]] =
                gain * static_peaks.normalization[ch] * response.scgc_frame[[ch, frame]];
        }
    }
    Ok(out)
}

pub fn gcfb_v23_sample_base(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
) -> Result<Array2<f64>> {
    gcfb_v23_sample_base_internal(pgc, scgc, param, response, None)
}

fn gcfb_v23_sample_base_internal(
    pgc: &Array2<f64>,
    scgc: &Array2<f64>,
    param: &GcParam,
    response: &mut GcResp,
    mut peak_lock: Option<&mut BandwidthPeakLock>,
) -> Result<Array2<f64>> {
    let (channels, samples) = pgc.dim();
    let channel_vectors_match = [
        param.fr1.len(),
        param.hloss.fb_compression_health.len(),
        param.lvl_est.n_ch_lvl_est.len(),
        response.b1_val.len(),
        response.c1_val.len(),
        response.fp1.len(),
        response.b2_val.len(),
        response.c2_val.len(),
        response.frat0_pc.len(),
        response.frat1_val.len(),
        response.pc_hpaf.len(),
    ]
    .into_iter()
    .all(|len| len == channels);
    if channels == 0
        || samples == 0
        || scgc.dim() != pgc.dim()
        || param.num_ch != channels
        || param.num_update_asym_cmp == 0
        || !channel_vectors_match
        || param
            .lvl_est
            .n_ch_lvl_est
            .iter()
            .any(|&source| source >= channels)
    {
        return Err(Error::InvalidParameter(
            "sample processing requires non-empty, equally shaped channel matrices and matching prepared parameters"
                .into(),
        ));
    }
    let mut out = Array2::zeros((channels, samples));
    response.fr2 = Array2::zeros((channels, samples));
    response.frat_val = Array2::zeros((channels, samples));
    response.lvl_db = Array2::zeros((channels, samples));
    let (initial_ratios, default_centers) = initial_asymmetric_ratio_and_centers(param, response);
    let centers = if let Some(lock) = peak_lock.as_deref_mut() {
        lock.centers_for_reference_peaks_at_ratios(&initial_ratios)?
    } else {
        default_centers
    };
    let mut coef = make_asym_cmp_filters_v2(
        param.fs,
        centers.as_slice().unwrap(),
        response.b2_val.as_slice().unwrap(),
        response.c2_val.as_slice().unwrap(),
    )?;
    let mut status = AcfStatus::new(&coef);
    let mut previous = Array2::<f64>::zeros((channels, 2));
    for sample in 0..samples {
        for ch in 0..channels {
            let source = param.lvl_est.n_ch_lvl_est[ch];
            let a = pgc[[source, sample]]
                .max(0.)
                .max(previous[[ch, 0]] * param.lvl_est.exp_decay_val);
            let b = scgc[[source, sample]]
                .max(0.)
                .max(previous[[ch, 1]] * param.lvl_est.exp_decay_val);
            previous[[ch, 0]] = a;
            previous[[ch, 1]] = b;
            let total = param.lvl_est.weight
                * param.lvl_est.lvl_lin_ref
                * (a / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[0])
                + (1. - param.lvl_est.weight)
                    * param.lvl_est.lvl_lin_ref
                    * (b / param.lvl_est.lvl_lin_ref).powf(param.lvl_est.pwr[1]);
            let db =
                20. * total.max(param.lvl_est.lvl_lin_min_lim).log10() + param.lvl_est.rms2spldb;
            response.lvl_db[[ch, sample]] = db;
            let ratio = response.frat0_pc[ch]
                + param.hloss.fb_compression_health[ch]
                    * response.frat1_val[ch]
                    * (db - response.pc_hpaf[ch]);
            response.frat_val[[ch, sample]] = ratio;
            if peak_lock.is_none() {
                response.fr2[[ch, sample]] = response.fp1[ch] * ratio;
            }
        }
        if sample % param.num_update_asym_cmp == 0 {
            let centers = if let Some(lock) = peak_lock.as_deref_mut() {
                let ratios = response.frat_val.column(sample).to_owned();
                let centers = lock.centers_for_reference_peaks_at_ratios(&ratios)?;
                response.fr2.column_mut(sample).assign(&centers);
                centers.to_vec()
            } else {
                response.fr2.column(sample).to_vec()
            };
            coef = make_asym_cmp_filters_v2(
                param.fs,
                &centers,
                response.b2_val.as_slice().unwrap(),
                response.c2_val.as_slice().unwrap(),
            )?;
        } else if let Some(lock) = peak_lock.as_deref() {
            response
                .fr2
                .column_mut(sample)
                .assign(lock.current_centers());
        }
        let value = status.process(&coef, &pgc.column(sample).to_vec(), false)?;
        out.column_mut(sample).assign(&value);
    }
    Ok(out)
}

pub fn gcfb_v234(snd_in: &[f64], gc_param: GcParam) -> Result<GcfbOutput> {
    gcfb_v234_internal(snd_in, gc_param, None, None)
}

pub(crate) fn gcfb_v234_with_bandwidth_peak_lock(
    snd_in: &[f64],
    gc_param: GcParam,
    hearing_loss: &HLoss,
    peak_grid: Arc<BandwidthPeakGrid>,
) -> Result<GcfbOutput> {
    gcfb_v234_internal(snd_in, gc_param, Some(hearing_loss), Some(peak_grid))
}

fn gcfb_v234_internal(
    snd_in: &[f64],
    gc_param: GcParam,
    preserved_hearing_loss: Option<&HLoss>,
    peak_grid: Option<Arc<BandwidthPeakGrid>>,
) -> Result<GcfbOutput> {
    if snd_in.is_empty() || snd_in.iter().any(|sample| !sample.is_finite()) {
        return Err(Error::InvalidParameter(
            "input sound must be non-empty and finite".into(),
        ));
    }
    let (param, mut response) =
        set_param_with_preserved_hearing_loss(gc_param, preserved_hearing_loss)?;
    let mut peak_lock = if let Some(peak_grid) = peak_grid {
        Some(prepare_bandwidth_peak_lock(
            &param,
            &mut response,
            peak_grid,
        )?)
    } else {
        None
    };
    let correction_fir = prepare_input_correction_fir(&param)?;
    let snd = dsp::lfilter(&correction_fir, &[1.0], snd_in)?;
    let channels = param.num_ch;
    let samples = snd.len();
    let mut pgc = Array2::zeros((channels, samples));
    let mut scgc = Array2::zeros((channels, samples));
    let passive_impulses = prepare_passive_impulses(&param, &response)?;
    let fixed_coef = prepare_time_invariant_response(&param, &mut response, &passive_impulses)?;
    for (ch, impulse) in passive_impulses.iter().enumerate() {
        let filtered = utils::fftfilt(impulse, &snd);
        pgc.row_mut(ch).assign(&filtered);
        let mut value = filtered.to_vec();
        for section in 0..4 {
            value = dsp::lfilter(
                &fixed_coef.bz.slice(s![ch, .., section]).to_vec(),
                &fixed_coef.ap.slice(s![ch, .., section]).to_vec(),
                &value,
            )?;
        }
        scgc.row_mut(ch).assign(&Array1::from(value));
    }
    let mut dcgc = match param.ctrl {
        ControlMode::Static => scgc.clone(),
        ControlMode::Dynamic if param.dyn_hpaf.str_prc.contains("frame") => {
            gcfb_v23_frame_base_internal(
                &pgc,
                &scgc,
                &param,
                &mut response,
                Some(&passive_impulses),
            )?
        }
        ControlMode::Dynamic if param.dyn_hpaf.str_prc.contains("sample") => {
            gcfb_v23_sample_base_internal(&pgc, &scgc, &param, &mut response, peak_lock.as_mut())?
        }
        ControlMode::Dynamic => {
            return Err(Error::InvalidParameter(
                "dynamic processing mode must be frame-base or sample-base".into(),
            ));
        }
        ControlMode::Level => scgc.clone(),
    };
    for ch in 0..channels {
        dcgc.row_mut(ch)
            .mapv_inplace(|value| value * response.gain_factor[ch]);
    }
    Ok(GcfbOutput {
        dcgc_out: dcgc,
        scgc_smpl: scgc,
        gc_param: param,
        gc_resp: response,
    })
}

pub fn set_frame4time_sequence(
    snd: &[f64],
    len_win: usize,
    len_shift: Option<usize>,
) -> Result<(Array2<f64>, Array1<usize>)> {
    dsp::frame_sequence(snd, len_win, len_shift.unwrap_or(len_win / 2))
}

pub fn gcfb_v23_synth_snd(gc_smpl: &Array2<f64>, param: &GcParam) -> Result<Array1<f64>> {
    let mean = gc_smpl.mean_axis(Axis(0)).ok_or_else(|| {
        Error::InvalidParameter("cannot synthesize empty filterbank output".into())
    })?;
    if mean.is_empty() {
        return Err(Error::InvalidParameter(
            "cannot synthesize empty filterbank output".into(),
        ));
    }
    if param.out_mid_crct.eq_ignore_ascii_case("no") {
        Ok(mean.mapv(|v| -15. * v))
    } else {
        let (fir, _) = utils::mk_filter_field2cochlea(&param.out_mid_crct, param.fs, false)?;
        // The legacy ELC inverse is linear phase. Pad before shifting so the
        // compensated output includes the FIR tail even for short signals.
        // The other correction filters are minimum phase and need no fixed
        // ELC delay compensation.
        let delay = if param.out_mid_crct.eq_ignore_ascii_case("ELC") {
            fir.len().saturating_sub(1) / 2
        } else {
            0
        };
        let mut padded = mean.to_vec();
        padded.resize(padded.len() + delay, 0.);
        let filtered = dsp::lfilter(fir.as_slice().unwrap(), &[1.], &padded)?;
        let out: Vec<f64> = (0..mean.len())
            .map(|i| -15. * filtered[i + delay])
            .collect();
        Ok(Array1::from(out))
    }
}

#[derive(Clone, Debug)]
pub struct EmParam {
    pub reduce_db: Array1<f64>,
    pub f_cutoff: Array1<f64>,
    pub fc_mod_list: Array1<f64>,
    pub fs: f64,
    pub fb_fr1: Array1<f64>,
    pub fb_reduce_db: Array1<f64>,
    pub fb_f_cutoff: Array1<f64>,
}
impl Default for EmParam {
    fn default() -> Self {
        Self {
            reduce_db: Array1::from_elem(7, 0.),
            f_cutoff: Array1::from_elem(7, 128.),
            fc_mod_list: Array1::from(vec![1., 2., 4., 8., 16., 32., 64., 128., 256.]),
            fs: 0.,
            fb_fr1: Array1::zeros(0),
            fb_reduce_db: Array1::zeros(0),
            fb_f_cutoff: Array1::zeros(0),
        }
    }
}

pub fn gcfb_v23_env_mod_loss(
    frames: &Array2<f64>,
    param: &GcParam,
    mut em: EmParam,
) -> Result<(Array2<f64>, EmParam)> {
    if !param.dyn_hpaf.str_prc.contains("frame") {
        return Err(Error::InvalidParameter(
            "envelope modulation loss requires frame processing".into(),
        ));
    }
    if em.reduce_db.len() != 7 || em.f_cutoff.len() != 7 {
        return Err(Error::InvalidParameter(
            "envelope parameters require seven audiogram values".into(),
        ));
    }
    em.fs = param.dyn_hpaf.fs;
    if !em.fs.is_finite()
        || em.fs <= 0.0
        || em.reduce_db.iter().any(|value| !value.is_finite())
        || em
            .f_cutoff
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= em.fs / 2.0)
        || frames.nrows() == 0
        || frames.ncols() == 0
        || frames.iter().any(|value| !value.is_finite())
        || frames.nrows() != param.num_ch
        || frames.nrows() != param.fr1.len()
    {
        return Err(Error::InvalidParameter(
            "envelope modulation parameters and frames must be finite and non-empty, cutoffs must be below Nyquist, and frame channels must match the prepared filterbank".into(),
        ));
    }
    let (aud, _) = utils::freq2erb(param.hloss.f_audgram_list.as_slice().unwrap());
    let (fb, _) = utils::freq2erb(param.fr1.as_slice().unwrap());
    em.fb_fr1 = param.fr1.clone();
    em.fb_reduce_db = utils::interp1(
        aud.as_slice().unwrap(),
        em.reduce_db.as_slice().unwrap(),
        fb.as_slice().unwrap(),
        true,
    )?;
    em.fb_f_cutoff = utils::interp1(
        aud.as_slice().unwrap(),
        em.f_cutoff.as_slice().unwrap(),
        fb.as_slice().unwrap(),
        true,
    )?;
    if em
        .fb_f_cutoff
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= em.fs / 2.0)
    {
        return Err(Error::InvalidParameter(
            "interpolated envelope cutoffs must be finite and below Nyquist".into(),
        ));
    }
    let mut out = Array2::zeros(frames.dim());
    for ch in 0..frames.nrows() {
        let env = frames.row(ch);
        let dc = (env.iter().map(|v| v * v).sum::<f64>() / env.len() as f64).sqrt();
        let hp: Vec<f64> = env.iter().map(|v| v - dc).collect();
        let (b, a) = dsp::first_order_lowpass(em.fb_f_cutoff[ch], em.fs);
        let filtered = dsp::lfilter(&b, &a, &hp)?;
        let gain = 10_f64.powf(-em.fb_reduce_db[ch] / 20.);
        for i in 0..frames.ncols().saturating_sub(1) {
            out[[ch, i]] = dc + gain * filtered[i + 1];
        }
    }
    Ok((out, em))
}

pub fn gcfb_v23_env_mod_fb(env: &[f64], em: &EmParam) -> Result<Array2<f64>> {
    if !em.fs.is_finite()
        || em.fs <= 0.
        || em.fc_mod_list.iter().any(|frequency| {
            !frequency.is_finite() || *frequency <= 0.0 || *frequency >= em.fs / 2.0
        })
    {
        return Err(Error::InvalidParameter(
            "modulation filterbank frequencies must be finite, positive, and below Nyquist".into(),
        ));
    }
    let mut out = Array2::zeros((em.fc_mod_list.len(), env.len()));
    for (ch, &fc) in em.fc_mod_list.iter().enumerate() {
        if ch == 0 {
            let (b, a) = dsp::third_order_butterworth_lowpass(fc, em.fs);
            out.row_mut(ch)
                .assign(&Array1::from(dsp::lfilter(&b, &a, env)?));
        } else {
            let warp = 2. * std::f64::consts::PI * fc / em.fs;
            let w0 = (warp / 2.).tan();
            let b0 = w0;
            let a0 = 1. + b0 + w0 * w0;
            let b = [b0 / a0, 0., -b0 / a0];
            let a = [1., (2. * w0 * w0 - 2.) / a0, (1. - b0 + w0 * w0) / a0];
            out.row_mut(ch)
                .assign(&Array1::from(dsp::lfilter(&b, &a, env)?));
        }
    }
    Ok(out)
}

pub fn gcfb_v23_ana_env_mod(
    frames: &Array2<f64>,
    param: &GcParam,
    mut em: EmParam,
) -> Result<(Array3<f64>, EmParam)> {
    if !param.dyn_hpaf.str_prc.contains("frame") {
        return Err(Error::InvalidParameter(
            "modulation analysis requires frame processing".into(),
        ));
    }
    if frames.nrows() == 0
        || frames.ncols() == 0
        || frames.iter().any(|value| !value.is_finite())
        || frames.nrows() != param.num_ch
        || frames.nrows() != param.fr1.len()
    {
        return Err(Error::InvalidParameter(
            "modulation analysis requires finite, non-empty frames whose channels match the prepared filterbank"
                .into(),
        ));
    }
    if em.fs == 0. {
        em.fs = param.dyn_hpaf.fs;
    }
    let mut out = Array3::zeros((frames.nrows(), em.fc_mod_list.len(), frames.ncols()));
    for ch in 0..frames.nrows() {
        let bank = gcfb_v23_env_mod_fb(frames.row(ch).as_slice().unwrap(), &em)?;
        out.slice_mut(s![ch, .., ..]).assign(&bank);
    }
    Ok((out, em))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcfb_v234::stream::{DcgcEvent, GcfbStream};
    use approx::assert_relative_eq;

    fn value_only_cascade_log_power(
        impulse: &[f64],
        coefficients: &AcfCoef,
        frequency_hz: f64,
        sample_rate: f64,
    ) -> f64 {
        let omega = std::f64::consts::TAU * frequency_hz / sample_rate;
        let fir = impulse.iter().enumerate().fold(
            Complex64::new(0.0, 0.0),
            |sum, (sample, &coefficient)| {
                sum + coefficient * Complex64::from_polar(1.0, -omega * sample as f64)
            },
        );
        let z1 = Complex64::from_polar(1.0, -omega);
        let z2 = z1 * z1;
        let mut log_power = fir.norm_sqr().ln();
        for section in 0..4 {
            let numerator = coefficients.bz[[0, 0, section]]
                + coefficients.bz[[0, 1, section]] * z1
                + coefficients.bz[[0, 2, section]] * z2;
            let denominator = coefficients.ap[[0, 0, section]]
                + coefficients.ap[[0, 1, section]] * z1
                + coefficients.ap[[0, 2, section]] * z2;
            log_power += numerator.norm_sqr().ln() - denominator.norm_sqr().ln();
        }
        log_power
    }

    #[test]
    fn continuous_dtft_derivatives_match_centered_finite_differences() {
        let sample_rate = 48_000.0;
        let impulse = passive_impulse(1_200.0, sample_rate, 4.0, 1.81, -2.96).unwrap();
        let coefficients =
            make_asym_cmp_filters_v2(sample_rate, &[1_300.0], &[2.17], &[2.2]).unwrap();
        let frequency = 1_050.25;
        let step = 1e-3;

        let fir = fir_log_power_and_derivative(&impulse, frequency, sample_rate).unwrap();
        let fir_difference =
            (fir_log_power_and_derivative(&impulse, frequency + step, sample_rate)
                .unwrap()
                .0
                - fir_log_power_and_derivative(&impulse, frequency - step, sample_rate)
                    .unwrap()
                    .0)
                / (2.0 * step);
        assert_relative_eq!(fir.1, fir_difference, epsilon = 2e-9);

        let hpaf = hpaf_log_power_and_derivative(&coefficients, frequency, sample_rate).unwrap();
        let hpaf_difference =
            (hpaf_log_power_and_derivative(&coefficients, frequency + step, sample_rate)
                .unwrap()
                .0
                - hpaf_log_power_and_derivative(&coefficients, frequency - step, sample_rate)
                    .unwrap()
                    .0)
                / (2.0 * step);
        assert_relative_eq!(hpaf.1, hpaf_difference, epsilon = 2e-9);
    }

    #[test]
    fn continuous_peak_matches_value_only_search_and_amplitude() {
        let sample_rate = 48_000.0;
        let carrier = 1_200.0;
        let impulse = passive_impulse(carrier, sample_rate, 4.0, 1.81, -2.96).unwrap();
        let fp1 = passive_peak_frequency(carrier, 4.0, 1.81, -2.96);
        let ratio = 1.08;
        let coefficients =
            make_asym_cmp_filters_v2(sample_rate, &[ratio * fp1], &[2.17], &[2.2]).unwrap();
        let analytic_peak = fr1_to_fp2(4.0, 1.81, -2.96, 2.17, 2.2, ratio, carrier)
            .unwrap()
            .0;
        let peak = continuous_peak_from_impulse(
            &impulse,
            &coefficients,
            analytic_peak,
            None,
            sample_rate,
            MINIMUM_PEAK_GRID_FFT_LEN,
        )
        .unwrap();

        let spacing = sample_rate / MINIMUM_PEAK_GRID_FFT_LEN as f64;
        let mut lower = peak.frequency_hz - spacing;
        let mut upper = peak.frequency_hz + spacing;
        let golden = (5.0_f64.sqrt() - 1.0) * 0.5;
        let mut left = upper - golden * (upper - lower);
        let mut right = lower + golden * (upper - lower);
        let mut left_value =
            value_only_cascade_log_power(&impulse, &coefficients, left, sample_rate);
        let mut right_value =
            value_only_cascade_log_power(&impulse, &coefficients, right, sample_rate);
        for _ in 0..128 {
            if left_value < right_value {
                lower = left;
                left = right;
                left_value = right_value;
                right = lower + golden * (upper - lower);
                right_value =
                    value_only_cascade_log_power(&impulse, &coefficients, right, sample_rate);
            } else {
                upper = right;
                right = left;
                right_value = left_value;
                left = upper - golden * (upper - lower);
                left_value =
                    value_only_cascade_log_power(&impulse, &coefficients, left, sample_rate);
            }
        }
        let searched_peak = lower + (upper - lower) * 0.5;
        assert!(
            (peak.frequency_hz - searched_peak).abs() < 1e-5,
            "derivative peak {} Hz versus value-only peak {} Hz",
            peak.frequency_hz,
            searched_peak
        );
        let value_only =
            value_only_cascade_log_power(&impulse, &coefficients, peak.frequency_hz, sample_rate);
        assert_relative_eq!(peak.log_power, value_only, epsilon = 2e-12);
        let amplitude = (0.5 * peak.log_power).exp();
        assert_relative_eq!(amplitude * amplitude, value_only.exp(), epsilon = 2e-12);
    }

    #[test]
    fn continuous_root_rejects_an_unreachable_frequency() {
        let error = continuous_frequency_root(1_000.0, 1_200.0, 48_000.0, "test parameter", |_| {
            Ok(1_000.0)
        })
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn nh_parameters_match_expected_health() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 10,
            ..GcParam::default()
        };
        let (p, r) = set_param(p).unwrap();
        assert!(
            p.hloss
                .fb_compression_health
                .iter()
                .all(|v| (*v - 1.).abs() < 1e-12)
        );
        assert_relative_eq!(r.fr1[0], 100., epsilon = 1e-9);
    }

    #[test]
    fn peak_lock_rejects_a_target_outside_the_physical_frequency_range() {
        let error = center_for_composite_peak(
            4.0, 1.81, -2.96, 2.17, 2.2, 1_000.0, 900.0, 4_000.0, 8_000.0, 900.0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn peak_grid_expands_to_contain_long_passive_impulses() {
        let parameters = GcParam {
            num_ch: 2,
            f_range: [100.0, 200.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            b1: [0.02, 0.0],
            ..GcParam::default()
        };
        let (reference_param, reference_response) = set_param(parameters.clone()).unwrap();
        let grid = prepare_bandwidth_peak_grid(
            &parameters,
            &[0.8, 1.0],
            &reference_param,
            &reference_response,
        )
        .unwrap();
        assert!(grid.fft_len() > MINIMUM_PEAK_GRID_FFT_LEN);
        assert!(grid.fft_len().is_power_of_two());
        let longest_reference = prepare_passive_impulses(&reference_param, &reference_response)
            .unwrap()
            .into_iter()
            .map(|impulse| impulse.len())
            .max()
            .unwrap();
        assert!(grid.fft_len() >= longest_reference);
    }

    #[test]
    fn peak_locked_dynamic_batch_and_stream_outputs_match() {
        let parameters = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [300.0, 1_800.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Dynamic,
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            num_update_asym_cmp: 3,
            ..GcParam::default()
        };
        let signal: Vec<f64> = (0..192)
            .map(|sample| {
                let amplitude = 0.02 + 0.48 * sample as f64 / 191.0;
                amplitude * (2.0 * std::f64::consts::PI * 750.0 * sample as f64 / 8_000.0).cos()
            })
            .collect();
        let baseline = gcfb_v234(&signal, parameters.clone()).unwrap();
        let grid = prepare_bandwidth_peak_grid(
            &parameters,
            &[1.0, 1.2],
            &baseline.gc_param,
            &baseline.gc_resp,
        )
        .unwrap();
        let scaled_parameters = scale_bandwidths(parameters, 1.2);
        let batch = gcfb_v234_with_bandwidth_peak_lock(
            &signal,
            scaled_parameters.clone(),
            &baseline.gc_param.hloss,
            grid.clone(),
        )
        .unwrap();
        let mut stream = GcfbStream::new_with_bandwidth_peak_lock(
            scaled_parameters,
            &baseline.gc_param.hloss,
            grid,
        )
        .unwrap();
        for (sample_index, &sample) in signal.iter().enumerate() {
            let step = stream.process_sample(sample).unwrap();
            for ch in 0..batch.gc_param.num_ch {
                assert_relative_eq!(
                    step.scgc_smpl[ch],
                    batch.scgc_smpl[[ch, sample_index]],
                    epsilon = 2e-11
                );
            }
            let Some(DcgcEvent::Sample {
                dcgc_out,
                fr2: Some(centers),
                ..
            }) = step.event
            else {
                panic!("sample-dynamic peak-locked stream must emit a sample event");
            };
            for ch in 0..batch.gc_param.num_ch {
                assert_relative_eq!(
                    dcgc_out[ch],
                    batch.dcgc_out[[ch, sample_index]],
                    epsilon = 2e-11
                );
                assert_relative_eq!(
                    centers[ch],
                    batch.gc_resp.fr2[[ch, sample_index]],
                    epsilon = peak_lock_tolerance_hz(batch.gc_param.fs)
                );
            }
        }
    }

    #[test]
    fn frame_output_shape() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200., 2000.],
            ..GcParam::default()
        };
        let out = gcfb_v234(&[1., 0., 0., 0., 0., 0., 0., 0.], p).unwrap();
        assert_eq!(out.dcgc_out.nrows(), 4);
        assert_eq!(out.dcgc_out.ncols(), 1);
        assert!(out.dcgc_out.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn hl3_audiogram_is_split_into_active_and_passive_loss() {
        let p = GcParam {
            out_mid_crct: "No".into(),
            num_ch: 10,
            hloss_type: "HL3".into(),
            hloss_compression_health: Some(0.5),
            ..GcParam::default()
        };
        let (p, _) = set_param(p).unwrap();
        for i in 0..p.hloss.hearing_level_db.len() {
            assert_relative_eq!(
                p.hloss.pin_loss_db_act[i] + p.hloss.pin_loss_db_pas[i],
                p.hloss.hearing_level_db[i],
                epsilon = 0.35
            );
        }
        assert!(
            p.hloss
                .fb_compression_health
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        );
    }

    #[test]
    fn sample_histories_are_channel_major_and_update_cadence_changes_filtering() {
        let signal: Vec<f64> = (0..96)
            .map(|sample| {
                let envelope = match sample {
                    0..=23 => 0.02,
                    24..=47 => 1.0,
                    48..=71 => 0.12,
                    _ => 0.55,
                };
                let transient = match sample {
                    8 => 1.0,
                    56 => -0.7,
                    _ => 0.0,
                };
                envelope * (2.0 * std::f64::consts::PI * 700.0 * sample as f64 / 8_000.0).sin()
                    + transient
            })
            .collect();
        let run = |num_update_asym_cmp| {
            gcfb_v234(
                &signal,
                GcParam {
                    fs: 8_000.0,
                    out_mid_crct: "No".into(),
                    num_ch: 4,
                    f_range: [200.0, 2_000.0],
                    num_update_asym_cmp,
                    dyn_hpaf: DynHpaf {
                        str_prc: "sample-base".into(),
                        ..DynHpaf::default()
                    },
                    ..GcParam::default()
                },
            )
            .unwrap()
        };
        let every_sample = run(1);
        let every_eighth_sample = run(8);

        for output in [&every_sample, &every_eighth_sample] {
            assert_eq!(output.dcgc_out.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.fr2.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.frat_val.dim(), (4, signal.len()));
            assert_eq!(output.gc_resp.lvl_db.dim(), (4, signal.len()));
            assert!(output.dcgc_out.iter().all(|value| value.is_finite()));
            for history in [
                &output.gc_resp.fr2,
                &output.gc_resp.frat_val,
                &output.gc_resp.lvl_db,
            ] {
                assert!(history.rows().into_iter().any(|row| {
                    row.windows(2)
                        .into_iter()
                        .any(|pair| (pair[1] - pair[0]).abs() > 1e-12)
                }));
            }
            for ch in 0..4 {
                for sample in 0..signal.len() {
                    assert_relative_eq!(
                        output.gc_resp.fr2[[ch, sample]],
                        output.gc_resp.fp1[ch] * output.gc_resp.frat_val[[ch, sample]],
                        epsilon = 1e-12
                    );
                }
            }
        }

        assert_eq!(
            every_sample.gc_resp.lvl_db,
            every_eighth_sample.gc_resp.lvl_db
        );
        assert_eq!(
            every_sample.gc_resp.frat_val,
            every_eighth_sample.gc_resp.frat_val
        );
        assert_eq!(every_sample.gc_resp.fr2, every_eighth_sample.gc_resp.fr2);
        assert!(
            every_sample
                .dcgc_out
                .iter()
                .zip(&every_eighth_sample.dcgc_out)
                .any(|(left, right)| (left - right).abs() > 1e-12)
        );
    }

    #[test]
    fn static_processing_populates_response_metadata() {
        let p = GcParam {
            fs: 8000.0,
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200.0, 1500.0],
            ctrl: ControlMode::Static,
            ..GcParam::default()
        };
        let out = gcfb_v234(&[1.0, 0.0, 0.0, 0.0], p).unwrap();

        assert_eq!(out.gc_resp.fr2.dim(), (4, 1));
        assert_eq!(out.gc_resp.frat_val.dim(), (4, 1));
        assert_eq!(out.gc_resp.lvl_db.dim(), (4, 1));
        assert!(out.gc_resp.fp2.iter().all(|value| *value > 0.0));
        for ch in 0..4 {
            assert_relative_eq!(
                out.gc_resp.fr2[[ch, 0]],
                out.gc_resp.frat_val[[ch, 0]] * out.gc_resp.fp1[ch],
                epsilon = 1e-10
            );
            assert_relative_eq!(out.gc_resp.lvl_db[[ch, 0]], 50.0);
        }
    }

    #[test]
    fn invalid_dynamic_processing_and_window_names_are_rejected() {
        let invalid_processing = GcParam {
            dyn_hpaf: DynHpaf {
                str_prc: "fram-base".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        };
        assert!(set_param(invalid_processing).is_err());

        let invalid_window = GcParam {
            dyn_hpaf: DynHpaf {
                name_win: "blackman".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        };
        assert!(set_param(invalid_window).is_err());
    }

    #[test]
    fn v234_channel_range_is_strictly_sub_nyquist() {
        let nyquist = 4_000.0_f64;
        let below_nyquist = f64::from_bits(nyquist.to_bits() - 1);
        let above_nyquist = f64::from_bits(nyquist.to_bits() + 1);

        let (_, response) = set_param(GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [100.0, below_nyquist],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        })
        .unwrap();
        assert_eq!(response.fr1[3], below_nyquist);

        for high in [nyquist, above_nyquist] {
            let error = set_param(GcParam {
                fs: 8_000.0,
                num_ch: 4,
                f_range: [100.0, high],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            })
            .unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message.contains("0 < low < high < fs / 2")
            ));
        }
    }

    #[test]
    fn v234_processes_a_range_above_the_former_fs_over_three_limit() {
        let output = gcfb_v234(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            GcParam {
                fs: 8_000.0,
                num_ch: 4,
                f_range: [2_800.0, 3_200.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
        )
        .unwrap();

        assert_eq!(output.scgc_smpl.dim(), (4, 8));
        assert!(output.scgc_smpl.iter().all(|value| value.is_finite()));
        assert!(output.dcgc_out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn v234_rejects_invalid_mode_specific_initial_centers_during_preparation() {
        let static_param = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [1_000.0, 2_000.0],
            out_mid_crct: "No".into(),
            ctrl: ControlMode::Static,
            frat: [[10.0, 0.0], [0.0109, 0.0]],
            ..GcParam::default()
        };
        let dynamic_param = GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [1_000.0, 2_000.0],
            out_mid_crct: "No".into(),
            lvl_est: LvlEst {
                frat: 10.0,
                ..LvlEst::default()
            },
            ..GcParam::default()
        };

        for param in [static_param, dynamic_param] {
            let error = set_param(param).unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message == "initial asymmetric filter centers must be finite, positive, and below Nyquist"
            ));
        }
    }

    #[test]
    fn v234_dynamic_centers_remain_runtime_validated() {
        let (param, mut response) = set_param(GcParam {
            fs: 8_000.0,
            num_ch: 4,
            f_range: [200.0, 1_500.0],
            out_mid_crct: "No".into(),
            dyn_hpaf: DynHpaf {
                str_prc: "sample-base".into(),
                ..DynHpaf::default()
            },
            ..GcParam::default()
        })
        .unwrap();
        response.frat0_pc.fill(100.0);
        let pgc = Array2::ones((4, 1));
        let scgc = Array2::ones((4, 1));

        assert!(gcfb_v23_sample_base(&pgc, &scgc, &param, &mut response).is_err());
    }

    #[test]
    fn v234_validates_derived_filter_centers_without_a_fixed_frequency_floor() {
        for param in [
            GcParam {
                num_ch: 4,
                f_range: [20.0, 1600.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
            GcParam {
                c1: [f64::NAN, 0.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
            GcParam {
                c1: [1000.0, 0.0],
                out_mid_crct: "No".into(),
                ..GcParam::default()
            },
        ] {
            let error = set_param(param).unwrap_err();
            assert!(matches!(
                error,
                Error::InvalidParameter(message)
                    if message == "derived filter centers must be finite, positive, and below Nyquist"
            ));
        }

        let valid = GcParam {
            num_ch: 4,
            f_range: [39.0, 1600.0],
            out_mid_crct: "No".into(),
            ..GcParam::default()
        };
        assert!(set_param(valid).is_ok());
    }

    #[test]
    fn modulation_filterbank_rejects_degenerate_or_aliased_frequencies() {
        for frequency in [0.0, 1000.0, 1500.0, f64::NAN] {
            let em = EmParam {
                fs: 2000.0,
                fc_mod_list: Array1::from(vec![frequency]),
                ..EmParam::default()
            };
            assert!(gcfb_v23_env_mod_fb(&[1.0, 0.0, 0.0], &em).is_err());
        }
    }

    #[test]
    fn hearing_pattern_rejects_ambiguous_names_and_non_finite_levels() {
        assert!(hearing_pattern("HL10", None).is_err());
        assert!(hearing_pattern("NHL3", None).is_err());
        assert!(hearing_pattern("HL3_ISO7029_70yr_male", None).is_ok());

        let manual = Array1::from(vec![0., 0., 0., f64::NAN, 0., 0., 0.]);
        assert!(hearing_pattern("HL0", Some(&manual)).is_err());
    }

    #[test]
    fn dynamic_processing_rejects_empty_or_mismatched_matrices() {
        let p = GcParam {
            fs: 8_000.,
            out_mid_crct: "No".into(),
            num_ch: 4,
            f_range: [200., 2_000.],
            ..GcParam::default()
        };
        let (param, mut response) = set_param(p).unwrap();

        assert!(
            gcfb_v23_frame_base(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                &param,
                &mut response,
            )
            .is_err()
        );
        assert!(
            gcfb_v23_frame_base(
                &Array2::zeros((4, 8)),
                &Array2::zeros((3, 8)),
                &param,
                &mut response,
            )
            .is_err()
        );
        assert!(
            gcfb_v23_sample_base(
                &Array2::zeros((4, 8)),
                &Array2::zeros((3, 8)),
                &param,
                &mut response,
            )
            .is_err()
        );
    }

    #[test]
    fn corrected_synthesis_preserves_short_signals() {
        let input = Array2::from_shape_vec((1, 8), vec![1., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        for correction in ["ELC", "EarDrum"] {
            let param = GcParam {
                fs: 8_000.,
                out_mid_crct: correction.into(),
                ..GcParam::default()
            };
            let output = gcfb_v23_synth_snd(&input, &param).unwrap();
            assert_eq!(output.len(), input.ncols());
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| value.abs() > 1e-12));
        }
    }
}
