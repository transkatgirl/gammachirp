//! Bounded-memory sample and centered-frame streaming for GCFB v2.34.

use std::collections::VecDeque;
use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::gcfb_v234::{
    AcfCoef, AcfStatus, BandwidthPeakGrid, BandwidthPeakLock, ControlMode, GcParam, GcResp, HLoss,
    asym_func_in_out_scalar, cmprs_gc_frsp, initial_asymmetric_ratio_and_centers,
    make_asym_cmp_filters_v2, prepare_bandwidth_peak_lock, prepare_input_correction_fir,
    prepare_passive_impulses, prepare_time_invariant_response, set_param,
    set_param_with_preserved_hearing_loss,
};
use crate::{Error, Result, dsp};

/// A normalized dcGC result made available by one streaming step.
#[derive(Clone, Debug)]
pub enum DcgcEvent {
    /// A sample-domain result for static, level, or dynamic sample-base control.
    Sample {
        sample_index: usize,
        dcgc_out: Array1<f64>,
        /// Dynamic level history; absent for static and level control.
        lvl_db: Option<Array1<f64>>,
        /// Dynamic frequency-ratio history; absent for static and level control.
        frat_val: Option<Array1<f64>>,
        /// Dynamic asymmetric centers; absent for static and level control.
        fr2: Option<Array1<f64>>,
    },
    /// A centered dynamic frame. The result already includes dcGC gain
    /// normalization selected by [`super::gcfb_v234::GainReference`].
    Frame {
        frame_index: usize,
        center_index: usize,
        dcgc_out: Array1<f64>,
        lvl_db: Array1<f64>,
        pgc_frame: Array1<f64>,
        scgc_frame: Array1<f64>,
        asym_func_gain: Array1<f64>,
    },
}

/// The immediate fixed-path output and any dcGC event released by one sample.
#[derive(Clone, Debug)]
pub struct StreamStep {
    pub sample_index: usize,
    pub scgc_smpl: Array1<f64>,
    pub event: Option<DcgcEvent>,
}

#[derive(Clone, Debug)]
struct FrameInput {
    pgc: Array1<f64>,
    scgc: Array1<f64>,
}

/// A bounded-memory GCFB v2.34 processor.
///
/// Dynamic frame mode delays each event until its centered window has enough
/// right-hand input. Call [`finish`](Self::finish) to emit zero-padded trailing
/// frames after the final sample.
#[derive(Clone, Debug)]
pub struct GcfbStream {
    param: GcParam,
    response: GcResp,
    correction: dsp::CausalFir,
    passive: dsp::CausalFirBank,
    fixed_coefficients: AcfCoef,
    fixed_status: AcfStatus,
    dynamic_coefficients: Option<AcfCoef>,
    dynamic_status: Option<AcfStatus>,
    peak_lock: Option<BandwidthPeakLock>,
    previous_sample_level: Array2<f64>,
    frame_normalization: Option<Array1<f64>>,
    previous_frame_level: Array2<f64>,
    frame_buffer: VecDeque<FrameInput>,
    frame_buffer_start: usize,
    next_frame_index: usize,
    samples_processed: usize,
    failed: bool,
}

impl GcfbStream {
    /// Prepare a stream using the batch implementation's derived parameters,
    /// hearing-loss model, fixed filters, and gain reference.
    pub fn new(gc_param: GcParam) -> Result<Self> {
        Self::new_internal(gc_param, None, None)
    }

    pub(crate) fn new_with_preserved_hearing_loss(
        gc_param: GcParam,
        hearing_loss: &HLoss,
    ) -> Result<Self> {
        Self::new_internal(gc_param, Some(hearing_loss), None)
    }

    pub(crate) fn new_with_bandwidth_peak_lock(
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
        let (param, mut response) = if let Some(hearing_loss) = hearing_loss {
            set_param_with_preserved_hearing_loss(gc_param, Some(hearing_loss))?
        } else {
            set_param(gc_param)?
        };
        let mut peak_lock = if let Some(peak_grid) = peak_grid {
            Some(prepare_bandwidth_peak_lock(
                &param,
                &mut response,
                peak_grid,
            )?)
        } else {
            None
        };
        let frame_mode = param.dyn_hpaf.str_prc.contains("frame");
        if param.ctrl == ControlMode::Dynamic
            && frame_mode
            && (!param.dyn_hpaf.len_frame.is_multiple_of(2)
                || !param
                    .dyn_hpaf
                    .len_frame
                    .is_multiple_of(param.dyn_hpaf.len_shift))
        {
            return Err(Error::InvalidParameter(
                "frame length must be positive and even; shift must be a positive divisor of it"
                    .into(),
            ));
        }
        let correction = dsp::CausalFir::new(prepare_input_correction_fir(&param)?);
        let passive = dsp::CausalFirBank::new(prepare_passive_impulses(&param, &response)?);
        let fixed_coefficients = prepare_time_invariant_response(&param, &mut response)?;
        let fixed_status = AcfStatus::new(&fixed_coefficients);

        let sample_dynamic =
            param.ctrl == ControlMode::Dynamic && param.dyn_hpaf.str_prc.contains("sample");
        let (dynamic_coefficients, dynamic_status) = if sample_dynamic {
            let (ratios, default_centers) = initial_asymmetric_ratio_and_centers(&param, &response);
            let centers = if let Some(lock) = peak_lock.as_mut() {
                lock.centers_for_reference_peaks_at_ratios(&ratios)?
            } else {
                default_centers
            };
            let coefficients = make_asym_cmp_filters_v2(
                param.fs,
                centers.as_slice().unwrap(),
                response.b2_val.as_slice().unwrap(),
                response.c2_val.as_slice().unwrap(),
            )?;
            let status = AcfStatus::new(&coefficients);
            (Some(coefficients), Some(status))
        } else {
            (None, None)
        };
        let frame_normalization = if param.ctrl == ControlMode::Dynamic && frame_mode {
            let c2 = &param.hloss.fb_compression_health * param.lvl_est.c2;
            Some(
                cmprs_gc_frsp(
                    response.fr1.as_slice().unwrap(),
                    param.fs,
                    param.n,
                    response.b1_val.as_slice().unwrap(),
                    response.c1_val.as_slice().unwrap(),
                    &[param.lvl_est.frat],
                    &[param.lvl_est.b2],
                    c2.as_slice().unwrap(),
                    2048,
                )?
                .norm_fct_fp2,
            )
        } else {
            None
        };
        let channels = param.num_ch;
        Ok(Self {
            param,
            response,
            correction,
            passive,
            fixed_coefficients,
            fixed_status,
            dynamic_coefficients,
            dynamic_status,
            peak_lock,
            previous_sample_level: Array2::zeros((channels, 2)),
            frame_normalization,
            previous_frame_level: Array2::zeros((channels, 2)),
            frame_buffer: VecDeque::new(),
            frame_buffer_start: 0,
            next_frame_index: 0,
            samples_processed: 0,
            failed: false,
        })
    }

    /// Process one finite input sample.
    ///
    /// A processing error after input validation makes the stream terminal,
    /// because causal filter history may already have advanced. Construct a
    /// new processor before submitting more samples after such an error.
    pub fn process_sample(&mut self, sample: f64) -> Result<StreamStep> {
        if self.failed {
            return Err(Error::Numerical(
                "v2.34 stream cannot continue after a previous processing error".into(),
            ));
        }
        // Reject invalid values before any FIR, IIR, counter, or frame state is
        // advanced, so callers can recover and submit the next valid sample.
        if !sample.is_finite() {
            return Err(Error::InvalidParameter(
                "input sound samples must be finite".into(),
            ));
        }
        match self.process_finite_sample(sample) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn process_finite_sample(&mut self, sample: f64) -> Result<StreamStep> {
        let sample_index = self.samples_processed;
        let corrected = self.correction.process_sample(sample);
        let pgc = self.passive.process_sample(corrected);
        let scgc =
            self.fixed_status
                .process(&self.fixed_coefficients, pgc.as_slice().unwrap(), false)?;

        let event = match self.param.ctrl {
            ControlMode::Static | ControlMode::Level => {
                let dcgc_out = &scgc * &self.response.gain_factor;
                Some(DcgcEvent::Sample {
                    sample_index,
                    dcgc_out,
                    lvl_db: None,
                    frat_val: None,
                    fr2: None,
                })
            }
            ControlMode::Dynamic if self.param.dyn_hpaf.str_prc.contains("sample") => {
                Some(self.process_dynamic_sample(sample_index, &pgc, &scgc)?)
            }
            ControlMode::Dynamic if self.param.dyn_hpaf.str_prc.contains("frame") => {
                if self.frame_buffer.is_empty() {
                    self.frame_buffer_start = sample_index;
                }
                self.frame_buffer.push_back(FrameInput {
                    pgc,
                    scgc: scgc.clone(),
                });
                if self.frame_is_ready(sample_index) {
                    Some(self.emit_next_frame())
                } else {
                    None
                }
            }
            ControlMode::Dynamic => unreachable!("processing mode is validated by set_param"),
        };
        self.samples_processed += 1;
        Ok(StreamStep {
            sample_index,
            scgc_smpl: scgc,
            event,
        })
    }

    fn process_dynamic_sample(
        &mut self,
        sample_index: usize,
        pgc: &Array1<f64>,
        scgc: &Array1<f64>,
    ) -> Result<DcgcEvent> {
        let channels = self.param.num_ch;
        let mut levels = Array1::zeros(channels);
        let mut ratios = Array1::zeros(channels);
        let mut centers = Array1::zeros(channels);
        for ch in 0..channels {
            let source = self.param.lvl_est.n_ch_lvl_est[ch];
            let passive_level = pgc[source]
                .max(0.0)
                .max(self.previous_sample_level[[ch, 0]] * self.param.lvl_est.exp_decay_val);
            let fixed_level = scgc[source]
                .max(0.0)
                .max(self.previous_sample_level[[ch, 1]] * self.param.lvl_est.exp_decay_val);
            self.previous_sample_level[[ch, 0]] = passive_level;
            self.previous_sample_level[[ch, 1]] = fixed_level;
            let total = self.param.lvl_est.weight
                * self.param.lvl_est.lvl_lin_ref
                * (passive_level / self.param.lvl_est.lvl_lin_ref).powf(self.param.lvl_est.pwr[0])
                + (1.0 - self.param.lvl_est.weight)
                    * self.param.lvl_est.lvl_lin_ref
                    * (fixed_level / self.param.lvl_est.lvl_lin_ref)
                        .powf(self.param.lvl_est.pwr[1]);
            levels[ch] = 20.0 * total.max(self.param.lvl_est.lvl_lin_min_lim).log10()
                + self.param.lvl_est.rms2spldb;
            ratios[ch] = self.response.frat0_pc[ch]
                + self.param.hloss.fb_compression_health[ch]
                    * self.response.frat1_val[ch]
                    * (levels[ch] - self.response.pc_hpaf[ch]);
            centers[ch] = self.response.fp1[ch] * ratios[ch];
        }
        if sample_index.is_multiple_of(self.param.num_update_asym_cmp) {
            if let Some(lock) = self.peak_lock.as_mut() {
                centers = lock.centers_for_reference_peaks_at_ratios(&ratios)?;
            }
            self.dynamic_coefficients = Some(make_asym_cmp_filters_v2(
                self.param.fs,
                centers.as_slice().unwrap(),
                self.response.b2_val.as_slice().unwrap(),
                self.response.c2_val.as_slice().unwrap(),
            )?);
        } else if let Some(lock) = self.peak_lock.as_ref() {
            centers.assign(lock.current_centers());
        }
        let mut dcgc_out = self.dynamic_status.as_mut().unwrap().process(
            self.dynamic_coefficients.as_ref().unwrap(),
            pgc.as_slice().unwrap(),
            false,
        )?;
        dcgc_out *= &self.response.gain_factor;
        Ok(DcgcEvent::Sample {
            sample_index,
            dcgc_out,
            lvl_db: Some(levels),
            frat_val: Some(ratios),
            fr2: Some(centers),
        })
    }

    fn frame_is_ready(&self, latest_sample: usize) -> bool {
        let center = self.next_frame_index * self.param.dyn_hpaf.len_shift;
        latest_sample >= center + self.param.dyn_hpaf.len_frame / 2 - 1
    }

    fn buffered_value(&self, sample: isize, channel: usize, passive: bool) -> f64 {
        if sample < 0 || sample as usize > self.samples_processed {
            return 0.0;
        }
        let index = sample as usize;
        if index < self.frame_buffer_start {
            return 0.0;
        }
        self.frame_buffer
            .get(index - self.frame_buffer_start)
            .map_or(0.0, |input| {
                if passive {
                    input.pgc[channel]
                } else {
                    input.scgc[channel]
                }
            })
    }

    fn weighted_frame_value(&self, center: usize, channel: usize, passive: bool) -> f64 {
        let half = self.param.dyn_hpaf.len_frame / 2;
        self.param
            .dyn_hpaf
            .val_win
            .iter()
            .enumerate()
            .map(|(offset, weight)| {
                let index = center as isize + offset as isize - half as isize;
                weight * self.buffered_value(index, channel, passive).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    fn emit_next_frame(&mut self) -> DcgcEvent {
        let frame_index = self.next_frame_index;
        let center_index = frame_index * self.param.dyn_hpaf.len_shift;
        let channels = self.param.num_ch;
        let mut pgc_frame = Array1::zeros(channels);
        let mut scgc_frame = Array1::zeros(channels);
        let mut source_pgc = Array1::zeros(channels);
        let mut source_scgc = Array1::zeros(channels);
        for ch in 0..channels {
            pgc_frame[ch] = self.weighted_frame_value(center_index, ch, true);
            scgc_frame[ch] = self.weighted_frame_value(center_index, ch, false);
            let source = self.param.lvl_est.n_ch_lvl_est[ch];
            source_pgc[ch] = self.weighted_frame_value(center_index, source, true);
            source_scgc[ch] = self.weighted_frame_value(center_index, source, false);
        }
        let decay = self
            .param
            .lvl_est
            .exp_decay_val
            .powf(self.param.dyn_hpaf.len_shift as f64);
        let mut levels = Array1::zeros(channels);
        let mut asym_func_gain = Array1::zeros(channels);
        let mut dcgc_out = Array1::zeros(channels);
        for ch in 0..channels {
            let passive_level = source_pgc[ch].max(self.previous_frame_level[[ch, 0]] * decay);
            let fixed_level = source_scgc[ch].max(self.previous_frame_level[[ch, 1]] * decay);
            self.previous_frame_level[[ch, 0]] = passive_level;
            self.previous_frame_level[[ch, 1]] = fixed_level;
            let total = self.param.lvl_est.weight
                * self.param.lvl_est.lvl_lin_ref
                * (passive_level / self.param.lvl_est.lvl_lin_ref).powf(self.param.lvl_est.pwr[0])
                + (1.0 - self.param.lvl_est.weight)
                    * self.param.lvl_est.lvl_lin_ref
                    * (fixed_level / self.param.lvl_est.lvl_lin_ref)
                        .powf(self.param.lvl_est.pwr[1]);
            levels[ch] = 20.0 * total.max(self.param.lvl_est.lvl_lin_min_lim).log10()
                + self.param.lvl_est.rms2spldb
                - 3.0;
            let (gain_db, _) = asym_func_in_out_scalar(
                &self.param,
                &self.response,
                self.param.fr1[ch],
                self.param.hloss.fb_compression_health[ch],
                levels[ch],
            );
            asym_func_gain[ch] = 10_f64.powf(gain_db / 20.0);
            dcgc_out[ch] = asym_func_gain[ch]
                * self.frame_normalization.as_ref().unwrap()[ch]
                * scgc_frame[ch]
                * self.response.gain_factor[ch];
        }

        self.next_frame_index += 1;
        let next_center = self.next_frame_index * self.param.dyn_hpaf.len_shift;
        let keep_from = next_center.saturating_sub(self.param.dyn_hpaf.len_frame / 2);
        while self.frame_buffer_start < keep_from && self.frame_buffer.pop_front().is_some() {
            self.frame_buffer_start += 1;
        }
        DcgcEvent::Frame {
            frame_index,
            center_index,
            dcgc_out,
            lvl_db: levels,
            pgc_frame,
            scgc_frame,
            asym_func_gain,
        }
    }

    /// Consume the stream and emit all centered frames whose right-hand side
    /// must be zero padded. For `N` samples, frame mode returns enough events
    /// to bring the total to `N / len_shift + 1`.
    pub fn finish(mut self) -> Result<Vec<DcgcEvent>> {
        if self.failed {
            return Err(Error::Numerical(
                "v2.34 stream cannot finish after a processing error".into(),
            ));
        }
        if self.samples_processed == 0 {
            return Err(Error::InvalidParameter(
                "input sound must be non-empty and finite".into(),
            ));
        }
        let mut events = Vec::new();
        if self.param.ctrl == ControlMode::Dynamic && self.param.dyn_hpaf.str_prc.contains("frame")
        {
            let total_frames = self.samples_processed / self.param.dyn_hpaf.len_shift + 1;
            while self.next_frame_index < total_frames {
                events.push(self.emit_next_frame());
            }
        }
        Ok(events)
    }

    /// Prepared parameters, including derived frame and hearing-loss fields.
    pub fn gc_param(&self) -> &GcParam {
        &self.param
    }

    /// Prepared, time-invariant response metadata and gain references.
    pub fn gc_resp(&self) -> &GcResp {
        &self.response
    }

    pub fn prepared_param(&self) -> &GcParam {
        self.gc_param()
    }

    pub fn prepared_response(&self) -> &GcResp {
        self.gc_resp()
    }

    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Number of raw sample slots currently retained for centered frame mode.
    pub fn buffered_frame_samples(&self) -> usize {
        self.frame_buffer.len()
    }
}
