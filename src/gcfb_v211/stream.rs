//! Bounded-memory sample streaming for GCFB v2.11.

use ndarray::{Array1, Array2};

use super::gcfb_v211::{
    AcfCoef, AcfStatus, ControlMode, GcParam, GcResp, initial_asymmetric_ratio_and_centers,
    make_asym_cmp_filters_v2, prepare_input_correction_fir, prepare_passive_impulses,
    prepare_time_invariant_response, set_param,
};
use crate::{Error, Result, dsp};

/// The channel-major outputs produced for one input sample.
#[derive(Clone, Debug)]
pub struct StreamSample {
    /// Zero-based index of this input sample.
    pub sample_index: usize,
    pub pgc_out: Array1<f64>,
    pub cgc_out: Array1<f64>,
    /// Dynamic level estimates, absent for static and level control.
    pub lvl_db: Option<Array1<f64>>,
    /// Dynamic asymmetric frequency ratios, absent for static and level control.
    pub frat_val: Option<Array1<f64>>,
    /// Dynamic asymmetric filter centers, absent for static and level control.
    pub fr2: Option<Array1<f64>>,
}

/// A bounded-memory GCFB v2.11 processor.
///
/// Each accepted sample immediately produces one [`StreamSample`]. Starting a
/// new signal requires constructing a new processor.
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
    previous_level: Array2<f64>,
    samples_processed: usize,
    failed: bool,
}

impl GcfbStream {
    /// Prepare a stream using the same derived parameters and responses as the
    /// optimized batch implementation.
    pub fn new(gc_param: GcParam) -> Result<Self> {
        let (param, mut response) = set_param(gc_param)?;
        let correction = dsp::CausalFir::new(prepare_input_correction_fir(&param)?);
        let passive = dsp::CausalFirBank::new(prepare_passive_impulses(&param, &response)?);
        let fixed_coefficients = prepare_time_invariant_response(&param, &mut response)?;
        let fixed_status = AcfStatus::new(&fixed_coefficients);
        let (dynamic_coefficients, dynamic_status) = if param.ctrl == ControlMode::Dynamic {
            let (_, centers) = initial_asymmetric_ratio_and_centers(&param, &response);
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
            previous_level: Array2::zeros((channels, 2)),
            samples_processed: 0,
            failed: false,
        })
    }

    /// Process one finite input sample.
    ///
    /// A processing error after input validation makes the stream terminal,
    /// because causal filter history may already have advanced. Construct a
    /// new processor before submitting more samples after such an error.
    pub fn process_sample(&mut self, sample: f64) -> Result<StreamSample> {
        if self.failed {
            return Err(Error::Numerical(
                "v2.11 stream cannot continue after a previous processing error".into(),
            ));
        }
        // Validate before advancing any FIR or IIR history.
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

    fn process_finite_sample(&mut self, sample: f64) -> Result<StreamSample> {
        let sample_index = self.samples_processed;
        let corrected = self.correction.process_sample(sample);
        let pgc_out = self.passive.process_sample(corrected);
        let fixed_out = self.fixed_status.process(
            &self.fixed_coefficients,
            pgc_out.as_slice().unwrap(),
            false,
        )?;

        let (mut cgc_out, lvl_db, frat_val, fr2) = if self.param.ctrl == ControlMode::Dynamic {
            let channels = self.param.num_ch;
            let mut levels = Array1::zeros(channels);
            let mut ratios = Array1::zeros(channels);
            let mut centers = Array1::zeros(channels);
            for ch in 0..channels {
                let source = self.param.lvl_est.n_ch_lvl_est[ch];
                let passive_level = pgc_out[source]
                    .max(0.0)
                    .max(self.previous_level[[ch, 0]] * self.param.lvl_est.exp_decay_val);
                let fixed_level = fixed_out[source]
                    .max(0.0)
                    .max(self.previous_level[[ch, 1]] * self.param.lvl_est.exp_decay_val);
                self.previous_level[[ch, 0]] = passive_level;
                self.previous_level[[ch, 1]] = fixed_level;
                let total = self.param.lvl_est.weight
                    * self.param.lvl_est.lvl_lin_ref
                    * (passive_level / self.param.lvl_est.lvl_lin_ref)
                        .powf(self.param.lvl_est.pwr[0])
                    + (1.0 - self.param.lvl_est.weight)
                        * self.param.lvl_est.lvl_lin_ref
                        * (fixed_level / self.param.lvl_est.lvl_lin_ref)
                            .powf(self.param.lvl_est.pwr[1]);
                levels[ch] = 20.0 * total.max(self.param.lvl_est.lvl_lin_min_lim).log10()
                    + self.param.lvl_est.rms2spldb;
                ratios[ch] = self.param.frat[0][0]
                    + self.param.frat[0][1] * self.response.ef[ch]
                    + (self.param.frat[1][0] + self.param.frat[1][1] * self.response.ef[ch])
                        * levels[ch];
                centers[ch] = self.response.fp1[ch] * ratios[ch];
            }
            if sample_index.is_multiple_of(self.param.num_update_asym_cmp) {
                self.dynamic_coefficients = Some(make_asym_cmp_filters_v2(
                    self.param.fs,
                    centers.as_slice().unwrap(),
                    self.response.b2_val.as_slice().unwrap(),
                    self.response.c2_val.as_slice().unwrap(),
                )?);
            }
            let output = self.dynamic_status.as_mut().unwrap().process(
                self.dynamic_coefficients.as_ref().unwrap(),
                pgc_out.as_slice().unwrap(),
                false,
            )?;
            (output, Some(levels), Some(ratios), Some(centers))
        } else {
            (fixed_out, None, None, None)
        };
        if self.param.ctrl == ControlMode::Dynamic {
            cgc_out *= &self.response.gain_factor;
        }
        self.samples_processed += 1;
        Ok(StreamSample {
            sample_index,
            pgc_out,
            cgc_out,
            lvl_db,
            frat_val,
            fr2,
        })
    }

    /// Prepared parameters, including derived level-estimator fields.
    pub fn gc_param(&self) -> &GcParam {
        &self.param
    }

    /// Prepared, time-invariant response metadata.
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
}
