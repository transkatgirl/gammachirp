//! GCFB v2.11: sample-by-sample dynamic compressive gammachirp filterbank.

pub mod gammachirp;
pub mod gcfb_v211;
mod stream;
pub mod utils;

pub use gcfb_v211::{ControlMode, GcParam, GcResp, GcfbOutput, gcfb_v211};
pub use stream::{GcfbStream, StreamSample};
