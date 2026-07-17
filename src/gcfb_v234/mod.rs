//! GCFB v2.34: frame processing and hearing-loss characteristics.

pub mod gammachirp;
pub mod gcfb_v234;
pub mod utils;

pub use gcfb_v234::{
    ControlMode, DynHpaf, EmParam, GainReference, GcParam, GcResp, GcfbOutput, HLoss, gcfb_v234,
};
