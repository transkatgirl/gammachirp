//! GCFB v2.34: frame processing and hearing-loss characteristics.

pub mod gammachirp;
pub mod gcfb_v234;
pub mod reassignment;
pub mod utils;

pub use gcfb_v234::{
    ControlMode, DynHpaf, EmParam, GainReference, GcParam, GcResp, GcfbOutput, HLoss, gcfb_v234,
};
pub use reassignment::{
    ReassignmentConfig, ReassignmentMode, ReassignmentResult, gcfb_v234_with_reassignment,
    reassign_gcfb_v234, reassign_gcfb_v234_with_config,
};
