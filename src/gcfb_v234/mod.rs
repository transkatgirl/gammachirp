//! GCFB v2.34: frame processing and hearing-loss characteristics.

mod base_utils;
mod common;
pub mod gammachirp;
pub mod gcfb_v234;
pub mod reassignment;
mod stream;
pub mod synchrosqueezing;
pub mod utils;

pub use gcfb_v234::{
    ControlMode, DynHpaf, EmParam, GainReference, GcParam, GcResp, GcfbOutput, HLoss, gcfb_v234,
};
pub use reassignment::{
    BandwidthConsensusConfig, BandwidthConsensusResult, BandwidthConsensusStream,
    BandwidthConsensusStreamConfig, BandwidthConsensusStreamFrame, BandwidthConsensusStreamStep,
    BandwidthScaleMetadata, PhaseReassignmentResult, ReassignmentConfig, ReassignmentMode,
    ReassignmentResult, ReassignmentStream, ReassignmentStreamStep, SparsityComparison,
    SparsityMetrics, gcfb_v234_with_bandwidth_consensus, gcfb_v234_with_phase_reassignment,
    gcfb_v234_with_reassignment, linear_weights, phase_reassign_gcfb_v234,
    phase_reassign_gcfb_v234_with_config, reassign_gcfb_v234, reassign_gcfb_v234_with_config,
};
pub use stream::{DcgcEvent, GcfbStream, StreamStep};
pub use synchrosqueezing::{
    SynchrosqueezingConfig, SynchrosqueezingMode, SynchrosqueezingResult, SynchrosqueezingStream,
    SynchrosqueezingStreamStep, gcfb_v234_with_synchrosqueezing, synchrosqueeze_gcfb_v234,
    synchrosqueeze_gcfb_v234_with_config,
};
