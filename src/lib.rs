//! Rust port of the GammachirPy dynamic compressive gammachirp filterbanks.
//!
//! [`gcfb_v234`] provides frame- and sample-based processing with hearing-loss
//! characteristics. [`breebaart2001`] adds offline and
//! bounded-memory streaming binaural excitation-inhibition stages. The crate
//! also provides causal streaming reassignment, an end-to-end GCFB/Breebaart
//! hybrid, and an ideal-observer template. Matrices are channel-major: rows
//! are filter channels and columns are samples or frames.

// These two lints conflict with the deliberately Python-compatible module and
// function layout retained by the v2.34 port.
#![allow(clippy::module_inception, clippy::too_many_arguments)]

mod dsp;
mod error;

pub mod breebaart2001;
pub mod gcfb_v234;

pub use error::{Error, Result};
