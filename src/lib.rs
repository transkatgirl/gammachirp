//! Rust port of the GammachirPy dynamic compressive gammachirp filterbanks.
//!
//! The module layout follows the Python project: [`gcfb_v211`] contains the
//! sample-by-sample model and [`gcfb_v234`] contains the frame-based model with
//! hearing-loss characteristics. [`breebaart2001`] adds offline and
//! bounded-memory streaming binaural excitation-inhibition stages. The crate
//! also provides causal streaming reassignment, an end-to-end GCFB/Breebaart
//! hybrid, and an ideal-observer template. Matrices are channel-major: rows
//! are filter channels and columns are samples or frames.

// These two lints conflict with the deliberately Python-compatible module and
// function layout retained by this port.
#![allow(clippy::module_inception, clippy::too_many_arguments)]

mod dsp;
mod error;

pub mod breebaart2001;
pub mod gcfb_v211;
pub mod gcfb_v234;

pub use error::{Error, Result};
