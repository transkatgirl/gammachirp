use thiserror::Error;

/// Errors returned by the filterbank and utility routines.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("numerical error: {0}")]
    Numerical(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid WAV file: {0}")]
    Wav(String),
}

pub type Result<T> = std::result::Result<T, Error>;
