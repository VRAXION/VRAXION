//! Deterministic, raw-data-free core types for the `AlphaSync` runtime.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(rust_2018_idioms)]
#![deny(unused_must_use)]
#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::unwrap_used
    )
)]

pub mod eval;
pub mod fabric;
pub mod ids;
pub mod progress;
