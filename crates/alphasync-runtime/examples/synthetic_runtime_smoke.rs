//! Synthetic release-mode runtime smoke wrapper.
//!
//! The canonical implementation lives in `alphasync_runtime::synthetic`; this
//! example remains as a convenient Cargo entrypoint for quick local checks.

#![forbid(unsafe_code)]
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

use std::error::Error;
use std::path::PathBuf;

use alphasync_runtime::synthetic::{
    DEFAULT_SMOKE_ITERATIONS, DEFAULT_SMOKE_RUN_DIR, SmokeRunConfig, run_smoke,
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_config()?;
    let result = run_smoke(&config)?;

    println!("alphasync_runtime_smoke=v1");
    println!("iterations={}", result.iterations());
    println!("proposal_total={}", result.proposal_total());
    println!("patch_total={}", result.patch_total());
    println!("elapsed_ms={}", result.elapsed().as_millis());
    println!("cycles_per_second={}", result.cycles_per_second());
    println!("run_dir={}", result.out_dir().display());
    println!(
        "artifact_checksum={}",
        result.artifact_checksum().fnv1a64_hex()
    );
    println!(
        "runtime_bundle_checksum={}",
        result.bundle_checksum().fnv1a64_hex()
    );

    Ok(())
}

fn parse_config() -> Result<SmokeRunConfig, Box<dyn Error>> {
    let mut iterations = DEFAULT_SMOKE_ITERATIONS;
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_SMOKE_RUN_DIR);
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iterations" => {
                let Some(value) = args.next() else {
                    return Err("missing --iterations value".into());
                };
                iterations = value.parse()?;
            }
            "--out" => {
                let Some(value) = args.next() else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
            }
            value => {
                iterations = value.parse()?;
            }
        }
    }

    Ok(SmokeRunConfig::new(iterations, out_dir))
}
