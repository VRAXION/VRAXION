//! Multi-rule synthetic runtime scene wrapper.
//!
//! The canonical implementation lives in `alphasync_runtime::synthetic`; this
//! example remains as a convenient Cargo entrypoint for GUI-loadable artifacts.

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

use alphasync_runtime::synthetic::{DEFAULT_SCENE_RUN_DIR, run_scene};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = parse_out_dir()?;
    let result = run_scene(&out_dir)?;

    println!("alphasync_runtime_scene=v1");
    println!("rules={}", result.rules());
    println!("activated_rules={}", result.activated_rules());
    println!("proposals={}", result.proposals());
    println!("patches={}", result.patches());
    println!("evidence_cells={}", result.evidence_cells());
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

fn parse_out_dir() -> Result<PathBuf, Box<dyn Error>> {
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_SCENE_RUN_DIR);
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let Some(value) = args.next() else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
            }
            value => return Err(format!("unexpected argument: {value}").into()),
        }
    }

    Ok(out_dir)
}
