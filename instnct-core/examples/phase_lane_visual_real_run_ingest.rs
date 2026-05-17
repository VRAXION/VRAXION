//! Smoke ingest for visual analysis real-run packages.

use instnct_core::visual_export::{
    bundle_from_049_adversarial_run, closure_bundle_from_049_adversarial_run, export_visual_bundle,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    let bundle = if cfg.closure_055 {
        closure_bundle_from_049_adversarial_run(&cfg.source)?
    } else {
        bundle_from_049_adversarial_run(&cfg.source)?
    };
    export_visual_bundle(&cfg.out, &bundle)?;
    println!(
        "visual_real_run_ingest_source={} out={} closure_055={}",
        cfg.source.display(),
        cfg.out.display(),
        cfg.closure_055
    );
    Ok(())
}

struct Config {
    source: PathBuf,
    out: PathBuf,
    closure_055: bool,
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut source = None;
    let mut out = None;
    let mut closure_055 = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source" => source = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--closure-055" => closure_055 = true,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(Config {
        source: source.unwrap_or_else(|| {
            PathBuf::from(
                "target/pilot_wave/stable_loop_phase_lock_049_adversarial_frozen_eval_scale/smoke",
            )
        }),
        out: out.unwrap_or_else(|| {
            PathBuf::from("target/pilot_wave/stable_loop_phase_lock_053_visual_analysis_real_run_ingest/smoke")
        }),
        closure_055,
    })
}
