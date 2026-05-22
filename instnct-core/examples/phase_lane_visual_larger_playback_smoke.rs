//! Smoke exporter for the 054 larger visual playback package.

use instnct_core::visual_export::{export_visual_bundle, larger_playback_visual_bundle};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = parse_out()?;
    export_visual_bundle(&out, &larger_playback_visual_bundle())?;
    println!("visual_larger_playback_smoke_out={}", out.display());
    Ok(())
}

fn parse_out() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(out.unwrap_or_else(|| {
        PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_054_visual_analysis_larger_run_playback/smoke",
        )
    }))
}
