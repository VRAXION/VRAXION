//! Canonical local `α-Sync` runtime CLI.
//!
//! This binary is the stable entrypoint for local runtime smoke/scene artifact
//! generation. It does not read datasets, perform training, publish results, or
//! call the network.

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
use std::path::{Path, PathBuf};

use alphasync_core::fabric::FabricShapeProfileKind;
use alphasync_runtime::logic_iq::{
    DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_CYCLES, DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_WRITE_EVERY,
    DEFAULT_LOGIC_IQ_CYCLES, DEFAULT_LOGIC_IQ_RUN_DIR, DEFAULT_LOGIC_IQ_WRITE_EVERY,
    LogicIqRunConfig, run_logic_iq_consensus_scene, run_logic_iq_zero,
};
use alphasync_runtime::synthetic::{
    DEFAULT_SCENE_RUN_DIR, DEFAULT_SESSION_CYCLES, DEFAULT_SESSION_RUN_DIR,
    DEFAULT_SESSION_WRITE_EVERY, DEFAULT_SMOKE_ITERATIONS, DEFAULT_SMOKE_RUN_DIR, SessionRunConfig,
    SmokeRunConfig, run_scene, run_session, run_smoke,
};

const DEFAULT_LOGIC_IQ_CLI_PROFILE: FabricShapeProfileKind = FabricShapeProfileKind::FrontierLocal;

fn main() -> Result<(), Box<dyn Error>> {
    match parse_command()? {
        Command::Help => {
            print_help();
            Ok(())
        }
        Command::Smoke {
            iterations,
            out_dir,
        } => run_smoke_command(iterations, out_dir),
        Command::Scene { out_dir } => run_scene_command(&out_dir),
        Command::Session {
            cycles,
            write_every,
            out_dir,
        } => run_session_command(cycles, write_every, out_dir),
        Command::LogicIqZero {
            cycles,
            write_every,
            profile,
            out_dir,
        } => run_logic_iq_zero_command(cycles, write_every, profile, out_dir),
        Command::LogicIqConsensusScene {
            cycles,
            write_every,
            profile,
            out_dir,
        } => run_logic_iq_consensus_scene_command(cycles, write_every, profile, out_dir),
    }
}

fn run_smoke_command(iterations: usize, out_dir: PathBuf) -> Result<(), Box<dyn Error>> {
    let result = run_smoke(&SmokeRunConfig::new(iterations, out_dir))?;
    println!("alphasync_runtime_cli=smoke_v1");
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

fn run_scene_command(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let result = run_scene(out_dir)?;
    println!("alphasync_runtime_cli=scene_v1");
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

fn run_session_command(
    cycles: usize,
    write_every: usize,
    out_dir: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let result = run_session(&SessionRunConfig::new(cycles, write_every, out_dir))?;
    println!("alphasync_runtime_cli=session_v1");
    println!("cycles={}", result.cycles());
    println!("writeout_count={}", result.writeout_count());
    println!("proposal_total={}", result.proposal_total());
    println!("patch_total={}", result.patch_total());
    println!("elapsed_ms={}", result.elapsed().as_millis());
    println!("cycles_per_second={}", result.cycles_per_second());
    println!("proposals_per_second={}", result.proposals_per_second());
    println!("patches_per_second={}", result.patches_per_second());
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

fn run_logic_iq_zero_command(
    cycles: usize,
    write_every: usize,
    profile: FabricShapeProfileKind,
    out_dir: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let result = run_logic_iq_zero(&LogicIqRunConfig::new_with_profile(
        cycles,
        write_every,
        profile,
        out_dir,
    ))?;
    print_logic_iq_result("logic_iq_zero_v1", &result);
    Ok(())
}

fn run_logic_iq_consensus_scene_command(
    cycles: usize,
    write_every: usize,
    profile: FabricShapeProfileKind,
    out_dir: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let result = run_logic_iq_consensus_scene(&LogicIqRunConfig::new_with_profile(
        cycles,
        write_every,
        profile,
        out_dir,
    ))?;
    print_logic_iq_result("logic_iq_consensus_scene_v1", &result);
    Ok(())
}

fn print_logic_iq_result(label: &str, result: &alphasync_runtime::logic_iq::LogicIqRunResult) {
    println!("alphasync_runtime_cli={label}");
    println!("cycles={}", result.cycles());
    println!("writeout_count={}", result.writeout_count());
    println!("exact_count={}", result.exact_count());
    println!("wrong_count={}", result.wrong_count());
    println!("false_commit_count={}", result.false_commit_count());
    println!(
        "conflict_or_unknown_count={}",
        result.conflict_or_unknown_count()
    );
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Command {
    Help,
    Smoke {
        iterations: usize,
        out_dir: PathBuf,
    },
    Scene {
        out_dir: PathBuf,
    },
    Session {
        cycles: usize,
        write_every: usize,
        out_dir: PathBuf,
    },
    LogicIqZero {
        cycles: usize,
        write_every: usize,
        profile: FabricShapeProfileKind,
        out_dir: PathBuf,
    },
    LogicIqConsensusScene {
        cycles: usize,
        write_every: usize,
        profile: FabricShapeProfileKind,
        out_dir: PathBuf,
    },
}

fn parse_command() -> Result<Command, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };

    let remaining = args.collect::<Vec<_>>();
    match command.as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "smoke" => parse_smoke(&remaining),
        "scene" => parse_scene(&remaining),
        "session" => parse_session(&remaining),
        "logic-iq-zero" => parse_logic_iq_zero(&remaining),
        "logic-iq-consensus-scene" => parse_logic_iq_consensus_scene(&remaining),
        value => Err(format!("unknown command: {value}").into()),
    }
}

fn parse_smoke(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut iterations = DEFAULT_SMOKE_ITERATIONS;
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_SMOKE_RUN_DIR);
    let mut index = 0usize;

    while index < args.len() {
        match args[index].as_str() {
            "--iterations" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --iterations value".into());
                };
                iterations = value.parse()?;
                index += 2;
            }
            "--out" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
                index += 2;
            }
            value if !value.starts_with('-') => {
                iterations = value.parse()?;
                index += 1;
            }
            value => return Err(format!("unexpected smoke argument: {value}").into()),
        }
    }

    Ok(Command::Smoke {
        iterations,
        out_dir,
    })
}

fn parse_scene(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_SCENE_RUN_DIR);
    let mut index = 0usize;

    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
                index += 2;
            }
            value => return Err(format!("unexpected scene argument: {value}").into()),
        }
    }

    Ok(Command::Scene { out_dir })
}

fn parse_session(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut cycles = DEFAULT_SESSION_CYCLES;
    let mut write_every = DEFAULT_SESSION_WRITE_EVERY;
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_SESSION_RUN_DIR);
    let mut index = 0usize;

    while index < args.len() {
        match args[index].as_str() {
            "--cycles" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --cycles value".into());
                };
                cycles = value.parse()?;
                index += 2;
            }
            "--write-every" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --write-every value".into());
                };
                write_every = value.parse()?;
                index += 2;
            }
            "--out" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
                index += 2;
            }
            value => return Err(format!("unexpected session argument: {value}").into()),
        }
    }

    Ok(Command::Session {
        cycles,
        write_every,
        out_dir,
    })
}

fn parse_logic_iq_zero(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut cycles = DEFAULT_LOGIC_IQ_CYCLES;
    let mut write_every = DEFAULT_LOGIC_IQ_WRITE_EVERY;
    let mut profile = DEFAULT_LOGIC_IQ_CLI_PROFILE;
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_LOGIC_IQ_RUN_DIR);
    let mut index = 0usize;

    while index < args.len() {
        match args[index].as_str() {
            "--cycles" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --cycles value".into());
                };
                cycles = value.parse()?;
                index += 2;
            }
            "--write-every" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --write-every value".into());
                };
                write_every = value.parse()?;
                index += 2;
            }
            "--out" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
                index += 2;
            }
            "--profile" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --profile value".into());
                };
                profile = parse_shape_profile(value)?;
                index += 2;
            }
            value => return Err(format!("unexpected logic-iq-zero argument: {value}").into()),
        }
    }

    Ok(Command::LogicIqZero {
        cycles,
        write_every,
        profile,
        out_dir,
    })
}

fn parse_logic_iq_consensus_scene(args: &[String]) -> Result<Command, Box<dyn Error>> {
    let mut cycles = DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_CYCLES;
    let mut write_every = DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_WRITE_EVERY;
    let mut profile = DEFAULT_LOGIC_IQ_CLI_PROFILE;
    let mut out_dir = std::env::current_dir()?.join(DEFAULT_LOGIC_IQ_RUN_DIR);
    let mut index = 0usize;

    while index < args.len() {
        match args[index].as_str() {
            "--cycles" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --cycles value".into());
                };
                cycles = value.parse()?;
                index += 2;
            }
            "--write-every" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --write-every value".into());
                };
                write_every = value.parse()?;
                index += 2;
            }
            "--out" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --out value".into());
                };
                out_dir = PathBuf::from(value);
                index += 2;
            }
            "--profile" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing --profile value".into());
                };
                profile = parse_shape_profile(value)?;
                index += 2;
            }
            value => {
                return Err(
                    format!("unexpected logic-iq-consensus-scene argument: {value}").into(),
                );
            }
        }
    }

    Ok(Command::LogicIqConsensusScene {
        cycles,
        write_every,
        profile,
        out_dir,
    })
}

fn parse_shape_profile(value: &str) -> Result<FabricShapeProfileKind, Box<dyn Error>> {
    match value {
        "logic-iq-canary" => Ok(FabricShapeProfileKind::LogicIqCanary),
        "frontier-local" => Ok(FabricShapeProfileKind::FrontierLocal),
        _ => Err(format!("unknown shape profile: {value}").into()),
    }
}

fn print_help() {
    println!("α-Sync Runtime CLI");
    println!();
    println!("Commands:");
    println!("  alphasync-runtime smoke [--iterations N] [--out DIR]");
    println!("  alphasync-runtime scene [--out DIR]");
    println!("  alphasync-runtime session [--cycles N] [--write-every N] [--out DIR]");
    println!(
        "  alphasync-runtime logic-iq-zero [--cycles N] [--write-every N] [--profile frontier-local|logic-iq-canary] [--out DIR]"
    );
    println!(
        "  alphasync-runtime logic-iq-consensus-scene [--cycles N] [--write-every N] [--profile frontier-local|logic-iq-canary] [--out DIR]"
    );
    println!("  Logic-IQ CLI default profile: frontier-local");
    println!();
    println!("Artifacts:");
    println!("  runtime_summary.json");
    println!("  runtime_frame.json");
    println!("  progress_snapshot.json");
    println!("  artifact_checksums.json");
    println!("  runtime_bundle.json");
    println!("  runtime_session.json");
}
