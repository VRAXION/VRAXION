//! Extract a plain UTF-8 `.txt` corpus from HuggingFace FineWeb-Edu parquet.
//!
//! Streams one or more parquet shards, filters by language / edu score,
//! and writes a single `.txt` file that [`instnct_core::load_corpus`] can
//! open as-is.
//!
//! # Usage
//!
//! Defaults (30 MB, `language=="en"`, `language_score>=0.95`, writes under
//! the input dir — NOT into the repo):
//!
//!   cargo run --release --features parquet --example extract_fineweb_txt
//!
//! Override anything:
//!
//!   cargo run --release --features parquet --example extract_fineweb_txt -- \
//!       --input  "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B" \
//!       --output "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt" \
//!       --max-bytes 30000000 \
//!       --min-language-score 0.95 \
//!       --min-int-score 3 \
//!       --max-files 2
//!
//! Supported flags: --input --output --max-bytes --max-files
//!                  --min-language-score --min-score --min-int-score
//!                  --no-english (disable en-only filter)
//!                  --no-separator (omit newline between docs)
//!                  --help
//!
//! Run from repo root, not from `instnct-core/`.

use instnct_core::{extract_to_file, find_parquet_files, ExtractConfig};
use std::path::PathBuf;

const DEFAULT_INPUT: &str = "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B";
const DEFAULT_OUTPUT: &str =
    "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt";

struct Args {
    input: PathBuf,
    output: PathBuf,
    cfg: ExtractConfig,
}

fn parse_args() -> Result<Args, String> {
    let mut input = PathBuf::from(DEFAULT_INPUT);
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut cfg = ExtractConfig::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("{}", HELP);
                std::process::exit(0);
            }
            "--input" => input = PathBuf::from(it.next().ok_or("--input needs value")?),
            "--output" => output = PathBuf::from(it.next().ok_or("--output needs value")?),
            "--max-bytes" => {
                cfg.max_output_bytes = it.next().ok_or("--max-bytes needs value")?
                    .parse().map_err(|e| format!("--max-bytes: {e}"))?;
            }
            "--max-files" => {
                cfg.max_files = Some(it.next().ok_or("--max-files needs value")?
                    .parse().map_err(|e| format!("--max-files: {e}"))?);
            }
            "--min-language-score" => {
                cfg.min_language_score = it.next().ok_or("--min-language-score needs value")?
                    .parse().map_err(|e| format!("--min-language-score: {e}"))?;
            }
            "--min-score" => {
                cfg.min_score = it.next().ok_or("--min-score needs value")?
                    .parse().map_err(|e| format!("--min-score: {e}"))?;
            }
            "--min-int-score" => {
                cfg.min_int_score = it.next().ok_or("--min-int-score needs value")?
                    .parse().map_err(|e| format!("--min-int-score: {e}"))?;
            }
            "--batch-size" => {
                cfg.batch_size = it.next().ok_or("--batch-size needs value")?
                    .parse().map_err(|e| format!("--batch-size: {e}"))?;
            }
            "--no-english" => cfg.require_english = false,
            "--no-separator" => cfg.separator_newline = false,
            other => return Err(format!("unknown arg: {other}  (see --help)")),
        }
    }
    Ok(Args { input, output, cfg })
}

const HELP: &str = r#"extract_fineweb_txt — FineWeb-Edu parquet -> plain .txt

Flags:
  --input  DIR       directory containing *.parquet (default: S:/AI/.../Fineweb edu 10B)
  --output PATH      output .txt path (default: <input>/fineweb_edu_30m.txt)
  --max-bytes  N     stop after N UTF-8 bytes (default 31457280 = 30 MiB)
  --max-files  N     cap on parquet files opened (default: all in dir)
  --min-language-score F   default 0.95
  --min-score F      min edu regressor score (0..5); default 0 (off)
  --min-int-score N  min rounded edu score (0..5); default 0 (off)
  --batch-size N     parquet record-batch rows (default 1024)
  --no-english       skip the language=="en" check
  --no-separator     do not insert \n between documents
  --help             this text
"#;

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    println!("=== FineWeb-Edu -> .txt extractor ===");
    println!("  input  : {}", args.input.display());
    println!("  output : {}", args.output.display());
    println!("  cfg    : {:?}", args.cfg);

    let files = match find_parquet_files(&args.input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("  found {} parquet file(s)", files.len());
    for p in &files { println!("    - {}", p.display()); }

    let stats = match extract_to_file(&files, &args.cfg, &args.output) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error during extract: {e}");
            std::process::exit(1);
        }
    };

    let raw = match std::fs::read(&args.output) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("warn: could not re-read output for stats: {e}");
            return;
        }
    };
    let (class27, letters, spaces) = post_filter_summary(&raw);

    println!();
    println!("--- done in {:.2}s ---", stats.elapsed_secs);
    println!("  raw bytes written      : {} ({:.2} MB)", stats.bytes_written, stats.bytes_written as f64 / 1e6);
    println!("  docs emitted / filtered / empty : {} / {} / {}",
        stats.docs_emitted, stats.docs_filtered, stats.docs_empty);
    println!("  pass rate              : {:.2}%", stats.pass_rate() * 100.0);
    println!("  row-groups skipped     : {}", stats.row_groups_skipped);
    println!("  files opened           : {}", stats.files_opened);
    println!();
    println!("  post 27-class filter   : {} chars ({:.2} MB)", class27, class27 as f64 / 1e6);
    println!("  letters / whitespace   : {} / {} ({:.1}% / {:.1}%)",
        letters, spaces,
        100.0 * letters as f64 / class27.max(1) as f64,
        100.0 * spaces as f64 / class27.max(1) as f64);
    println!();
    println!("ready to consume:");
    println!("  let corpus = instnct_core::load_corpus({:?}).unwrap();", args.output.display().to_string());
}

/// Mirror of `load_corpus`'s 27-class filter, purely for a post-run sanity
/// summary. Returns (total, letters, whitespace).
fn post_filter_summary(raw: &[u8]) -> (usize, usize, usize) {
    // Mirror `load_corpus` exactly: letters a..z/A..Z -> 0..25, whitespace
    // (space, \n, \t) -> 26. Note: load_corpus does NOT count \r, so we
    // must not either, or the reported post-filter size overstates what
    // load_corpus actually produces.
    let mut total = 0usize;
    let mut letters = 0usize;
    let mut spaces = 0usize;
    for &b in raw {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' => { total += 1; letters += 1; }
            b' ' | b'\n' | b'\t' => { total += 1; spaces += 1; }
            _ => {}
        }
    }
    (total, letters, spaces)
}
