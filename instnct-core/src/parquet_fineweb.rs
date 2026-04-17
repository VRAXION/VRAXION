//! FineWeb-Edu parquet ingestion pipeline.
//!
//! Streams one or more HuggingFace FineWeb-Edu parquet files row-group
//! by row-group, pulls the `text` column, applies cheap metadata filters
//! (`language`, `language_score`, `score`, `int_score`) and writes plain
//! UTF-8 bytes to a `Vec<u8>` or directly to a `.txt` file that the
//! existing [`crate::load_corpus`] helper can consume.
//!
//! Design choices:
//! * **Feature-gated.** The arrow-rs tree is ~40 transitive crates and
//!   ~30s cold compile; core-only users do not pay that cost.
//! * **Row-group streaming.** We never materialise a whole 2 GB file in
//!   RAM. The builder projects only the columns we need; other columns
//!   (url, file_path, id, dump, token_count) are not even decoded.
//! * **Cheap filters first.** `language` / `language_score` / `int_score`
//!   are all tiny-width columns — we apply those predicates per-row-group
//!   before concatenating any UTF-8.
//! * **Schema robustness.** Missing optional columns (e.g. older FineWeb
//!   dumps without `int_score`) degrade to "filter off", not error.
//! * **Deterministic.** Files are iterated in lexicographic order; rows
//!   are iterated in on-disk order. Given the same `ExtractConfig` and
//!   file set, output is byte-identical.
//! * **Fail-loud per file, skip-with-log on soft errors.** A corrupted
//!   row-group in file N does not abort the whole extract — it emits a
//!   warning and moves on to the next row-group.
//!
//! FineWeb-Edu canonical schema (as of 2025-02 sample-10BT):
//! ```text
//!   text             : string       (the document body — the only heavy column)
//!   id               : string
//!   dump             : string       (CC-MAIN-YYYY-WW)
//!   url              : string
//!   file_path        : string       (WARC path)
//!   language         : string       (typically "en")
//!   language_score   : double       (fastText confidence, ~0.5..1.0)
//!   token_count      : int64
//!   score            : double       (edu-quality regressor, ~0..5)
//!   int_score        : int64        (round(score) clamped to 0..5)
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::{Array, Float64Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

/// Knobs for one extraction run. `Default::default()` is the sensible
/// "give me ~30 MB of clean English" preset.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// Stop after this many raw UTF-8 bytes have been emitted. Default 30 MB —
    /// ~300× the Alice corpus, enough to bulletproof the 5-way gauntlet.
    pub max_output_bytes: u64,
    /// Cap on number of parquet files opened (after sort). `None` = all.
    pub max_files: Option<usize>,
    /// Require `language == "en"`. If the column is absent this check is skipped.
    pub require_english: bool,
    /// Minimum `language_score`. FineWeb-Edu population has only ~4.5% of docs
    /// at ≥0.98 and ~44% at ≥0.95, so 0.95 is the practical default.
    pub min_language_score: f64,
    /// Minimum `score` (edu-quality regressor, 0..5). 0.0 disables the filter.
    pub min_score: f64,
    /// Minimum `int_score` (rounded score, 0..5). 0 disables the filter.
    pub min_int_score: i64,
    /// Read-batch size in rows. Smaller = more allocator churn, larger = more RAM.
    /// 1024 is a sweet spot for FineWeb (5 MB row-groups of 1000 rows each).
    pub batch_size: usize,
    /// If true, append a single `\n` between consecutive documents so the
    /// 27-class corpus filter sees document boundaries as whitespace.
    pub separator_newline: bool,
    /// If true, skip documents whose raw `text` is empty or all-whitespace.
    pub skip_empty: bool,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            max_output_bytes: 30 * 1024 * 1024,
            max_files: None,
            require_english: true,
            min_language_score: 0.95,
            min_score: 0.0,
            min_int_score: 0,
            batch_size: 1024,
            separator_newline: true,
            skip_empty: true,
        }
    }
}

/// Post-run diagnostics. Not serialised — just printed.
#[derive(Debug, Default, Clone)]
pub struct ExtractStats {
    /// Raw UTF-8 bytes written (before the 27-class filter).
    pub bytes_written: u64,
    /// Documents that passed all filters and were emitted.
    pub docs_emitted: u64,
    /// Documents rejected by filters.
    pub docs_filtered: u64,
    /// Documents skipped as empty (when `skip_empty` is on).
    pub docs_empty: u64,
    /// Row-groups we failed to decode (corruption / truncation) — logged and skipped.
    pub row_groups_skipped: u64,
    /// Number of parquet files opened.
    pub files_opened: u64,
    /// Wall-clock seconds.
    pub elapsed_secs: f64,
}

impl ExtractStats {
    /// Fraction of documents that survived all filters.
    pub fn pass_rate(&self) -> f64 {
        let total = self.docs_emitted + self.docs_filtered + self.docs_empty;
        if total == 0 { 0.0 } else { self.docs_emitted as f64 / total as f64 }
    }
}

/// Errors that abort the whole extraction run (per-file / per-row-group
/// errors are logged and do not surface here).
#[derive(Debug)]
pub enum ExtractError {
    /// IO error opening a file or writing output.
    Io(std::io::Error),
    /// Parquet metadata could not be parsed — file is likely not a parquet at all.
    Parquet(parquet::errors::ParquetError),
    /// Input directory does not exist.
    InputMissing(PathBuf),
    /// No parquet files found under the input directory.
    NoFilesFound(PathBuf),
    /// Required column `text` is missing from a file's schema.
    MissingTextColumn(PathBuf),
    /// `max_output_bytes` was configured to zero — refuse to produce an
    /// empty corpus silently.
    ZeroBudget,
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Io(e) => write!(f, "I/O error: {e}"),
            ExtractError::Parquet(e) => write!(f, "parquet error: {e}"),
            ExtractError::InputMissing(p) => write!(f, "input dir does not exist: {}", p.display()),
            ExtractError::NoFilesFound(p) => write!(f, "no .parquet files found under {}", p.display()),
            ExtractError::MissingTextColumn(p) => write!(f, "file {} has no 'text' column", p.display()),
            ExtractError::ZeroBudget => write!(f, "max_output_bytes is 0 — refusing to produce empty corpus"),
        }
    }
}

impl std::error::Error for ExtractError {}

impl From<std::io::Error> for ExtractError {
    fn from(e: std::io::Error) -> Self { ExtractError::Io(e) }
}

impl From<parquet::errors::ParquetError> for ExtractError {
    fn from(e: parquet::errors::ParquetError) -> Self { ExtractError::Parquet(e) }
}

/// Discover every `*.parquet` file under `dir` (non-recursive, sorted).
pub fn find_parquet_files(dir: &Path) -> Result<Vec<PathBuf>, ExtractError> {
    if !dir.exists() {
        return Err(ExtractError::InputMissing(dir.to_path_buf()));
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("parquet"))
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(ExtractError::NoFilesFound(dir.to_path_buf()));
    }
    Ok(out)
}

/// Stream parquet files through `writer`. Stops once `cfg.max_output_bytes`
/// are written or `cfg.max_files` files have been consumed.
pub fn extract_to_writer<W: Write>(
    files: &[PathBuf],
    cfg: &ExtractConfig,
    writer: &mut W,
) -> Result<ExtractStats, ExtractError> {
    if cfg.max_output_bytes == 0 {
        return Err(ExtractError::ZeroBudget);
    }
    let t0 = Instant::now();
    let mut stats = ExtractStats::default();
    let budget = cfg.max_output_bytes;
    let file_cap = cfg.max_files.unwrap_or(files.len());

    'files: for (f_idx, path) in files.iter().take(file_cap).enumerate() {
        if stats.bytes_written >= budget { break; }
        stats.files_opened += 1;

        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[parquet_fineweb] WARN file {}: open failed: {e} — skipping", path.display());
                continue;
            }
        };

        // Build a reader with metadata, then narrow projection to only the
        // columns we will actually look at. Missing optional columns are
        // silently dropped from the projection.
        let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[parquet_fineweb] WARN file {}: parquet metadata bad: {e} — skipping", path.display());
                continue;
            }
        };
        let schema = builder.schema().clone();

        // Resolve column indices once per file. Only `text` is mandatory.
        let idx_of = |name: &str| -> Option<usize> {
            schema.fields().iter().position(|f| f.name() == name)
        };
        let text_col = match idx_of("text") {
            Some(i) => i,
            None => {
                eprintln!("[parquet_fineweb] WARN file {}: no 'text' column — skipping", path.display());
                continue;
            }
        };
        let lang_col = idx_of("language");
        let ls_col = idx_of("language_score");
        let score_col = idx_of("score");
        let is_col = idx_of("int_score");

        // Build projection mask (leaf indices). parquet's ProjectionMask::roots
        // takes logical root column indices matching `schema.fields()` order.
        let mut proj_roots = vec![text_col];
        if let Some(i) = lang_col { proj_roots.push(i); }
        if let Some(i) = ls_col { proj_roots.push(i); }
        if let Some(i) = score_col { proj_roots.push(i); }
        if let Some(i) = is_col { proj_roots.push(i); }
        proj_roots.sort();
        proj_roots.dedup();

        let mask = ProjectionMask::roots(builder.parquet_schema(), proj_roots);
        let batch_size = cfg.batch_size.max(1);

        let reader = match builder
            .with_projection(mask)
            .with_batch_size(batch_size)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[parquet_fineweb] WARN file {}: reader build failed: {e} — skipping file", path.display());
                continue;
            }
        };

        eprintln!(
            "[parquet_fineweb] opened {} (file {}/{}, bytes so far {:.2} MB / {:.2} MB)",
            path.display(),
            f_idx + 1,
            files.len().min(file_cap),
            stats.bytes_written as f64 / 1e6,
            budget as f64 / 1e6,
        );

        for batch_res in reader {
            if stats.bytes_written >= budget { break 'files; }
            let batch = match batch_res {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[parquet_fineweb] WARN row-group decode: {e} — skipping");
                    stats.row_groups_skipped += 1;
                    continue;
                }
            };

            // Column name -> position in THIS batch (projection reorders).
            let batch_schema = batch.schema();
            let col_by_name = |n: &str| batch_schema.fields().iter().position(|f| f.name() == n);

            let text_arr = match col_by_name("text").and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>()) {
                Some(a) => a,
                None => {
                    eprintln!("[parquet_fineweb] WARN batch: 'text' column missing/non-string — skipping batch");
                    stats.row_groups_skipped += 1;
                    continue;
                }
            };
            let lang_arr = col_by_name("language")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>());
            let ls_arr = col_by_name("language_score")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>());
            let score_arr = col_by_name("score")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>());
            let is_arr = col_by_name("int_score")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());

            for row in 0..batch.num_rows() {
                if stats.bytes_written >= budget { break; }

                // Cheap filters first — never touch the text column on reject.
                if cfg.require_english {
                    if let Some(arr) = lang_arr {
                        if arr.is_null(row) || arr.value(row) != "en" {
                            stats.docs_filtered += 1;
                            continue;
                        }
                    }
                }
                if cfg.min_language_score > 0.0 {
                    if let Some(arr) = ls_arr {
                        if arr.is_null(row) || arr.value(row) < cfg.min_language_score {
                            stats.docs_filtered += 1;
                            continue;
                        }
                    }
                }
                if cfg.min_score > 0.0 {
                    if let Some(arr) = score_arr {
                        if arr.is_null(row) || arr.value(row) < cfg.min_score {
                            stats.docs_filtered += 1;
                            continue;
                        }
                    }
                }
                if cfg.min_int_score > 0 {
                    if let Some(arr) = is_arr {
                        if arr.is_null(row) || arr.value(row) < cfg.min_int_score {
                            stats.docs_filtered += 1;
                            continue;
                        }
                    }
                }

                if text_arr.is_null(row) {
                    stats.docs_empty += 1;
                    continue;
                }
                let text = text_arr.value(row);
                if cfg.skip_empty && text.trim().is_empty() {
                    stats.docs_empty += 1;
                    continue;
                }

                // Remaining budget check so we never overshoot by an entire doc.
                let remaining = budget.saturating_sub(stats.bytes_written);
                if remaining == 0 { break; }
                let write_bytes = (text.len() as u64).min(remaining);
                // Preserve UTF-8 by truncating at the original text's char boundary
                // nearest to `write_bytes` from below.
                let safe_bytes = if write_bytes as usize == text.len() {
                    text.as_bytes()
                } else {
                    let mut end = write_bytes as usize;
                    while end > 0 && !text.is_char_boundary(end) { end -= 1; }
                    &text.as_bytes()[..end]
                };
                // Edge case: the remaining budget is smaller than the width of
                // the next UTF-8 char in this doc, so the char-boundary walk
                // collapsed to 0. Emitting 0 doc bytes + a separator would
                // produce a phantom doc record and a spurious trailing `\n`.
                // Bail out of the whole extract — budget is exhausted for any
                // practical purpose.
                if safe_bytes.is_empty() {
                    break;
                }
                writer.write_all(safe_bytes)?;
                stats.bytes_written += safe_bytes.len() as u64;

                if cfg.separator_newline && stats.bytes_written < budget {
                    writer.write_all(b"\n")?;
                    stats.bytes_written += 1;
                }
                stats.docs_emitted += 1;
            }
        }
    }

    stats.elapsed_secs = t0.elapsed().as_secs_f64();
    Ok(stats)
}

/// Convenience wrapper: open `out_path`, stream, close. Creates parent dirs.
pub fn extract_to_file(
    files: &[PathBuf],
    cfg: &ExtractConfig,
    out_path: &Path,
) -> Result<ExtractStats, ExtractError> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(out_path)?;
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f);
    let s = extract_to_writer(files, cfg, &mut w)?;
    w.flush()?;
    Ok(s)
}

/// Convenience wrapper: stream into memory and return the raw UTF-8 buffer.
/// For the spiking experiments, prefer the file variant — the buffer can be
/// hundreds of MB and is friendlier to put on disk once and mmap / re-read.
pub fn extract_to_bytes(
    files: &[PathBuf],
    cfg: &ExtractConfig,
) -> Result<(Vec<u8>, ExtractStats), ExtractError> {
    // Cap pre-allocation at 64 MiB so a config with a massive budget does not
    // immediately claim that much RAM before any data is read. The Vec will
    // still grow as needed; this just avoids a front-loaded OOM on Windows
    // when `max_output_bytes` is e.g. 2 GiB.
    const PREALLOC_CAP: usize = 64 * 1024 * 1024;
    let cap = (cfg.max_output_bytes as usize).min(PREALLOC_CAP);
    let mut buf = Vec::with_capacity(cap);
    let s = extract_to_writer(files, cfg, &mut buf)?;
    Ok((buf, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sensible() {
        let c = ExtractConfig::default();
        assert_eq!(c.max_output_bytes, 30 * 1024 * 1024);
        assert!(c.require_english);
        assert!(c.min_language_score > 0.9);
        assert!(c.batch_size >= 256);
    }

    #[test]
    fn pass_rate_handles_zero_total() {
        let s = ExtractStats::default();
        assert_eq!(s.pass_rate(), 0.0);
    }

    #[test]
    fn pass_rate_math() {
        let s = ExtractStats {
            docs_emitted: 3,
            docs_filtered: 7,
            docs_empty: 0,
            ..Default::default()
        };
        assert!((s.pass_rate() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn find_parquet_files_rejects_missing_dir() {
        let p = std::path::Path::new("/definitely/not/a/real/path/xyzzy");
        assert!(matches!(
            find_parquet_files(p),
            Err(ExtractError::InputMissing(_))
        ));
    }

    #[test]
    fn extract_rejects_zero_budget() {
        // No files needed — the zero-budget check fires before file iteration.
        let cfg = ExtractConfig { max_output_bytes: 0, ..Default::default() };
        let mut sink: Vec<u8> = Vec::new();
        let r = extract_to_writer(&[], &cfg, &mut sink);
        assert!(matches!(r, Err(ExtractError::ZeroBudget)));
    }
}
