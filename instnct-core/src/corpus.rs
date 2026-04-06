//! Corpus loading and bigram table construction.
//!
//! Standard preprocessing for English text: maps ASCII letters to `0–25`,
//! whitespace to `26`, and discards everything else. This is the same
//! mapping used across all INSTNCT language experiments.

use std::fs;
use std::io;

/// Load a text file and map it to the standard 27-symbol alphabet.
///
/// - `a`–`z` → `0`–`25`
/// - `A`–`Z` → `0`–`25` (lowercased)
/// - space, newline, tab → `26`
/// - everything else is discarded
///
/// # Errors
///
/// Returns [`io::Error`] if the file cannot be read.
///
/// # Example
///
/// ```no_run
/// let corpus = instnct_core::load_corpus("corpus.txt").unwrap();
/// assert!(corpus.iter().all(|&b| b < 27));
/// ```
pub fn load_corpus(path: &str) -> Result<Vec<u8>, io::Error> {
    let raw = fs::read(path)?;
    Ok(raw
        .iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() {
                Some(b - b'a')
            } else if b.is_ascii_uppercase() {
                Some(b.to_ascii_lowercase() - b'a')
            } else if b == b' ' || b == b'\n' || b == b'\t' {
                Some(26)
            } else {
                None
            }
        })
        .collect())
}

/// Build an N×N bigram probability table from a corpus.
///
/// `bigram[i][j]` = P(next symbol = j | current symbol = i).
/// Each row sums to 1.0 (or all zeros if the symbol never appears).
///
/// # Example
///
/// ```
/// let corpus = vec![0u8, 1, 0, 1, 0]; // "ababa"
/// let bigram = instnct_core::build_bigram_table(&corpus, 3);
/// assert_eq!(bigram.len(), 3);
/// // P(b | a) should be high
/// assert!(bigram[0][1] > 0.5);
/// ```
pub fn build_bigram_table(corpus: &[u8], num_classes: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u64; num_classes]; num_classes];
    for pair in corpus.windows(2) {
        let a = pair[0] as usize;
        let b = pair[1] as usize;
        if a < num_classes && b < num_classes {
            counts[a][b] += 1;
        }
    }
    let mut bigram = vec![vec![0.0f64; num_classes]; num_classes];
    for (i, row) in counts.iter().enumerate() {
        let total: u64 = row.iter().sum();
        if total > 0 {
            for (j, &c) in row.iter().enumerate() {
                bigram[i][j] = c as f64 / total as f64;
            }
        }
    }
    bigram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_corpus_maps_correctly() {
        // Write a temp file
        let dir = std::env::temp_dir().join("instnct_corpus_test");
        fs::create_dir_all(&dir).ok();
        let path = dir.join("test.txt");
        fs::write(&path, "Hello World! 123\n\tBye.").unwrap();
        let corpus = load_corpus(path.to_str().unwrap()).unwrap();
        // 'H' → 7, 'e' → 4, 'l' → 11, 'l' → 11, 'o' → 14
        assert_eq!(corpus[0], 7); // h
        assert_eq!(corpus[1], 4); // e
        assert_eq!(corpus[5], 26); // space
        // '1','2','3' discarded, '.' discarded
        assert!(corpus.iter().all(|&b| b < 27));
    }

    #[test]
    fn load_corpus_discards_non_ascii() {
        let dir = std::env::temp_dir().join("instnct_corpus_test2");
        fs::create_dir_all(&dir).ok();
        let path = dir.join("test2.txt");
        fs::write(&path, "abc123!@#xyz").unwrap();
        let corpus = load_corpus(path.to_str().unwrap()).unwrap();
        assert_eq!(corpus, vec![0, 1, 2, 23, 24, 25]); // abc...xyz
    }

    #[test]
    fn bigram_rows_sum_to_one() {
        let corpus: Vec<u8> = (0..1000).map(|i| (i % 27) as u8).collect();
        let bigram = build_bigram_table(&corpus, 27);
        for (i, row) in bigram.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "row {i} sums to {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn bigram_empty_corpus() {
        let bigram = build_bigram_table(&[], 27);
        assert_eq!(bigram.len(), 27);
        for row in &bigram {
            assert!(row.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn bigram_deterministic() {
        let corpus = vec![0u8, 1, 2, 0, 1, 2, 0];
        let b1 = build_bigram_table(&corpus, 3);
        let b2 = build_bigram_table(&corpus, 3);
        assert_eq!(b1, b2);
    }
}
