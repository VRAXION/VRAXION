//! Sparse Distributed Representation (SDR) encoding table.
//!
//! Maps discrete symbols to sparse binary patterns for spiking network input.
//! Each pattern has exactly `active_bits` neurons set to 1 in the input zone
//! `[0, input_dim)`, with the rest zero.

use rand::Rng;
use std::error::Error;
use std::fmt;

/// SDR encoding table: `num_symbols` patterns of `neuron_count` length.
///
/// # Example
///
/// ```
/// use instnct_core::SdrTable;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let sdr = SdrTable::new(27, 256, 158, 20, &mut rng).unwrap();
/// assert_eq!(sdr.num_symbols(), 27);
/// assert_eq!(sdr.active_bits(), 31); // 158 * 20 / 100
/// let pattern = sdr.pattern(0);
/// assert_eq!(pattern.len(), 256);
/// ```
pub struct SdrTable {
    patterns: Vec<Vec<i32>>, // num_symbols × neuron_count, values 0 or 1
    neuron_count: usize,
    input_dim: usize,        // active bits confined to [0, input_dim)
    active_bits: usize,      // K active bits per pattern
}

/// Errors from [`SdrTable::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SdrError {
    /// `input_dim` exceeds `neuron_count`.
    InputDimExceedsNeuronCount {
        /// Requested input dimension.
        input_dim: usize,
        /// Network neuron count.
        neuron_count: usize,
    },
    /// `active_pct` exceeds 100.
    ActivePctTooHigh {
        /// Requested active percentage.
        active_pct: usize,
    },
    /// Computed `active_bits` is zero (input_dim or active_pct too small).
    ZeroActiveBits {
        /// Input dimension used.
        input_dim: usize,
        /// Active percentage used.
        active_pct: usize,
    },
}

impl fmt::Display for SdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputDimExceedsNeuronCount {
                input_dim,
                neuron_count,
            } => write!(
                f,
                "input_dim {input_dim} exceeds neuron_count {neuron_count}"
            ),
            Self::ActivePctTooHigh { active_pct } => {
                write!(f, "active_pct {active_pct} exceeds 100")
            }
            Self::ZeroActiveBits {
                input_dim,
                active_pct,
            } => write!(
                f,
                "active_bits is zero (input_dim={input_dim}, active_pct={active_pct})"
            ),
        }
    }
}

impl Error for SdrError {}

impl SdrTable {
    /// Build a random SDR table with validated parameters.
    ///
    /// Each of `num_symbols` patterns has exactly `input_dim * active_pct / 100`
    /// active bits (value 1), all within `[0, input_dim)`. Remaining neurons are 0.
    ///
    /// # Errors
    ///
    /// Returns [`SdrError`] if parameters are invalid (would cause infinite loop,
    /// index panic, or silently empty patterns).
    pub fn new(
        num_symbols: usize,
        neuron_count: usize,
        input_dim: usize,
        active_pct: usize,
        rng: &mut impl Rng,
    ) -> Result<Self, SdrError> {
        if input_dim > neuron_count {
            return Err(SdrError::InputDimExceedsNeuronCount {
                input_dim,
                neuron_count,
            });
        }
        if active_pct > 100 {
            return Err(SdrError::ActivePctTooHigh { active_pct });
        }
        let active_bits = input_dim * active_pct / 100;
        if active_bits == 0 {
            return Err(SdrError::ZeroActiveBits {
                input_dim,
                active_pct,
            });
        }

        let mut patterns = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            let mut pattern = vec![0i32; neuron_count];
            let mut activated = 0;
            while activated < active_bits {
                let idx = rng.gen_range(0..input_dim);
                if pattern[idx] == 0 {
                    pattern[idx] = 1;
                    activated += 1;
                }
            }
            patterns.push(pattern);
        }

        Ok(Self {
            patterns,
            neuron_count,
            input_dim,
            active_bits,
        })
    }

    /// Get the SDR pattern for a symbol.
    ///
    /// Returns a slice of length `neuron_count` with exactly `active_bits`
    /// entries set to 1, all within `[0, input_dim)`.
    ///
    /// # Panics
    ///
    /// Panics if `symbol >= num_symbols`.
    pub fn pattern(&self, symbol: usize) -> &[i32] {
        &self.patterns[symbol]
    }

    /// Number of symbols in the table.
    #[inline]
    pub fn num_symbols(&self) -> usize {
        self.patterns.len()
    }

    /// Number of active bits per pattern.
    #[inline]
    pub fn active_bits(&self) -> usize {
        self.active_bits
    }

    /// Total neuron count (pattern length).
    #[inline]
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Input dimension (active zone width, `[0, input_dim)`).
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn new_creates_correct_dimensions() {
        let mut rng = StdRng::seed_from_u64(42);
        let sdr = SdrTable::new(27, 256, 158, 20, &mut rng).unwrap();
        assert_eq!(sdr.num_symbols(), 27);
        assert_eq!(sdr.neuron_count(), 256);
        assert_eq!(sdr.input_dim(), 158);
        assert_eq!(sdr.active_bits(), 31); // 158 * 20 / 100
    }

    #[test]
    fn patterns_have_exact_active_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let sdr = SdrTable::new(27, 256, 158, 20, &mut rng).unwrap();
        for sym in 0..27 {
            let active: usize = sdr.pattern(sym).iter().filter(|&&v| v == 1).count();
            assert_eq!(active, 31, "symbol {sym} has {active} active bits, expected 31");
        }
    }

    #[test]
    fn active_bits_confined_to_input_dim() {
        let mut rng = StdRng::seed_from_u64(42);
        let sdr = SdrTable::new(27, 256, 158, 20, &mut rng).unwrap();
        for sym in 0..27 {
            let outside: usize = sdr.pattern(sym)[158..].iter().filter(|&&v| v != 0).count();
            assert_eq!(outside, 0, "symbol {sym} has active bits outside input zone");
        }
    }

    /// Verify the library SdrTable produces byte-identical output to the legacy helper.
    #[test]
    fn rng_flow_matches_legacy_helper() {
        // Legacy helper (copied from old evolve_language.rs)
        fn legacy_build(rng: &mut StdRng) -> Vec<Vec<i32>> {
            let input_end = 158;
            let active_count = input_end * 20 / 100; // 31
            let mut table = Vec::with_capacity(27);
            for _ in 0..27 {
                let mut pattern = vec![0i32; 256];
                let mut activated = 0;
                while activated < active_count {
                    let idx = rng.gen_range(0..input_end);
                    if pattern[idx] == 0 {
                        pattern[idx] = 1;
                        activated += 1;
                    }
                }
                table.push(pattern);
            }
            table
        }

        let mut rng_old = StdRng::seed_from_u64(100);
        let old_table = legacy_build(&mut rng_old);

        let mut rng_new = StdRng::seed_from_u64(100);
        let new_table = SdrTable::new(27, 256, 158, 20, &mut rng_new).unwrap();

        for (i, old_pattern) in old_table.iter().enumerate() {
            assert_eq!(
                new_table.pattern(i),
                old_pattern.as_slice(),
                "pattern {i} differs between legacy and SdrTable"
            );
        }
    }

    #[test]
    fn constructor_rejects_invalid_params() {
        let mut rng = StdRng::seed_from_u64(1);

        // input_dim > neuron_count
        assert!(matches!(
            SdrTable::new(10, 100, 200, 20, &mut rng),
            Err(SdrError::InputDimExceedsNeuronCount { .. })
        ));

        // active_pct > 100
        assert!(matches!(
            SdrTable::new(10, 100, 50, 150, &mut rng),
            Err(SdrError::ActivePctTooHigh { .. })
        ));

        // active_bits == 0 (small input_dim × small pct rounds to 0)
        assert!(matches!(
            SdrTable::new(10, 100, 3, 1, &mut rng), // 3 * 1 / 100 = 0
            Err(SdrError::ZeroActiveBits { .. })
        ));
    }
}
