//! # Network Parameters — Centralized Hyperparameter Registry
//!
//! Every tunable constant and default value lives here. No magic numbers
//! in other modules — they all import from this single source of truth.
//!
//! ## Naming Convention
//!
//! | Prefix | Meaning | Example |
//! |--------|---------|---------|
//! | `NEURON_*` | Per-neuron default (learnable, can diverge per neuron) | `NEURON_DEFAULT_THRESHOLD` |
//! | `GLOBAL_*` | Network-wide setting (same for all neurons) | `GLOBAL_TICKS_PER_TOKEN` |
//! | `LIMIT_*` | Hard constraint (never exceeded, enforced in code) | `LIMIT_MAX_CHARGE` |
//!
//! ## Categories
//!
//! | Category | What it controls |
//! |----------|-----------------|
//! | Neuron dynamics | Charge bounds, firing threshold, polarity ratio |
//! | Wave gating | Temporal specialization via cosine-modulated thresholds |
//! | Propagation timing | Ticks per token, input duration, decay schedule |
//! | Topology | Initial density, connection capacity |
//! | I/O geometry | Golden-ratio dimensioning, sparse distributed representations |

#![allow(dead_code)]

// =========================================================================
// LIMIT_* — hard constraints, enforced everywhere
// =========================================================================

/// Hard upper bound on accumulated charge per neuron.
///
/// Charge is clamped to `[0, LIMIT_MAX_CHARGE]` after each propagation step.
/// Fits in 4 bits (int4). The forward pass uses `u32` for speed but the
/// value never exceeds 15.
pub(crate) const LIMIT_MAX_CHARGE: u32 = 15;

// =========================================================================
// NEURON_* — per-neuron defaults (learnable, each neuron can diverge)
// =========================================================================

/// Default per-neuron firing threshold.
///
/// Range: `[1, 15]`. A neuron fires when its charge exceeds
/// `threshold * wave_multiplier`. Validated sweep converged to ~6 as optimal.
/// Each neuron's threshold evolves independently via `theta` mutation.
pub(crate) const NEURON_DEFAULT_THRESHOLD: u32 = 6;

/// Percentage of neurons initialized as inhibitory (polarity = -1).
///
/// Each neuron's polarity is learnable via `flip` mutation.
/// Fly-realistic: inhibitory neurons are fewer (~10%) but have higher
/// out-degree (2x), acting as broad-range hubs. Matches FlyWire connectome data.
pub(crate) const NEURON_INHIBITORY_PERCENT: u32 = 10;

// =========================================================================
// GLOBAL_* — network-wide settings (same for all neurons)
// =========================================================================

/// Number of distinct temporal channels.
///
/// Each neuron is assigned a channel in `[1, GLOBAL_WAVE_CHANNEL_COUNT]`
/// that determines its preferred firing tick.
pub(crate) const GLOBAL_WAVE_CHANNEL_COUNT: usize = 8;

/// Ticks per wave period. The cosine modulation repeats every this many ticks.
pub(crate) const GLOBAL_WAVE_TICKS_PER_PERIOD: usize = 8;

/// Amplitude of the cosine threshold modulation, as permille (parts per 1000).
///
/// At 300 permille (0.3): threshold range is `[0.7x, 1.3x]`, giving 1.86x
/// selectivity. Used only during LUT precomputation; the runtime forward pass
/// is integer-only.
pub(crate) const GLOBAL_WAVE_AMPLITUDE_PERMILLE: u32 = 300;

/// Default simulation ticks per token.
///
/// More ticks allow signals to traverse longer paths through the graph.
/// A loop of length `N` needs at least `N` ticks for one full revolution.
pub(crate) const GLOBAL_TICKS_PER_TOKEN: usize = 12;

/// Default number of initial ticks during which the input is injected.
pub(crate) const GLOBAL_INPUT_DURATION: usize = 2;

/// Default charge decay interval: subtract 1 from all charges every `N` ticks.
///
/// Prevents unbounded charge accumulation in high-in-degree neurons.
pub(crate) const GLOBAL_CHARGE_DECAY_PERIOD: usize = 6;

/// Default initial connection density as percentage (`5 = 5%`).
pub(crate) const GLOBAL_INITIAL_DENSITY_PERCENT: u32 = 5;

// =========================================================================
// I/O Geometry
// =========================================================================

/// Golden ratio as an integer ratio for phi-overlap I/O computation.
pub(crate) const GLOBAL_PHI_NUMERATOR: u32 = 1618;
/// Golden ratio denominator for phi-overlap I/O computation.
pub(crate) const GLOBAL_PHI_DENOMINATOR: u32 = 1000;

/// Compute input/output dimension for a given neuron count.
///
/// `io_dim(H) = round(H / PHI)`
#[inline]
pub(crate) fn io_dimension(neuron_count: u32) -> u32 {
    (neuron_count as u64 * GLOBAL_PHI_DENOMINATOR as u64 + GLOBAL_PHI_NUMERATOR as u64 / 2) as u32
        / GLOBAL_PHI_NUMERATOR
}

/// SDR active neuron percentage (`20 = 20%`).
pub(crate) const GLOBAL_SDR_ACTIVE_PERCENT: u32 = 20;
