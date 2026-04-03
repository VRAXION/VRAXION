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

// =========================================================================
// LIMIT_* — hard constraints, enforced everywhere
// =========================================================================

/// Hard upper bound on accumulated charge per neuron.
///
/// Charge is clamped to `[0, LIMIT_MAX_CHARGE]` after each propagation step.
/// Fits in 4 bits (int4). The forward pass uses `u32` for speed but the
/// value never exceeds 15.
pub const LIMIT_MAX_CHARGE: u32 = 15;

// =========================================================================
// NEURON_* — per-neuron defaults (learnable, each neuron can diverge)
// =========================================================================

/// Default per-neuron firing threshold.
///
/// Range: `[1, 15]` (int4). A neuron fires when its charge exceeds
/// `threshold * wave_multiplier`. Validated sweep converged to ~6 as optimal.
/// Each neuron's threshold evolves independently via `theta` mutation.
pub const NEURON_DEFAULT_THRESHOLD: u32 = 6;

/// Percentage of neurons initialized as inhibitory (polarity = -1).
///
/// Each neuron's polarity is learnable via `flip` mutation.
/// Fly-realistic: inhibitory neurons are fewer (~10%) but have higher
/// out-degree (2x), acting as broad-range hubs. Matches FlyWire connectome data.
pub const NEURON_INHIBITORY_PERCENT: u32 = 10;

// =========================================================================
// GLOBAL_* — network-wide settings (same for all neurons)
// =========================================================================

/// Number of distinct temporal channels. Each neuron is assigned a channel
/// in `[1, GLOBAL_WAVE_CHANNEL_COUNT]` that determines its preferred firing tick.
///
/// Channel N peaks (lowest threshold) at tick N-1 within each period.
/// Validated: learnable channel (23.8%) > sine-wave gating (21.4%) > none (6.7%).
pub const GLOBAL_WAVE_CHANNEL_COUNT: usize = 8;

/// Ticks per wave period. The cosine modulation repeats every this many ticks.
pub const GLOBAL_WAVE_TICKS_PER_PERIOD: usize = 8;

/// Amplitude of the cosine threshold modulation, as permille (parts per 1000).
///
/// At 300 permille (0.3): threshold range is [0.7x, 1.3x], giving 1.86x selectivity.
/// Used only during LUT precomputation; the runtime forward pass is integer-only.
pub const GLOBAL_WAVE_AMPLITUDE_PERMILLE: u32 = 300;

/// Default simulation ticks per token.
///
/// More ticks allow signals to traverse longer paths through the graph.
/// A loop of length N needs at least N ticks for one full revolution.
/// Typical: 8 (H=256), 12 (canonical), 16 (H=1024+).
pub const GLOBAL_TICKS_PER_TOKEN: usize = 12;

/// Default number of initial ticks during which the input is injected.
pub const GLOBAL_INPUT_DURATION: usize = 2;

/// Default charge decay interval: subtract 1 from all charges every N ticks.
///
/// Prevents unbounded charge accumulation in high-in-degree neurons.
/// At N=6: leak rate ~0.167 per tick.
pub const GLOBAL_CHARGE_DECAY_PERIOD: usize = 6;

/// Default initial connection density as percentage (5 = 5%).
///
/// At 5%, a network of H=256 starts with ~3,200 edges out of 65,280 possible.
pub const GLOBAL_INITIAL_DENSITY_PERCENT: u32 = 5;

// =========================================================================
// I/O Geometry
// =========================================================================

/// Golden ratio as integer ratio for phi-overlap I/O computation.
/// PHI = 1618/1000 = 1.618
pub const GLOBAL_PHI_NUMERATOR: u32 = 1618;
pub const GLOBAL_PHI_DENOMINATOR: u32 = 1000;

/// Compute input/output dimension for a given neuron count.
///
/// `io_dim(H) = round(H / PHI)`
#[inline]
pub fn io_dimension(neuron_count: u32) -> u32 {
    (neuron_count as u64 * GLOBAL_PHI_DENOMINATOR as u64 + GLOBAL_PHI_NUMERATOR as u64 / 2) as u32
        / GLOBAL_PHI_NUMERATOR
}

/// SDR active neuron percentage (20 = 20%).
///
/// Each byte value activates this percentage of input neurons.
pub const GLOBAL_SDR_ACTIVE_PERCENT: u32 = 20;
