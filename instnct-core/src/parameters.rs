//! # Network Parameters — Centralized Hyperparameter Registry
//!
//! Every tunable constant and default value lives here. No magic numbers
//! in other modules — they all import from this single source of truth.
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
// Neuron Dynamics
// =========================================================================

/// Upper bound on accumulated charge per neuron.
///
/// Charge is clamped to `[0, MAX_CHARGE]` after each propagation step.
/// Matched to the C edge-device port which uses `int8` charge in `[0, 15]`.
pub const MAX_CHARGE: f32 = 15.0;

/// Default per-neuron firing threshold (theta).
///
/// Range: `[1, 15]` (int4). A neuron fires when its charge exceeds
/// `theta * wave_multiplier`. Validated sweep converged to ~6 as optimal.
pub const DEFAULT_FIRING_THRESHOLD: f32 = 6.0;

/// Fraction of neurons initialized as inhibitory (polarity = -1).
///
/// Fly-realistic: inhibitory neurons are fewer (~10%) but have higher
/// out-degree (2x), acting as broad-range hubs. Matches FlyWire connectome data.
pub const INHIBITORY_FRACTION: f32 = 0.10;

// =========================================================================
// Wave Gating (C19 Mechanism)
// =========================================================================

/// Number of distinct temporal channels. Each neuron is assigned a channel
/// in `[1, WAVE_CHANNEL_COUNT]` that determines its preferred firing tick.
///
/// Channel N peaks (lowest threshold) at tick N-1 within each period.
/// Validated: learnable channel (23.8%) > sine-wave gating (21.4%) > none (6.7%).
pub const WAVE_CHANNEL_COUNT: usize = 8;

/// Ticks per wave period. The cosine modulation repeats every this many ticks.
/// Equal to `WAVE_CHANNEL_COUNT` so each channel has exactly one preferred tick.
pub const WAVE_TICKS_PER_PERIOD: usize = 8;

/// Amplitude of the cosine threshold modulation.
///
/// The effective threshold multiplier is `1.0 - WAVE_AMPLITUDE * cos(phase)`,
/// ranging from `1 - A` (easiest firing) to `1 + A` (hardest).
/// At 0.3: range is [0.7, 1.3], giving a 1.86x selectivity ratio.
pub const WAVE_AMPLITUDE: f32 = 0.3;

// =========================================================================
// Propagation Timing
// =========================================================================

/// Default simulation ticks per token.
///
/// More ticks allow signals to traverse longer paths through the graph.
/// Typical values: 8 (H=256), 12 (canonical), 16 (H=1024+).
/// A loop of length N needs at least N ticks for one full signal revolution.
pub const DEFAULT_TICKS_PER_TOKEN: usize = 12;

/// Default number of initial ticks during which the input is injected.
///
/// The input vector is added to neuron activations for this many ticks,
/// giving the external signal time to enter the recurrent substrate before
/// the network runs freely on internal dynamics.
pub const DEFAULT_INPUT_DURATION: usize = 2;

/// Default charge decay interval: subtract 1 from all charges every N ticks.
///
/// Prevents unbounded charge accumulation in high-in-degree neurons.
/// Equivalent to a leak rate of `1/N` per tick. At N=6: ~0.167 per tick,
/// close to phi/10 (0.1618).
pub const DEFAULT_CHARGE_DECAY_PERIOD: usize = 6;

// =========================================================================
// Topology
// =========================================================================

/// Default initial connection density (fraction of possible directed edges).
///
/// At 5%, a network of H=256 starts with ~3,200 edges out of 65,280 possible.
/// The BUILD+CRYSTAL ratchet typically converges to ~1,200 edges (~1.8% density).
pub const DEFAULT_INITIAL_DENSITY: f32 = 0.05;

// =========================================================================
// I/O Geometry
// =========================================================================

/// Golden ratio, used for phi-overlap I/O dimensioning.
///
/// Input and output dimensions are both `round(H / PHI)`, creating an
/// overlap zone where some neurons serve as both input receivers and output
/// emitters. Validated: 20.8% with overlap > 20.0% without.
pub const PHI: f32 = 1.618_034;

/// Fraction of input-dimension neurons activated per byte (SDR sparsity).
///
/// Each byte value maps to a fixed Sparse Distributed Representation where
/// this fraction of input neurons are set to 1.0. At 20%: ~63 active neurons
/// out of IN_DIM=316 for H=512.
pub const SDR_ACTIVE_FRACTION: f32 = 0.20;
