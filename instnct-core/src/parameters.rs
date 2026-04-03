//! # Network Parameters
//!
//! Canonical constant registry for `instnct-core`.
//! Prefixes:
//! - `LIMIT_*`: hard bounds enforced in code
//! - `NEURON_*`: per-neuron initialization defaults
//! - `GLOBAL_*`: network-wide defaults shared across forward passes

// Some registry entries are reserved for upcoming modules and are
// intentionally unused for now.
#![allow(dead_code)]

// =========================================================================
// LIMIT_* — hard bounds
// =========================================================================
//
// Absolute ceilings consumed by validation and clamp paths.

/// Maximum accumulated charge per neuron.
///
/// Range: `[0, LIMIT_MAX_CHARGE]`. Fits in 4 bits; runtime stores it as `u32`.
pub(crate) const LIMIT_MAX_CHARGE: u32 = 15;

// =========================================================================
// NEURON_* — per-neuron initialization defaults
// =========================================================================
//
// Construction-time defaults that fan out into per-neuron mutable state.

/// Initial firing threshold for a neuron.
///
/// Range: `[1, 15]`. Serves as the default starting point before mutation.
pub(crate) const NEURON_INIT_THRESHOLD: u32 = 6;

/// Initial share of inhibitory neurons.
///
/// Unit: percent of neurons at network construction time.
pub(crate) const NEURON_INHIBITORY_PERCENT: u32 = 10;

// =========================================================================
// GLOBAL_* — network-wide defaults
// =========================================================================
//
// Shared defaults that apply uniformly across propagation and topology.

/// Number of wave-gating channels.
///
/// Unit: distinct channel slots assigned across neurons.
pub(crate) const GLOBAL_WAVE_CHANNEL_COUNT: usize = 8;

/// Length of one wave-gating period.
///
/// Unit: ticks per period.
pub(crate) const GLOBAL_WAVE_TICKS_PER_PERIOD: usize = 8;

/// Wave-gating threshold amplitude.
///
/// Unit: permille of the cosine coefficient (`300 = 0.3`).
pub(crate) const GLOBAL_WAVE_AMPLITUDE_PERMILLE: u32 = 300;

/// Default simulation length for one token.
///
/// Unit: ticks. Longer recurrent paths require more ticks to propagate.
pub(crate) const GLOBAL_TICKS_PER_TOKEN: usize = 12;

/// Duration of external input injection.
///
/// Unit: initial ticks per token.
pub(crate) const GLOBAL_INPUT_DURATION_TICKS: usize = 2;

/// Charge leak interval.
///
/// Unit: ticks between `-1` decay steps.
pub(crate) const GLOBAL_CHARGE_DECAY_INTERVAL_TICKS: usize = 6;

/// Initial edge density target.
///
/// Unit: percent of possible directed edges at network initialization.
pub(crate) const GLOBAL_INITIAL_DENSITY_PERCENT: u32 = 5;

// =========================================================================
// I/O geometry
// =========================================================================
//
// Phi-derived sizing and SDR sparsity defaults for input/output layout.

/// Phi numerator for integer I/O sizing.
///
/// Used with `GLOBAL_PHI_DENOMINATOR` to approximate `1.618`.
pub(crate) const GLOBAL_PHI_NUMERATOR: u32 = 1618;

/// Phi denominator for integer I/O sizing.
///
/// Used with `GLOBAL_PHI_NUMERATOR` to approximate `1.618`.
pub(crate) const GLOBAL_PHI_DENOMINATOR: u32 = 1000;

/// Compute the phi-derived input/output dimension.
///
/// Returns `round(neuron_count / PHI)`. Uses a `u64` intermediate for headroom.
#[inline]
pub(crate) fn io_dimension(neuron_count: u32) -> u32 {
    (neuron_count as u64 * GLOBAL_PHI_DENOMINATOR as u64 + GLOBAL_PHI_NUMERATOR as u64 / 2) as u32
        / GLOBAL_PHI_NUMERATOR
}

/// Default SDR activation density.
///
/// Unit: percent of I/O neurons active per token representation.
pub(crate) const GLOBAL_SDR_ACTIVE_PERCENT: u32 = 20;
