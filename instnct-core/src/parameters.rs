//! # Network Parameters
//!
//! Canonical constant registry for `instnct-core`.
//! Prefixes:
//! - `LIMIT_*`: hard bounds enforced in code
//! - `GLOBAL_*`: network-wide defaults shared across forward passes

// =========================================================================
// LIMIT_* — hard bounds
// =========================================================================

/// Maximum accumulated charge per neuron.
///
/// Range: `[0, LIMIT_MAX_CHARGE]`. Fits in 4 bits; runtime stores it as `u32`.
pub(crate) const LIMIT_MAX_CHARGE: u32 = 15;

// =========================================================================
// GLOBAL_* — network-wide defaults
// =========================================================================

/// Number of phase-gating channels.
pub(crate) const GLOBAL_PHASE_CHANNEL_COUNT: usize = 8;

/// Length of one phase-gating period (ticks).
pub(crate) const GLOBAL_PHASE_TICKS_PER_PERIOD: usize = 8;

/// Default simulation length for one token (ticks).
pub(crate) const GLOBAL_TICKS_PER_TOKEN: usize = 12;

/// Duration of external input injection (initial ticks per token).
pub(crate) const GLOBAL_INPUT_DURATION_TICKS: usize = 2;

/// Charge leak interval (ticks between -1 decay steps).
pub(crate) const GLOBAL_CHARGE_DECAY_INTERVAL_TICKS: usize = 6;
