//! Deterministic router for choosing the next mutation lane.
//!
//! E142/E143 showed that full-range `u8` search needs guided structure, and
//! E144 showed that a small history-based controller can keep best-baseline
//! quality while avoiding unnecessarily expensive lanes. This module bakes
//! that controller shape into the production Rust core.
//!
//! The router is deliberately narrow:
//!
//! ```text
//! lane history snapshots
//!   -> quality gate
//!   -> cheapest passing lane
//!   -> fallback best-score lane when nothing passes the gate
//! ```
//!
//! It does not inspect raw observations, does not generate mutations, and does
//! not mutate the live library. It only chooses which bounded mutation lane
//! should produce the next [`super::evolution::EvolutionProposal`].

use super::CONFIDENCE_PPM_DENOMINATOR;
use super::confidence::ConfidencePpm;
use super::error::FabricError;
use super::evolution::EvolutionLane;

/// Aggregated calibration/history statistics for one mutation lane.
///
/// All values are deterministic integers. The router intentionally avoids
/// floats because lane choice becomes part of replayable experiment evidence.
///
/// Changeability: medium policy surface. Fields can be extended by adding a
/// new versioned stats type, but existing persisted runs must keep their
/// current interpretation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvolutionLaneStats {
    lane: EvolutionLane,
    evaluated_count: u32,
    pass_count: u32,
    mean_score: ConfidencePpm,
    min_score: ConfidencePpm,
    total_cost_units: u64,
}

impl EvolutionLaneStats {
    /// Creates one lane-history snapshot.
    ///
    /// `mean_score` and `min_score` are validation-quality scores in parts per
    /// million. `total_cost_units` is deterministic search work, not wall-clock
    /// time; wall-clock time is hardware-dependent and must not decide replayed
    /// router behavior.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when the lane has no evaluated history, when the
    /// deterministic cost is zero, when the pass count exceeds the evaluated
    /// count, or when `min_score` is greater than `mean_score`.
    pub const fn new(
        lane: EvolutionLane,
        evaluated_count: u32,
        pass_count: u32,
        mean_score: ConfidencePpm,
        min_score: ConfidencePpm,
        total_cost_units: u64,
    ) -> Result<Self, FabricError> {
        if evaluated_count == 0 {
            return Err(FabricError::EmptyEvolutionLaneStats);
        }

        if total_cost_units == 0 {
            return Err(FabricError::ZeroEvolutionLaneCost);
        }

        if pass_count > evaluated_count || min_score.as_ppm() > mean_score.as_ppm() {
            return Err(FabricError::InvalidEvolutionLaneStats);
        }

        Ok(Self {
            lane,
            evaluated_count,
            pass_count,
            mean_score,
            min_score,
            total_cost_units,
        })
    }

    /// Returns the lane described by this snapshot.
    #[must_use]
    pub const fn lane(self) -> EvolutionLane {
        self.lane
    }

    /// Returns the number of evaluated calibration/evidence cases.
    #[must_use]
    pub const fn evaluated_count(self) -> u32 {
        self.evaluated_count
    }

    /// Returns the number of cases that crossed the lane's pass threshold.
    #[must_use]
    pub const fn pass_count(self) -> u32 {
        self.pass_count
    }

    /// Returns the mean validation score.
    #[must_use]
    pub const fn mean_score(self) -> ConfidencePpm {
        self.mean_score
    }

    /// Returns the worst validation score observed in the snapshot.
    #[must_use]
    pub const fn min_score(self) -> ConfidencePpm {
        self.min_score
    }

    /// Returns deterministic accumulated work units.
    #[must_use]
    pub const fn total_cost_units(self) -> u64 {
        self.total_cost_units
    }

    /// Returns conservative floor-rounded pass rate in parts per million.
    ///
    /// Floor rounding is intentional: the router should not cross a quality
    /// gate because of optimistic rounding.
    #[must_use]
    pub fn pass_rate_ppm(self) -> u32 {
        let value = (u64::from(self.pass_count) * u64::from(CONFIDENCE_PPM_DENOMINATOR))
            / u64::from(self.evaluated_count);

        match u32::try_from(value) {
            Ok(value) => value,
            Err(_) => CONFIDENCE_PPM_DENOMINATOR,
        }
    }
}

/// Reasoned lane choice emitted by [`EvolutionRouterPolicy`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvolutionRouterDecision {
    lane: EvolutionLane,
    reason: EvolutionRouterReason,
}

impl EvolutionRouterDecision {
    /// Creates a decision for the cheapest lane that passed all quality gates.
    #[must_use]
    pub const fn cheapest_quality_lane(lane: EvolutionLane) -> Self {
        Self {
            lane,
            reason: EvolutionRouterReason::CheapestQualityLane,
        }
    }

    /// Creates a fallback decision for the best observed score.
    #[must_use]
    pub const fn fallback_best_score(lane: EvolutionLane) -> Self {
        Self {
            lane,
            reason: EvolutionRouterReason::FallbackBestScore,
        }
    }

    /// Returns the selected lane.
    #[must_use]
    pub const fn lane(self) -> EvolutionLane {
        self.lane
    }

    /// Returns the fixed reason code for the selection.
    #[must_use]
    pub const fn reason(self) -> EvolutionRouterReason {
        self.reason
    }
}

/// Stable router selection reason.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvolutionRouterReason {
    /// At least one lane passed quality gates; the cheapest passing lane won.
    CheapestQualityLane,
    /// No lane passed all gates; the best observed score won as a safe
    /// exploration fallback.
    FallbackBestScore,
}

/// Deterministic policy for selecting the next mutation lane.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvolutionRouterPolicy {
    min_pass_rate: ConfidencePpm,
    min_score: ConfidencePpm,
}

impl EvolutionRouterPolicy {
    /// Creates a router policy from quality gates.
    #[must_use]
    pub const fn new(min_pass_rate: ConfidencePpm, min_score: ConfidencePpm) -> Self {
        Self {
            min_pass_rate,
            min_score,
        }
    }

    /// Returns the minimum accepted pass rate.
    #[must_use]
    pub const fn min_pass_rate(self) -> ConfidencePpm {
        self.min_pass_rate
    }

    /// Returns the minimum accepted worst-case score.
    #[must_use]
    pub const fn min_score(self) -> ConfidencePpm {
        self.min_score
    }

    /// Chooses one mutation lane from deterministic lane-history snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::EmptyEvolutionRouterStats`] when no lane
    /// snapshots are provided.
    pub fn choose(
        &self,
        stats: &[EvolutionLaneStats],
    ) -> Result<EvolutionRouterDecision, FabricError> {
        let Some(first) = stats.first().copied() else {
            return Err(FabricError::EmptyEvolutionRouterStats);
        };
        Self::reject_duplicate_lanes(stats)?;

        let mut cheapest_quality: Option<EvolutionLaneStats> = None;
        let mut fallback_best = first;

        for candidate in stats.iter().copied() {
            if Self::is_better_fallback(candidate, fallback_best) {
                fallback_best = candidate;
            }

            if !self.passes_quality_gate(candidate) {
                continue;
            }

            cheapest_quality = Some(match cheapest_quality {
                Some(current) if !Self::is_cheaper_passing_lane(candidate, current) => current,
                _ => candidate,
            });
        }

        Ok(match cheapest_quality {
            Some(selected) => EvolutionRouterDecision::cheapest_quality_lane(selected.lane()),
            None => EvolutionRouterDecision::fallback_best_score(fallback_best.lane()),
        })
    }

    fn passes_quality_gate(self, stats: EvolutionLaneStats) -> bool {
        stats.pass_count() > 0
            && stats.pass_rate_ppm() >= self.min_pass_rate.as_ppm()
            && stats.min_score().as_ppm() >= self.min_score.as_ppm()
    }

    fn is_cheaper_passing_lane(
        candidate: EvolutionLaneStats,
        incumbent: EvolutionLaneStats,
    ) -> bool {
        let candidate_cost =
            u128::from(candidate.total_cost_units()) * u128::from(incumbent.pass_count());
        let incumbent_cost =
            u128::from(incumbent.total_cost_units()) * u128::from(candidate.pass_count());

        (
            candidate_cost,
            // If cost per pass ties, prefer stronger quality and then stable
            // enum order for deterministic replay.
            core::cmp::Reverse(candidate.mean_score().as_ppm()),
            core::cmp::Reverse(candidate.min_score().as_ppm()),
            core::cmp::Reverse(candidate.pass_rate_ppm()),
            lane_rank(candidate.lane()),
        ) < (
            incumbent_cost,
            core::cmp::Reverse(incumbent.mean_score().as_ppm()),
            core::cmp::Reverse(incumbent.min_score().as_ppm()),
            core::cmp::Reverse(incumbent.pass_rate_ppm()),
            lane_rank(incumbent.lane()),
        )
    }

    fn is_better_fallback(candidate: EvolutionLaneStats, incumbent: EvolutionLaneStats) -> bool {
        (
            candidate.mean_score().as_ppm(),
            candidate.pass_rate_ppm(),
            candidate.min_score().as_ppm(),
            core::cmp::Reverse(candidate.total_cost_units()),
            core::cmp::Reverse(lane_rank(candidate.lane())),
        ) > (
            incumbent.mean_score().as_ppm(),
            incumbent.pass_rate_ppm(),
            incumbent.min_score().as_ppm(),
            core::cmp::Reverse(incumbent.total_cost_units()),
            core::cmp::Reverse(lane_rank(incumbent.lane())),
        )
    }

    fn reject_duplicate_lanes(stats: &[EvolutionLaneStats]) -> Result<(), FabricError> {
        for (index, left) in stats.iter().enumerate() {
            if stats[index + 1..]
                .iter()
                .any(|right| right.lane() == left.lane())
            {
                return Err(FabricError::DuplicateEvolutionLaneStats);
            }
        }

        Ok(())
    }
}

const fn lane_rank(lane: EvolutionLane) -> u8 {
    match lane {
        EvolutionLane::Binary => 0,
        EvolutionLane::Guided => 1,
        EvolutionLane::Hybrid => 2,
        EvolutionLane::Predicted => 3,
        EvolutionLane::Random => 4,
    }
}
