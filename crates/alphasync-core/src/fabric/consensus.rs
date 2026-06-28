//! Passive consensus field contract with Agency ingress and egress gates.
//!
//! The consensus field is not a stable-state writer. It can aggregate already
//! validated worker/Prismion signals into a stronger proposal-shaped decision,
//! but both boundaries remain gated:
//!
//! ```text
//! worker signal -> Agency ingress -> consensus accumulation
//! consensus accumulation -> Agency egress -> commit/reject/defer decision
//! ```
//!
//! Changeability: medium. The exact thresholds are policy, but the two-gate
//! shape is an architecture invariant after E149/E150.

use std::collections::BTreeSet;

use super::CONFIDENCE_PPM_DENOMINATOR;
use super::confidence::ConfidencePpm;
use super::error::FabricError;
use super::proposal::PrismionProposal;

const CONSENSUS_DAMPING_PPM: u128 = 750_000;
const MAX_REPRESENTABLE_CONSENSUS_VOTES: usize = u16::MAX as usize;
const CONSENSUS_SOURCE_KIND_SPACE: usize = 4;
const PROPOSAL_FIELD_SOURCE_KEY: u64 = 0;
const TRACE_CONTEXT_SOURCE_KEY: u64 = 1;
const GROUND_CONTEXT_SOURCE_KEY: u64 = 2;

/// Direction of one source vote before consensus accumulation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConsensusVoteKind {
    /// The source supports committing the candidate.
    Support,
    /// The source rejects the candidate.
    Reject,
}

/// Independent source-family class for one consensus vote.
///
/// `source_slot` proves that one exact lane did not vote twice. This field is
/// the stronger release-boundary provenance check: a quorum may require
/// multiple source families, so many correlated Prismion proposal slots cannot
/// impersonate independent agreement by themselves.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConsensusSourceKind {
    /// Materialized Proposal Field candidate lane.
    ProposalField,
    /// Trace/evidence compatibility context lane.
    TraceContext,
    /// Ground, shape, target, and value compatibility context lane.
    GroundContext,
    /// Individual Prismion proposal lane.
    PrismionProposal,
}

/// Non-authoritative recommendation emitted by the passive consensus scorer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConsensusRecommendation {
    /// The candidate is strong enough to become a commit proposal.
    RecommendCommit,
    /// The candidate is strong enough to become a reject proposal.
    RecommendReject,
    /// The candidate is not safe or strong enough to decide.
    RecommendDefer,
}

/// One bounded worker/Prismion vote offered to the consensus field.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConsensusVote {
    source_key: u64,
    source_kind: ConsensusSourceKind,
    kind: ConsensusVoteKind,
    reputation: ConfidencePpm,
    confidence: ConfidencePpm,
    evidence_quality: ConfidencePpm,
    age_ticks: u16,
}

impl ConsensusVote {
    /// Creates one raw vote inside the core crate.
    ///
    /// This constructor is intentionally not public API. Public callers may
    /// inspect votes and pass existing votes into the pure consensus primitive,
    /// but they cannot freely mint arbitrary `source_kind/source_slot`
    /// provenance. Runtime admission code must use the lane-specific
    /// constructors below so source classes are fixed and Prismion vote slots
    /// are derived from the emitted proposal source identity.
    #[must_use]
    pub(crate) const fn new(
        source_key: u64,
        source_kind: ConsensusSourceKind,
        kind: ConsensusVoteKind,
        reputation: ConfidencePpm,
        confidence: ConfidencePpm,
        evidence_quality: ConfidencePpm,
        age_ticks: u16,
    ) -> Self {
        Self {
            source_key,
            source_kind,
            kind,
            reputation,
            confidence,
            evidence_quality,
            age_ticks,
        }
    }

    /// Creates the fixed Proposal Field context vote.
    #[must_use]
    pub const fn proposal_field(
        kind: ConsensusVoteKind,
        reputation: ConfidencePpm,
        confidence: ConfidencePpm,
        evidence_quality: ConfidencePpm,
        age_ticks: u16,
    ) -> Self {
        Self::new(
            PROPOSAL_FIELD_SOURCE_KEY,
            ConsensusSourceKind::ProposalField,
            kind,
            reputation,
            confidence,
            evidence_quality,
            age_ticks,
        )
    }

    /// Creates the fixed trace/evidence compatibility context vote.
    #[must_use]
    pub const fn trace_context(
        kind: ConsensusVoteKind,
        reputation: ConfidencePpm,
        confidence: ConfidencePpm,
        evidence_quality: ConfidencePpm,
        age_ticks: u16,
    ) -> Self {
        Self::new(
            TRACE_CONTEXT_SOURCE_KEY,
            ConsensusSourceKind::TraceContext,
            kind,
            reputation,
            confidence,
            evidence_quality,
            age_ticks,
        )
    }

    /// Creates the fixed ground/shape compatibility context vote.
    #[must_use]
    pub const fn ground_context(
        kind: ConsensusVoteKind,
        reputation: ConfidencePpm,
        confidence: ConfidencePpm,
        evidence_quality: ConfidencePpm,
        age_ticks: u16,
    ) -> Self {
        Self::new(
            GROUND_CONTEXT_SOURCE_KEY,
            ConsensusSourceKind::GroundContext,
            kind,
            reputation,
            confidence,
            evidence_quality,
            age_ticks,
        )
    }

    /// Creates a Prismion proposal vote from an already emitted proposal.
    ///
    /// The vote source slot is derived from the proposal source ID. The caller
    /// chooses only the vote direction and measured evidence quality; it cannot
    /// impersonate a context lane or pick an arbitrary independent source slot.
    #[must_use]
    pub fn from_prismion_proposal(
        proposal: &PrismionProposal,
        kind: ConsensusVoteKind,
        reputation: ConfidencePpm,
        evidence_quality: ConfidencePpm,
        age_ticks: u16,
    ) -> Self {
        Self::new(
            prismion_proposal_source_key(proposal.source().as_str()),
            ConsensusSourceKind::PrismionProposal,
            kind,
            reputation,
            proposal.confidence(),
            evidence_quality,
            age_ticks,
        )
    }

    /// Returns the caller-assigned source slot used for duplicate detection.
    #[must_use]
    pub const fn source_slot(self) -> u64 {
        self.source_key
    }

    /// Returns the independent source-family class.
    #[must_use]
    pub const fn source_kind(self) -> ConsensusSourceKind {
        self.source_kind
    }

    /// Returns the vote direction.
    #[must_use]
    pub const fn kind(self) -> ConsensusVoteKind {
        self.kind
    }

    /// Returns source reputation in parts-per-million space.
    #[must_use]
    pub const fn reputation(self) -> ConfidencePpm {
        self.reputation
    }

    /// Returns source confidence in parts-per-million space.
    #[must_use]
    pub const fn confidence(self) -> ConfidencePpm {
        self.confidence
    }

    /// Returns evidence quality in parts-per-million space.
    #[must_use]
    pub const fn evidence_quality(self) -> ConfidencePpm {
        self.evidence_quality
    }

    /// Returns vote age in consensus ticks.
    #[must_use]
    pub const fn age_ticks(self) -> u16 {
        self.age_ticks
    }
}

fn prismion_proposal_source_key(source: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in source.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    match hash {
        PROPOSAL_FIELD_SOURCE_KEY | TRACE_CONTEXT_SOURCE_KEY | GROUND_CONTEXT_SOURCE_KEY => {
            hash.saturating_add(3)
        }
        _ => hash,
    }
}

/// Fixed Agency+consensus policy.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConsensusPolicy {
    min_workers: u16,
    min_source_kinds: u8,
    accept_score: ConfidencePpm,
    reject_score: ConfidencePpm,
    max_conflict: ConfidencePpm,
    max_stale: ConfidencePpm,
    max_ingress_age_ticks: u16,
    stale_after_ticks: u16,
    min_signal: ConfidencePpm,
}

impl ConsensusPolicy {
    /// Creates a deterministic Agency+consensus policy.
    ///
    /// `accept_score` and `reject_score` are absolute consensus score gates.
    /// `max_conflict` and `max_stale` are ratio gates. `min_signal` is the
    /// minimum `confidence * evidence_quality` ingress strength.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::InvalidConsensusPolicy`] when the minimum worker
    /// count is zero.
    pub const fn new(
        min_workers: u16,
        accept_score: ConfidencePpm,
        reject_score: ConfidencePpm,
        max_conflict: ConfidencePpm,
        max_stale: ConfidencePpm,
        max_ingress_age_ticks: u16,
        min_signal: ConfidencePpm,
    ) -> Result<Self, FabricError> {
        Self::new_with_source_diversity(
            min_workers,
            1,
            accept_score,
            reject_score,
            max_conflict,
            max_stale,
            max_ingress_age_ticks,
            max_ingress_age_ticks / 2,
            min_signal,
        )
    }

    /// Creates a deterministic Agency+consensus policy with an explicit
    /// independent-source-family gate.
    ///
    /// `min_workers` is an independent Prismion-source gate. Context lanes can
    /// support, reject, veto, or explain a consensus score, but they cannot
    /// satisfy this quorum. `min_source_kinds` is an optional provenance-family
    /// gate for future policies.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::InvalidConsensusPolicy`] when the worker or
    /// source-kind minimum is zero, when more source families are required
    /// than admitted workers, or when the requested source-family count exceeds
    /// the fixed consensus source taxonomy.
    #[allow(clippy::too_many_arguments)]
    pub const fn new_with_source_diversity(
        min_workers: u16,
        min_source_kinds: u8,
        accept_score: ConfidencePpm,
        reject_score: ConfidencePpm,
        max_conflict: ConfidencePpm,
        max_stale: ConfidencePpm,
        max_ingress_age_ticks: u16,
        stale_after_ticks: u16,
        min_signal: ConfidencePpm,
    ) -> Result<Self, FabricError> {
        Self::new_with_source_diversity_and_stale_after(
            min_workers,
            min_source_kinds,
            accept_score,
            reject_score,
            max_conflict,
            max_stale,
            max_ingress_age_ticks,
            stale_after_ticks,
            min_signal,
        )
    }

    /// Creates a deterministic Agency+consensus policy with explicit source
    /// diversity and staleness semantics.
    ///
    /// `stale_after_ticks` is separated from `max_ingress_age_ticks`: votes
    /// older than the former still enter the score but contribute to stale
    /// pressure, while votes older than the latter are blocked at ingress.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::InvalidConsensusPolicy`] when the worker or
    /// source-kind minimum is invalid, when the source-family minimum exceeds
    /// available taxonomy or worker count, or when `stale_after_ticks` exceeds
    /// `max_ingress_age_ticks`.
    #[allow(clippy::too_many_arguments)]
    pub const fn new_with_source_diversity_and_stale_after(
        min_workers: u16,
        min_source_kinds: u8,
        accept_score: ConfidencePpm,
        reject_score: ConfidencePpm,
        max_conflict: ConfidencePpm,
        max_stale: ConfidencePpm,
        max_ingress_age_ticks: u16,
        stale_after_ticks: u16,
        min_signal: ConfidencePpm,
    ) -> Result<Self, FabricError> {
        if min_workers == 0 {
            return Err(FabricError::InvalidConsensusPolicy);
        }
        if min_source_kinds == 0
            || (min_source_kinds as u16) > min_workers
            || (min_source_kinds as usize) > CONSENSUS_SOURCE_KIND_SPACE
        {
            return Err(FabricError::InvalidConsensusPolicy);
        }
        if stale_after_ticks > max_ingress_age_ticks {
            return Err(FabricError::InvalidConsensusPolicy);
        }

        Ok(Self {
            min_workers,
            min_source_kinds,
            accept_score,
            reject_score,
            max_conflict,
            max_stale,
            max_ingress_age_ticks,
            stale_after_ticks,
            min_signal,
        })
    }

    /// E150 best-safe policy candidate:
    ///
    /// ```text
    /// min_workers = 2
    /// min_source_kinds = 1
    /// accept_score = 0.34
    /// reject_score = 0.34
    /// max_conflict = 0.32
    /// max_stale = 0.44
    /// ```
    ///
    /// Changeability: medium. This is a policy default, not a law. Changing it
    /// requires rerunning the consensus threshold sweep.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::InvalidConfidence`] only if these static
    /// constants are edited into invalid parts-per-million values.
    pub fn e150_best_safe() -> Result<Self, FabricError> {
        Self::new_with_source_diversity(
            2,
            1,
            ConfidencePpm::new(340_000)?,
            ConfidencePpm::new(340_000)?,
            ConfidencePpm::new(320_000)?,
            ConfidencePpm::new(440_000)?,
            32,
            16,
            ConfidencePpm::new(160_000)?,
        )
    }

    /// Evaluates votes through Agency ingress, passive consensus, and Agency
    /// egress.
    ///
    /// Duplicate source slots are rejected before accumulation so one source
    /// cannot inflate consensus by repeating the same vote.
    ///
    /// A positive egress action also requires enough independent Prismion
    /// proposal sources. Proposal-field, trace, and ground context lanes remain
    /// useful diagnostics and score inputs, but they cannot satisfy the worker
    /// quorum by themselves.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::TooManyConsensusVotes`] when the vote set cannot
    /// be represented by compact `u16` counters.
    ///
    /// Returns [`FabricError::DuplicateConsensusSource`] when two votes use the
    /// same source slot.
    pub fn decide(self, votes: &[ConsensusVote]) -> Result<ConsensusDecision, FabricError> {
        reject_duplicate_sources(votes)?;

        let accumulation = ConsensusAccumulation::from_votes(self, votes)?;
        let total_weight = accumulation.total_weight();
        if total_weight == 0 {
            return Ok(ConsensusDecision::new(
                ConsensusRecommendation::RecommendDefer,
                0,
                0,
                0,
                accumulation.allowed_votes,
                accumulation.blocked_votes,
                accumulation.allowed_source_kinds,
                accumulation.allowed_independent_sources,
                0,
            ));
        }

        let absolute_delta = accumulation
            .support_weight
            .abs_diff(accumulation.reject_weight);
        let score_ppm = ratio_ppm(
            absolute_delta,
            total_weight.saturating_add(CONSENSUS_DAMPING_PPM),
        );
        let conflict_ppm = ratio_ppm(
            core::cmp::min(accumulation.support_weight, accumulation.reject_weight),
            total_weight,
        );
        let stale_ppm = ratio_ppm(accumulation.stale_weight, total_weight);
        let raw_action = if accumulation.support_weight > accumulation.reject_weight
            && score_ppm >= self.accept_score.as_ppm()
        {
            ConsensusRecommendation::RecommendCommit
        } else if accumulation.reject_weight > accumulation.support_weight
            && score_ppm >= self.reject_score.as_ppm()
        {
            ConsensusRecommendation::RecommendReject
        } else {
            ConsensusRecommendation::RecommendDefer
        };
        let winner_independent_sources = accumulation.winner_independent_sources(raw_action);
        let winner_source_kinds = accumulation.winner_source_kinds(raw_action);

        let action = if raw_action == ConsensusRecommendation::RecommendDefer
            || accumulation.allowed_votes < self.min_workers
            || winner_independent_sources < self.min_workers
            || winner_source_kinds < self.min_source_kinds
            || conflict_ppm > self.max_conflict.as_ppm()
            || stale_ppm > self.max_stale.as_ppm()
        {
            ConsensusRecommendation::RecommendDefer
        } else {
            raw_action
        };

        Ok(ConsensusDecision::new(
            action,
            score_ppm,
            conflict_ppm,
            stale_ppm,
            accumulation.allowed_votes,
            accumulation.blocked_votes,
            accumulation.allowed_source_kinds,
            accumulation.allowed_independent_sources,
            winner_independent_sources,
        ))
    }

    /// Returns whether this policy admits one vote through Agency ingress.
    #[must_use]
    pub fn admits_vote(self, vote: ConsensusVote) -> bool {
        self.accumulation_allows(vote)
    }

    /// Returns the quorum-counted independent source minimum.
    #[must_use]
    pub const fn min_distinct_proposal_sources(self) -> u16 {
        self.min_workers
    }

    /// Returns the conflict ratio ceiling.
    #[must_use]
    pub const fn max_conflict(self) -> ConfidencePpm {
        self.max_conflict
    }

    /// Returns the stale ratio ceiling.
    #[must_use]
    pub const fn max_stale(self) -> ConfidencePpm {
        self.max_stale
    }

    /// Returns the age after which an admitted vote contributes stale pressure.
    #[must_use]
    pub const fn stale_after_ticks(self) -> u16 {
        self.stale_after_ticks
    }

    fn ingress_allows(self, vote: ConsensusVote) -> bool {
        if vote.age_ticks > self.max_ingress_age_ticks {
            return false;
        }

        multiply_ratio_ppm(vote.confidence, vote.evidence_quality) >= self.min_signal.as_ppm()
    }

    fn accumulation_allows(self, vote: ConsensusVote) -> bool {
        self.ingress_allows(vote) && vote_weight(vote) > 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ConsensusAccumulation {
    support_weight: u128,
    reject_weight: u128,
    stale_weight: u128,
    allowed_votes: u16,
    blocked_votes: u16,
    allowed_source_kinds: u8,
    support_source_kinds: u8,
    reject_source_kinds: u8,
    allowed_independent_sources: u16,
    support_independent_sources: u16,
    reject_independent_sources: u16,
}

impl ConsensusAccumulation {
    fn from_votes(policy: ConsensusPolicy, votes: &[ConsensusVote]) -> Result<Self, FabricError> {
        let mut accumulation = Self::default();
        let mut allowed_source_kind_seen = [false; CONSENSUS_SOURCE_KIND_SPACE];
        let mut support_source_kind_seen = [false; CONSENSUS_SOURCE_KIND_SPACE];
        let mut reject_source_kind_seen = [false; CONSENSUS_SOURCE_KIND_SPACE];

        for vote in votes {
            accumulation.accumulate_vote(
                policy,
                *vote,
                &mut allowed_source_kind_seen,
                &mut support_source_kind_seen,
                &mut reject_source_kind_seen,
            )?;
        }

        Ok(accumulation)
    }

    const fn total_weight(self) -> u128 {
        self.support_weight.saturating_add(self.reject_weight)
    }

    const fn winner_independent_sources(self, action: ConsensusRecommendation) -> u16 {
        match action {
            ConsensusRecommendation::RecommendCommit => self.support_independent_sources,
            ConsensusRecommendation::RecommendReject => self.reject_independent_sources,
            ConsensusRecommendation::RecommendDefer => 0,
        }
    }

    const fn winner_source_kinds(self, action: ConsensusRecommendation) -> u8 {
        match action {
            ConsensusRecommendation::RecommendCommit => self.support_source_kinds,
            ConsensusRecommendation::RecommendReject => self.reject_source_kinds,
            ConsensusRecommendation::RecommendDefer => 0,
        }
    }

    fn accumulate_vote(
        &mut self,
        policy: ConsensusPolicy,
        vote: ConsensusVote,
        allowed_source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
        support_source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
        reject_source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
    ) -> Result<(), FabricError> {
        if !policy.accumulation_allows(vote) {
            self.blocked_votes = checked_consensus_count(self.blocked_votes)?;
            return Ok(());
        }

        self.allowed_votes = checked_consensus_count(self.allowed_votes)?;
        self.record_allowed_source_kind(vote.source_kind, allowed_source_kind_seen)?;
        let weight = vote_weight(vote);
        if vote.age_ticks > policy.stale_after_ticks {
            self.stale_weight = self.stale_weight.saturating_add(weight);
        }
        self.record_direction(
            vote.kind,
            vote.source_kind,
            weight,
            support_source_kind_seen,
            reject_source_kind_seen,
        )
    }

    fn record_allowed_source_kind(
        &mut self,
        source_kind: ConsensusSourceKind,
        source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
    ) -> Result<(), FabricError> {
        let source_kind_index = consensus_source_kind_index(source_kind);
        if !source_kind_seen[source_kind_index] {
            source_kind_seen[source_kind_index] = true;
            self.allowed_source_kinds = self
                .allowed_source_kinds
                .checked_add(1)
                .ok_or(FabricError::TooManyConsensusVotes)?;
        }
        Ok(())
    }

    fn record_direction(
        &mut self,
        kind: ConsensusVoteKind,
        source_kind: ConsensusSourceKind,
        weight: u128,
        support_source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
        reject_source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
    ) -> Result<(), FabricError> {
        match kind {
            ConsensusVoteKind::Support => {
                self.support_weight = self.support_weight.saturating_add(weight);
                self.record_direction_source_kind(source_kind, true, support_source_kind_seen)?;
                self.record_independent_source(source_kind, true)
            }
            ConsensusVoteKind::Reject => {
                self.reject_weight = self.reject_weight.saturating_add(weight);
                self.record_direction_source_kind(source_kind, false, reject_source_kind_seen)?;
                self.record_independent_source(source_kind, false)
            }
        }
    }

    fn record_direction_source_kind(
        &mut self,
        source_kind: ConsensusSourceKind,
        is_support: bool,
        source_kind_seen: &mut [bool; CONSENSUS_SOURCE_KIND_SPACE],
    ) -> Result<(), FabricError> {
        let source_kind_index = consensus_source_kind_index(source_kind);
        if source_kind_seen[source_kind_index] {
            return Ok(());
        }
        source_kind_seen[source_kind_index] = true;
        if is_support {
            self.support_source_kinds = self
                .support_source_kinds
                .checked_add(1)
                .ok_or(FabricError::TooManyConsensusVotes)?;
        } else {
            self.reject_source_kinds = self
                .reject_source_kinds
                .checked_add(1)
                .ok_or(FabricError::TooManyConsensusVotes)?;
        }
        Ok(())
    }

    fn record_independent_source(
        &mut self,
        source_kind: ConsensusSourceKind,
        is_support: bool,
    ) -> Result<(), FabricError> {
        if !source_kind.counts_as_independent_source() {
            return Ok(());
        }
        self.allowed_independent_sources =
            checked_consensus_count(self.allowed_independent_sources)?;
        if is_support {
            self.support_independent_sources =
                checked_consensus_count(self.support_independent_sources)?;
        } else {
            self.reject_independent_sources =
                checked_consensus_count(self.reject_independent_sources)?;
        }
        Ok(())
    }
}

/// Result of the Agency-sandwiched consensus evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConsensusDecision {
    action: ConsensusRecommendation,
    score_ppm: u32,
    conflict_ppm: u32,
    stale_ppm: u32,
    allowed_votes: u16,
    blocked_votes: u16,
    allowed_source_kinds: u8,
    allowed_independent_sources: u16,
    winner_independent_sources: u16,
}

impl ConsensusDecision {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        action: ConsensusRecommendation,
        score_ppm: u32,
        conflict_ppm: u32,
        stale_ppm: u32,
        allowed_votes: u16,
        blocked_votes: u16,
        allowed_source_kinds: u8,
        allowed_independent_sources: u16,
        winner_independent_sources: u16,
    ) -> Self {
        Self {
            action,
            score_ppm,
            conflict_ppm,
            stale_ppm,
            allowed_votes,
            blocked_votes,
            allowed_source_kinds,
            allowed_independent_sources,
            winner_independent_sources,
        }
    }

    /// Returns the gated Agency action.
    #[must_use]
    pub const fn action(self) -> ConsensusRecommendation {
        self.action
    }

    /// Returns the absolute consensus score in parts per million.
    #[must_use]
    pub const fn score_ppm(self) -> u32 {
        self.score_ppm
    }

    /// Returns the conflict ratio in parts per million.
    #[must_use]
    pub const fn conflict_ppm(self) -> u32 {
        self.conflict_ppm
    }

    /// Returns the stale-weight ratio in parts per million.
    #[must_use]
    pub const fn stale_ppm(self) -> u32 {
        self.stale_ppm
    }

    /// Returns the number of votes admitted by Agency ingress.
    #[must_use]
    pub const fn allowed_votes(self) -> u16 {
        self.allowed_votes
    }

    /// Returns the number of votes blocked by Agency ingress.
    #[must_use]
    pub const fn blocked_votes(self) -> u16 {
        self.blocked_votes
    }

    /// Returns independent source-family count admitted by Agency ingress.
    #[must_use]
    pub const fn allowed_source_kinds(self) -> u8 {
        self.allowed_source_kinds
    }

    /// Returns quorum-counted independent source lanes admitted by Agency.
    #[must_use]
    pub const fn allowed_independent_sources(self) -> u16 {
        self.allowed_independent_sources
    }

    /// Returns non-zero-weight independent Prismion lanes on the winning side.
    ///
    /// This is the quorum-authority value. `allowed_independent_sources` is a
    /// diagnostic total across both directions; only this winner-side count can
    /// authorize a non-defer action.
    #[must_use]
    pub const fn winner_independent_sources(self) -> u16 {
        self.winner_independent_sources
    }
}

const fn checked_consensus_count(value: u16) -> Result<u16, FabricError> {
    match value.checked_add(1) {
        Some(next) => Ok(next),
        None => Err(FabricError::TooManyConsensusVotes),
    }
}

fn reject_duplicate_sources(votes: &[ConsensusVote]) -> Result<(), FabricError> {
    if votes.len() > MAX_REPRESENTABLE_CONSENSUS_VOTES {
        return Err(FabricError::TooManyConsensusVotes);
    }

    let mut seen = BTreeSet::new();
    for vote in votes {
        if !seen.insert((vote.source_kind, vote.source_key)) {
            return Err(FabricError::DuplicateConsensusSource);
        }
    }

    Ok(())
}

const fn consensus_source_kind_index(source_kind: ConsensusSourceKind) -> usize {
    match source_kind {
        ConsensusSourceKind::ProposalField => 0,
        ConsensusSourceKind::TraceContext => 1,
        ConsensusSourceKind::GroundContext => 2,
        ConsensusSourceKind::PrismionProposal => 3,
    }
}

impl ConsensusSourceKind {
    const fn counts_as_independent_source(self) -> bool {
        matches!(self, Self::PrismionProposal)
    }
}

fn vote_weight(vote: ConsensusVote) -> u128 {
    let reputation_confidence = multiply_ratio_ppm(vote.reputation, vote.confidence);
    (u128::from(reputation_confidence) * u128::from(vote.evidence_quality.as_ppm()))
        / u128::from(CONFIDENCE_PPM_DENOMINATOR)
}

fn multiply_ratio_ppm(left: ConfidencePpm, right: ConfidencePpm) -> u32 {
    let product = u128::from(left.as_ppm()) * u128::from(right.as_ppm());
    let value = product / u128::from(CONFIDENCE_PPM_DENOMINATOR);
    ratio_u128_to_ppm(value)
}

fn ratio_ppm(numerator: u128, denominator: u128) -> u32 {
    if denominator == 0 {
        return 0;
    }
    let value = numerator.saturating_mul(u128::from(CONFIDENCE_PPM_DENOMINATOR)) / denominator;
    ratio_u128_to_ppm(value)
}

fn ratio_u128_to_ppm(value: u128) -> u32 {
    let capped = core::cmp::min(value, u128::from(CONFIDENCE_PPM_DENOMINATOR));
    match u32::try_from(capped) {
        Ok(result) => result,
        Err(_error) => CONFIDENCE_PPM_DENOMINATOR,
    }
}
