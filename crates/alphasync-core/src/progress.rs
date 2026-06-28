//! Minimal progress/completion gate primitives.
//!
//! This module exists to prevent premature completion claims. It does not plan,
//! reason, schedule work, or decide scientific truth. It only enforces a small
//! mechanical rule:
//!
//! ```text
//! COMPLETE is allowed only when:
//!   every required evidence slot is satisfied
//!   no blocker is active
//!   the completion check uses the current plan revision
//!
//! QUIT is allowed only when:
//!   COMPLETE is already allowed
//!   no required partial writeout is pending
//!   no explicit next action remains
//! ```
//!
//! Changeability: medium. The exact slot cap is a compact-core tradeoff, but
//! the "no completion without required evidence" invariant is architecture
//! level.

use core::fmt;

/// Maximum number of evidence slots in one minimal completion gate.
///
/// Changeability: medium. This is a compact `u128` bitset cap. If a runtime
/// needs more slots, it should compose multiple gates or add a versioned wider
/// representation.
pub const MAX_EVIDENCE_SLOTS: u8 = 128;

/// Longest allowed wall-clock writeout interval, in milliseconds.
///
/// Changeability: medium. This is intentionally conservative because long
/// frontier runs must never become black-box jobs. Runners may write more
/// often, but a cadence above this cap is rejected by the core policy.
pub const MAX_WRITEOUT_INTERVAL_MS: u64 = 300_000;

/// Stable progress-gate error codes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProgressError {
    /// A completion gate was created without any required evidence.
    EmptyRequiredEvidence,
    /// An evidence slot index was outside the compact bitset range.
    EvidenceSlotOutOfRange,
    /// A blocker clear was requested when no blocker was active.
    NoActiveBlocker,
    /// A blocker add would overflow the compact blocker counter.
    BlockerOverflow,
    /// A run writeout cadence used a zero threshold.
    InvalidWriteoutCadence,
    /// A progress cursor moved backward relative to the previous writeout.
    NonMonotonicProgress,
    /// The plan revision counter reached its maximum value.
    RevisionOverflow,
}

impl ProgressError {
    /// Returns the stable machine-readable error code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyRequiredEvidence => "empty_required_evidence",
            Self::EvidenceSlotOutOfRange => "evidence_slot_out_of_range",
            Self::NoActiveBlocker => "no_active_blocker",
            Self::BlockerOverflow => "blocker_overflow",
            Self::InvalidWriteoutCadence => "invalid_writeout_cadence",
            Self::NonMonotonicProgress => "non_monotonic_progress",
            Self::RevisionOverflow => "revision_overflow",
        }
    }
}

impl fmt::Display for ProgressError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::error::Error for ProgressError {}

/// One fixed evidence slot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvidenceSlot(u8);

impl EvidenceSlot {
    /// Creates an evidence slot in `0..128`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::EvidenceSlotOutOfRange`] when the slot index is
    /// outside the compact gate representation.
    pub const fn new(index: u8) -> Result<Self, ProgressError> {
        if index >= MAX_EVIDENCE_SLOTS {
            return Err(ProgressError::EvidenceSlotOutOfRange);
        }

        Ok(Self(index))
    }

    /// Returns the slot index.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    const fn bit(self) -> u128 {
        1u128 << self.0
    }
}

/// Compact set of evidence slots.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvidenceSet(u128);

impl EvidenceSet {
    /// Returns an empty evidence set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns a set containing one slot.
    #[must_use]
    pub const fn single(slot: EvidenceSlot) -> Self {
        Self(slot.bit())
    }

    /// Returns true when the set contains no slots.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true when the set contains `slot`.
    #[must_use]
    pub const fn contains(self, slot: EvidenceSlot) -> bool {
        self.0 & slot.bit() != 0
    }

    /// Returns a set with `slot` inserted.
    #[must_use]
    pub const fn with(self, slot: EvidenceSlot) -> Self {
        Self(self.0 | slot.bit())
    }

    /// Returns `self - other`.
    #[must_use]
    pub const fn without_all(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns true when all slots in `required` are also in this set.
    #[must_use]
    pub const fn covers(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// Returns the number of slots in the set.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

/// Monotonic progress cursor used by long-running experiment runners.
///
/// This type contains only counters supplied by the runtime. It does not read
/// clocks, write files, or perform I/O. The purpose is to make the required
/// partial-writeout cadence testable as a pure core rule.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RunProgressCursor {
    elapsed_ms: u64,
    documents_seen: u64,
    frames_seen: u64,
}

impl RunProgressCursor {
    /// Creates a monotonic progress cursor.
    #[must_use]
    pub const fn new(elapsed_ms: u64, documents_seen: u64, frames_seen: u64) -> Self {
        Self {
            elapsed_ms,
            documents_seen,
            frames_seen,
        }
    }

    /// Returns elapsed runtime milliseconds.
    #[must_use]
    pub const fn elapsed_ms(self) -> u64 {
        self.elapsed_ms
    }

    /// Returns documents seen by the run.
    #[must_use]
    pub const fn documents_seen(self) -> u64 {
        self.documents_seen
    }

    /// Returns frames seen by the run.
    #[must_use]
    pub const fn frames_seen(self) -> u64 {
        self.frames_seen
    }

    const fn checked_delta_since(self, previous: Self) -> Result<Self, ProgressError> {
        if self.elapsed_ms < previous.elapsed_ms
            || self.documents_seen < previous.documents_seen
            || self.frames_seen < previous.frames_seen
        {
            return Err(ProgressError::NonMonotonicProgress);
        }

        Ok(Self {
            elapsed_ms: self.elapsed_ms - previous.elapsed_ms,
            documents_seen: self.documents_seen - previous.documents_seen,
            frames_seen: self.frames_seen - previous.frames_seen,
        })
    }
}

/// Reason a partial run writeout is due.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WriteoutReason {
    /// The run has not emitted its first partial state yet.
    Initial,
    /// The elapsed runtime threshold was reached.
    Elapsed,
    /// The document counter threshold was reached.
    Documents,
    /// The frame counter threshold was reached.
    Frames,
}

/// Writeout decision for a long-running experiment.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WriteoutDecision {
    /// Runtime must emit a partial artifact now.
    Write {
        /// Reason the writeout is required.
        reason: WriteoutReason,
    },
    /// Runtime may continue without writing yet.
    Wait,
}

/// Reason a long-running runtime must keep going instead of quitting.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ContinuationReason {
    /// A partial artifact is due before the runtime consumes or discards more
    /// work.
    PartialWriteoutDue {
        /// Reason the writeout cadence fired.
        reason: WriteoutReason,
    },
    /// Required evidence is still missing from the completion gate.
    CompletionIncomplete {
        /// Number of required slots not yet satisfied.
        missing_count: u32,
    },
    /// At least one blocker is still active.
    CompletionBlocked {
        /// Number of active blockers.
        blocker_count: u16,
    },
    /// The caller used a stale completion-gate revision.
    CompletionStale {
        /// Current plan revision.
        current: u64,
        /// Revision supplied by the caller.
        checked: u64,
    },
    /// The caller still has explicit next work items to run.
    PendingNextAction {
        /// Count of pending next actions.
        action_count: u16,
    },
}

impl ContinuationReason {
    /// Returns the stable machine-readable reason code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PartialWriteoutDue { .. } => "partial_writeout_due",
            Self::CompletionIncomplete { .. } => "completion_incomplete",
            Self::CompletionBlocked { .. } => "completion_blocked",
            Self::CompletionStale { .. } => "completion_stale",
            Self::PendingNextAction { .. } => "pending_next_action",
        }
    }
}

/// Decision produced by [`ContinuationGate::quit_decision`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ContinuationDecision {
    /// Runtime must continue or write state before quitting.
    Continue {
        /// Mechanical reason quitting is not allowed yet.
        reason: ContinuationReason,
    },
    /// Runtime may stop because completion is current, materialized enough, and
    /// no next action remains.
    QuitAllowed,
}

/// Minimal anti-premature-quit gate.
///
/// This gate is intentionally stateless. Runners pass the already-computed
/// completion decision, writeout decision, and pending next-action count. The
/// gate only applies fixed precedence, making "do not stop yet" a pure,
/// testable rule.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContinuationGate;

impl ContinuationGate {
    /// Decides whether a runtime is allowed to quit now.
    ///
    /// Precedence is conservative: a due writeout wins first, then incomplete
    /// or stale completion, then remaining next actions. This guarantees a
    /// runner writes observable partial state before silently stopping.
    #[must_use]
    pub const fn quit_decision(
        completion_decision: CompletionDecision,
        writeout_decision: WriteoutDecision,
        pending_next_action_count: u16,
    ) -> ContinuationDecision {
        if let WriteoutDecision::Write { reason } = writeout_decision {
            return ContinuationDecision::Continue {
                reason: ContinuationReason::PartialWriteoutDue { reason },
            };
        }

        match completion_decision {
            CompletionDecision::Complete => {}
            CompletionDecision::Incomplete { missing_count } => {
                return ContinuationDecision::Continue {
                    reason: ContinuationReason::CompletionIncomplete { missing_count },
                };
            }
            CompletionDecision::Blocked { blocker_count } => {
                return ContinuationDecision::Continue {
                    reason: ContinuationReason::CompletionBlocked { blocker_count },
                };
            }
            CompletionDecision::StaleRevision { current, checked } => {
                return ContinuationDecision::Continue {
                    reason: ContinuationReason::CompletionStale { current, checked },
                };
            }
        }

        if pending_next_action_count > 0 {
            return ContinuationDecision::Continue {
                reason: ContinuationReason::PendingNextAction {
                    action_count: pending_next_action_count,
                },
            };
        }

        ContinuationDecision::QuitAllowed
    }
}

/// Partial-writeout cadence for a long-running experiment.
///
/// This does not replace run-specific artifact schemas. It only enforces that
/// a runner can be asked, mechanically, whether it owes the user a partial
/// progress artifact before more work is consumed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WriteoutCadence {
    elapsed_interval_ms: u64,
    document_interval: u64,
    frame_interval: u64,
}

impl WriteoutCadence {
    /// Creates a writeout cadence with non-zero thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::InvalidWriteoutCadence`] when any threshold is
    /// zero or when the elapsed-time threshold exceeds
    /// [`MAX_WRITEOUT_INTERVAL_MS`].
    pub const fn new(
        max_elapsed_ms: u64,
        max_documents: u64,
        max_frames: u64,
    ) -> Result<Self, ProgressError> {
        if max_elapsed_ms == 0
            || max_documents == 0
            || max_frames == 0
            || max_elapsed_ms > MAX_WRITEOUT_INTERVAL_MS
        {
            return Err(ProgressError::InvalidWriteoutCadence);
        }

        Ok(Self {
            elapsed_interval_ms: max_elapsed_ms,
            document_interval: max_documents,
            frame_interval: max_frames,
        })
    }

    /// Returns the default frontier-run cadence.
    #[must_use]
    pub const fn frontier_default() -> Self {
        Self {
            elapsed_interval_ms: 20_000,
            document_interval: 10_000,
            frame_interval: 25_000,
        }
    }

    /// Returns the maximum elapsed milliseconds allowed between writeouts.
    #[must_use]
    pub const fn max_elapsed_ms(self) -> u64 {
        self.elapsed_interval_ms
    }

    /// Returns the maximum documents allowed between writeouts.
    #[must_use]
    pub const fn max_documents(self) -> u64 {
        self.document_interval
    }

    /// Returns the maximum frames allowed between writeouts.
    #[must_use]
    pub const fn max_frames(self) -> u64 {
        self.frame_interval
    }

    /// Decides whether a partial writeout is due.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::NonMonotonicProgress`] when `current` moved
    /// backward relative to `last_writeout`.
    pub fn writeout_decision(
        self,
        last_writeout: Option<RunProgressCursor>,
        current: RunProgressCursor,
    ) -> Result<WriteoutDecision, ProgressError> {
        let Some(last_writeout) = last_writeout else {
            return Ok(WriteoutDecision::Write {
                reason: WriteoutReason::Initial,
            });
        };

        let delta = current.checked_delta_since(last_writeout)?;
        if delta.elapsed_ms >= self.elapsed_interval_ms {
            return Ok(WriteoutDecision::Write {
                reason: WriteoutReason::Elapsed,
            });
        }
        if delta.documents_seen >= self.document_interval {
            return Ok(WriteoutDecision::Write {
                reason: WriteoutReason::Documents,
            });
        }
        if delta.frames_seen >= self.frame_interval {
            return Ok(WriteoutDecision::Write {
                reason: WriteoutReason::Frames,
            });
        }

        Ok(WriteoutDecision::Wait)
    }
}

/// Completion decision produced by [`CompletionGate::completion_decision`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompletionDecision {
    /// All required evidence is present, no blocker is active, and revision is
    /// current.
    Complete,
    /// Required evidence is still missing.
    Incomplete {
        /// Number of required slots not yet satisfied.
        missing_count: u32,
    },
    /// At least one blocker is active.
    Blocked {
        /// Number of active blockers.
        blocker_count: u16,
    },
    /// The caller checked an older plan revision.
    StaleRevision {
        /// Current plan revision.
        current: u64,
        /// Revision supplied by the caller.
        checked: u64,
    },
}

/// Minimal evidence-backed completion gate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompletionGate {
    required: EvidenceSet,
    satisfied: EvidenceSet,
    blocker_count: u16,
    revision: u64,
}

impl CompletionGate {
    /// Creates a completion gate with at least one required evidence slot.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::EmptyRequiredEvidence`] when `required` is
    /// empty, because a zero-evidence completion gate would always be able to
    /// complete.
    pub const fn new(required: EvidenceSet) -> Result<Self, ProgressError> {
        if required.is_empty() {
            return Err(ProgressError::EmptyRequiredEvidence);
        }

        Ok(Self {
            required,
            satisfied: EvidenceSet::empty(),
            blocker_count: 0,
            revision: 0,
        })
    }

    /// Returns the current revision.
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }

    /// Returns the required evidence set.
    #[must_use]
    pub const fn required(self) -> EvidenceSet {
        self.required
    }

    /// Returns the satisfied evidence set.
    #[must_use]
    pub const fn satisfied(self) -> EvidenceSet {
        self.satisfied
    }

    /// Marks one required evidence slot as satisfied.
    ///
    /// Non-required evidence is ignored. This lets a runtime report extra
    /// observations without making completion easier.
    pub fn satisfy(&mut self, slot: EvidenceSlot) {
        if self.required.contains(slot) {
            self.satisfied = self.satisfied.with(slot);
        }
    }

    /// Adds one active blocker.
    ///
    /// Blockers are represented in the completion decision. Saturating the
    /// count would hide blockers from downstream status and could let a caller
    /// clear the represented count while unrepresented blockers still exist.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::BlockerOverflow`] if the compact blocker
    /// counter is already at its maximum value.
    pub fn add_blocker(&mut self) -> Result<(), ProgressError> {
        self.blocker_count = self
            .blocker_count
            .checked_add(1)
            .ok_or(ProgressError::BlockerOverflow)?;
        Ok(())
    }

    /// Clears one active blocker.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::NoActiveBlocker`] when no blocker is active.
    pub fn clear_blocker(&mut self) -> Result<(), ProgressError> {
        if self.blocker_count == 0 {
            return Err(ProgressError::NoActiveBlocker);
        }

        self.blocker_count -= 1;
        Ok(())
    }

    /// Records that the plan changed and invalidates previous evidence.
    ///
    /// Evidence is revision-bound. A caller may not satisfy one plan, touch the
    /// plan, and then reuse the old satisfied slots under the new revision.
    /// Future selective carry-over must be represented by an explicit
    /// revalidation operation rather than this broad invalidation path.
    ///
    /// # Errors
    ///
    /// Returns [`ProgressError::RevisionOverflow`] if the revision counter is
    /// already at its maximum value.
    pub fn touch_plan(&mut self) -> Result<(), ProgressError> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(ProgressError::RevisionOverflow)?;
        self.satisfied = EvidenceSet::empty();
        Ok(())
    }

    /// Returns the current completion decision.
    ///
    /// `checked_revision` must be the revision observed by the caller when it
    /// performed its evidence check. If the plan changed afterward, completion
    /// is stale and must be rechecked.
    #[must_use]
    pub fn completion_decision(self, checked_revision: u64) -> CompletionDecision {
        if checked_revision != self.revision {
            return CompletionDecision::StaleRevision {
                current: self.revision,
                checked: checked_revision,
            };
        }
        if self.blocker_count > 0 {
            return CompletionDecision::Blocked {
                blocker_count: self.blocker_count,
            };
        }
        if !self.satisfied.covers(self.required) {
            return CompletionDecision::Incomplete {
                missing_count: self.required.without_all(self.satisfied).count(),
            };
        }

        CompletionDecision::Complete
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompletionDecision, CompletionGate, ContinuationDecision, ContinuationGate,
        ContinuationReason, EvidenceSet, EvidenceSlot, MAX_EVIDENCE_SLOTS,
        MAX_WRITEOUT_INTERVAL_MS, ProgressError, RunProgressCursor, WriteoutCadence,
        WriteoutDecision, WriteoutReason,
    };

    fn slot(index: u8) -> EvidenceSlot {
        EvidenceSlot::new(index).expect("test slot is in range")
    }

    #[test]
    fn progress_error_codes_are_unique() {
        let codes = [
            ProgressError::EmptyRequiredEvidence.as_str(),
            ProgressError::EvidenceSlotOutOfRange.as_str(),
            ProgressError::NoActiveBlocker.as_str(),
            ProgressError::BlockerOverflow.as_str(),
            ProgressError::InvalidWriteoutCadence.as_str(),
            ProgressError::NonMonotonicProgress.as_str(),
            ProgressError::RevisionOverflow.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn continuation_reason_codes_are_unique() {
        let codes = [
            ContinuationReason::PartialWriteoutDue {
                reason: WriteoutReason::Elapsed,
            }
            .as_str(),
            ContinuationReason::CompletionIncomplete { missing_count: 1 }.as_str(),
            ContinuationReason::CompletionBlocked { blocker_count: 1 }.as_str(),
            ContinuationReason::CompletionStale {
                current: 2,
                checked: 1,
            }
            .as_str(),
            ContinuationReason::PendingNextAction { action_count: 1 }.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn completion_gate_rejects_empty_required_evidence() {
        assert_eq!(
            CompletionGate::new(EvidenceSet::empty()).expect_err("empty gate rejected"),
            ProgressError::EmptyRequiredEvidence
        );
    }

    #[test]
    fn completion_gate_stays_incomplete_until_all_required_slots_are_satisfied() {
        let required = EvidenceSet::single(slot(1)).with(slot(2));
        let mut gate = CompletionGate::new(required).expect("valid completion gate");

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Incomplete { missing_count: 2 }
        );

        gate.satisfy(slot(1));

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Incomplete { missing_count: 1 }
        );

        gate.satisfy(slot(2));

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Complete
        );
    }

    #[test]
    fn non_required_evidence_does_not_unlock_completion() {
        let mut gate =
            CompletionGate::new(EvidenceSet::single(slot(1))).expect("valid completion gate");

        gate.satisfy(slot(3));

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Incomplete { missing_count: 1 }
        );
    }

    #[test]
    fn blockers_prevent_completion_even_when_evidence_is_satisfied() {
        let mut gate =
            CompletionGate::new(EvidenceSet::single(slot(1))).expect("valid completion gate");
        gate.satisfy(slot(1));
        gate.add_blocker().expect("blocker can be represented");

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Blocked { blocker_count: 1 }
        );

        gate.clear_blocker().expect("blocker can be cleared");

        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Complete
        );
    }

    #[test]
    fn blocker_overflow_is_rejected_instead_of_saturating() {
        let mut gate = CompletionGate {
            required: EvidenceSet::single(slot(1)),
            satisfied: EvidenceSet::single(slot(1)),
            blocker_count: u16::MAX,
            revision: 0,
        };

        assert_eq!(
            gate.add_blocker().expect_err("overflow must be explicit"),
            ProgressError::BlockerOverflow
        );
        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Blocked {
                blocker_count: u16::MAX
            }
        );
    }

    #[test]
    fn stale_revision_prevents_done_reuse_after_plan_change() {
        let mut gate =
            CompletionGate::new(EvidenceSet::single(slot(1))).expect("valid completion gate");
        gate.satisfy(slot(1));
        let checked_revision = gate.revision();

        gate.touch_plan().expect("revision can advance");

        assert_eq!(
            gate.completion_decision(checked_revision),
            CompletionDecision::StaleRevision {
                current: 1,
                checked: 0
            }
        );
    }

    #[test]
    fn plan_change_invalidates_previously_satisfied_evidence() {
        let mut gate =
            CompletionGate::new(EvidenceSet::single(slot(1))).expect("valid completion gate");
        gate.satisfy(slot(1));

        gate.touch_plan().expect("revision can advance");

        assert_eq!(gate.satisfied(), EvidenceSet::empty());
        assert_eq!(
            gate.completion_decision(gate.revision()),
            CompletionDecision::Incomplete { missing_count: 1 }
        );
    }

    #[test]
    fn evidence_slot_cap_is_enforced() {
        assert_eq!(MAX_EVIDENCE_SLOTS, 128);
        assert_eq!(
            EvidenceSlot::new(127).expect("last slot accepted").index(),
            127
        );
        assert_eq!(
            EvidenceSlot::new(128).expect_err("slot beyond compact bitset rejected"),
            ProgressError::EvidenceSlotOutOfRange
        );
    }

    #[test]
    fn writeout_cadence_rejects_zero_or_too_long_thresholds() {
        assert_eq!(
            WriteoutCadence::new(0, 1, 1).expect_err("zero elapsed rejected"),
            ProgressError::InvalidWriteoutCadence
        );
        assert_eq!(
            WriteoutCadence::new(1, 0, 1).expect_err("zero document interval rejected"),
            ProgressError::InvalidWriteoutCadence
        );
        assert_eq!(
            WriteoutCadence::new(1, 1, 0).expect_err("zero frame interval rejected"),
            ProgressError::InvalidWriteoutCadence
        );
        assert_eq!(
            WriteoutCadence::new(MAX_WRITEOUT_INTERVAL_MS + 1, 1, 1)
                .expect_err("too long cadence rejected"),
            ProgressError::InvalidWriteoutCadence
        );
    }

    #[test]
    fn writeout_cadence_accepts_exact_elapsed_cap() {
        let cadence =
            WriteoutCadence::new(MAX_WRITEOUT_INTERVAL_MS, 1, 1).expect("exact cap accepted");

        assert_eq!(cadence.max_elapsed_ms(), MAX_WRITEOUT_INTERVAL_MS);
    }

    #[test]
    fn plan_revision_overflow_is_rejected() {
        let mut gate = CompletionGate {
            required: EvidenceSet::single(slot(1)),
            satisfied: EvidenceSet::empty(),
            blocker_count: 0,
            revision: u64::MAX,
        };

        assert_eq!(
            gate.touch_plan().expect_err("revision overflow rejected"),
            ProgressError::RevisionOverflow
        );
    }

    #[test]
    fn writeout_cadence_requires_initial_partial_artifact() {
        let cadence = WriteoutCadence::frontier_default();

        assert_eq!(
            cadence
                .writeout_decision(None, RunProgressCursor::new(0, 0, 0))
                .expect("initial decision is valid"),
            WriteoutDecision::Write {
                reason: WriteoutReason::Initial
            }
        );
    }

    #[test]
    fn writeout_cadence_triggers_on_elapsed_documents_or_frames() {
        let cadence = WriteoutCadence::new(20_000, 10, 25).expect("valid cadence");
        let last = RunProgressCursor::new(100, 5, 7);

        assert_eq!(
            cadence
                .writeout_decision(Some(last), RunProgressCursor::new(20_100, 5, 7))
                .expect("elapsed decision is valid"),
            WriteoutDecision::Write {
                reason: WriteoutReason::Elapsed
            }
        );
        assert_eq!(
            cadence
                .writeout_decision(Some(last), RunProgressCursor::new(100, 15, 7))
                .expect("document decision is valid"),
            WriteoutDecision::Write {
                reason: WriteoutReason::Documents
            }
        );
        assert_eq!(
            cadence
                .writeout_decision(Some(last), RunProgressCursor::new(100, 5, 32))
                .expect("frame decision is valid"),
            WriteoutDecision::Write {
                reason: WriteoutReason::Frames
            }
        );
    }

    #[test]
    fn writeout_cadence_waits_below_thresholds() {
        let cadence = WriteoutCadence::new(20_000, 10, 25).expect("valid cadence");

        assert_eq!(
            cadence
                .writeout_decision(
                    Some(RunProgressCursor::new(0, 0, 0)),
                    RunProgressCursor::new(19_999, 9, 24),
                )
                .expect("wait decision is valid"),
            WriteoutDecision::Wait
        );
    }

    #[test]
    fn writeout_cadence_rejects_non_monotonic_progress() {
        let cadence = WriteoutCadence::frontier_default();

        assert_eq!(
            cadence
                .writeout_decision(
                    Some(RunProgressCursor::new(10, 10, 10)),
                    RunProgressCursor::new(9, 10, 10),
                )
                .expect_err("backward elapsed rejected"),
            ProgressError::NonMonotonicProgress
        );
    }

    #[test]
    fn continuation_gate_requires_writeout_before_quit() {
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::Complete,
                WriteoutDecision::Write {
                    reason: WriteoutReason::Elapsed
                },
                0,
            ),
            ContinuationDecision::Continue {
                reason: ContinuationReason::PartialWriteoutDue {
                    reason: WriteoutReason::Elapsed
                }
            }
        );
        assert_eq!(
            ContinuationReason::PartialWriteoutDue {
                reason: WriteoutReason::Elapsed
            }
            .as_str(),
            "partial_writeout_due"
        );
    }

    #[test]
    fn continuation_gate_rejects_incomplete_blocked_or_stale_completion() {
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::Incomplete { missing_count: 2 },
                WriteoutDecision::Wait,
                0,
            ),
            ContinuationDecision::Continue {
                reason: ContinuationReason::CompletionIncomplete { missing_count: 2 }
            }
        );
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::Blocked { blocker_count: 1 },
                WriteoutDecision::Wait,
                0,
            ),
            ContinuationDecision::Continue {
                reason: ContinuationReason::CompletionBlocked { blocker_count: 1 }
            }
        );
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::StaleRevision {
                    current: 4,
                    checked: 3
                },
                WriteoutDecision::Wait,
                0,
            ),
            ContinuationDecision::Continue {
                reason: ContinuationReason::CompletionStale {
                    current: 4,
                    checked: 3
                }
            }
        );
    }

    #[test]
    fn continuation_gate_rejects_pending_next_action_after_complete() {
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::Complete,
                WriteoutDecision::Wait,
                3,
            ),
            ContinuationDecision::Continue {
                reason: ContinuationReason::PendingNextAction { action_count: 3 }
            }
        );
    }

    #[test]
    fn continuation_gate_allows_quit_only_when_complete_written_and_idle() {
        assert_eq!(
            ContinuationGate::quit_decision(
                CompletionDecision::Complete,
                WriteoutDecision::Wait,
                0,
            ),
            ContinuationDecision::QuitAllowed
        );
    }
}
