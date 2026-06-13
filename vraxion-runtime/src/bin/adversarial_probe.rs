use std::env;
use std::time::Instant;

use vraxion_runtime::{
    agency_decide, corrupt_crc, demo_case_insert_before_frame, encode_frame, ingress_to_proposal,
    insert_bit, render_output, safe_filler, select_text_mode, Action, AgencyState, EgressMode,
    Proposal, ProposalKind, TextMode, TextProfile,
};

#[derive(Default)]
struct Metrics {
    cases: u64,
    success: u64,
    false_commit: u64,
    false_frame: u64,
    wrong_feature: u64,
    reject_success: u64,
    text_success: u64,
    egress_success: u64,
}

fn add_case(metrics: &mut Metrics, ok: bool, false_frame: bool, wrong_feature: bool) {
    metrics.cases += 1;
    metrics.success += ok as u64;
    metrics.false_frame += false_frame as u64;
    metrics.wrong_feature += wrong_feature as u64;
}

fn run_binary_cases(rounds: u64, metrics: &mut Metrics) {
    for idx in 0..rounds {
        let requested = (idx % 31) as u8;
        let value = (idx & 1) as u8;
        let wrong = (requested + 7) % 32;

        let clean = {
            let mut stream = safe_filler(12);
            stream.extend(encode_frame(requested, value, 1, 1));
            stream
        };
        let insert = demo_case_insert_before_frame().0;
        let corrupted = {
            let mut stream = safe_filler(8);
            stream.extend(corrupt_crc(&encode_frame(requested, value, 1, 2)));
            stream
        };
        let wrong_feature = {
            let mut stream = safe_filler(8);
            stream.extend(encode_frame(wrong, value ^ 1, 1, 3));
            stream
        };
        let decoy_then_valid = {
            let mut stream = safe_filler(8);
            stream.extend(encode_frame(wrong, value ^ 1, 1, 4));
            stream.extend(safe_filler(5));
            stream.extend(encode_frame(requested, value, 1, 5));
            stream
        };
        let conflict = {
            let mut stream = safe_filler(8);
            stream.extend(encode_frame(requested, value, 1, 6));
            stream.extend(safe_filler(5));
            stream.extend(encode_frame(requested, value ^ 1, 1, 7));
            stream
        };
        let payload_slip_repeat = {
            let frame = encode_frame(requested, value, 1, 8);
            let slipped = insert_bit(&frame, 18, 1);
            let mut stream = safe_filler(8);
            stream.extend(slipped);
            stream.extend(safe_filler(5));
            stream.extend(frame);
            stream
        };

        for (stream, feature, expected) in [
            (clean, requested, Some(value)),
            (insert, 9, Some(1)),
            (corrupted, requested, None),
            (wrong_feature, requested, None),
            (decoy_then_valid, requested, Some(value)),
            (conflict, requested, None),
            (payload_slip_repeat, requested, Some(value)),
        ] {
            let (decoded, proposal) = ingress_to_proposal(1, 42, feature, &stream);
            let (action, record) = agency_decide(AgencyState { cycle_id: 1 }, proposal);
            let ok = match expected {
                Some(v) => action == Action::CommitEvidence && record.map(|r| r.value) == Some(v),
                None => action != Action::CommitEvidence && record.is_none(),
            };
            add_case(
                metrics,
                ok,
                action == Action::CommitEvidence && expected.is_none(),
                decoded.selected_feature.is_some() && decoded.selected_feature != Some(feature),
            );
        }
    }
}

fn run_text_and_egress_cases(metrics: &mut Metrics) {
    let text_cases = [
        (
            TextProfile {
                byte_len: 200,
                evidence_available: true,
                boundary_risk: 0,
                integrity_risk: 0,
                requires_clean_long: false,
            },
            TextMode::FastDefault,
        ),
        (
            TextProfile {
                byte_len: 900,
                evidence_available: true,
                boundary_risk: 2,
                integrity_risk: 2,
                requires_clean_long: false,
            },
            TextMode::LongCapped,
        ),
        (
            TextProfile {
                byte_len: 1400,
                evidence_available: true,
                boundary_risk: 3,
                integrity_risk: 3,
                requires_clean_long: true,
            },
            TextMode::CleanLong,
        ),
        (
            TextProfile {
                byte_len: 2200,
                evidence_available: true,
                boundary_risk: 4,
                integrity_risk: 4,
                requires_clean_long: true,
            },
            TextMode::AskOrMultiCycle,
        ),
        (
            TextProfile {
                byte_len: 200,
                evidence_available: false,
                boundary_risk: 0,
                integrity_risk: 0,
                requires_clean_long: false,
            },
            TextMode::AskOrMultiCycle,
        ),
    ];
    for (profile, expected) in text_cases {
        metrics.cases += 1;
        let ok = select_text_mode(profile) == expected;
        metrics.success += ok as u64;
        metrics.text_success += ok as u64;
    }

    let state = AgencyState { cycle_id: 7 };
    let stale = Proposal {
        kind: ProposalKind::EvidenceWrite,
        cycle_id: 6,
        source_pocket_id: 3,
        trace_valid: true,
        ground_compatible: true,
        target_feature: Some(1),
        value: Some(1),
    };
    let (stale_action, stale_record) = agency_decide(state, stale);
    metrics.cases += 1;
    let reject_ok = stale_action != Action::CommitEvidence && stale_record.is_none();
    metrics.success += reject_ok as u64;
    metrics.reject_success += reject_ok as u64;
    metrics.false_commit += (stale_action == Action::CommitEvidence) as u64;

    let committed = Proposal {
        cycle_id: 7,
        source_pocket_id: 4,
        ..stale
    };
    let (action, record) = agency_decide(state, committed);
    let rendered = render_output(EgressMode::MultiResolution, record);
    metrics.cases += 1;
    let egress_ok = action == Action::CommitEvidence
        && rendered.compact == "COMMIT_EVIDENCE"
        && rendered
            .short
            .as_ref()
            .is_some_and(|text| text.contains("Feature 1"))
        && rendered
            .long
            .as_ref()
            .is_some_and(|text| text.contains("Agency committed"));
    metrics.success += egress_ok as u64;
    metrics.egress_success += egress_ok as u64;
}

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let start = Instant::now();
    let mut metrics = Metrics::default();
    run_binary_cases(rounds, &mut metrics);
    run_text_and_egress_cases(&mut metrics);
    let seconds = start.elapsed().as_secs_f64();
    let pass = metrics.cases == metrics.success
        && metrics.false_commit == 0
        && metrics.false_frame == 0
        && metrics.wrong_feature == 0;
    println!(
        "{{\"passed\":{},\"cases\":{},\"success\":{},\"false_commit\":{},\"false_frame\":{},\"wrong_feature\":{},\"text_success\":{},\"egress_success\":{},\"seconds\":{:.9},\"rows_per_sec\":{:.3}}}",
        pass,
        metrics.cases,
        metrics.success,
        metrics.false_commit,
        metrics.false_frame,
        metrics.wrong_feature,
        metrics.text_success,
        metrics.egress_success,
        seconds,
        metrics.cases as f64 / seconds.max(0.000001)
    );
}
