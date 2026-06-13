//! Egress rendering from committed Agency state only.

use crate::agency::CommitRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressMode {
    CompactAction,
    ShortText,
    LongText,
    MultiResolution,
    NeedMoreInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedOutput {
    pub compact: String,
    pub short: Option<String>,
    pub long: Option<String>,
}

pub fn render_output(mode: EgressMode, record: Option<CommitRecord>) -> RenderedOutput {
    match (mode, record) {
        (EgressMode::NeedMoreInfo, _) | (_, None) => RenderedOutput {
            compact: "NEED_MORE_INFO".to_string(),
            short: Some("The committed state is unresolved; request more evidence.".to_string()),
            long: None,
        },
        (EgressMode::CompactAction, Some(_)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: None,
            long: None,
        },
        (EgressMode::ShortText, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: Some(format!(
                "Feature {} is committed as {}.",
                record.feature_id, record.value
            )),
            long: None,
        },
        (EgressMode::LongText, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: None,
            long: Some(format!(
                "Trace: Agency committed feature {} = {} from validated pocket {} after the proposal passed cycle, trace, ground, and ingress guards.",
                record.feature_id, record.value, record.source_pocket_id
            )),
        },
        (EgressMode::MultiResolution, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: Some(format!(
                "Feature {} is committed as {}.",
                record.feature_id, record.value
            )),
            long: Some(format!(
                "Trace: Agency committed feature {} = {} from validated pocket {} after the proposal passed cycle, trace, ground, and ingress guards.",
                record.feature_id, record.value, record.source_pocket_id
            )),
        },
    }
}
