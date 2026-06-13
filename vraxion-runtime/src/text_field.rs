//! Agency-selected Text Field modes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextMode {
    FastDefault,
    LongCapped,
    CleanLong,
    AskOrMultiCycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextProfile {
    pub byte_len: usize,
    pub evidence_available: bool,
    pub boundary_risk: u8,
    pub integrity_risk: u8,
    pub requires_clean_long: bool,
}

pub fn select_text_mode(profile: TextProfile) -> TextMode {
    if !profile.evidence_available {
        return TextMode::AskOrMultiCycle;
    }
    if profile.byte_len <= 416 && profile.boundary_risk <= 1 && profile.integrity_risk <= 1 {
        return TextMode::FastDefault;
    }
    if profile.byte_len <= 1024 && profile.integrity_risk <= 2 && !profile.requires_clean_long {
        return TextMode::LongCapped;
    }
    if profile.byte_len <= 1664 && profile.integrity_risk <= 3 {
        return TextMode::CleanLong;
    }
    TextMode::AskOrMultiCycle
}
