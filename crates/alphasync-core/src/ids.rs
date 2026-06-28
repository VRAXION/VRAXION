//! Strongly typed identifiers used by the deterministic core.
//!
//! Parsing validates representation only. Code that creates IDs from hashes or
//! positional metadata must also use the type's domain separator and must not
//! derive observation/document IDs from raw corpus text.

use core::fmt;
use core::str::FromStr;

/// Number of lowercase hexadecimal characters in opaque 128-bit identifiers.
pub const OPAQUE_ID_HEX_LEN: usize = 32;

/// Number of lowercase hexadecimal characters in full SHA-256 digests.
pub const SHA256_HEX_LEN: usize = 64;

const TRANSACTION_PREFIX: &str = "tx_";

/// Human-readable category for an identifier parse failure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IdKind {
    /// Canonical operator-definition identity.
    OperatorDefinition,
    /// Runtime identity of one operator instance.
    OperatorInstance,
    /// Runtime identity of one Prismion matrix reader.
    Prismion,
    /// Proposal identity inside a proposal/decision cycle.
    Proposal,
    /// Active registry epoch identity.
    RegistryEpoch,
    /// Durable replacement transaction identity.
    Transaction,
    /// Hash-chained ledger event identity.
    LedgerEvent,
    /// Safe frame-observation identity.
    Observation,
    /// Safe document identity derived from positional metadata.
    Document,
    /// Parent/challenger comparison identity.
    Comparison,
    /// Atomic challenger-bundle identity.
    ChallengerBundle,
    /// Immutable lineage event identity.
    LineageEvent,
    /// Semantic state digest.
    SemanticDigest,
    /// Persisted artifact digest.
    ArtifactDigest,
}

impl IdKind {
    /// Returns the stable machine-readable identifier-kind code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OperatorDefinition => "operator_definition",
            Self::OperatorInstance => "operator_instance",
            Self::Prismion => "prismion",
            Self::Proposal => "proposal",
            Self::RegistryEpoch => "registry_epoch",
            Self::Transaction => "transaction",
            Self::LedgerEvent => "ledger_event",
            Self::Observation => "observation",
            Self::Document => "document",
            Self::Comparison => "comparison",
            Self::ChallengerBundle => "challenger_bundle",
            Self::LineageEvent => "lineage_event",
            Self::SemanticDigest => "semantic_digest",
            Self::ArtifactDigest => "artifact_digest",
        }
    }
}

impl fmt::Display for IdKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Stable reason for an identifier parse failure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IdParseReason {
    /// Input was empty.
    Empty,
    /// Input had the wrong byte length.
    WrongLength,
    /// A required type prefix was absent.
    MissingPrefix,
    /// Input contained a byte outside lowercase ASCII hexadecimal.
    NotLowerHex,
}

impl IdParseReason {
    /// Returns the stable machine-readable reason code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::WrongLength => "wrong_length",
            Self::MissingPrefix => "missing_prefix",
            Self::NotLowerHex => "not_lower_hex",
        }
    }
}

impl fmt::Display for IdParseReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned when a typed identifier string is malformed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IdParseError {
    kind: IdKind,
    reason: IdParseReason,
}

impl IdParseError {
    #[must_use]
    const fn new(kind: IdKind, reason: IdParseReason) -> Self {
        Self { kind, reason }
    }

    /// Returns the identifier category that failed validation.
    #[must_use]
    pub const fn kind(&self) -> IdKind {
        self.kind
    }

    /// Returns the stable machine-readable reason code.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason.as_str()
    }

    /// Returns the typed parse-reason value.
    #[must_use]
    pub const fn reason_kind(&self) -> IdParseReason {
        self.reason
    }
}

impl fmt::Display for IdParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.kind, self.reason)
    }
}

impl std::error::Error for IdParseError {}

macro_rules! define_lower_hex_id {
    (
        $(#[$meta:meta])*
        $name:ident,
        $kind:expr,
        $expected_len:expr,
        $domain_separator:expr
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            /// Domain separator that ID-generation code must prepend before
            /// hashing canonical bytes or safe positional metadata.
            pub const DOMAIN_SEPARATOR: &'static [u8] = $domain_separator;

            /// Parses and validates a lowercase hexadecimal identifier.
            ///
            /// Validation occurs before allocation when parsing from `&str`.
            ///
            /// # Errors
            ///
            /// Returns [`IdParseError`] when the value is empty, has the wrong
            /// length, or contains a non-lowercase-hexadecimal byte.
            pub fn parse(value: &str) -> Result<Self, IdParseError> {
                validate_lower_hex($kind, value, $expected_len)?;
                Ok(Self(value.to_owned()))
            }

            /// Returns the validated identifier as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consumes the identifier and returns the owned string.
            #[must_use]
            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdParseError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::parse(value)
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdParseError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                validate_lower_hex($kind, &value, $expected_len)?;
                Ok(Self(value))
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdParseError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::parse(value)
            }
        }
    };
}

define_lower_hex_id!(
    /// Full SHA-256 identity of a canonical operator definition.
    OperatorId,
    IdKind::OperatorDefinition,
    SHA256_HEX_LEN,
    b"vraxion.operator_definition.v1\0"
);

define_lower_hex_id!(
    /// Opaque runtime identity of one operator instance.
    OperatorInstanceId,
    IdKind::OperatorInstance,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.operator_instance.v1\0"
);

define_lower_hex_id!(
    /// Opaque runtime identity of one Prismion matrix reader.
    PrismionId,
    IdKind::Prismion,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.prismion.v1\0"
);

define_lower_hex_id!(
    /// Identity of a proposal written for governance review.
    ProposalId,
    IdKind::Proposal,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.proposal.v1\0"
);

define_lower_hex_id!(
    /// Identity of an active registry epoch.
    RegistryEpochId,
    IdKind::RegistryEpoch,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.registry_epoch.v1\0"
);

define_lower_hex_id!(
    /// Full SHA-256 identity of a hash-chained ledger event.
    LedgerEventId,
    IdKind::LedgerEvent,
    SHA256_HEX_LEN,
    b"vraxion.ledger_event.v1\0"
);

define_lower_hex_id!(
    /// Opaque identity of a safe observation frame.
    ObservationId,
    IdKind::Observation,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.observation_position.v1\0"
);

define_lower_hex_id!(
    /// Opaque document identity derived only from safe positional metadata.
    DocumentId,
    IdKind::Document,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.document_position.v1\0"
);

define_lower_hex_id!(
    /// Opaque identity of one parent/challenger comparison.
    ComparisonId,
    IdKind::Comparison,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.comparison.v1\0"
);

define_lower_hex_id!(
    /// Opaque identity of one atomic challenger bundle.
    ChallengerBundleId,
    IdKind::ChallengerBundle,
    OPAQUE_ID_HEX_LEN,
    b"vraxion.challenger_bundle.v1\0"
);

define_lower_hex_id!(
    /// Full SHA-256 identity of an immutable lineage event.
    LineageEventId,
    IdKind::LineageEvent,
    SHA256_HEX_LEN,
    b"vraxion.lineage_event.v1\0"
);

define_lower_hex_id!(
    /// Full semantic state digest.
    SemanticDigest,
    IdKind::SemanticDigest,
    SHA256_HEX_LEN,
    b"vraxion.semantic_state.v1\0"
);

define_lower_hex_id!(
    /// Full persisted artifact digest.
    ArtifactDigest,
    IdKind::ArtifactDigest,
    SHA256_HEX_LEN,
    b"vraxion.artifact.v1\0"
);

/// Durable replacement transaction identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransactionId(String);

impl TransactionId {
    /// Domain separator for transaction-ID generation.
    pub const DOMAIN_SEPARATOR: &'static [u8] = b"vraxion.replacement_transaction.v1\0";

    /// Parses and validates a transaction ID in `tx_<32 lowercase hex>` form.
    ///
    /// Validation occurs before allocation when parsing from `&str`.
    ///
    /// # Errors
    ///
    /// Returns [`IdParseError`] when the value is empty, misses the `tx_`
    /// prefix, has the wrong suffix length, or contains a non-lowercase-
    /// hexadecimal suffix byte.
    pub fn parse(value: &str) -> Result<Self, IdParseError> {
        validate_prefixed_lower_hex(
            IdKind::Transaction,
            value,
            TRANSACTION_PREFIX,
            OPAQUE_ID_HEX_LEN,
        )?;
        Ok(Self(value.to_owned()))
    }

    /// Returns the validated transaction ID as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the transaction ID and returns the owned string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for TransactionId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for TransactionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for TransactionId {
    type Err = IdParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl TryFrom<String> for TransactionId {
    type Error = IdParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        validate_prefixed_lower_hex(
            IdKind::Transaction,
            &value,
            TRANSACTION_PREFIX,
            OPAQUE_ID_HEX_LEN,
        )?;
        Ok(Self(value))
    }
}

impl TryFrom<&str> for TransactionId {
    type Error = IdParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

fn validate_lower_hex(kind: IdKind, value: &str, expected_len: usize) -> Result<(), IdParseError> {
    if value.is_empty() {
        return Err(IdParseError::new(kind, IdParseReason::Empty));
    }
    if value.len() != expected_len {
        return Err(IdParseError::new(kind, IdParseReason::WrongLength));
    }
    if !value.bytes().all(is_lower_hex_byte) {
        return Err(IdParseError::new(kind, IdParseReason::NotLowerHex));
    }
    Ok(())
}

fn validate_prefixed_lower_hex(
    kind: IdKind,
    value: &str,
    prefix: &str,
    suffix_len: usize,
) -> Result<(), IdParseError> {
    if value.is_empty() {
        return Err(IdParseError::new(kind, IdParseReason::Empty));
    }
    let Some(suffix) = value.strip_prefix(prefix) else {
        return Err(IdParseError::new(kind, IdParseReason::MissingPrefix));
    };
    if suffix.len() != suffix_len {
        return Err(IdParseError::new(kind, IdParseReason::WrongLength));
    }
    if !suffix.bytes().all(is_lower_hex_byte) {
        return Err(IdParseError::new(kind, IdParseReason::NotLowerHex));
    }
    Ok(())
}

const fn is_lower_hex_byte(byte: u8) -> bool {
    matches!(byte, b'0'..=b'9' | b'a'..=b'f')
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactDigest, ChallengerBundleId, ComparisonId, DocumentId, IdKind, IdParseReason,
        LedgerEventId, LineageEventId, OPAQUE_ID_HEX_LEN, ObservationId, OperatorId,
        OperatorInstanceId, PrismionId, ProposalId, RegistryEpochId, SHA256_HEX_LEN,
        SemanticDigest, TransactionId,
    };

    const OPAQUE_HEX: &str = "0123456789abcdef0123456789abcdef";
    const OLD_SHORT_HEX: &str = "0123456789abcdef01234567";
    const FULL_HEX: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn opaque_ids_accept_exact_128_bit_lowercase_hex() {
        let observation = ObservationId::parse(OPAQUE_HEX).expect("valid observation ID");
        let epoch = RegistryEpochId::parse(OPAQUE_HEX).expect("valid epoch ID");

        assert_eq!(observation.as_str().len(), OPAQUE_ID_HEX_LEN);
        assert_eq!(epoch.as_str(), OPAQUE_HEX);
    }

    #[test]
    fn content_addressed_ids_require_full_sha256() {
        let operator = OperatorId::parse(FULL_HEX).expect("valid operator ID");
        let semantic = SemanticDigest::parse(FULL_HEX).expect("valid semantic digest");
        let artifact = ArtifactDigest::parse(FULL_HEX).expect("valid artifact digest");

        assert_eq!(operator.as_str().len(), SHA256_HEX_LEN);
        assert_eq!(semantic.as_str(), FULL_HEX);
        assert_eq!(artifact.as_str(), FULL_HEX);
    }

    #[test]
    fn old_96_bit_operator_ids_are_rejected() {
        let error = OperatorId::parse(OLD_SHORT_HEX).expect_err("truncated hash rejected");

        assert_eq!(error.kind(), IdKind::OperatorDefinition);
        assert_eq!(error.reason_kind(), IdParseReason::WrongLength);
    }

    #[test]
    fn empty_identifier_is_rejected_without_echoing_input() {
        let error = ProposalId::parse("").expect_err("empty ID rejected");

        assert_eq!(error.kind(), IdKind::Proposal);
        assert_eq!(error.reason_kind(), IdParseReason::Empty);
        assert_eq!(error.to_string(), "proposal:empty");
    }

    #[test]
    fn non_hex_ascii_is_rejected_without_echoing_input() {
        let error = ObservationId::parse("g123456789abcdef0123456789abcdef")
            .expect_err("non-hex byte rejected");

        assert_eq!(error.kind(), IdKind::Observation);
        assert_eq!(error.reason_kind(), IdParseReason::NotLowerHex);
        assert_eq!(error.to_string(), "observation:not_lower_hex");
    }

    #[test]
    fn unicode_byte_length_trick_is_rejected_without_echoing_input() {
        let unicode_payload = "é".repeat(SHA256_HEX_LEN / 2);
        let error =
            OperatorId::parse(&unicode_payload).expect_err("non-ASCII byte payload rejected");

        assert_eq!(unicode_payload.len(), SHA256_HEX_LEN);
        assert_eq!(error.kind(), IdKind::OperatorDefinition);
        assert_eq!(error.reason_kind(), IdParseReason::NotLowerHex);
        assert_eq!(error.to_string(), "operator_definition:not_lower_hex");
    }

    #[test]
    fn transaction_id_requires_prefix_and_128_bit_suffix() {
        let transaction = TransactionId::parse("tx_0123456789abcdef0123456789abcdef")
            .expect("valid transaction ID");

        assert_eq!(transaction.as_str(), "tx_0123456789abcdef0123456789abcdef");
    }

    #[test]
    fn transaction_suffix_must_be_lowercase_hex() {
        let error = TransactionId::parse("tx_0123456789ABCDEF0123456789abcdef")
            .expect_err("uppercase transaction suffix rejected");

        assert_eq!(error.kind(), IdKind::Transaction);
        assert_eq!(error.reason_kind(), IdParseReason::NotLowerHex);
        assert_eq!(error.to_string(), "transaction:not_lower_hex");
    }

    #[test]
    fn uppercase_hex_is_rejected_without_echoing_input() {
        let error =
            OperatorId::parse("0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef")
                .expect_err("uppercase rejected");

        assert_eq!(error.kind(), IdKind::OperatorDefinition);
        assert_eq!(error.reason(), "not_lower_hex");
        assert_eq!(error.to_string(), "operator_definition:not_lower_hex");
    }

    #[test]
    fn missing_transaction_prefix_is_rejected() {
        let error = TransactionId::parse(OPAQUE_HEX).expect_err("missing prefix rejected");

        assert_eq!(error.kind(), IdKind::Transaction);
        assert_eq!(error.reason_kind(), IdParseReason::MissingPrefix);
    }

    #[test]
    fn id_kind_codes_are_unique() {
        let codes = [
            IdKind::OperatorDefinition.as_str(),
            IdKind::OperatorInstance.as_str(),
            IdKind::Prismion.as_str(),
            IdKind::Proposal.as_str(),
            IdKind::RegistryEpoch.as_str(),
            IdKind::Transaction.as_str(),
            IdKind::LedgerEvent.as_str(),
            IdKind::Observation.as_str(),
            IdKind::Document.as_str(),
            IdKind::Comparison.as_str(),
            IdKind::ChallengerBundle.as_str(),
            IdKind::LineageEvent.as_str(),
            IdKind::SemanticDigest.as_str(),
            IdKind::ArtifactDigest.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn id_parse_reason_codes_are_unique() {
        let codes = [
            IdParseReason::Empty.as_str(),
            IdParseReason::WrongLength.as_str(),
            IdParseReason::MissingPrefix.as_str(),
            IdParseReason::NotLowerHex.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn domain_separators_are_distinct() {
        let separators = [
            OperatorId::DOMAIN_SEPARATOR,
            OperatorInstanceId::DOMAIN_SEPARATOR,
            PrismionId::DOMAIN_SEPARATOR,
            ProposalId::DOMAIN_SEPARATOR,
            RegistryEpochId::DOMAIN_SEPARATOR,
            TransactionId::DOMAIN_SEPARATOR,
            LedgerEventId::DOMAIN_SEPARATOR,
            ObservationId::DOMAIN_SEPARATOR,
            DocumentId::DOMAIN_SEPARATOR,
            ComparisonId::DOMAIN_SEPARATOR,
            ChallengerBundleId::DOMAIN_SEPARATOR,
            LineageEventId::DOMAIN_SEPARATOR,
            SemanticDigest::DOMAIN_SEPARATOR,
            ArtifactDigest::DOMAIN_SEPARATOR,
        ];

        for (index, left) in separators.iter().enumerate() {
            for right in &separators[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }
}
