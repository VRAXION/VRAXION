//! Deterministic confidence values used by proposal-like records.

use super::CONFIDENCE_PPM_DENOMINATOR;
use super::error::FabricError;

/// Validated proposal confidence in parts per million.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConfidencePpm(u32);

impl ConfidencePpm {
    /// Creates a confidence value in the inclusive range `0..=1_000_000`.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::InvalidConfidence`] when the value is outside
    /// the accepted parts-per-million range.
    pub const fn new(value: u32) -> Result<Self, FabricError> {
        if value > CONFIDENCE_PPM_DENOMINATOR {
            return Err(FabricError::InvalidConfidence);
        }

        Ok(Self(value))
    }

    /// Returns the confidence as parts per million.
    #[must_use]
    pub const fn as_ppm(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{CONFIDENCE_PPM_DENOMINATOR, ConfidencePpm, FabricError};

    #[test]
    fn confidence_accepts_inclusive_ppm_bounds() {
        assert_eq!(ConfidencePpm::new(0).expect("zero accepted").as_ppm(), 0);
        assert_eq!(
            ConfidencePpm::new(CONFIDENCE_PPM_DENOMINATOR)
                .expect("denominator accepted")
                .as_ppm(),
            CONFIDENCE_PPM_DENOMINATOR
        );
    }

    #[test]
    fn confidence_rejects_values_above_denominator() {
        assert_eq!(
            ConfidencePpm::new(CONFIDENCE_PPM_DENOMINATOR + 1).unwrap_err(),
            FabricError::InvalidConfidence
        );
        assert_eq!(
            ConfidencePpm::new(u32::MAX).unwrap_err(),
            FabricError::InvalidConfidence
        );
    }

    #[test]
    fn confidence_round_trips_representative_values() {
        for value in [0, 1, 500_000, 999_999, CONFIDENCE_PPM_DENOMINATOR] {
            assert_eq!(
                ConfidencePpm::new(value).expect("valid ppm").as_ppm(),
                value
            );
        }
    }

    #[test]
    fn confidence_ordering_tracks_ppm_value() {
        assert!(ConfidencePpm::new(1).unwrap() < ConfidencePpm::new(2).unwrap());
        assert_eq!(ConfidencePpm::new(7), ConfidencePpm::new(7));
    }
}
