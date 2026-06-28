//! Versioned matrix-shape profiles for the anonymous fabric.
//!
//! A [`MatrixShape`] says only how large one matrix is. A shape profile says
//! how large the major runtime fields are allowed to be in one named operating
//! mode. This keeps tiny canaries from accidentally becoming the perceived
//! production limit, while also preventing experimental runs from inventing
//! incompatible dimensions ad hoc.

use super::error::FabricError;
use super::matrix::MatrixShape;

/// Named runtime matrix layer inside the anonymous fabric.
///
/// This enum is descriptive only. The matrix cells remain anonymous `u8`
/// coordinates; layer names describe the runtime boundary that owns the matrix.
///
/// Changeability: medium. Adding a layer is easy mechanically, but it changes
/// profile completeness and GUI/runtime expectations, so every new layer needs
/// an explicit shape in every supported profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FabricMatrixLayer {
    /// Raw observation workspace read by Prismions.
    Observation,
    /// Temporary proposal workspace written by Prismions and read by Agency.
    Proposal,
    /// Agency-committed active state workspace.
    Flow,
    /// Stable anchor/context workspace.
    Ground,
}

/// Named matrix-shape profile.
///
/// Changeability: medium/hard. Existing profile meanings should stay stable
/// for benchmark comparability. Add a new profile when changing a profile would
/// reinterpret old run artifacts.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FabricShapeProfileKind {
    /// Tiny deterministic profile used by the Logic-IQ canary.
    LogicIqCanary,
    /// High-capacity local frontier profile for multi-week search.
    FrontierLocal,
}

/// Resolved matrix shapes for the major anonymous fabric fields.
///
/// The current storage path is dense `Vec<u8>`, so the frontier profile is a
/// capacity lock, not a command to run every toy search at maximum size. Large
/// searches should use active-cell sampling, evidence cells, local windows, or
/// sparse candidate addressing instead of uniformly mutating over every cell.
///
/// Changeability: medium. The struct is small and copyable, but profile values
/// become benchmark metadata once run artifacts reference them.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FabricShapeProfile {
    kind: FabricShapeProfileKind,
    observation: MatrixShape,
    proposal: MatrixShape,
    flow: MatrixShape,
    ground: MatrixShape,
}

impl FabricShapeProfile {
    /// Creates a profile from its named kind.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] if the selected baked profile violates the
    /// global matrix cap. This should only happen after an incompatible cap
    /// change.
    pub fn from_kind(kind: FabricShapeProfileKind) -> Result<Self, FabricError> {
        match kind {
            FabricShapeProfileKind::LogicIqCanary => Self::logic_iq_canary(),
            FabricShapeProfileKind::FrontierLocal => Self::frontier_local(),
        }
    }

    /// Creates the tiny Logic-IQ canary profile.
    ///
    /// This profile intentionally keeps the curriculum cheap:
    ///
    /// ```text
    /// observation = 1 x 16 x 16
    /// proposal    = 1 x 4  x 4
    /// flow        = 1 x 4  x 4
    /// ground      = 1 x 16 x 16
    /// ```
    ///
    /// It is not the production size of `AlphaSync`; it is a deterministic smoke
    /// and curriculum profile.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] if any baked shape violates the global matrix
    /// cap. This should only happen after an incompatible cap change.
    pub fn logic_iq_canary() -> Result<Self, FabricError> {
        Ok(Self {
            kind: FabricShapeProfileKind::LogicIqCanary,
            observation: MatrixShape::new(1, 16, 16)?,
            proposal: MatrixShape::new(1, 4, 4)?,
            flow: MatrixShape::new(1, 4, 4)?,
            ground: MatrixShape::new(1, 16, 16)?,
        })
    }

    /// Creates the high-capacity local frontier profile.
    ///
    /// This is the current "do not saturate too early, but keep multi-week
    /// search plausible" profile:
    ///
    /// ```text
    /// observation = 16 x 512 x 512 = 4,194,304 cells
    /// proposal    = 4  x 256 x 256 =   262,144 cells
    /// flow        = 4  x 512 x 512 = 1,048,576 cells
    /// ground      = 4  x 512 x 512 = 1,048,576 cells
    /// ```
    ///
    /// Rationale:
    ///
    /// ```text
    /// Observation:
    ///   uses the full current cap so text, audio, image, and later video
    ///   projections have room without another ABI change.
    ///
    /// Flow/Ground:
    ///   large enough for persistent state, but below the hard cap so dense
    ///   copies remain practical during local search.
    ///
    /// Proposal:
    ///   intentionally smaller because it is temporary and Agency-read. If it
    ///   saturates, the first fix should be better routing, multiple cycles,
    ///   or adapter growth, not blind patch-field expansion.
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] if any baked shape violates the global matrix
    /// cap. This should only happen after an incompatible cap change.
    pub fn frontier_local() -> Result<Self, FabricError> {
        Ok(Self {
            kind: FabricShapeProfileKind::FrontierLocal,
            observation: MatrixShape::new(16, 512, 512)?,
            proposal: MatrixShape::new(4, 256, 256)?,
            flow: MatrixShape::new(4, 512, 512)?,
            ground: MatrixShape::new(4, 512, 512)?,
        })
    }

    /// Returns the profile kind.
    #[must_use]
    pub const fn kind(self) -> FabricShapeProfileKind {
        self.kind
    }

    /// Returns the observation matrix shape.
    #[must_use]
    pub const fn observation(self) -> MatrixShape {
        self.observation
    }

    /// Returns the proposal matrix shape.
    #[must_use]
    pub const fn proposal(self) -> MatrixShape {
        self.proposal
    }

    /// Returns the Flow matrix shape.
    #[must_use]
    pub const fn flow(self) -> MatrixShape {
        self.flow
    }

    /// Returns the Ground matrix shape.
    #[must_use]
    pub const fn ground(self) -> MatrixShape {
        self.ground
    }

    /// Returns one layer shape from the profile.
    #[must_use]
    pub const fn layer(self, layer: FabricMatrixLayer) -> MatrixShape {
        match layer {
            FabricMatrixLayer::Observation => self.observation,
            FabricMatrixLayer::Proposal => self.proposal,
            FabricMatrixLayer::Flow => self.flow,
            FabricMatrixLayer::Ground => self.ground,
        }
    }
}

impl FabricShapeProfileKind {
    /// Returns the stable CLI/artifact code for this profile kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LogicIqCanary => "logic-iq-canary",
            Self::FrontierLocal => "frontier-local",
        }
    }
}
