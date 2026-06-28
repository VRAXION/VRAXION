//! Anonymous observation matrix geometry and storage.

use super::MAX_MATRIX_CELLS;
use super::error::FabricError;

/// Shape of an anonymous observation matrix.
///
/// A shape is only the geometry of a matrix, not the matrix contents. The
/// runtime interprets cells by coordinate:
///
/// ```text
/// matrix[plane, row, column] -> u8 value
/// ```
///
/// The three axes are stored as `u16` because that gives large local canvases
/// without making every address a wide `u32` record. The actual allocation
/// limit is [`MAX_MATRIX_CELLS`], not the theoretical `u16::MAX` axis value.
///
/// `cell_count` is cached after validation so downstream code can allocate and
/// compare matrix sizes without recomputing the product or risking a different
/// overflow path.
///
/// Changeability: medium/hard ABI surface. The global cell cap is an easy
/// policy limit, but changing the axis types or coordinate model requires a
/// new versioned matrix/address representation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatrixShape {
    /// Number of independent matrix planes.
    ///
    /// Plane meaning is not encoded here. A caller may use planes for raw
    /// observations, masks, derived lanes, audio/image channels, or other
    /// view layers, but Prismions still see only anonymous coordinates.
    planes: u16,
    /// Number of rows in every plane.
    rows: u16,
    /// Number of columns in every row.
    columns: u16,
    /// Validated total cell count: `planes * rows * columns`.
    ///
    /// This value is guaranteed to be non-zero and no larger than
    /// [`MAX_MATRIX_CELLS`] for every constructed `MatrixShape`.
    cell_count: usize,
}

impl MatrixShape {
    /// Creates a non-empty bounded matrix shape.
    ///
    /// This is the first hard gate before an observation matrix can exist. It
    /// rejects three failure classes:
    ///
    /// ```text
    /// zero axis:
    ///   would create edge-case matrices that look valid but contain no work
    ///
    /// arithmetic overflow:
    ///   would corrupt the cell count before allocation or indexing
    ///
    /// cap overflow:
    ///   would allocate a matrix larger than the current local fabric budget
    /// ```
    ///
    /// A successful value means:
    ///
    /// ```text
    /// planes >= 1
    /// rows >= 1
    /// columns >= 1
    /// cell_count == planes * rows * columns
    /// cell_count <= MAX_MATRIX_CELLS
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when any axis is zero, multiplication overflows,
    /// or the resulting matrix is larger than [`MAX_MATRIX_CELLS`].
    pub fn new(planes: u16, rows: u16, columns: u16) -> Result<Self, FabricError> {
        // A zero-sized axis is not a useful neutral observation. Neutral
        // evidence is represented by a valid zero-valued matrix, not by an
        // empty geometry that would special-case every downstream loop.
        if planes == 0 || rows == 0 || columns == 0 {
            return Err(FabricError::ZeroDimension);
        }

        // Use checked arithmetic even though the current cap is small. This
        // keeps the constructor correct if the cap or axis type changes later,
        // and prevents silent wraparound from becoming an allocator/index bug.
        let cell_count = usize::from(planes)
            .checked_mul(usize::from(rows))
            .and_then(|partial| partial.checked_mul(usize::from(columns)))
            .ok_or(FabricError::MatrixTooLarge)?;

        // The cap is a runtime policy knob. It protects search cost and memory
        // pressure; it is not the semantic maximum of the coordinate model.
        if cell_count > MAX_MATRIX_CELLS {
            return Err(FabricError::MatrixTooLarge);
        }

        Ok(Self {
            planes,
            rows,
            columns,
            cell_count,
        })
    }

    /// Returns the plane count.
    #[must_use]
    pub const fn planes(self) -> u16 {
        self.planes
    }

    /// Returns the row count.
    #[must_use]
    pub const fn rows(self) -> u16 {
        self.rows
    }

    /// Returns the column count.
    #[must_use]
    pub const fn columns(self) -> u16 {
        self.columns
    }

    /// Returns the total bounded cell count.
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.cell_count
    }

    /// Returns whether the address is inside this shape.
    ///
    /// A [`MatrixAddress`] is intentionally shape-independent. This method is
    /// the explicit boundary where a raw coordinate becomes valid or invalid
    /// for a particular matrix geometry.
    #[must_use]
    pub const fn contains(self, address: MatrixAddress) -> bool {
        address.plane < self.planes && address.row < self.rows && address.column < self.columns
    }

    /// Returns the canonical dense-cell index for an address inside this shape.
    pub(crate) fn dense_index(self, address: MatrixAddress) -> Option<usize> {
        if !self.contains(address) {
            return None;
        }

        let plane_stride = usize::from(self.rows) * usize::from(self.columns);
        let row_stride = usize::from(self.columns);
        Some(
            usize::from(address.plane) * plane_stride
                + usize::from(address.row) * row_stride
                + usize::from(address.column),
        )
    }
}

/// Address of one cell in an anonymous matrix.
///
/// A matrix address is only a coordinate:
///
/// ```text
/// plane, row, column
/// ```
///
/// It does not carry semantic labels and it does not know which
/// [`MatrixShape`] it belongs to. That is deliberate: Prismion rules, proposal
/// patches, and sparse samples can be built as coordinate-level artifacts, then
/// validated against the concrete matrix shape at the access boundary.
///
/// Changeability: medium/hard ABI surface. The `u16,u16,u16` coordinate model
/// is compact and sufficient for current local canvases; larger axes should be
/// introduced as a versioned address type rather than silently widening this
/// one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatrixAddress {
    /// Matrix plane index.
    plane: u16,
    /// Matrix row index within the plane.
    row: u16,
    /// Matrix column index within the row.
    column: u16,
}

impl MatrixAddress {
    /// Creates a shape-independent address.
    ///
    /// This constructor does not reject large coordinates. Bounds are checked
    /// only when the address is evaluated against a [`MatrixShape`]. This keeps
    /// address construction deterministic and cheap while preserving a single
    /// validation boundary for matrix access.
    #[must_use]
    pub const fn new(plane: u16, row: u16, column: u16) -> Self {
        Self { plane, row, column }
    }

    /// Returns the plane index.
    #[must_use]
    pub const fn plane(self) -> u16 {
        self.plane
    }

    /// Returns the row index.
    #[must_use]
    pub const fn row(self) -> u16 {
        self.row
    }

    /// Returns the column index.
    #[must_use]
    pub const fn column(self) -> u16 {
        self.column
    }
}

/// Compatibility alias for older call sites that still say `CellAddress`.
pub type CellAddress = MatrixAddress;

/// Non-zero sparse matrix sample used to build a canonical matrix.
///
/// Sparse input is a compact way to say:
///
/// ```text
/// every missing cell is zero
/// these listed cells are non-zero
/// ```
///
/// Therefore an explicit zero sample is forbidden. Allowing both "missing" and
/// "present with value 0" would create two byte representations for the same
/// logical matrix, which would weaken deterministic replay, hashing, and
/// mutation/evidence comparison.
///
/// Address bounds are intentionally not checked here. A sample is a
/// shape-independent atom; [`ObservationMatrix::from_sparse`] validates that
/// each sample address fits the concrete matrix shape.
///
/// Changeability: hard canonicalization invariant. The non-zero-only sparse
/// rule should not be relaxed in place.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatrixSample {
    /// Shape-independent address of the sampled non-zero cell.
    address: MatrixAddress,
    /// Non-zero byte value at the sampled address.
    value: u8,
}

impl MatrixSample {
    /// Creates one sparse matrix sample.
    ///
    /// The value is accepted only when it carries information not already
    /// implied by the sparse default. Use a missing sample, not an explicit
    /// zero sample, to represent an inactive/default cell.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::NonCanonicalZeroSparseCell`] when the value is
    /// zero, because missing sparse samples already represent zero.
    pub const fn new(address: MatrixAddress, value: u8) -> Result<Self, FabricError> {
        if value == 0 {
            return Err(FabricError::NonCanonicalZeroSparseCell);
        }

        Ok(Self { address, value })
    }

    /// Returns the sampled address.
    #[must_use]
    pub const fn address(self) -> MatrixAddress {
        self.address
    }

    /// Returns the sampled byte value.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.value
    }
}

/// Compatibility alias for older call sites that still say `CellSample`.
pub type CellSample = MatrixSample;

/// Anonymous observation matrix consumed by Prismions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationMatrix {
    shape: MatrixShape,
    cells: Vec<u8>,
}

impl ObservationMatrix {
    /// Creates a dense matrix from exact cell values.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::DenseCellCountMismatch`] when `cells.len()` does
    /// not match the shape's cell count.
    pub fn from_dense(shape: MatrixShape, cells: Vec<u8>) -> Result<Self, FabricError> {
        if cells.len() != shape.cell_count() {
            return Err(FabricError::DenseCellCountMismatch);
        }

        Ok(Self { shape, cells })
    }

    /// Creates a zero-filled matrix.
    #[must_use]
    pub fn zeroed(shape: MatrixShape) -> Self {
        Self {
            shape,
            cells: vec![0; shape.cell_count()],
        }
    }

    /// Creates a canonical sparse matrix from non-zero samples.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when a sample is out of bounds, duplicated, or
    /// non-canonical.
    pub fn from_sparse(
        shape: MatrixShape,
        mut samples: Vec<MatrixSample>,
    ) -> Result<Self, FabricError> {
        if samples.len() > shape.cell_count() {
            return Err(FabricError::TooManySparseCells);
        }

        if samples.iter().any(|sample| !shape.contains(sample.address)) {
            return Err(FabricError::CellOutOfBounds);
        }

        samples.sort_unstable_by_key(|sample| sample.address);

        if samples
            .windows(2)
            .any(|pair| pair[0].address == pair[1].address)
        {
            return Err(FabricError::DuplicateSparseCell);
        }

        let mut matrix = Self::zeroed(shape);

        for sample in samples {
            let index = shape
                .dense_index(sample.address)
                .ok_or(FabricError::CellOutOfBounds)?;
            matrix.cells[index] = sample.value;
        }

        Ok(matrix)
    }

    /// Returns the matrix shape.
    #[must_use]
    pub const fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Reads one matrix cell.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::CellOutOfBounds`] when the address is outside
    /// this matrix.
    pub fn get(&self, address: MatrixAddress) -> Result<u8, FabricError> {
        let index = self
            .shape
            .dense_index(address)
            .ok_or(FabricError::CellOutOfBounds)?;
        Ok(self.cells[index])
    }

    /// Returns the dense backing cells in canonical plane-row-column order.
    #[must_use]
    pub fn dense_cells(&self) -> &[u8] {
        &self.cells
    }
}
