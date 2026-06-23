//! Golden Refactor Step 01: Observation Matrix
//!
//! Purpose:
//! This file owns the canonical observation boundary.
//! It turns incoming dense or sparse cell data into one valid, bounded,
//! deterministic, anonymous matrix frame before any Prismion reads it.
//!
//! Non-goals:
//! - This file does not interpret meaning.
//! - This file does not run Prismions.
//! - This file does not create proposals.
//! - This file does not mutate runtime state or the SkillStore.
//!
//! Flow:
//! raw observation -> shape gate -> address gate -> canonical cells -> ObservationMatrix
//!
//! Why this file is first:
//! every later stage trusts that the same observation has exactly one internal form.
//! If this boundary is unstable, evidence, consensus, replay, and promotion can drift.

/// Hard cap for one observation frame.
///
/// This keeps the observation boundary explicit instead of allowing callers to
/// allocate arbitrary world-sized inputs.
pub const MAX_OBSERVATION_CELLS: usize = 1_048_576;

/// A bounded observation window.
///
/// Shape answers one question only: how large is the current cell-space?
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MatrixShape {
    planes: usize,
    rows: usize,
    columns: usize,
}

impl MatrixShape {
    pub fn new(planes: usize, rows: usize, columns: usize) -> Result<Self, ObservationMatrixError> {
        if planes == 0 {
            return Err(ObservationMatrixError::EmptyAxis { axis: "planes" });
        }
        if rows == 0 {
            return Err(ObservationMatrixError::EmptyAxis { axis: "rows" });
        }
        if columns == 0 {
            return Err(ObservationMatrixError::EmptyAxis { axis: "columns" });
        }

        let plane_cells = rows
            .checked_mul(columns)
            .ok_or(ObservationMatrixError::CellCountOverflow)?;
        let cell_count = planes
            .checked_mul(plane_cells)
            .ok_or(ObservationMatrixError::CellCountOverflow)?;

        if cell_count > MAX_OBSERVATION_CELLS {
            return Err(ObservationMatrixError::CellCountLimitExceeded {
                cell_count,
                limit: MAX_OBSERVATION_CELLS,
            });
        }

        Ok(Self {
            planes,
            rows,
            columns,
        })
    }

    pub fn planes(self) -> usize {
        self.planes
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn columns(self) -> usize {
        self.columns
    }

    pub fn cell_count(self) -> usize {
        self.planes * self.rows * self.columns
    }

    pub fn contains(self, address: CellAddress) -> bool {
        address.plane < self.planes && address.row < self.rows && address.column < self.columns
    }

    pub fn linear_index(self, address: CellAddress) -> Result<usize, ObservationMatrixError> {
        if !self.contains(address) {
            return Err(ObservationMatrixError::AddressOutOfBounds { address });
        }

        Ok((address.plane * self.rows * self.columns) + (address.row * self.columns) + address.column)
    }
}

/// Coordinate token for one cell.
///
/// A CellAddress is not valid by itself. It becomes valid only when checked
/// against a concrete MatrixShape.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CellAddress {
    pub plane: usize,
    pub row: usize,
    pub column: usize,
}

impl CellAddress {
    pub const fn new(plane: usize, row: usize, column: usize) -> Self {
        Self { plane, row, column }
    }
}

/// One explicit non-zero sparse observation.
///
/// In sparse input, missing means zero. Explicit zero samples are rejected so
/// the same matrix cannot be represented in two different ways.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CellSample {
    pub address: CellAddress,
    pub value: u8,
}

impl CellSample {
    pub fn new(address: CellAddress, value: u8) -> Result<Self, ObservationMatrixError> {
        if value == 0 {
            return Err(ObservationMatrixError::ZeroSparseSample { address });
        }

        Ok(Self { address, value })
    }
}

/// Canonical internal observation frame.
///
/// Dense and sparse inputs both end here: one plane-row-column byte vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationMatrix {
    shape: MatrixShape,
    cells: Vec<u8>,
}

impl ObservationMatrix {
    pub fn from_dense(shape: MatrixShape, cells: Vec<u8>) -> Result<Self, ObservationMatrixError> {
        let expected = shape.cell_count();
        let actual = cells.len();

        if actual != expected {
            return Err(ObservationMatrixError::DenseLengthMismatch { expected, actual });
        }

        Ok(Self { shape, cells })
    }

    pub fn from_sparse(
        shape: MatrixShape,
        samples: impl IntoIterator<Item = CellSample>,
    ) -> Result<Self, ObservationMatrixError> {
        let mut cells = vec![0; shape.cell_count()];
        let mut seen = vec![false; shape.cell_count()];

        for sample in samples {
            let index = shape.linear_index(sample.address)?;

            if sample.value == 0 {
                return Err(ObservationMatrixError::ZeroSparseSample {
                    address: sample.address,
                });
            }

            if seen[index] {
                return Err(ObservationMatrixError::DuplicateSparseCell {
                    address: sample.address,
                });
            }

            seen[index] = true;
            cells[index] = sample.value;
        }

        Ok(Self { shape, cells })
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    pub fn get(&self, address: CellAddress) -> Result<u8, ObservationMatrixError> {
        let index = self.shape.linear_index(address)?;
        Ok(self.cells[index])
    }

    pub fn dense_cells(&self) -> &[u8] {
        &self.cells
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationMatrixError {
    EmptyAxis { axis: &'static str },
    CellCountOverflow,
    CellCountLimitExceeded { cell_count: usize, limit: usize },
    DenseLengthMismatch { expected: usize, actual: usize },
    AddressOutOfBounds { address: CellAddress },
    DuplicateSparseCell { address: CellAddress },
    ZeroSparseSample { address: CellAddress },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_input_must_match_shape() {
        let shape = MatrixShape::new(1, 2, 2).unwrap();
        let err = ObservationMatrix::from_dense(shape, vec![1, 2, 3]).unwrap_err();

        assert_eq!(
            err,
            ObservationMatrixError::DenseLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn sparse_input_rejects_duplicate_cells() {
        let shape = MatrixShape::new(1, 2, 2).unwrap();
        let address = CellAddress::new(0, 1, 1);
        let samples = [
            CellSample::new(address, 7).unwrap(),
            CellSample::new(address, 8).unwrap(),
        ];

        let err = ObservationMatrix::from_sparse(shape, samples).unwrap_err();

        assert_eq!(err, ObservationMatrixError::DuplicateSparseCell { address });
    }

    #[test]
    fn sparse_input_canonicalizes_to_dense_cells() {
        let shape = MatrixShape::new(1, 2, 3).unwrap();
        let samples = [
            CellSample::new(CellAddress::new(0, 0, 2), 5).unwrap(),
            CellSample::new(CellAddress::new(0, 1, 1), 9).unwrap(),
        ];

        let matrix = ObservationMatrix::from_sparse(shape, samples).unwrap();

        assert_eq!(matrix.dense_cells(), &[0, 0, 5, 0, 9, 0]);
    }
}
