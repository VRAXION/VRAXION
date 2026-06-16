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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTextFieldError {
    CapacityOverflow,
    NulByteRejected,
    InvalidBitCell,
    InvalidUtf8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTextField {
    byte_capacity: usize,
    cells: Vec<[u8; 8]>,
    committed_byte_len: usize,
    checksum: u32,
}

impl OutputTextField {
    pub fn new(byte_capacity: usize) -> Self {
        Self {
            byte_capacity,
            cells: vec![[0; 8]; byte_capacity],
            committed_byte_len: 0,
            checksum: checksum32(&[]),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.byte_capacity, 8)
    }

    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }

    pub fn committed_byte_len(&self) -> usize {
        self.committed_byte_len
    }

    pub fn active_bit_count(&self) -> usize {
        self.cells
            .iter()
            .flatten()
            .filter(|cell| **cell != 0)
            .count()
    }

    pub fn commit_text_from_agency(&mut self, text: &str) -> Result<(), OutputTextFieldError> {
        self.commit_bytes_from_agency(text.as_bytes())
    }

    pub fn commit_bytes_from_agency(&mut self, bytes: &[u8]) -> Result<(), OutputTextFieldError> {
        if bytes.len() > self.byte_capacity {
            return Err(OutputTextFieldError::CapacityOverflow);
        }
        if bytes.contains(&0) {
            return Err(OutputTextFieldError::NulByteRejected);
        }
        self.cells.fill([0; 8]);
        for (index, byte) in bytes.iter().copied().enumerate() {
            self.cells[index] = byte_to_bits(byte);
        }
        self.committed_byte_len = bytes.len();
        self.checksum = checksum32(bytes);
        Ok(())
    }

    pub fn as_bytes(&self) -> Result<Vec<u8>, OutputTextFieldError> {
        let mut bytes = Vec::with_capacity(self.committed_byte_len);
        for row in self.cells.iter().take(self.committed_byte_len) {
            bytes.push(bits_to_byte(*row)?);
        }
        Ok(bytes)
    }

    pub fn as_text(&self) -> Result<String, OutputTextFieldError> {
        String::from_utf8(self.as_bytes()?).map_err(|_| OutputTextFieldError::InvalidUtf8)
    }

    pub fn verify_checksum(&self) -> bool {
        match self.as_bytes() {
            Ok(bytes) => checksum32(&bytes) == self.checksum,
            Err(_) => false,
        }
    }

    pub fn zero_fill_after_commit(&self) -> bool {
        self.cells
            .iter()
            .skip(self.committed_byte_len)
            .all(|row| row.iter().all(|bit| *bit == 0))
    }

    pub fn bit_at(&self, row: usize, bit: usize) -> Option<u8> {
        self.cells.get(row).and_then(|bits| bits.get(bit)).copied()
    }

    #[cfg(test)]
    pub fn flip_bit_for_test(&mut self, row: usize, bit: usize) -> bool {
        if let Some(cell) = self.cells.get_mut(row).and_then(|bits| bits.get_mut(bit)) {
            *cell ^= 1;
            return true;
        }
        false
    }
}

fn byte_to_bits(byte: u8) -> [u8; 8] {
    [
        (byte >> 7) & 1,
        (byte >> 6) & 1,
        (byte >> 5) & 1,
        (byte >> 4) & 1,
        (byte >> 3) & 1,
        (byte >> 2) & 1,
        (byte >> 1) & 1,
        byte & 1,
    ]
}

fn bits_to_byte(bits: [u8; 8]) -> Result<u8, OutputTextFieldError> {
    let mut byte = 0_u8;
    for bit in bits {
        if bit > 1 {
            return Err(OutputTextFieldError::InvalidBitCell);
        }
        byte = (byte << 1) | bit;
    }
    Ok(byte)
}

fn checksum32(bytes: &[u8]) -> u32 {
    let mut hash = 0x811c9dc5_u32;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{OutputTextField, OutputTextFieldError};

    #[test]
    fn output_text_field_roundtrips_ascii_as_n_by_8_matrix() {
        let mut field = OutputTextField::new(16);
        field.commit_text_from_agency("Szia!").unwrap();

        assert_eq!(field.shape(), (16, 8));
        assert_eq!(field.committed_byte_len(), 5);
        assert_eq!(field.as_text().unwrap(), "Szia!");
        assert_eq!(field.bit_at(0, 0), Some(0));
        assert_eq!(field.bit_at(0, 1), Some(1));
        assert!(field.zero_fill_after_commit());
        assert!(field.verify_checksum());
    }

    #[test]
    fn output_text_field_roundtrips_utf8_bytes() {
        let mut field = OutputTextField::new(64);
        let text = "2251-ben leszel 250 \u{00e9}ves.";
        field.commit_text_from_agency(text).unwrap();

        assert!(field.committed_byte_len() > text.chars().count());
        assert_eq!(field.as_text().unwrap(), text);
        assert!(field.verify_checksum());
    }

    #[test]
    fn output_text_field_rejects_overflow_without_mutating_existing_text() {
        let mut field = OutputTextField::new(8);
        field.commit_text_from_agency("ok").unwrap();

        let result = field.commit_text_from_agency("0123456789");

        assert_eq!(result, Err(OutputTextFieldError::CapacityOverflow));
        assert_eq!(field.as_text().unwrap(), "ok");
        assert!(field.verify_checksum());
    }

    #[test]
    fn output_text_field_rejects_nul_text() {
        let mut field = OutputTextField::new(16);

        let result = field.commit_text_from_agency("bad\0text");

        assert_eq!(result, Err(OutputTextFieldError::NulByteRejected));
        assert_eq!(field.committed_byte_len(), 0);
        assert!(field.zero_fill_after_commit());
    }

    #[test]
    fn output_text_field_detects_tampered_bits() {
        let mut field = OutputTextField::new(16);
        field.commit_text_from_agency("ok").unwrap();

        assert!(field.flip_bit_for_test(0, 0));

        assert!(!field.verify_checksum());
    }
}
