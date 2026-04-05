//! Wire format for genome persistence (topology + learned parameters).
//!
//! The disk structs are private DTO types, separate from the runtime
//! `Network` / `ConnectionGraph`. This keeps the on-disk contract stable
//! even if internal runtime layouts change.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Current wire format version. Bump on breaking layout changes.
pub(crate) const CURRENT_VERSION: u8 = 1;

/// On-disk representation of a network genome.
#[derive(Serialize, Deserialize)]
pub(crate) struct NetworkDiskV1 {
    pub version: u8,
    pub graph: ConnectionGraphDiskV1,
    pub threshold: Vec<u32>,
    pub channel: Vec<u8>,
    pub polarity: Vec<i32>,
}

/// On-disk representation of the connection graph.
#[derive(Serialize, Deserialize)]
pub(crate) struct ConnectionGraphDiskV1 {
    pub neuron_count: usize,
    pub sources: Vec<usize>,
    pub targets: Vec<usize>,
}

/// Validate a deserialized genome before constructing runtime types.
///
/// Returns `Ok(())` if all invariants hold, or a descriptive error string.
pub(crate) fn validate(disk: &NetworkDiskV1) -> Result<(), String> {
    let n = disk.graph.neuron_count;

    // Edge array length match
    if disk.graph.sources.len() != disk.graph.targets.len() {
        return Err(format!(
            "edge array length mismatch: {} sources vs {} targets",
            disk.graph.sources.len(),
            disk.graph.targets.len()
        ));
    }

    // Edge endpoint bounds + no self-loops + no duplicates
    let mut seen = HashSet::with_capacity(disk.graph.sources.len());
    for (i, (&s, &t)) in disk.graph.sources.iter().zip(&disk.graph.targets).enumerate() {
        if s >= n {
            return Err(format!("edge {i}: source {s} >= neuron_count {n}"));
        }
        if t >= n {
            return Err(format!("edge {i}: target {t} >= neuron_count {n}"));
        }
        if s == t {
            return Err(format!("edge {i}: self-loop at neuron {s}"));
        }
        if !seen.insert((s, t)) {
            return Err(format!("edge {i}: duplicate ({s}, {t})"));
        }
    }

    // Param array lengths
    if disk.threshold.len() != n {
        return Err(format!(
            "threshold length {}, expected {n}",
            disk.threshold.len()
        ));
    }
    if disk.channel.len() != n {
        return Err(format!(
            "channel length {}, expected {n}",
            disk.channel.len()
        ));
    }
    if disk.polarity.len() != n {
        return Err(format!(
            "polarity length {}, expected {n}",
            disk.polarity.len()
        ));
    }

    // Value ranges
    for (i, &t) in disk.threshold.iter().enumerate() {
        if t > 15 {
            return Err(format!("threshold[{i}] = {t}, max 15"));
        }
    }
    for (i, &c) in disk.channel.iter().enumerate() {
        if !(1..=8).contains(&c) {
            return Err(format!("channel[{i}] = {c}, must be 1..=8"));
        }
    }
    for (i, &p) in disk.polarity.iter().enumerate() {
        if p != 1 && p != -1 {
            return Err(format!("polarity[{i}] = {p}, must be +1 or -1"));
        }
    }

    Ok(())
}
