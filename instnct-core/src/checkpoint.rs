//! Checkpoint persistence — bundle network + projection + metadata into one file.
//!
//! Every experiment should call [`save_checkpoint`] at the end of a run (or
//! periodically). This ensures results are never lost and can be resumed,
//! compared, or loaded for further evolution.
//!
//! # Quick start
//!
//! ```no_run
//! use instnct_core::{save_checkpoint, load_checkpoint, CheckpointMeta, Network, Int8Projection};
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! # let net = Network::new(8);
//! # let proj = Int8Projection::new(5, 27, &mut StdRng::seed_from_u64(42));
//! // Save
//! save_checkpoint("run_42.ckpt", &net, &proj, CheckpointMeta {
//!     step: 30_000,
//!     accuracy: 0.175,
//!     label: "pocket_chain 2p seed=42".into(),
//! }).unwrap();
//!
//! // Load
//! let (net, proj, meta) = load_checkpoint("run_42.ckpt").unwrap();
//! assert_eq!(meta.step, 30_000);
//! ```

use crate::{Int8Projection, Network, NetworkError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

const CHECKPOINT_VERSION: u8 = 1;

/// Metadata stored alongside a checkpoint.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CheckpointMeta {
    /// Evolution step at time of save.
    pub step: usize,
    /// Last known accuracy (0.0–1.0).
    pub accuracy: f64,
    /// Free-form label for identification.
    pub label: String,
}

/// On-disk bundle: network genome + projection weights + metadata.
#[derive(Serialize, Deserialize)]
struct CheckpointDisk {
    version: u8,
    network_bytes: Vec<u8>,
    projection_bytes: Vec<u8>,
    meta: CheckpointMeta,
}

/// Save a checkpoint (network + projection + metadata) to a single file.
///
/// Written atomically (temp + rename) to avoid corruption on crash.
pub fn save_checkpoint(
    path: impl AsRef<Path>,
    net: &Network,
    proj: &Int8Projection,
    meta: CheckpointMeta,
) -> io::Result<()> {
    let disk = CheckpointDisk {
        version: CHECKPOINT_VERSION,
        network_bytes: net.genome_to_bytes(),
        projection_bytes: bincode::serialize(proj)
            .map_err(io::Error::other)?,
        meta,
    };
    let bytes = bincode::serialize(&disk).map_err(io::Error::other)?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a checkpoint from disk. Returns (Network, Int8Projection, metadata).
///
/// # Errors
///
/// Returns [`NetworkError::Io`] on file/deserialization errors.
/// Returns [`NetworkError::Genome`] if the network genome is malformed.
pub fn load_checkpoint(
    path: impl AsRef<Path>,
) -> Result<(Network, Int8Projection, CheckpointMeta), NetworkError> {
    let bytes = fs::read(path).map_err(NetworkError::Io)?;
    let disk: CheckpointDisk = bincode::deserialize(&bytes)
        .map_err(|e| NetworkError::Genome(format!("checkpoint deserialize: {e}")))?;
    if disk.version != CHECKPOINT_VERSION {
        return Err(NetworkError::Genome(format!(
            "checkpoint version {}, expected {CHECKPOINT_VERSION}",
            disk.version
        )));
    }
    let net = Network::genome_from_bytes(&disk.network_bytes)?;
    let proj: Int8Projection = bincode::deserialize(&disk.projection_bytes)
        .map_err(|e| NetworkError::Genome(format!("projection deserialize: {e}")))?;
    Ok((net, proj, disk.meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_network, InitConfig, SdrTable};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_dir() -> std::path::PathBuf {
        std::env::temp_dir().join("instnct_ckpt_test")
    }

    /// Verify every single edge, weight, and parameter survives the round trip.
    #[test]
    fn round_trip_bit_exact() {
        let cfg = InitConfig::phi(256);
        let net = build_network(&cfg, &mut StdRng::seed_from_u64(42));
        let proj = Int8Projection::new(cfg.phi_dim, 27, &mut StdRng::seed_from_u64(99));
        let meta = CheckpointMeta {
            step: 50_000,
            accuracy: 0.175,
            label: "bit-exact test".into(),
        };

        let path = test_dir().join("bit_exact.ckpt");
        save_checkpoint(&path, &net, &proj, meta).unwrap();
        let (net2, proj2, meta2) = load_checkpoint(&path).unwrap();

        // Topology: every edge must match
        assert_eq!(net.edge_count(), net2.edge_count());
        let edges_orig: Vec<_> = net.graph().iter_edges().collect();
        let edges_load: Vec<_> = net2.graph().iter_edges().collect();
        assert_eq!(edges_orig.len(), edges_load.len());
        for (a, b) in edges_orig.iter().zip(&edges_load) {
            assert_eq!(a.source, b.source, "edge source mismatch");
            assert_eq!(a.target, b.target, "edge target mismatch");
        }

        // Parameters: bit-exact
        assert_eq!(net.threshold(), net2.threshold());
        assert_eq!(net.channel(), net2.channel());
        assert_eq!(net.polarity(), net2.polarity());

        // Projection: compare raw genome bytes (catches ALL weight differences)
        let proj_bytes1 = bincode::serialize(&proj).unwrap();
        let proj_bytes2 = bincode::serialize(&proj2).unwrap();
        assert_eq!(proj_bytes1, proj_bytes2, "projection weights differ");

        // Meta
        assert_eq!(meta2.step, 50_000);
        assert!((meta2.accuracy - 0.175).abs() < 1e-15);
        assert_eq!(meta2.label, "bit-exact test");

        let _ = fs::remove_file(&path);
    }

    /// Loaded network produces identical propagation output.
    #[test]
    fn functional_propagation_match() {
        let cfg = InitConfig::phi(256);
        let mut net = build_network(&cfg, &mut StdRng::seed_from_u64(7));
        let proj = Int8Projection::new(cfg.phi_dim, 27, &mut StdRng::seed_from_u64(7));
        let sdr = SdrTable::new(27, cfg.neuron_count, cfg.input_end(), 20,
            &mut StdRng::seed_from_u64(7)).unwrap();

        // Run 10 tokens, record charge + predictions
        let tokens = [0u8, 4, 19, 26, 7, 14, 12, 4, 0, 19];
        let mut charges_orig = Vec::new();
        let mut preds_orig = Vec::new();
        for &t in &tokens {
            net.propagate(sdr.pattern(t as usize), &cfg.propagation).unwrap();
            charges_orig.push(net.charge().to_vec());
            preds_orig.push(proj.predict(&net.charge()[cfg.output_start()..cfg.neuron_count]));
        }

        // Save + load
        let path = test_dir().join("functional.ckpt");
        save_checkpoint(&path, &net, &proj, CheckpointMeta {
            step: 0, accuracy: 0.0, label: "func".into(),
        }).unwrap();
        let (mut net2, proj2, _) = load_checkpoint(&path).unwrap();

        // Replay same tokens — must produce identical results
        // (loaded net is reset, so we reset original too)
        net.reset();
        net2.reset();
        for (i, &t) in tokens.iter().enumerate() {
            net.propagate(sdr.pattern(t as usize), &cfg.propagation).unwrap();
            net2.propagate(sdr.pattern(t as usize), &cfg.propagation).unwrap();
            assert_eq!(net.charge(), net2.charge(), "charge mismatch at token {i}");
            let p1 = proj.predict(&net.charge()[cfg.output_start()..cfg.neuron_count]);
            let p2 = proj2.predict(&net2.charge()[cfg.output_start()..cfg.neuron_count]);
            assert_eq!(p1, p2, "prediction mismatch at token {i}");
        }

        let _ = fs::remove_file(&path);
    }

    /// Empty network (0 edges) round-trips correctly.
    #[test]
    fn empty_network() {
        let net = Network::new(16);
        assert_eq!(net.edge_count(), 0);
        let proj = Int8Projection::new(10, 27, &mut StdRng::seed_from_u64(1));

        let path = test_dir().join("empty.ckpt");
        save_checkpoint(&path, &net, &proj, CheckpointMeta {
            step: 0, accuracy: 0.0, label: "empty".into(),
        }).unwrap();
        let (net2, _, _) = load_checkpoint(&path).unwrap();
        assert_eq!(net2.edge_count(), 0);
        assert_eq!(net2.neuron_count(), 16);

        let _ = fs::remove_file(&path);
    }

    /// Overwrite existing checkpoint file.
    #[test]
    fn overwrite_existing() {
        let net1 = Network::new(8);
        let proj = Int8Projection::new(5, 27, &mut StdRng::seed_from_u64(1));
        let path = test_dir().join("overwrite.ckpt");

        save_checkpoint(&path, &net1, &proj, CheckpointMeta {
            step: 100, accuracy: 0.1, label: "first".into(),
        }).unwrap();

        let cfg = InitConfig::phi(64);
        let net2 = build_network(&cfg, &mut StdRng::seed_from_u64(42));
        let proj2 = Int8Projection::new(cfg.phi_dim, 27, &mut StdRng::seed_from_u64(2));
        save_checkpoint(&path, &net2, &proj2, CheckpointMeta {
            step: 200, accuracy: 0.2, label: "second".into(),
        }).unwrap();

        let (loaded, _, meta) = load_checkpoint(&path).unwrap();
        assert_eq!(meta.step, 200);
        assert_eq!(meta.label, "second");
        assert_eq!(loaded.neuron_count(), 64);

        let _ = fs::remove_file(&path);
    }

    /// Corrupted bytes produce clean error, not panic.
    #[test]
    fn corrupted_bytes_no_panic() {
        let path = test_dir().join("corrupt.ckpt");

        // Garbage
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not a checkpoint").unwrap();
        assert!(load_checkpoint(&path).is_err());

        // Truncated (write valid then chop)
        let net = Network::new(8);
        let proj = Int8Projection::new(5, 27, &mut StdRng::seed_from_u64(1));
        save_checkpoint(&path, &net, &proj, CheckpointMeta {
            step: 0, accuracy: 0.0, label: "x".into(),
        }).unwrap();
        let full = fs::read(&path).unwrap();
        fs::write(&path, &full[..full.len() / 2]).unwrap();
        assert!(load_checkpoint(&path).is_err());

        // Empty file
        fs::write(&path, b"").unwrap();
        assert!(load_checkpoint(&path).is_err());

        let _ = fs::remove_file(&path);
    }

    /// Non-existent file returns error.
    #[test]
    fn missing_file() {
        let result = load_checkpoint("/nonexistent/path/foo.ckpt");
        assert!(result.is_err());
    }

    /// Unicode label survives round trip.
    #[test]
    fn unicode_label() {
        let net = Network::new(8);
        let proj = Int8Projection::new(5, 27, &mut StdRng::seed_from_u64(1));
        let path = test_dir().join("unicode.ckpt");

        let label = "pocket_chain 🧠 ékezetes — 日本語".to_string();
        save_checkpoint(&path, &net, &proj, CheckpointMeta {
            step: 0, accuracy: 0.0, label: label.clone(),
        }).unwrap();
        let (_, _, meta) = load_checkpoint(&path).unwrap();
        assert_eq!(meta.label, label);

        let _ = fs::remove_file(&path);
    }
}
