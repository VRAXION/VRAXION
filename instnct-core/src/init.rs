//! Network initialization — the "headquarters" for init decisions.
//!
//! Encodes all proven defaults in one place: phi-overlap geometry,
//! chain highway seeding, density fill, and parameter randomization.
//!
//! # Quick start
//!
//! ```
//! use instnct_core::{build_network, InitConfig};
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! let cfg = InitConfig::phi(256);
//! let mut rng = StdRng::seed_from_u64(42);
//! let net = build_network(&cfg, &mut rng);
//! assert!(net.edge_count() > 0);
//! ```

use crate::{EvolutionConfig, Network, PropagationConfig};
use rand::Rng;
use std::ops::Range;

const PHI: f64 = 1.618_033_988_749_895;

/// Central "headquarters" configuration for network initialization.
///
/// All fields are public — override any individual parameter after construction.
///
/// # Proven defaults (from 87-config highway + forest sweep)
///
/// - Chain-50 init at H=256 raises the worst-seed floor from 6.5% to 16.1%
/// - H≥512 does not benefit from chains (random init is sufficient)
/// - 5% density fill, threshold \[0,7\], channel \[1,8\], 10% inhibitory
/// - Propagation: ticks=6, input=2, decay=6 (no refractory)
#[derive(Clone, Debug)]
pub struct InitConfig {
    /// Total neuron count (H).
    pub neuron_count: usize,
    /// Phi-overlap dimension: `round(H / phi)`.
    /// Input zone: `0..phi_dim`. Output zone: `(H - phi_dim)..H`.
    pub phi_dim: usize,

    /// Number of 3-hop chain highways through the overlap zone.
    /// Set to 0 to disable. Default: 50 for H<512, 0 for H≥512.
    pub chain_count: usize,

    /// Target edge density as percentage of H² (e.g. 5 means 5%).
    pub density_pct: usize,

    /// Inclusive upper bound for random threshold init. Range \[0, max\].
    pub threshold_max: u8,
    /// Inclusive upper bound for random channel init. Range \[1, max\].
    pub channel_max: u8,
    /// Percentage of neurons initialized as inhibitory (0–100).
    pub inhibitory_pct: u32,

    /// Edge cap percentage for evolution. `edge_cap = H² × pct / 100`.
    /// Mutations that increase edge count above this hard cap are rejected.
    pub edge_cap_pct: usize,

    /// Whether evolution accepts fitness ties (after == before).
    /// `false` (default) = strict, only genuine improvements accepted.
    /// `true` = permissive, ties also accepted (risk of neutral drift at cap).
    pub accept_ties: bool,

    /// Propagation config with proven defaults (ticks=6, input=2, decay=6).
    pub propagation: PropagationConfig,
}

impl InitConfig {
    /// Create config for phi-overlap geometry with proven defaults.
    ///
    /// Chain-50 is auto-enabled for H < 512 and disabled for H ≥ 512
    /// (based on 87-config highway sweep: chains help small networks,
    /// larger networks have sufficient random reachability).
    pub fn phi(neuron_count: usize) -> Self {
        let phi_dim = (neuron_count as f64 / PHI).round() as usize;
        let chain_count = if neuron_count >= 512 { 0 } else { 50 };
        Self {
            neuron_count,
            phi_dim,
            chain_count,
            density_pct: 5,
            threshold_max: 7,
            channel_max: 8,
            inhibitory_pct: 10,
            edge_cap_pct: 7,
            accept_ties: false,
            propagation: PropagationConfig {
                ticks_per_token: 6,
                input_duration_ticks: 2,
                decay_interval_ticks: 6,
                use_refractory: false,
            },
        }
    }

    /// Create config for an empty network — zero density, zero chains.
    ///
    /// The evolution builds every edge from scratch. Proven to produce
    /// better circuits than prefilled networks on the addition task
    /// (80% with 83 edges vs 64% with 3400 prefilled edges).
    pub fn empty(neuron_count: usize) -> Self {
        let mut cfg = Self::phi(neuron_count);
        cfg.chain_count = 0;
        cfg.density_pct = 0;
        cfg
    }

    /// Input zone end index (exclusive). Equal to `phi_dim`.
    #[inline]
    pub fn input_end(&self) -> usize {
        self.phi_dim
    }

    /// Output zone start index. Equal to `neuron_count - phi_dim`.
    #[inline]
    pub fn output_start(&self) -> usize {
        self.neuron_count - self.phi_dim
    }

    /// Overlap zone: neurons that serve as both input and output.
    #[inline]
    pub fn overlap_range(&self) -> Range<usize> {
        self.output_start()..self.input_end()
    }

    /// Edge cap for density-capped evolution acceptance.
    #[inline]
    pub fn edge_cap(&self) -> usize {
        self.neuron_count * self.neuron_count * self.edge_cap_pct / 100
    }

    /// Build an [`EvolutionConfig`] derived from this init config.
    #[inline]
    pub fn evolution_config(&self) -> EvolutionConfig {
        EvolutionConfig {
            edge_cap: self.edge_cap(),
            accept_ties: self.accept_ties,
        }
    }
}

/// Build a fully-initialized network from an [`InitConfig`].
///
/// Performs the proven 3-phase initialization:
/// 1. **Chain highways** — `chain_count` random 3-hop paths through the overlap
///    zone (src → hub\_low → hub\_high → tgt). Skipped when `chain_count == 0`
///    or the overlap zone is too small (< 2 neurons).
/// 2. **Random density fill** — adds random edges until `density_pct`% of H²
///    edges exist.
/// 3. **Parameter randomization** — threshold in \[0, `threshold_max`\],
///    channel in \[1, `channel_max`\], `inhibitory_pct`% inhibitory polarity.
///
/// # Example
///
/// ```
/// use instnct_core::{build_network, InitConfig};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let cfg = InitConfig::phi(256);
/// let mut rng = StdRng::seed_from_u64(42);
/// let net = build_network(&cfg, &mut rng);
/// assert!(net.edge_count() > 3000);
/// ```
pub fn build_network(config: &InitConfig, rng: &mut impl Rng) -> Network {
    let h = config.neuron_count;
    let mut net = Network::new(h);

    // Phase 1: chain highways through overlap zone
    let overlap_start = config.output_start();
    let overlap_end = config.input_end();
    if config.chain_count > 0 && overlap_end > overlap_start + 1 {
        let overlap_mid = (overlap_start + overlap_end) / 2;
        for _ in 0..config.chain_count {
            let src = rng.gen_range(0..overlap_start) as u16;
            let hub1 = rng.gen_range(overlap_start..overlap_mid) as u16;
            let hub2 = rng.gen_range(overlap_mid..overlap_end) as u16;
            let tgt = rng.gen_range(overlap_end..h) as u16;
            net.graph_mut().add_edge(src, hub1);
            net.graph_mut().add_edge(hub1, hub2);
            net.graph_mut().add_edge(hub2, tgt);
        }
    }

    // Phase 2: fill to target density with random edges
    let target_edges = h * h * config.density_pct / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target_edges {
            break;
        }
    }

    // Phase 3: randomize neuron parameters
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=config.threshold_max);
        net.channel_mut()[i] = rng.gen_range(1..=config.channel_max);
        if config.inhibitory_pct > 0 && rng.gen_ratio(config.inhibitory_pct, 100) {
            net.polarity_mut()[i] = -1;
        }
    }

    net
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn phi_geometry_256() {
        let cfg = InitConfig::phi(256);
        assert_eq!(cfg.neuron_count, 256);
        assert_eq!(cfg.phi_dim, 158);
        assert_eq!(cfg.input_end(), 158);
        assert_eq!(cfg.output_start(), 98);
        assert_eq!(cfg.overlap_range(), 98..158);
        assert_eq!(cfg.chain_count, 50);
        assert_eq!(cfg.edge_cap(), 4587);
    }

    #[test]
    fn phi_geometry_512_disables_chains() {
        let cfg = InitConfig::phi(512);
        assert_eq!(cfg.phi_dim, 316);
        assert_eq!(cfg.chain_count, 0);
    }

    #[test]
    fn build_network_produces_edges() {
        let cfg = InitConfig::phi(256);
        let mut rng = StdRng::seed_from_u64(42);
        let net = build_network(&cfg, &mut rng);
        assert!(net.edge_count() > 3000, "got {}", net.edge_count());
        assert_eq!(net.neuron_count(), 256);
    }

    #[test]
    fn build_network_has_randomized_params() {
        let cfg = InitConfig::phi(256);
        let mut rng = StdRng::seed_from_u64(42);
        let net = build_network(&cfg, &mut rng);

        let distinct_thresholds: HashSet<u8> = net.threshold().iter().copied().collect();
        assert!(distinct_thresholds.len() > 1, "threshold not randomized");

        let distinct_channels: HashSet<u8> = net.channel().iter().copied().collect();
        assert!(distinct_channels.len() > 1, "channel not randomized");

        let inhibitory = net.polarity().iter().filter(|&&p| p == -1).count();
        assert!(inhibitory > 0, "no inhibitory neurons");
        assert!(inhibitory < 256, "all inhibitory");
    }

    #[test]
    fn build_network_chains_disabled() {
        let mut cfg = InitConfig::phi(256);
        cfg.chain_count = 0;
        let mut rng = StdRng::seed_from_u64(42);
        let net = build_network(&cfg, &mut rng);
        assert!(net.edge_count() > 0);
    }

    #[test]
    fn build_network_deterministic() {
        let cfg = InitConfig::phi(256);
        let net1 = build_network(&cfg, &mut StdRng::seed_from_u64(42));
        let net2 = build_network(&cfg, &mut StdRng::seed_from_u64(42));
        assert_eq!(net1.edge_count(), net2.edge_count());
        assert_eq!(net1.threshold(), net2.threshold());
        assert_eq!(net1.channel(), net2.channel());
        assert_eq!(net1.polarity(), net2.polarity());
    }

    #[test]
    fn evolution_config_derives_correctly() {
        let cfg = InitConfig::phi(256);
        let evo = cfg.evolution_config();
        assert_eq!(evo.edge_cap, 256 * 256 * 7 / 100);
        assert!(!evo.accept_ties);
    }

    #[test]
    fn propagation_config_is_proven() {
        let cfg = InitConfig::phi(256);
        assert_eq!(cfg.propagation.ticks_per_token, 6);
        assert_eq!(cfg.propagation.input_duration_ticks, 2);
        assert_eq!(cfg.propagation.decay_interval_ticks, 6);
        assert!(!cfg.propagation.use_refractory);
    }

    #[test]
    fn smoke_propagate() {
        let cfg = InitConfig::phi(256);
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = build_network(&cfg, &mut rng);

        let mut input = vec![0i32; cfg.neuron_count];
        input[0] = 1;
        input[10] = 1;

        net.propagate(&input, &cfg.propagation).unwrap();

        let total_charge: u32 = net.charge().iter().map(|&c| c as u32).sum();
        assert!(total_charge > 0, "network is dead after propagation");
    }
}
