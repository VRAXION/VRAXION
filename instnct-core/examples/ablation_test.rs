//! Deep ablation test: aggressive dominant-neuron surgery + evolution from ablated state.
//!
//! The 6 dominant neurons [104, 116, 131, 161, 219, 247] are ALWAYS active
//! regardless of input, suppressing all other output neurons.  Setting threshold
//! to 15 didn't kill them because their high in-degree (10-13 edges each)
//! overwhelms any threshold.
//!
//! This test performs 4 experiments:
//!   1. FULL EDGE REMOVAL: delete ALL edges to/from the 6 dominant neurons.
//!      Do other neurons activate?  Is output now diverse across inputs?
//!   2. EVOLUTION FROM ABLATED STATE: if diverse, run 5K evolution steps on the
//!      ablated network.  Does accuracy improve with dominant neurons gone?
//!      If still constant, check for second-tier dominance hierarchy.
//!   3. DOMINANCE HIERARCHY: scan for 2nd/3rd tier dominant neurons that take
//!      over once the top-6 are removed.  How deep does the collapse go?
//!   4. RE-INITIALIZATION: clone the original network, randomly re-wire the
//!      dominant neurons (fresh edges), and test if fresh wiring helps.

use instnct_core::{
    evolution_step_jackpot, load_checkpoint, InitConfig, Int8Projection, Network, StepOutcome,
    VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::time::Instant;

const MAX_CHARGE: i32 = 7;
const DOMINANT_NEURONS: [usize; 6] = [104, 116, 131, 161, 219, 247];

// ── Helpers ─────────────────────────────────────────────────────────

/// Run all test pairs through the network and return per-input charge vectors + predictions.
fn run_test_battery(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    test_pairs: &[(u16, &str)],
    init: &InitConfig,
    h: usize,
) -> (Vec<Vec<u8>>, Vec<usize>, Vec<Vec<usize>>) {
    let e = table.e;
    let output_start = init.output_start();
    let output_count = h - output_start;
    let mut all_charges: Vec<Vec<u8>> = Vec::new();
    let mut all_preds: Vec<usize> = Vec::new();
    let mut all_active: Vec<Vec<usize>> = Vec::new();

    for (pid, _label) in test_pairs {
        net.reset();
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();

        let charges = net.charge_vec(output_start..h);
        let pred = proj.predict(&charges);
        let active: Vec<usize> = (0..output_count)
            .filter(|&i| charges[i] > 0)
            .map(|i| i + output_start)
            .collect();

        all_charges.push(charges);
        all_preds.push(pred);
        all_active.push(active);
    }

    (all_charges, all_preds, all_active)
}

/// Compute diversity metrics from a set of charge vectors and predictions.
fn diversity_metrics(
    charges: &[Vec<u8>],
    preds: &[usize],
    output_count: usize,
) -> (f64, usize, usize) {
    let mut diffs = 0usize;
    let mut pairs = 0usize;
    for i in 0..charges.len() {
        for j in (i + 1)..charges.len() {
            let d = charges[i]
                .iter()
                .zip(charges[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            diffs += d;
            pairs += 1;
        }
    }
    let unique_preds: HashSet<usize> = preds.iter().cloned().collect();
    let avg_diff = if pairs > 0 {
        diffs as f64 / pairs as f64
    } else {
        0.0
    };
    (avg_diff, unique_preds.len(), output_count)
}

/// Print battery results in a compact table.
fn print_battery(
    test_pairs: &[(u16, &str)],
    charges: &[Vec<u8>],
    preds: &[usize],
    active: &[Vec<usize>],
    output_count: usize,
) {
    for (idx, (_, label)) in test_pairs.iter().enumerate() {
        let alive = charges[idx].iter().filter(|&&c| c > 0).count();
        let top_active: Vec<usize> = active[idx].iter().take(15).cloned().collect();
        println!(
            "  '{}' -> pred={:>3}, alive={}/{}, active={:?}{}",
            label,
            preds[idx],
            alive,
            output_count,
            top_active,
            if active[idx].len() > 15 { "..." } else { "" }
        );
    }
}

/// Count edges to/from a set of neurons.
fn count_edges_involving(net: &Network, neurons: &[usize]) -> (usize, usize) {
    let set: HashSet<usize> = neurons.iter().cloned().collect();
    let mut to_count = 0usize;
    let mut from_count = 0usize;
    for edge in net.graph().iter_edges() {
        if set.contains(&(edge.target as usize)) {
            to_count += 1;
        }
        if set.contains(&(edge.source as usize)) {
            from_count += 1;
        }
    }
    (to_count, from_count)
}

/// Remove ALL edges to/from a set of neurons.  Returns the removed edges.
fn remove_all_edges_involving(net: &mut Network, neurons: &[usize]) -> Vec<(u16, u16)> {
    let set: HashSet<usize> = neurons.iter().cloned().collect();
    // Collect first to avoid iterator invalidation
    let to_remove: Vec<(u16, u16)> = net
        .graph()
        .iter_edges()
        .filter(|e| set.contains(&(e.source as usize)) || set.contains(&(e.target as usize)))
        .map(|e| (e.source, e.target))
        .collect();
    for &(src, tgt) in &to_remove {
        net.graph_mut().remove_edge(src, tgt);
    }
    to_remove
}

/// Find neurons active for ALL inputs (dominant) and their tier.
fn find_dominant_neurons(
    net: &mut Network,
    table: &VcbpTable,
    test_pairs: &[(u16, &str)],
    init: &InitConfig,
    h: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let e = table.e;
    let output_start = init.output_start();
    let n_inputs = test_pairs.len() as u32;
    let mut neuron_active_count = vec![0u32; h];

    for (pid, _) in test_pairs {
        net.reset();
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();

        for i in output_start..h {
            if net.spike_data()[i].charge > 0 {
                neuron_active_count[i] += 1;
            }
        }
    }

    let always: Vec<usize> = (output_start..h)
        .filter(|&i| neuron_active_count[i] == n_inputs)
        .collect();
    let sometimes: Vec<usize> = (output_start..h)
        .filter(|&i| neuron_active_count[i] > 0 && neuron_active_count[i] < n_inputs)
        .collect();
    let never: Vec<usize> = (output_start..h)
        .filter(|&i| neuron_active_count[i] == 0)
        .collect();

    (always, sometimes, never)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: ablation_test <checkpoint.bin> <packed.bin>");
        std::process::exit(1);
    }

    let (mut net, proj, meta) = load_checkpoint(&args[0]).expect("load checkpoint");
    let table = VcbpTable::from_packed(Path::new(&args[1])).unwrap();
    let h = net.neuron_count();
    let init = InitConfig::phi(h);
    let e = table.e;
    let output_start = init.output_start();
    let output_count = h - output_start;

    println!("================================================================");
    println!("  DEEP ABLATION TEST — Dominant Neuron Surgery");
    println!("================================================================");
    println!(
        "Checkpoint: step={}, acc={:.2}%",
        meta.step,
        meta.accuracy * 100.0
    );
    println!(
        "Network: H={}, phi_dim={}, output_zone={}..{} ({} neurons)",
        h,
        init.phi_dim,
        output_start,
        h,
        output_count
    );
    println!(
        "Dominant neurons (known): {:?}",
        DOMINANT_NEURONS
    );
    println!("Edges: {}\n", net.edge_count());

    let test_pairs: Vec<(u16, &str)> = vec![
        (VcbpTable::pair_id(b't', b'h'), "th"),
        (VcbpTable::pair_id(b'e', b' '), "e_"),
        (VcbpTable::pair_id(b' ', b't'), "_t"),
        (VcbpTable::pair_id(b'a', b'l'), "al"),
        (VcbpTable::pair_id(b'.', b' '), "._"),
        (VcbpTable::pair_id(b'i', b'n'), "in"),
        (VcbpTable::pair_id(b's', b't'), "st"),
        (VcbpTable::pair_id(b'o', b'n'), "on"),
    ];

    // ════════════════════════════════════════════════════════════════
    // STEP 0: BASELINE — normal output before any surgery
    // ════════════════════════════════════════════════════════════════
    println!("================================================================");
    println!("  STEP 0: BASELINE (unmodified network)");
    println!("================================================================\n");

    let (base_charges, base_preds, base_active) =
        run_test_battery(&mut net, &proj, &table, &test_pairs, &init, h);
    print_battery(&test_pairs, &base_charges, &base_preds, &base_active, output_count);

    let (base_avg_diff, base_unique, _) = diversity_metrics(&base_charges, &base_preds, output_count);
    println!(
        "\n  Diversity: avg_diff={:.1}/{}, unique_preds={}/{}",
        base_avg_diff, output_count, base_unique, test_pairs.len()
    );

    // Confirm the known dominant neurons
    let (always, sometimes, never) = find_dominant_neurons(&mut net, &table, &test_pairs, &init, h);
    println!(
        "\n  Always-active: {} neurons {:?}",
        always.len(),
        &always[..always.len().min(20)]
    );
    println!("  Sometimes-active: {} neurons", sometimes.len());
    println!("  Never-active: {} neurons", never.len());

    // Edge stats for dominant neurons
    for &n in &DOMINANT_NEURONS {
        let sd = &net.spike_data()[n];
        let pol = if net.polarity()[n] > 0 { "+" } else { "-" };
        let in_edges = net
            .graph()
            .iter_edges()
            .filter(|e| e.target as usize == n)
            .count();
        let out_edges = net
            .graph()
            .iter_edges()
            .filter(|e| e.source as usize == n)
            .count();
        println!(
            "    neuron {:>3}: th={:>2}, ch={}, pol={}, in={:>2}, out={:>2}",
            n, sd.threshold, sd.channel, pol, in_edges, out_edges
        );
    }

    // ════════════════════════════════════════════════════════════════
    // EXPERIMENT 1: FULL EDGE REMOVAL
    // ════════════════════════════════════════════════════════════════
    println!("\n================================================================");
    println!("  EXPERIMENT 1: FULL EDGE REMOVAL (all edges to/from dominant 6)");
    println!("================================================================\n");

    // Save original state
    let original_snapshot = net.save_state();

    let (to_dom, from_dom) = count_edges_involving(&net, &DOMINANT_NEURONS);
    println!("  Edges TO dominant: {}", to_dom);
    println!("  Edges FROM dominant: {}", from_dom);
    println!("  Total edges before: {}", net.edge_count());

    let removed = remove_all_edges_involving(&mut net, &DOMINANT_NEURONS);
    println!(
        "  Removed {} edges ({} unique)",
        removed.len(),
        {
            let mut deduped = removed.clone();
            deduped.sort();
            deduped.dedup();
            deduped.len()
        }
    );
    println!("  Total edges after: {}\n", net.edge_count());

    // Test the ablated network
    let (abl_charges, abl_preds, abl_active) =
        run_test_battery(&mut net, &proj, &table, &test_pairs, &init, h);
    print_battery(&test_pairs, &abl_charges, &abl_preds, &abl_active, output_count);

    let (abl_avg_diff, abl_unique, _) = diversity_metrics(&abl_charges, &abl_preds, output_count);
    println!(
        "\n  Diversity: avg_diff={:.1}/{}, unique_preds={}/{}",
        abl_avg_diff, output_count, abl_unique, test_pairs.len()
    );

    // Find NEW dominant neurons after ablation
    let (abl_always, abl_sometimes, abl_never) =
        find_dominant_neurons(&mut net, &table, &test_pairs, &init, h);
    println!(
        "\n  After ablation — Always-active: {} neurons {:?}",
        abl_always.len(),
        &abl_always[..abl_always.len().min(20)]
    );
    println!("  After ablation — Sometimes-active: {}", abl_sometimes.len());
    println!("  After ablation — Never-active: {}", abl_never.len());

    let output_diverse = abl_avg_diff > 0.5 && abl_unique > 1;
    let still_constant = abl_avg_diff < 0.5;

    if output_diverse {
        println!("\n  ** DIVERSE OUTPUT: the dominant neurons WERE suppressing diversity! **");
    } else if still_constant {
        println!("\n  ** STILL CONSTANT: problem is deeper than the top-6 **");
    } else {
        println!("\n  ~ Partial diversity: charges differ but predictions mostly same ~");
    }

    // ════════════════════════════════════════════════════════════════
    // EXPERIMENT 2: EVOLUTION FROM ABLATED STATE  (or)
    //               DOMINANCE HIERARCHY ANALYSIS
    // ════════════════════════════════════════════════════════════════

    if !still_constant {
        // ── Path A: output is diverse — try evolution ──
        println!("\n================================================================");
        println!("  EXPERIMENT 2A: EVOLUTION FROM ABLATED STATE (5K steps)");
        println!("================================================================\n");

        let t_evo = Instant::now();
        let mut evo_net = net.clone();
        let mut evo_proj = proj.clone();
        let evo_config = init.evolution_config();
        let mut rng = StdRng::seed_from_u64(777);
        let mut eval_rng = StdRng::seed_from_u64(778);
        let mut accepted = 0u32;
        let mut rejected = 0u32;

        for step in 0..5000 {
            let outcome = evolution_step_jackpot(
                &mut evo_net,
                &mut evo_proj,
                &mut rng,
                &mut eval_rng,
                |n, p, _er| {
                    // Simple fitness: run 20 test inputs, count unique predictions
                    // + check how many output neurons are alive
                    n.reset();
                    let mut pred_set = HashSet::new();
                    let mut total_alive = 0usize;
                    for i in 0..test_pairs.len().min(6) {
                        n.reset();
                        let emb = table.embed_id(test_pairs[i].0);
                        let mut input = vec![0i32; h];
                        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
                        n.propagate(&input, &init.propagation).unwrap();
                        let charges = n.charge_vec(output_start..h);
                        let pred = p.predict(&charges);
                        pred_set.insert(pred);
                        total_alive += charges.iter().filter(|&&c| c > 0).count();
                    }
                    // Reward: diversity of predictions + alive neurons
                    let diversity = pred_set.len() as f64 / 6.0;
                    let alive_frac = total_alive as f64 / (output_count * 6) as f64;
                    diversity * 0.7 + alive_frac * 0.3
                },
                &evo_config,
                9,
            );
            match outcome {
                StepOutcome::Accepted => accepted += 1,
                StepOutcome::Rejected => rejected += 1,
                StepOutcome::Skipped => {}
            }
            if (step + 1) % 1000 == 0 {
                let (evo_charges, evo_preds, evo_active) =
                    run_test_battery(&mut evo_net, &evo_proj, &table, &test_pairs, &init, h);
                let (evo_diff, evo_unique, _) =
                    diversity_metrics(&evo_charges, &evo_preds, output_count);
                let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
                let avg_alive = evo_active.iter().map(|a| a.len()).sum::<usize>() as f64
                    / test_pairs.len() as f64;
                println!(
                    "  step {:>5}: unique_preds={}/{}, avg_diff={:.1}, avg_alive={:.1}, accept={:.0}%, edges={}",
                    step + 1,
                    evo_unique,
                    test_pairs.len(),
                    evo_diff,
                    avg_alive,
                    rate,
                    evo_net.edge_count()
                );
            }
        }

        println!(
            "\n  Evolution time: {:.1}s",
            t_evo.elapsed().as_secs_f64()
        );

        // Final evolved battery
        println!("\n  Final evolved output:");
        let (evo_charges, evo_preds, evo_active) =
            run_test_battery(&mut evo_net, &evo_proj, &table, &test_pairs, &init, h);
        print_battery(
            &test_pairs,
            &evo_charges,
            &evo_preds,
            &evo_active,
            output_count,
        );
        let (evo_diff, evo_unique, _) = diversity_metrics(&evo_charges, &evo_preds, output_count);
        println!(
            "\n  Diversity: avg_diff={:.1}/{}, unique_preds={}/{}",
            evo_diff, output_count, evo_unique, test_pairs.len()
        );

        // Check for new dominants after evolution
        let (evo_always, _, _) =
            find_dominant_neurons(&mut evo_net, &table, &test_pairs, &init, h);
        println!(
            "  New always-active after evolution: {} {:?}",
            evo_always.len(),
            &evo_always[..evo_always.len().min(20)]
        );
    }

    // ── Path B: DOMINANCE HIERARCHY (always runs) ──
    println!("\n================================================================");
    println!("  EXPERIMENT 3: DOMINANCE HIERARCHY ANALYSIS");
    println!("================================================================\n");

    // Start from original network, iteratively remove dominant tiers
    net.restore_state(&original_snapshot);
    let mut tier = 0usize;
    let mut all_removed: Vec<usize> = Vec::new();
    let max_tiers = 5;

    loop {
        tier += 1;
        if tier > max_tiers {
            println!("  Stopping after {} tiers (max reached).\n", max_tiers);
            break;
        }

        let (tier_always, _tier_sometimes, _tier_never) =
            find_dominant_neurons(&mut net, &table, &test_pairs, &init, h);

        if tier_always.is_empty() {
            println!(
                "  Tier {}: NO always-active neurons remain. Hierarchy exhausted.\n",
                tier
            );
            break;
        }

        println!(
            "  Tier {}: {} always-active neurons: {:?}",
            tier,
            tier_always.len(),
            &tier_always[..tier_always.len().min(20)]
        );

        // Show their stats
        for &n in tier_always.iter().take(10) {
            let in_edges = net
                .graph()
                .iter_edges()
                .filter(|e| e.target as usize == n)
                .count();
            let out_edges = net
                .graph()
                .iter_edges()
                .filter(|e| e.source as usize == n)
                .count();
            let sd = &net.spike_data()[n];
            println!(
                "    neuron {:>3}: th={}, ch={}, in={}, out={}",
                n, sd.threshold, sd.channel, in_edges, out_edges
            );
        }

        // Remove this tier
        let removed = remove_all_edges_involving(&mut net, &tier_always);
        all_removed.extend_from_slice(&tier_always);
        println!(
            "  Removed {} edges from tier {} ({} neurons). Edges remaining: {}",
            removed.len(),
            tier,
            tier_always.len(),
            net.edge_count()
        );

        // Test diversity after this removal
        let (tier_charges, tier_preds, _) =
            run_test_battery(&mut net, &proj, &table, &test_pairs, &init, h);
        let (tier_diff, tier_unique, _) =
            diversity_metrics(&tier_charges, &tier_preds, output_count);
        println!(
            "  After tier {} removal: avg_diff={:.1}, unique_preds={}/{}",
            tier, tier_diff, tier_unique, test_pairs.len()
        );

        if tier_diff > 1.0 && tier_unique > 1 {
            println!("  ** Diversity achieved after removing {} tiers ({} neurons total) **\n",
                tier, all_removed.len());
            break;
        } else {
            println!("  Still constant, going deeper...\n");
        }
    }

    println!(
        "  Total dominant neurons removed across all tiers: {} {:?}",
        all_removed.len(),
        &all_removed[..all_removed.len().min(40)]
    );

    // ════════════════════════════════════════════════════════════════
    // EXPERIMENT 4: RE-INITIALIZATION of dominant neuron edges
    // ════════════════════════════════════════════════════════════════
    println!("\n================================================================");
    println!("  EXPERIMENT 4: RE-INITIALIZATION (fresh random wiring for dominant 6)");
    println!("================================================================\n");

    // Restore original, then replace dominant neuron edges with fresh random ones
    net.restore_state(&original_snapshot);
    let mut rng = StdRng::seed_from_u64(999);

    // Count original edges per dominant neuron
    let mut orig_in_counts = Vec::new();
    let mut orig_out_counts = Vec::new();
    for &n in &DOMINANT_NEURONS {
        let ic = net
            .graph()
            .iter_edges()
            .filter(|e| e.target as usize == n)
            .count();
        let oc = net
            .graph()
            .iter_edges()
            .filter(|e| e.source as usize == n)
            .count();
        orig_in_counts.push(ic);
        orig_out_counts.push(oc);
    }

    // Remove all edges involving dominant neurons
    let removed = remove_all_edges_involving(&mut net, &DOMINANT_NEURONS);
    println!(
        "  Removed {} edges from dominant neurons",
        removed.len()
    );

    // Re-wire each dominant neuron with FEWER random edges (half the original count)
    // to prevent immediate re-dominance, and randomize thresholds higher
    let dom_set: HashSet<usize> = DOMINANT_NEURONS.iter().cloned().collect();
    let non_dom_neurons: Vec<usize> = (0..h).filter(|n| !dom_set.contains(n)).collect();

    for (idx, &n) in DOMINANT_NEURONS.iter().enumerate() {
        // Fresh in-edges: half the original, from random non-dominant neurons
        let new_in = (orig_in_counts[idx] / 2).max(1);
        let new_out = (orig_out_counts[idx] / 2).max(1);

        let mut added_in = 0;
        for _ in 0..new_in * 3 {
            // try up to 3x
            if added_in >= new_in {
                break;
            }
            let src = non_dom_neurons[rng.gen_range(0..non_dom_neurons.len())] as u16;
            if net.graph_mut().add_edge(src, n as u16) {
                added_in += 1;
            }
        }

        let mut added_out = 0;
        for _ in 0..new_out * 3 {
            if added_out >= new_out {
                break;
            }
            let tgt = non_dom_neurons[rng.gen_range(0..non_dom_neurons.len())] as u16;
            if net.graph_mut().add_edge(n as u16, tgt) {
                added_out += 1;
            }
        }

        // Raise threshold to make it harder to fire
        net.spike_data_mut()[n].threshold = rng.gen_range(3..=6);

        println!(
            "    neuron {:>3}: was in={}, out={} -> now in={}, out={}, th={}",
            n,
            orig_in_counts[idx],
            orig_out_counts[idx],
            added_in,
            added_out,
            net.spike_data()[n].threshold
        );
    }

    println!("  Total edges after re-init: {}\n", net.edge_count());

    // Test re-initialized network
    let (reinit_charges, reinit_preds, reinit_active) =
        run_test_battery(&mut net, &proj, &table, &test_pairs, &init, h);
    print_battery(
        &test_pairs,
        &reinit_charges,
        &reinit_preds,
        &reinit_active,
        output_count,
    );

    let (reinit_diff, reinit_unique, _) =
        diversity_metrics(&reinit_charges, &reinit_preds, output_count);
    println!(
        "\n  Diversity: avg_diff={:.1}/{}, unique_preds={}/{}",
        reinit_diff, output_count, reinit_unique, test_pairs.len()
    );

    // Check for new dominants
    let (reinit_always, reinit_sometimes, _) =
        find_dominant_neurons(&mut net, &table, &test_pairs, &init, h);
    println!(
        "  After re-init — Always-active: {} {:?}",
        reinit_always.len(),
        &reinit_always[..reinit_always.len().min(20)]
    );
    println!(
        "  After re-init — Sometimes-active: {}",
        reinit_sometimes.len()
    );

    // ── BONUS: evolution from re-initialized state ──
    if reinit_diff > 0.5 || reinit_unique > 1 {
        println!("\n  Re-init shows promise! Running 3K evolution steps...\n");
        let t_evo = Instant::now();
        let mut evo_proj = proj.clone();
        let evo_config = init.evolution_config();
        let mut evo_rng = StdRng::seed_from_u64(888);
        let mut eval_rng = StdRng::seed_from_u64(889);
        let mut accepted = 0u32;
        let mut rejected = 0u32;

        for step in 0..3000 {
            let outcome = evolution_step_jackpot(
                &mut net,
                &mut evo_proj,
                &mut evo_rng,
                &mut eval_rng,
                |n, p, _er| {
                    let mut pred_set = HashSet::new();
                    let mut total_alive = 0usize;
                    for i in 0..test_pairs.len().min(6) {
                        n.reset();
                        let emb = table.embed_id(test_pairs[i].0);
                        let mut input = vec![0i32; h];
                        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
                        n.propagate(&input, &init.propagation).unwrap();
                        let charges = n.charge_vec(output_start..h);
                        let pred = p.predict(&charges);
                        pred_set.insert(pred);
                        total_alive += charges.iter().filter(|&&c| c > 0).count();
                    }
                    let diversity = pred_set.len() as f64 / 6.0;
                    let alive_frac = total_alive as f64 / (output_count * 6) as f64;
                    diversity * 0.7 + alive_frac * 0.3
                },
                &evo_config,
                9,
            );
            match outcome {
                StepOutcome::Accepted => accepted += 1,
                StepOutcome::Rejected => rejected += 1,
                StepOutcome::Skipped => {}
            }
            if (step + 1) % 1000 == 0 {
                let (evo_charges, evo_preds, evo_active) =
                    run_test_battery(&mut net, &evo_proj, &table, &test_pairs, &init, h);
                let (evo_diff, evo_unique, _) =
                    diversity_metrics(&evo_charges, &evo_preds, output_count);
                let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
                let avg_alive = evo_active.iter().map(|a| a.len()).sum::<usize>() as f64
                    / test_pairs.len() as f64;
                println!(
                    "  step {:>5}: unique_preds={}/{}, avg_diff={:.1}, avg_alive={:.1}, accept={:.0}%, edges={}",
                    step + 1,
                    evo_unique,
                    test_pairs.len(),
                    evo_diff,
                    avg_alive,
                    rate,
                    net.edge_count()
                );
            }
        }
        println!(
            "\n  Re-init evolution time: {:.1}s",
            t_evo.elapsed().as_secs_f64()
        );

        // Final re-init evolved output
        println!("\n  Final re-init evolved output:");
        let (evo_charges, evo_preds, evo_active) =
            run_test_battery(&mut net, &evo_proj, &table, &test_pairs, &init, h);
        print_battery(
            &test_pairs,
            &evo_charges,
            &evo_preds,
            &evo_active,
            output_count,
        );
        let (evo_diff, evo_unique, _) = diversity_metrics(&evo_charges, &evo_preds, output_count);
        println!(
            "\n  Diversity: avg_diff={:.1}/{}, unique_preds={}/{}",
            evo_diff, output_count, evo_unique, test_pairs.len()
        );
    } else {
        println!("\n  Re-init didn't produce diversity either.");
    }

    // ════════════════════════════════════════════════════════════════
    // SUMMARY
    // ════════════════════════════════════════════════════════════════
    println!("\n================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!(
        "  Baseline: avg_diff={:.1}, unique_preds={}/{}",
        base_avg_diff, base_unique, test_pairs.len()
    );
    println!(
        "  After full edge removal (exp 1): avg_diff={:.1}, unique_preds={}/{}",
        abl_avg_diff, abl_unique, test_pairs.len()
    );
    println!(
        "  Dominance hierarchy depth: {} tiers, {} neurons total",
        tier.min(max_tiers),
        all_removed.len()
    );
    println!(
        "  Re-init (exp 4): avg_diff={:.1}, unique_preds={}/{}",
        reinit_diff, reinit_unique, test_pairs.len()
    );

    if abl_avg_diff < 0.5 && reinit_diff < 0.5 {
        println!("\n  CONCLUSION: The constant-output problem is NOT just the 6 dominant neurons.");
        println!("  The entire network topology has collapsed into a single attractor basin.");
        println!("  Next steps: re-init from scratch with anti-dominance constraints,");
        println!("  or add explicit inhibition/competition mechanisms.");
    } else if abl_avg_diff > 1.0 {
        println!("\n  CONCLUSION: Removing dominant neurons RESTORED diversity!");
        println!("  The network has useful structure underneath the dominant collapse.");
        println!("  Next steps: evolve from ablated state with dominance penalty.");
    } else {
        println!("\n  CONCLUSION: Partial effect. The dominant neurons contribute to collapse");
        println!("  but are not the sole cause. Structural intervention needed.");
    }
}
