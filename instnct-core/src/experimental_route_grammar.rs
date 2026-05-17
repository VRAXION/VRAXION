//! Experimental route-grammar construction primitives.
//!
//! This module is intentionally small and explicitly experimental. It exists so
//! research runners can exercise the route-grammar loop through an
//! `instnct-core` owned API surface without promoting the mechanism to the
//! production beta contract.

use std::collections::VecDeque;

/// Directed edge used by the experimental route-grammar subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouteGrammarEdge {
    /// Source node id.
    pub from: usize,
    /// Destination node id.
    pub to: usize,
}

/// Diagnostic label strategy used to repair and prune a candidate route field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteGrammarLabelPolicy {
    /// Prefer frontier expansion labels, then fall back to candidate reachability.
    Frontier,
    /// Prefer prune-residual missing-link labels.
    PruneResidual,
    /// Prefer structural successor and continuity labels.
    GraphInvariant,
    /// Use the best available diagnostic label source.
    Mixed,
}

/// Configuration for one experimental route-grammar construction pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteGrammarConfig {
    /// Maximum repair/prune iterations before returning the best current route.
    pub max_iterations: usize,
    /// Diagnostic label policy used during repair.
    pub label_policy: RouteGrammarLabelPolicy,
    /// Whether exact diagnostic successor labels can override the candidate path.
    pub prefer_diagnostic_successors: bool,
}

impl Default for RouteGrammarConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1,
            label_policy: RouteGrammarLabelPolicy::Mixed,
            prefer_diagnostic_successors: true,
        }
    }
}

/// Input graph for the experimental route-grammar constructor.
pub struct RouteGrammarTask<'a> {
    /// Number of nodes in the route graph.
    pub node_count: usize,
    /// Source node where the route must start.
    pub source: usize,
    /// Target node where the route must end.
    pub target: usize,
    /// Dense public candidate route field.
    pub candidate_edges: &'a [RouteGrammarEdge],
    /// Optional seed successor field, usually produced by a weak route policy.
    pub seed_successors: &'a [RouteGrammarEdge],
    /// Optional autonomous graph-diagnostic successor labels.
    pub diagnostic_successors: &'a [RouteGrammarEdge],
}

/// Structural quality of a constructed route grammar.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteGrammarQuality {
    /// Whether the constructed route reaches the target from the source.
    pub source_to_target_reachable: bool,
    /// Number of nodes with more than one retained successor.
    pub branch_count: usize,
    /// Whether the retained successor field contains a directed cycle.
    pub has_cycle: bool,
    /// Number of retained directed edges.
    pub edge_count: usize,
    /// Number of retained route nodes.
    pub path_len: usize,
}

/// Output of the experimental route-grammar constructor.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteGrammarReport {
    /// Ordered source-to-target node path retained by the constructor.
    pub ordered_path: Vec<usize>,
    /// Directed successor edges retained from `ordered_path`.
    pub ordered_edges: Vec<RouteGrammarEdge>,
    /// Structural quality of the retained route.
    pub quality: RouteGrammarQuality,
    /// Number of repair/prune iterations used.
    pub iterations: usize,
}

/// Error returned for malformed route-grammar inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteGrammarError {
    /// Source node is outside the task graph.
    SourceOutOfBounds {
        /// Number of nodes in the task graph.
        node_count: usize,
        /// Invalid source node.
        source: usize,
    },
    /// Target node is outside the task graph.
    TargetOutOfBounds {
        /// Number of nodes in the task graph.
        node_count: usize,
        /// Invalid target node.
        target: usize,
    },
    /// A supplied edge references a node outside the task graph.
    EdgeOutOfBounds {
        /// Number of nodes in the task graph.
        node_count: usize,
        /// Invalid edge.
        edge: RouteGrammarEdge,
    },
}

/// Construct an ordered successor route from dense candidates and graph diagnostics.
///
/// The constructor is deterministic and deliberately conservative: diagnostic
/// successor labels are allowed to repair the candidate field, but the final
/// output is always a single source-to-target successor chain with no retained
/// branches. This is a research API, not a production routing interface.
pub fn construct_route_grammar(
    task: &RouteGrammarTask<'_>,
    config: RouteGrammarConfig,
) -> Result<RouteGrammarReport, RouteGrammarError> {
    validate_task(task)?;

    let max_iterations = config.max_iterations.max(1);
    if config.prefer_diagnostic_successors {
        if let Some(path) = first_reachable_path(task, task.diagnostic_successors) {
            return Ok(report_from_path(task, path, 1));
        }
    }

    let mut best_path = first_reachable_path(task, task.seed_successors)
        .or_else(|| first_reachable_path(task, task.candidate_edges))
        .unwrap_or_else(|| vec![task.source]);

    for iteration in 0..max_iterations {
        if path_reaches_target(&best_path, task.target) {
            return Ok(report_from_path(task, best_path, iteration + 1));
        }

        let candidate = repair_candidate_path(task, config.label_policy);
        if candidate.len() > best_path.len() || path_reaches_target(&candidate, task.target) {
            best_path = candidate;
        }

        if config.prefer_diagnostic_successors {
            if let Some(path) = first_reachable_path(task, task.diagnostic_successors) {
                best_path = path;
            }
        }
    }

    Ok(report_from_path(task, best_path, max_iterations))
}

fn validate_task(task: &RouteGrammarTask<'_>) -> Result<(), RouteGrammarError> {
    if task.source >= task.node_count {
        return Err(RouteGrammarError::SourceOutOfBounds {
            node_count: task.node_count,
            source: task.source,
        });
    }
    if task.target >= task.node_count {
        return Err(RouteGrammarError::TargetOutOfBounds {
            node_count: task.node_count,
            target: task.target,
        });
    }
    for edge in task
        .candidate_edges
        .iter()
        .chain(task.seed_successors)
        .chain(task.diagnostic_successors)
    {
        if edge.from >= task.node_count || edge.to >= task.node_count {
            return Err(RouteGrammarError::EdgeOutOfBounds {
                node_count: task.node_count,
                edge: *edge,
            });
        }
    }
    Ok(())
}

fn repair_candidate_path(
    task: &RouteGrammarTask<'_>,
    policy: RouteGrammarLabelPolicy,
) -> Vec<usize> {
    match policy {
        RouteGrammarLabelPolicy::Frontier => first_reachable_path(task, task.diagnostic_successors)
            .or_else(|| first_reachable_path(task, task.candidate_edges)),
        RouteGrammarLabelPolicy::PruneResidual => {
            first_reachable_path(task, task.diagnostic_successors)
                .or_else(|| first_reachable_path(task, task.seed_successors))
        }
        RouteGrammarLabelPolicy::GraphInvariant => {
            let path = first_reachable_path(task, task.seed_successors);
            if path
                .as_ref()
                .is_some_and(|p| path_reaches_target(p, task.target))
            {
                path
            } else {
                first_reachable_path(task, task.diagnostic_successors)
            }
        }
        RouteGrammarLabelPolicy::Mixed => first_reachable_path(task, task.diagnostic_successors)
            .or_else(|| first_reachable_path(task, task.seed_successors))
            .or_else(|| first_reachable_path(task, task.candidate_edges)),
    }
    .unwrap_or_else(|| vec![task.source])
}

fn first_reachable_path(
    task: &RouteGrammarTask<'_>,
    edges: &[RouteGrammarEdge],
) -> Option<Vec<usize>> {
    let mut outgoing = vec![Vec::new(); task.node_count];
    for edge in edges {
        if edge.from != edge.to {
            outgoing[edge.from].push(edge.to);
        }
    }
    for next in &mut outgoing {
        next.sort_unstable();
        next.dedup();
    }

    let mut parent = vec![None; task.node_count];
    let mut seen = vec![false; task.node_count];
    let mut queue = VecDeque::new();
    seen[task.source] = true;
    queue.push_back(task.source);
    while let Some(node) = queue.pop_front() {
        if node == task.target {
            break;
        }
        for &next in &outgoing[node] {
            if !seen[next] {
                seen[next] = true;
                parent[next] = Some(node);
                queue.push_back(next);
            }
        }
    }
    if !seen[task.target] {
        return None;
    }

    let mut path = Vec::new();
    let mut cur = task.target;
    path.push(cur);
    while cur != task.source {
        let prev = parent[cur]?;
        cur = prev;
        path.push(cur);
    }
    path.reverse();
    Some(path)
}

fn report_from_path(
    task: &RouteGrammarTask<'_>,
    ordered_path: Vec<usize>,
    iterations: usize,
) -> RouteGrammarReport {
    let ordered_edges = ordered_path
        .windows(2)
        .map(|pair| RouteGrammarEdge {
            from: pair[0],
            to: pair[1],
        })
        .collect::<Vec<_>>();
    let quality = quality_for_path(task, &ordered_path, &ordered_edges);
    RouteGrammarReport {
        ordered_path,
        ordered_edges,
        quality,
        iterations,
    }
}

fn quality_for_path(
    task: &RouteGrammarTask<'_>,
    ordered_path: &[usize],
    ordered_edges: &[RouteGrammarEdge],
) -> RouteGrammarQuality {
    RouteGrammarQuality {
        source_to_target_reachable: path_reaches_target(ordered_path, task.target),
        branch_count: 0,
        has_cycle: has_path_cycle(ordered_path),
        edge_count: ordered_edges.len(),
        path_len: ordered_path.len(),
    }
}

fn path_reaches_target(path: &[usize], target: usize) -> bool {
    path.last().copied() == Some(target)
}

fn has_path_cycle(path: &[usize]) -> bool {
    let mut seen = Vec::new();
    for &node in path {
        if seen.contains(&node) {
            return true;
        }
        seen.push(node);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        construct_route_grammar, RouteGrammarConfig, RouteGrammarEdge, RouteGrammarLabelPolicy,
        RouteGrammarTask,
    };

    #[test]
    fn diagnostic_successors_repair_missing_seed_route() {
        let candidates = [
            RouteGrammarEdge { from: 0, to: 1 },
            RouteGrammarEdge { from: 1, to: 2 },
            RouteGrammarEdge { from: 2, to: 3 },
        ];
        let seed = [RouteGrammarEdge { from: 0, to: 1 }];
        let task = RouteGrammarTask {
            node_count: 4,
            source: 0,
            target: 3,
            candidate_edges: &candidates,
            seed_successors: &seed,
            diagnostic_successors: &candidates,
        };

        let report = construct_route_grammar(
            &task,
            RouteGrammarConfig {
                max_iterations: 2,
                label_policy: RouteGrammarLabelPolicy::Mixed,
                prefer_diagnostic_successors: true,
            },
        )
        .unwrap();

        assert_eq!(report.ordered_path, vec![0, 1, 2, 3]);
        assert!(report.quality.source_to_target_reachable);
        assert!(!report.quality.has_cycle);
    }
}
