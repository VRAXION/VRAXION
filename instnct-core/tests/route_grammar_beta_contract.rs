use instnct_core::experimental_route_grammar::{
    construct_route_grammar, RouteGrammarConfig, RouteGrammarEdge, RouteGrammarLabelPolicy,
    RouteGrammarTask,
};

#[test]
fn beta_contract_snapshot_keeps_expected_symbols_and_regression_behavior() {
    let _config = RouteGrammarConfig {
        max_iterations: 4,
        label_policy: RouteGrammarLabelPolicy::Mixed,
        prefer_diagnostic_successors: true,
    };

    let candidates = [
        RouteGrammarEdge { from: 0, to: 2 },
        RouteGrammarEdge { from: 0, to: 1 },
        RouteGrammarEdge { from: 1, to: 2 },
        RouteGrammarEdge { from: 2, to: 3 },
    ];
    let reachable_but_wrong_seed = [RouteGrammarEdge { from: 0, to: 2 }];
    let diagnostics = [
        RouteGrammarEdge { from: 0, to: 1 },
        RouteGrammarEdge { from: 1, to: 2 },
        RouteGrammarEdge { from: 2, to: 3 },
    ];
    let task = RouteGrammarTask {
        node_count: 4,
        source: 0,
        target: 3,
        candidate_edges: &candidates,
        seed_successors: &reachable_but_wrong_seed,
        diagnostic_successors: &diagnostics,
    };

    let report = construct_route_grammar(&task, _config).unwrap();
    assert_eq!(report.ordered_path, vec![0, 1, 2, 3]);
    assert!(report.quality.source_to_target_reachable);
    assert_eq!(report.quality.branch_count, 0);
    assert!(!report.quality.has_cycle);
}
