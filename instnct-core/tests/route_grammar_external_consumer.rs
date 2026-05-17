use instnct_core::experimental_route_grammar::{
    construct_route_grammar, RouteGrammarConfig, RouteGrammarEdge, RouteGrammarError,
    RouteGrammarTask,
};

fn sample_edges() -> Vec<RouteGrammarEdge> {
    vec![
        RouteGrammarEdge { from: 0, to: 1 },
        RouteGrammarEdge { from: 1, to: 2 },
        RouteGrammarEdge { from: 2, to: 3 },
        RouteGrammarEdge { from: 0, to: 2 },
    ]
}

#[test]
fn external_consumer_gets_ordered_route_and_rejects_bad_inputs() {
    let candidates = sample_edges();
    let seed = [RouteGrammarEdge { from: 0, to: 2 }];
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
        seed_successors: &seed,
        diagnostic_successors: &diagnostics,
    };

    let first = construct_route_grammar(&task, RouteGrammarConfig::default()).unwrap();
    let second = construct_route_grammar(&task, RouteGrammarConfig::default()).unwrap();
    assert_eq!(first.ordered_path, vec![0, 1, 2, 3]);
    assert_eq!(first, second);

    let bad_edge = [RouteGrammarEdge { from: 0, to: 99 }];
    let bad_task = RouteGrammarTask {
        node_count: 4,
        source: 0,
        target: 3,
        candidate_edges: &bad_edge,
        seed_successors: &[],
        diagnostic_successors: &[],
    };
    assert!(matches!(
        construct_route_grammar(&bad_task, RouteGrammarConfig::default()),
        Err(RouteGrammarError::EdgeOutOfBounds { .. })
    ));
}
