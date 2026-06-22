fn main() {
    // Detect `--duckdb` (and `--duck-native-uf`) in argv before
    // `egglog::cli` parses args. The CLI's own duckdb branch rebuilds
    // the egraph from scratch through `EGraph::with_duckdb_backend`,
    // which would drop the experimental commands (`run-schedule`,
    // `multi-extract`, …) and primitives we register here. Building
    // the duckdb-backed egraph ourselves and extending it with the
    // experimental surface up front keeps those alive — the CLI then
    // sees an already-correct egraph and (since `--duckdb` is set on
    // both sides) short-circuits its own rebuild.
    let argv: Vec<String> = std::env::args().collect();
    let want_duckdb = argv.iter().any(|a| a == "--duckdb");
    // FlowLog / Feldera: like `--duckdb`, cli.rs rebuilds a fresh
    // backend-specific egraph and would DROP the experimental primitives
    // (`get-size!`, the rational sort, set-cost) and commands registered here.
    // Pre-build the backend egraph + extend it up front so they survive; cli.rs
    // then short-circuits its rebuild (it sees an already-flowlog/feldera-backed
    // egraph). Without this, `(run R :until (<= N (get-size!)))` fails
    // "Unbound function get-size!" on these backends.
    let want_flowlog = argv.iter().any(|a| a == "--flowlog");
    let want_feldera = argv.iter().any(|a| a == "--feldera");
    // Honor both the duckdb-specific `--duck-native-uf` and the unified
    // `--native-uf` (PR #782) so the duckdb egraph is built in native-UF mode
    // when either is set — otherwise the `--native-uf` encoding would emit
    // UF-backed functions against a relational duckdb backend.
    let want_native_uf = argv
        .iter()
        .any(|a| a == "--duck-native-uf" || a == "--native-uf");
    // `--fast-rebuild` engages the duckdb backend's delta-scoped rebuild; the
    // pre-built duckdb egraph must carry it so the flag survives cli.rs's
    // short-circuit (mirror `want_native_uf`).
    let want_fast_rebuild = argv.iter().any(|a| a == "--fast-rebuild");
    // `--proof-testing` implies proofs — the desugar pass rewrites
    // `(check ...)` into `(prove-exists ...)` which needs the proof
    // encoding active. Without this, cli.rs's `args.proof_testing`
    // branch would try `with_proofs_enabled()` after construction,
    // and that path clones the egraph for `original_typechecking`,
    // hitting duckdb's unimplemented `clone_boxed`.
    let want_proofs = argv
        .iter()
        .any(|a| a == "--proofs" || a == "--proof-testing");
    let want_wcoj = argv.iter().any(|a| a == "--wcoj");
    let egraph = if want_duckdb {
        egglog_experimental::new_experimental_egraph_duckdb(egglog::DuckBackendConfig {
            native_uf: want_native_uf,
            fast_rebuild: want_fast_rebuild,
            proofs: want_proofs,
        })
        .expect("failed to start DuckDB-backed experimental egraph")
    } else if want_flowlog {
        egglog_experimental::new_experimental_egraph_flowlog(egglog::FlowlogBackendConfig {
            native_uf: want_native_uf,
            fast_rebuild: want_fast_rebuild,
            wcoj: want_wcoj,
        })
        .expect("failed to start FlowLog-backed experimental egraph")
    } else if want_feldera {
        egglog_experimental::new_experimental_egraph_feldera(egglog::FelderaBackendConfig {
            native_uf: want_native_uf,
            fast_rebuild: want_fast_rebuild,
        })
        .expect("failed to start Feldera-backed experimental egraph")
    } else {
        egglog_experimental::new_experimental_egraph()
    };
    egglog::cli(egraph)
}
