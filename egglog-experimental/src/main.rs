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
    // `--native-merge` does congruence inline via the backend's UnionId merge and
    // injects the union edge into the in-core union-find, so it REQUIRES native-UF
    // — provision the backend in native-UF mode when it is set (mirrors cli.rs's
    // `args.native_uf = true` fold for `--native-merge`).
    //
    // PROOF-MODE EXCEPTION (FlowLog / Feldera): `--flowlog --proofs --native-merge`
    // and `--feldera --proofs --native-merge` use the RELATIONAL proof-UF + a proof
    // side-table (the 2-table proof-congruence encoding), NOT the displaced
    // native-UF (which both backends' `add_uf_function` reject in proof mode). So
    // `--native-merge` must NOT imply native-UF there. Mirror cli.rs's identical
    // Step-0 carve-out so the pre-built backend egraph agrees with what cli.rs
    // computes.
    let want_proofs_early = argv
        .iter()
        .any(|a| a == "--proofs" || a == "--proof-testing");
    // FLIP: native `:merge` is the DEFAULT for the term encoding. These
    // experimental backends always term-encode, so native-merge is ON unless
    // `--no-native-merge` (an explicit `--native-merge` is honored too, but
    // redundant). This MUST mirror cli.rs's flip so the pre-built backend's
    // native-UF mode agrees with the encoding cli.rs emits.
    let any_experimental = want_duckdb || want_flowlog || want_feldera;
    let explicit_native_uf = argv
        .iter()
        .any(|a| a == "--duck-native-uf" || a == "--native-uf");
    // Mirror cli.rs's flip: native-merge auto-ons for an experimental backend
    // UNLESS `--no-native-merge` or an EXPLICIT `--native-uf` (which selects the
    // rule-encoded + native-uf path — native-merge + native-uf conflict on these
    // backends, esp. proofs which need the relational proof-UF). An explicit
    // `--native-merge` still force-enables it.
    //
    // DUCKDB + FELDERA + FLOWLOG FAST-RELATIONAL (mirror cli.rs): plain `--duckdb`
    // / `--feldera` / `--flowlog` (non-proof, no explicit native-merge/native-uf)
    // is migrated OFF native-UF onto the fast relational term-encoding — native-
    // merge does NOT auto-on there (congruence stays rule-encoded; duckdb/feldera
    // use the relational δuf fast-rebuild, flowlog lowers the rebuild rules through
    // its general DD rule engine). An explicit `--native-merge` / `--native-uf`
    // with any of these (non-proof) is REJECTED by `egglog::cli` (the old native
    // path is gone on all three).
    let duckdb_fast_relational_default = want_duckdb && !want_proofs_early;
    let feldera_fast_relational_default = want_feldera && !want_proofs_early;
    let flowlog_fast_relational_default = want_flowlog && !want_proofs_early;
    let effective_native_merge = (any_experimental
        && !argv.iter().any(|a| a == "--no-native-merge")
        && !explicit_native_uf
        && !duckdb_fast_relational_default
        && !feldera_fast_relational_default
        && !flowlog_fast_relational_default)
        || argv.iter().any(|a| a == "--native-merge");
    // Proof-mode native-merge on the dataflow/SQL backends (flowlog/feldera/DUCKDB)
    // uses the RELATIONAL proof-UF + a proof side-table (the 2-table encoding), NOT
    // the displaced native-UF (which their `add_uf_function` rejects in proof mode).
    // So native-merge must NOT imply native-UF there. (duckdb included — it now has
    // native-merge proof congruence via `emit_native_congruence_proof`.)
    let single_output_proof_native_merge =
        any_experimental && want_proofs_early && effective_native_merge;
    // Explicit `--native-uf` always wins; otherwise non-proof native-merge on
    // these backends injects the union via the in-core UF-backed function, so it
    // REQUIRES native-UF (and the proof carve-out keeps it OFF for proof mode).
    let want_native_uf =
        explicit_native_uf || (!single_output_proof_native_merge && effective_native_merge);
    // `--fast-rebuild` engages the backend's delta-scoped rebuild; the pre-built
    // backend egraph must carry it so the flag survives cli.rs's short-circuit
    // (mirror `want_native_uf`). DUCKDB + FELDERA FAST-RELATIONAL: default it ON for
    // the migrated plain-`--duckdb` / `--feldera` path (non-proof, no
    // native-uf/native-merge) so rule-encoded congruence is canonicalized by the
    // relational δuf fast-rebuild — mirrors cli.rs's `args.fast_rebuild = true` fold.
    let want_fast_rebuild = argv.iter().any(|a| a == "--fast-rebuild")
        || ((want_duckdb || want_feldera)
            && !want_native_uf
            && !effective_native_merge
            && !want_proofs_early);
    // `--proof-testing` implies proofs — the desugar pass rewrites
    // `(check ...)` into `(prove-exists ...)` which needs the proof
    // encoding active. Without this, cli.rs's `args.proof_testing`
    // branch would try `with_proofs_enabled()` after construction,
    // and that path clones the egraph for `original_typechecking`,
    // hitting the duckdb/feldera/flowlog backends' unimplemented
    // `clone_boxed`. Threading `proofs` into each backend config
    // provisions a proof-enabled typechecker at construction instead.
    let want_proofs = argv
        .iter()
        .any(|a| a == "--proofs" || a == "--proof-testing");
    let want_wcoj = argv.iter().any(|a| a == "--wcoj");
    // Configure the rayon global thread pool BEFORE constructing the egraph.
    // `egglog::cli` sets it too, but only AFTER we hand it an already-built
    // egraph -- and constructing the experimental/backend egraph touches rayon,
    // which lazily initializes the global pool at its default (all logical
    // CPUs). That makes cli's later `set_num_threads` a no-op
    // (GlobalPoolAlreadyInitialized), so `--threads`/`-j` would be silently
    // ignored and everything would run at full width. Setting it here first
    // makes the flag take effect (cli's duplicate call is now an idempotent
    // no-op). Mirrors cli's `--threads` default of 1 (0 = max).
    egglog::EGraph::set_num_threads(parse_num_threads(&argv));
    let egraph = if want_duckdb {
        // DuckDB is migrated OFF native-UF onto the fast relational term-encoding:
        // there is no `native_uf` config knob anymore. If `--native-uf` /
        // `--duck-native-uf` / `--native-merge` were passed with `--duckdb`,
        // `egglog::cli` (below) rejects them with a clear error; pre-build the
        // fast-relational egraph regardless (cli exits before running it).
        egglog_experimental::new_experimental_egraph_duckdb(egglog::DuckBackendConfig {
            fast_rebuild: want_fast_rebuild,
            proofs: want_proofs,
        })
        .expect("failed to start DuckDB-backed experimental egraph")
    } else if want_flowlog {
        // FlowLog is migrated OFF native-UF onto the fast relational term-encoding
        // (the LAST backend): there is no `native_uf` config knob anymore. If
        // `--native-uf` / non-proof `--native-merge` were passed with `--flowlog`,
        // `egglog::cli` (below) rejects them with a clear error; pre-build the
        // fast-relational egraph regardless (cli exits before running it).
        // `fast_rebuild` is a no-op on flowlog (retained for CLI symmetry).
        egglog_experimental::new_experimental_egraph_flowlog(egglog::FlowlogBackendConfig {
            fast_rebuild: want_fast_rebuild,
            wcoj: want_wcoj,
            proofs: want_proofs,
        })
        .expect("failed to start FlowLog-backed experimental egraph")
    } else if want_feldera {
        // Feldera is migrated OFF native-UF onto the fast relational term-encoding:
        // there is no `native_uf` config knob anymore. If `--native-uf` /
        // non-proof `--native-merge` were passed with `--feldera`, `egglog::cli`
        // (below) rejects them with a clear error; pre-build the fast-relational
        // egraph regardless (cli exits before running it).
        egglog_experimental::new_experimental_egraph_feldera(egglog::FelderaBackendConfig {
            fast_rebuild: want_fast_rebuild,
            proofs: want_proofs,
        })
        .expect("failed to start Feldera-backed experimental egraph")
    } else {
        egglog_experimental::new_experimental_egraph()
    };
    egglog::cli(egraph)
}

/// Parse the `--threads`/`-j` value from argv (matching `egglog::cli`'s clap
/// option), defaulting to 1. Accepts `--threads N`, `--threads=N`, `-j N`, and
/// `-jN`. `0` means "use the maximum" (rayon's default), same as cli.
fn parse_num_threads(argv: &[String]) -> usize {
    for (i, a) in argv.iter().enumerate() {
        let val = if a == "--threads" || a == "-j" {
            argv.get(i + 1).map(String::as_str)
        } else if let Some(v) = a.strip_prefix("--threads=") {
            Some(v)
        } else if a.starts_with("-j") && a.len() > 2 {
            Some(&a[2..])
        } else {
            None
        };
        if let Some(v) = val {
            if let Ok(n) = v.parse::<usize>() {
                return n;
            }
        }
    }
    1
}
