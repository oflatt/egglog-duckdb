use crate::*;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};

use clap::Parser;
use env_logger::Env;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(version = env!("FULL_VERSION"), about = env!("CARGO_PKG_DESCRIPTION"))]
struct Args {
    /// Directory for files when using `input` and `output` commands
    #[clap(short = 'F', long)]
    fact_directory: Option<PathBuf>,
    /// Turns off the seminaive optimization
    #[clap(long)]
    naive: bool,
    /// Skips tree-decomposition during query planning. Tree decomposition
    /// tries to decompose complex queries into smaller independent subqueries,
    /// and evaluate them separately. It has a better theoretical guarantee,
    /// but sometimes the decomposed subqueries (called "bags") can be much larger
    /// than the final output, which leads to worse performance sometimes.
    ///
    /// Setting this flag forces the query planner to skip tree decomposition and
    /// evaluate the query as a single bag.
    ///
    /// You can also disable tree decomposition on a per-rule basis with the `:no-decomp` label
    /// on rules.
    #[clap(long)]
    no_decomp: bool,
    /// Prints extra information, which can be useful for debugging
    #[clap(long, default_value_t = RunMode::Normal)]
    mode: RunMode,
    /// The file names for the egglog files to run
    inputs: Vec<PathBuf>,
    /// Serializes the egraph for each egglog file as JSON
    #[clap(long)]
    to_json: bool,
    /// Serializes the egraph for each egglog file as a dot file
    #[clap(long)]
    to_dot: bool,
    /// Serializes the egraph for each egglog file as an SVG
    #[clap(long)]
    to_svg: bool,
    /// Splits the serialized egraph into primitives and non-primitives
    #[clap(long)]
    serialize_split_primitive_outputs: bool,
    /// Maximum number of function nodes to render in dot/svg output
    #[clap(long, default_value = "40")]
    max_functions: usize,
    /// Maximum number of calls per function to render in dot/svg output
    #[clap(long, default_value = "40")]
    max_calls_per_function: usize,
    /// Number of times to inline leaves
    #[clap(long, default_value = "0")]
    serialize_n_inline_leaves: usize,
    #[clap(short = 'j', long, default_value = "1")]
    /// Number of threads to use for parallel execution. Passing `0` will use the maximum
    /// inferred parallelism available on the current system.
    threads: usize,
    #[arg(value_enum)]
    #[clap(long, default_value_t = ReportLevel::TimeOnly)]
    report_level: ReportLevel,
    #[clap(long)]
    save_report: Option<PathBuf>,
    /// Treat missing `$` prefixes on globals as errors instead of warnings
    #[clap(long = "strict-mode")]
    strict_mode: bool,
    /// Run the terms encoding of equality saturation
    #[clap(long)]
    term_encoding: bool,
    /// Run with proof generation enabled
    #[clap(long)]
    proofs: bool,
    /// Enable proof testing, turning all `check` statements into `prove` statements
    #[clap(long)]
    proof_testing: bool,
    /// Run the program on the experimental DuckDB backend (Phase 1.3).
    /// Bypasses the default `egglog-bridge` execution path. Many features
    /// are not yet supported; see `src/backend_duckdb.rs` for current scope.
    #[clap(long = "duckdb")]
    duckdb_backend: bool,
    /// Use a native union-find data structure for the term-encoding
    /// UF function (`@_u_f___<sort>f`) instead of a SQL table. The
    /// table-backed UF requires three maintenance rulesets
    /// (singleparent, path_compress, uf_function_index) that together
    /// dominate runtime on saturating workloads; the native UF
    /// replaces them with O(α(n)) memory ops behind a DuckDB UDF.
    /// `--duckdb` only; off by default.
    #[clap(long = "duck-native-uf")]
    duck_native_uf: bool,
    /// Run the program on the experimental Feldera/DBSP backend.
    /// Like `--duckdb`, this bypasses the default `egglog-bridge`
    /// execution path and is term-encoding only. Mutually exclusive
    /// with `--duckdb` and `--flowlog`.
    #[clap(long = "feldera")]
    feldera_backend: bool,
    /// Run the program on the experimental FlowLog/Differential-Dataflow
    /// backend. Like `--duckdb`, this bypasses the default
    /// `egglog-bridge` execution path and is term-encoding only.
    /// Mutually exclusive with `--duckdb` and `--feldera`.
    #[clap(long = "flowlog")]
    flowlog_backend: bool,
    /// Use a native union-find data structure for the term-encoding UF
    /// function instead of the relational `@UF_S` parent table +
    /// singleparent/path_compress/uf_function_index maintenance rulesets.
    /// Emits PR #782's `:impl displaced-union-find` UF-backed function plus an
    /// onchange relation driving the rebuild. Runs on the default native bridge
    /// AND on `--flowlog` / `--feldera` / `--duckdb`: each honours the UF-backed
    /// encoding but drives its own fast HOST-PASS rebuild instead of the
    /// onchange-driven rebuild rules (Feldera additionally keeps the
    /// `view ⋈ @UF_Sf` integral out of the DBSP circuit; DuckDB uses a find UDF
    /// + demote pass). Term-encoding only on the dataflow/SQL backends. Supports
    /// `--proofs` on the native bridge (provenance-tracking UF; leader-change
    /// callback composes the onchange proof) — but `--native-uf` with
    /// `--flowlog`/`--feldera`/`--duckdb` is TERM mode only.
    /// Off by default; the relational encoding is unchanged when off.
    #[clap(long = "native-uf")]
    native_uf: bool,
    /// Enable each dataflow backend's RELATIONAL δuf fast-rebuild: drop the
    /// always-empty `δview ⋈ uf_old` rebuild term (sound under
    /// canonicalize-at-creation). This is the relational (non-native-uf) path
    /// — on `--flowlog` / `--feldera` / `--duckdb` WITHOUT `--native-uf` it
    /// engages the backend's δuf fast-rebuild (the `enable_fast_rebuild()`
    /// hook, also reachable via the `FELDERA_DELTA_REBUILD` /
    /// `DUCK_DELTA_REBUILD` / `FLOWLOG_DELTA_REBUILD` env vars). With
    /// `--native-uf` it is a no-op: native-UF already drives the `view ⋈ δuf`
    /// onchange-delta rebuild, which has no `δview ⋈ uf_old` term to drop.
    /// Bit-exact with the full rebuild; off by default.
    #[clap(long = "fast-rebuild")]
    fast_rebuild: bool,
}

/// Start a command-line interface for the E-graph.
///
/// This is what vanilla egglog uses, and custom egglog builds (i.e., "egglog batteries included")
/// should also call this function.
#[allow(clippy::disallowed_macros)]
pub fn cli(mut egraph: EGraph) {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .parse_default_env()
        .init();

    let args = Args::parse();

    // The experimental backends are mutually exclusive: each swaps in a
    // different `Backend` impl for the run, so at most one may be selected.
    if (args.duckdb_backend as u8) + (args.feldera_backend as u8) + (args.flowlog_backend as u8) > 1
    {
        log::error!("at most one of --duckdb, --feldera, --flowlog may be passed");
        std::process::exit(1);
    }

    // The native-UF encoding mode (PR #782's `:impl displaced-union-find`
    // UF-backed function) runs on the native bridge AND all three experimental
    // backends (`--flowlog`/`--feldera`/`--duckdb`): each honours the UF-backed
    // encoding but drives its own fast HOST-PASS rebuild instead of the
    // onchange-driven rebuild rules. All backends support `--native-uf`.
    // Native-UF on the dataflow/SQL backends is TERM mode only (proofs are a
    // later step; `--proofs` is supported only on the native bridge).
    if args.native_uf
        && (args.flowlog_backend || args.feldera_backend || args.duckdb_backend)
        && args.proofs
    {
        log::error!(
            "--native-uf with --flowlog/--feldera/--duckdb is TERM mode only; --proofs is not yet supported"
        );
        std::process::exit(1);
    }

    // `--native-uf` is a term-encoding-only mode, so it implies term
    // encoding (enabling it here covers the case where `--term-encoding`
    // was not also passed).
    //
    // Skip this when the caller already handed us a duckdb-backed egraph
    // (egglog-experimental's `main` pre-builds one so its commands /
    // primitives survive). `with_duckdb_backend` already provisions term
    // encoding (a bridge-backed typechecker) without cloning the backend,
    // whereas `with_term_encoding_enabled` clones `self` — and the duckdb
    // backend's `clone_boxed` is unimplemented, so cloning it panics.
    if (args.term_encoding || args.native_uf) && !egraph.has_duckdb_backend() {
        egraph = egraph.with_term_encoding_enabled();
    }

    if args.native_uf {
        egraph = egraph.with_native_uf();
    }

    // `--fast-rebuild` on the dataflow/SQL backends is wired into each backend's
    // config (`config.fast_rebuild` → `enable_fast_rebuild()`) below. On the
    // NATIVE BRIDGE (no `--duckdb`/`--feldera`/`--flowlog`) there is no separate
    // host-pass rebuild, so the flag instead drives the encoding-level drop of
    // the always-empty `δview ⋈ uf_old` term: relationally the rebuild rule's
    // view atom is excluded from delta-focus; under `--native-uf` the δview
    // probe rule is dropped (leaving `view ⋈ δuf`). Bit-exact with the full
    // rebuild under canonicalize-at-creation.
    if args.fast_rebuild && !args.duckdb_backend && !args.feldera_backend && !args.flowlog_backend {
        egraph = egraph.with_fast_rebuild();
    }

    if args.proofs && !egraph.are_proofs_enabled() {
        egraph = egraph.with_proofs_enabled();
    }

    if args.proof_testing {
        if !egraph.are_proofs_enabled() {
            egraph = egraph.with_proofs_enabled();
        }
        egraph = egraph.with_proof_testing();
    }

    EGraph::set_num_threads(args.threads);
    egraph.fact_directory.clone_from(&args.fact_directory);
    egraph.seminaive = !args.naive;
    egraph.no_decomp = args.no_decomp;
    egraph.set_report_level(args.report_level);
    if args.strict_mode {
        egraph.set_strict_mode(true);
    }
    if args.inputs.is_empty() {
        match egraph.repl(args.mode) {
            Ok(()) => std::process::exit(0),
            Err(err) => {
                log::error!("{err}");
                std::process::exit(1)
            }
        }
    } else {
        for input in &args.inputs {
            let program = std::fs::read_to_string(input).unwrap_or_else(|_| {
                let arg = input.to_string_lossy();
                panic!("Failed to read file {arg}")
            });

            // DuckDB backend: route the program through the same egglog
            // frontend (parser, typechecker, term encoding) but with the
            // DuckDB-backed `Backend` impl swapped in. The
            // `EGraph::with_duckdb_backend` constructor builds an
            // ordinary `EGraph` whose `backend` is the DuckDB engine —
            // no parallel typechecker, no separate dispatch loop. The
            // caller's `egraph` (which may carry an experimental parser
            // or other customizations) is replaced wholesale for this
            // input; we preserve the parser so any
            // `egglog-experimental` parse-time macros still resolve.
            if args.duckdb_backend {
                // If the caller already handed us a duckdb-backed
                // egraph (e.g. `egglog-experimental`'s main does this
                // up front so its commands / primitives survive),
                // route the program through it directly. Otherwise
                // build a fresh duckdb egraph here. Either way the
                // program runs through the same trait pipeline.
                if egraph.has_duckdb_backend() {
                    egraph.fact_directory.clone_from(&args.fact_directory);
                    // Dump-on-panic so even when (check ...) /
                    // prove-exists / extract panics mid-run we still
                    // see the duck table contents.
                    if std::env::var("DUCK_DUMP_TABLES").is_ok() {
                        let input_str = input.to_str().unwrap().to_owned();
                        let mut tmp = std::mem::replace(
                            &mut egraph,
                            EGraph::with_duckdb_backend(egglog::DuckBackendConfig::default())
                                .unwrap(),
                        );
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tmp.parse_and_run_program(Some(input_str), &program)
                        }));
                        if let Err(e) = result {
                            eprintln!("== panicked during run: {e:?} ==");
                        }
                        tmp.dump_debug_info();
                        continue;
                    }
                    let _duck_wall0 = std::time::Instant::now();
                    let result = egraph
                        .parse_and_run_program(Some(input.to_str().unwrap().into()), &program);
                    let _duck_wall_ns = _duck_wall0.elapsed().as_nanos() as u64;
                    // Dump per-rule timing if DUCK_PERF_DUMP is set
                    // (env-flag activated; no CLI flag). Reads the
                    // duckdb backend's per-rule counters directly.
                    if std::env::var("DUCK_PERF_DUMP").is_ok()
                        && let Some(duck) = egraph
                            .backend_for_diagnostics()
                            .as_any()
                            .downcast_ref::<egglog_bridge_duckdb::EGraph>()
                    {
                        duck_perf_dump(duck, _duck_wall_ns);
                    }
                    match result {
                        Ok(msgs) => {
                            // Print command output (`print-size`, `extract`, …)
                            // like the fresh-duckdb / flowlog / feldera paths,
                            // so the pre-built (egglog-experimental) duckdb
                            // egraph is observable from the CLI too.
                            if args.mode != RunMode::NoMessages {
                                use std::io::Write;
                                let mut out = io::stdout();
                                for msg in msgs {
                                    let _ = write!(out, "{msg}");
                                }
                            }
                        }
                        Err(err) => {
                            log::error!("{err}");
                            std::process::exit(1);
                        }
                    }
                    continue;
                }
                // `--native-uf --duckdb`: enable the DuckDB backend's in-process
                // UF host-pass (`enable_native_uf`, via the config flag) AND the
                // matching PR #782 encoding (`with_native_uf` below) — both are
                // needed (the encoding emits the UF-backed program; the backend
                // interception runs it through the SQL host-pass rebuild). The
                // legacy `--duck-native-uf` flag drives the same backend
                // interception on the *relational* (non-#782) encoding.
                let mut duck_eg = egglog::EGraph::with_duckdb_backend(egglog::DuckBackendConfig {
                    native_uf: args.duck_native_uf || args.native_uf,
                    fast_rebuild: args.fast_rebuild,
                    proofs: false,
                })
                .unwrap_or_else(|err| {
                    log::error!("failed to start DuckDB backend: {err}");
                    std::process::exit(1);
                });
                if args.native_uf {
                    duck_eg = duck_eg.with_native_uf();
                }
                duck_eg.parser = std::mem::take(&mut egraph.parser);
                duck_eg.fact_directory.clone_from(&args.fact_directory);
                duck_eg.ensure_no_reserved_symbols(false);
                let duck_wall0 = std::time::Instant::now();
                let duck_result =
                    duck_eg.parse_and_run_program(Some(input.to_str().unwrap().into()), &program);
                let duck_wall_ns = duck_wall0.elapsed().as_nanos() as u64;
                if std::env::var("DUCK_PERF_DUMP").is_ok()
                    && let Some(duck) = duck_eg
                        .backend_for_diagnostics()
                        .as_any()
                        .downcast_ref::<egglog_bridge_duckdb::EGraph>()
                {
                    duck_perf_dump(duck, duck_wall_ns);
                }
                match duck_result {
                    Ok(msgs) => {
                        // Print command output (`print-size`, `extract`, …) like
                        // the flowlog/feldera paths below, so `--duckdb` is
                        // observable from the CLI (it previously discarded msgs).
                        if args.mode != RunMode::NoMessages {
                            use std::io::Write;
                            let mut out = io::stdout();
                            for msg in msgs {
                                let _ = write!(out, "{msg}");
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("{err}");
                        std::process::exit(1);
                    }
                }
                continue;
            }

            // Feldera / FlowLog backends: same idea as the duckdb path but
            // simpler — these constructors take no config and enable term
            // encoding internally, and there are no table-dump/perf env
            // hooks to replicate. Build a fresh egraph with the requested
            // backend and route the program through the shared frontend.
            if args.feldera_backend || args.flowlog_backend {
                let mut backend_eg = if args.feldera_backend {
                    // `--native-uf --feldera`: enable the Feldera backend's
                    // in-process UF host-pass (`enable_native_uf`) here, and the
                    // matching #782 encoding (`with_native_uf`) below — both are
                    // needed (the encoding emits the program; the backend
                    // interception runs it through the host-pass rebuild, keeping
                    // the `view ⋈ @UF_Sf` integral out of the DBSP circuit).
                    egglog::EGraph::with_feldera_backend_config(egglog::FelderaBackendConfig {
                        native_uf: args.native_uf,
                        fast_rebuild: args.fast_rebuild,
                    })
                } else {
                    // `--native-uf --flowlog`: enable the FlowLog backend's
                    // in-process UF host-pass (`enable_native_uf`) here, and the
                    // matching #782 encoding (`with_native_uf`) below — both are
                    // needed (the encoding emits the program; the backend
                    // interception runs it through the host-pass rebuild).
                    egglog::EGraph::with_flowlog_backend_config(egglog::FlowlogBackendConfig {
                        native_uf: args.native_uf,
                        fast_rebuild: args.fast_rebuild,
                    })
                }
                .unwrap_or_else(|err| {
                    log::error!("failed to start experimental backend: {err}");
                    std::process::exit(1);
                });
                if args.native_uf && (args.flowlog_backend || args.feldera_backend) {
                    backend_eg = backend_eg.with_native_uf();
                }
                backend_eg.parser = std::mem::take(&mut egraph.parser);
                backend_eg.fact_directory.clone_from(&args.fact_directory);
                backend_eg.ensure_no_reserved_symbols(false);
                match backend_eg
                    .parse_and_run_program(Some(input.to_str().unwrap().into()), &program)
                {
                    Ok(msgs) => {
                        if args.mode != RunMode::NoMessages {
                            use std::io::Write;
                            let mut out = io::stdout();
                            for msg in msgs {
                                let _ = write!(out, "{msg}");
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("{err}");
                        std::process::exit(1);
                    }
                }
                continue;
            }

            match run_commands(
                &mut egraph,
                Some(input.to_str().unwrap().into()),
                &program,
                io::stdout(),
                args.mode,
            ) {
                Ok(None) => {}
                _ => std::process::exit(1),
            }

            if args.to_json || args.to_dot || args.to_svg {
                let serialized_output = egraph.serialize(SerializeConfig {
                    max_functions: Some(args.max_functions),
                    max_calls_per_function: Some(args.max_calls_per_function),
                    ..SerializeConfig::default()
                });
                if !serialized_output.is_complete() {
                    log::warn!("{}", serialized_output.omitted_description());
                }
                let mut serialized = serialized_output.egraph;
                if args.serialize_split_primitive_outputs {
                    serialized.split_classes(|id, _| egraph.from_node_id(id).is_primitive())
                }
                for _ in 0..args.serialize_n_inline_leaves {
                    serialized.inline_leaves();
                }

                // if we are splitting primitive outputs, add `-split` to the end of the file name
                let serialize_filename = if args.serialize_split_primitive_outputs {
                    input.with_file_name(format!(
                        "{}-split",
                        input.file_stem().unwrap().to_str().unwrap()
                    ))
                } else {
                    input.clone()
                };
                if args.to_dot {
                    let dot_path = serialize_filename.with_extension("dot");
                    serialized
                        .to_dot_file(dot_path.clone())
                        .unwrap_or_else(|_| panic!("Failed to write dot file to {dot_path:?}"));
                }
                if args.to_svg {
                    let svg_path = serialize_filename.with_extension("svg");
                    serialized.to_svg_file(svg_path.clone()).unwrap_or_else( |_|
                        panic!("Failed to write svg file to {svg_path:?}. Make sure you have the `dot` executable installed")
                    );
                }
                if args.to_json {
                    let json_path = serialize_filename.with_extension("json");
                    serialized
                        .to_json_file(json_path.clone())
                        .unwrap_or_else(|_| panic!("Failed to write json file to {json_path:?}"));
                }
            }
        }
    }

    if let Some(report_path) = args.save_report {
        let report = egraph.get_overall_run_report();
        serde_json::to_writer(
            std::fs::File::create(&report_path)
                .unwrap_or_else(|_| panic!("Failed to create report file at {report_path:?}")),
            &report,
        )
        .expect("Failed to serialize report");
        log::info!("Saved report to {report_path:?}");
    }

    // no need to drop the egraph if we are going to exit
    std::mem::forget(egraph)
}

impl EGraph {
    /// Start a Read-Eval-Print Loop with standard I/O.
    pub fn repl(&mut self, mode: RunMode) -> io::Result<()> {
        self.repl_with(io::stdin(), io::stdout(), mode, io::stdin().is_terminal())
    }

    /// Start a Read-Eval-Print Loop with the given input and output channel.
    pub fn repl_with<R, W>(
        &mut self,
        input: R,
        mut output: W,
        mode: RunMode,
        is_terminal: bool,
    ) -> io::Result<()>
    where
        R: Read,
        W: Write,
    {
        // https://doc.rust-lang.org/beta/std/io/trait.IsTerminal.html#examples
        if is_terminal {
            output.write_all(welcome_prompt().as_bytes())?;
            output.write_all(b"\n> ")?;
            output.flush()?;
        }
        let mut cmd_buffer = String::new();

        for line in BufReader::new(input).lines() {
            let line_str = line?;
            cmd_buffer.push_str(&line_str);
            cmd_buffer.push('\n');
            // handles multi-line commands
            if should_eval(&cmd_buffer) {
                run_commands(self, None, &cmd_buffer, &mut output, mode)?;
                cmd_buffer = String::new();
                if is_terminal {
                    output.write_all(b"> ")?;
                    output.flush()?;
                }
            }
        }

        if !cmd_buffer.is_empty() {
            run_commands(self, None, &cmd_buffer, &mut output, mode)?;
        }

        Ok(())
    }
}

/// Print the DUCK_PERF_DUMP breakdown to stderr: top rules, global
/// search(mat)/apply(act) totals, a per-ruleset rollup classified into
/// rebuild / UF-maintenance / congruence / cleanup / user, and an
/// "other" bucket = wall time not attributed to any rule SQL (planning,
/// snapshotting, watermark/skip bookkeeping, parse). `wall_ns` is the
/// measured wall time of `parse_and_run_program`.
fn duck_perf_dump(duck: &egglog_bridge_duckdb::EGraph, wall_ns: u64) {
    eprintln!("\n=== DUCK_PERF_DUMP: top 20 rules by total ns ===");
    let mut rows = duck.perf_per_rule();
    rows.truncate(20);
    eprintln!(
        "{:>10} {:>10} {:>10}  {:<40} rule",
        "total_s", "mat_s", "act_s", "ruleset"
    );
    for (rn, rs, m, a) in &rows {
        eprintln!(
            "{:>10.3} {:>10.3} {:>10.3}  {:<40} {}",
            (m + a) as f64 / 1e9,
            *m as f64 / 1e9,
            *a as f64 / 1e9,
            rs,
            rn
        );
    }
    let (mat, mat_act, act) = duck.perf_timings_ns();
    eprintln!(
        "totals: materialize {:.3}s, mat_action {:.3}s, action {:.3}s",
        mat as f64 / 1e9,
        mat_act as f64 / 1e9,
        act as f64 / 1e9,
    );

    let wall_s = wall_ns as f64 / 1e9;
    let rs_rows = duck.perf_per_ruleset();
    let accounted_ns = mat + mat_act + act;
    let other_ns = wall_ns.saturating_sub(accounted_ns);
    let pct = |x: u64| {
        if wall_ns > 0 {
            x as f64 / wall_ns as f64 * 100.0
        } else {
            0.0
        }
    };
    eprintln!("\n=== DUCK_PERF_DUMP: per-ruleset rollup ===");
    eprintln!(
        "wall: {wall_s:.3}s (accounted {:.3}s in rule SQL, other {:.3}s = planning/snapshot/skip-gates/parse)",
        accounted_ns as f64 / 1e9,
        other_ns as f64 / 1e9
    );
    eprintln!(
        "{:>9} {:>9} {:>9} {:>6} {:>5}  {:<28} kind",
        "total_s", "search_s", "apply_s", "%wall", "rules", "ruleset"
    );
    let mut cls: std::collections::HashMap<&'static str, (u64, u64)> =
        std::collections::HashMap::new();
    for (rs, kind, m, a, n) in &rs_rows {
        let tot = m + a;
        eprintln!(
            "{:>9.3} {:>9.3} {:>9.3} {:>5.1}% {:>5}  {:<28} {}",
            tot as f64 / 1e9,
            *m as f64 / 1e9,
            *a as f64 / 1e9,
            pct(tot),
            n,
            rs,
            kind,
        );
        let e = cls.entry(kind).or_insert((0, 0));
        e.0 = e.0.wrapping_add(*m);
        e.1 = e.1.wrapping_add(*a);
    }
    eprintln!("\n--- by class (search=mat, apply=act) ---");
    let mut cls_rows: Vec<(&'static str, u64, u64)> =
        cls.into_iter().map(|(k, (m, a))| (k, m, a)).collect();
    cls_rows.sort_by(|x, y| (y.1 + y.2).cmp(&(x.1 + x.2)));
    for (k, m, a) in cls_rows {
        let tot = m + a;
        eprintln!(
            "{:>9.3}s  search {:>7.3}s  apply {:>7.3}s  {:>5.1}%wall  {}",
            tot as f64 / 1e9,
            m as f64 / 1e9,
            a as f64 / 1e9,
            pct(tot),
            k,
        );
    }
    eprintln!(
        "  search(mat) {:.1}%  apply(act+mat_act) {:.1}%  other {:.1}%  of wall",
        pct(mat),
        pct(act + mat_act),
        pct(other_ns),
    );
    eprintln!(
        "  rule firings skipped by watermark gate: {}; rows affected total: {}",
        duck.rules_skipped(),
        duck.rules_affected_total(),
    );
}

fn welcome_prompt() -> String {
    format!("Welcome to Egglog REPL! (build: {})", env!("FULL_VERSION"))
}

fn should_eval(curr_cmd: &str) -> bool {
    all_sexps(SexpParser::new(None, curr_cmd)).is_ok()
}

fn run_commands<W>(
    egraph: &mut EGraph,
    filename: Option<String>,
    command: &str,
    mut output: W,
    mode: RunMode,
) -> io::Result<Option<Error>>
where
    W: Write,
{
    if mode == RunMode::ShowDesugaredEgglog {
        return Ok(match egraph.resolve_program(filename, command) {
            Ok(resolved) => {
                let sanitized = sanitize_internal_names(&resolved);

                for line in sanitized {
                    writeln!(output, "{line}")?;
                }
                None
            }
            Err(err) => {
                log::error!("{err}");
                Some(err)
            }
        });
    };

    Ok(match egraph.parse_and_run_program(filename, command) {
        Ok(msgs) => {
            if mode != RunMode::NoMessages {
                for msg in msgs {
                    write!(output, "{msg}")?;
                }
            }
            if mode == RunMode::Interactive {
                writeln!(output, "(done)")?;
            }
            None
        }
        Err(err) => {
            log::error!("{err}");
            if mode == RunMode::Interactive {
                writeln!(output, "(error)")?;
            }
            Some(err)
        }
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum RunMode {
    Normal,
    ShowDesugaredEgglog,
    Interactive,
    NoMessages,
}

impl Display for RunMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RunMode::Normal => write!(f, "normal"),
            RunMode::ShowDesugaredEgglog => write!(f, "desugar"),
            RunMode::Interactive => write!(f, "interactive"),
            RunMode::NoMessages => write!(f, "no-messages"),
        }
    }
}

impl FromStr for RunMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(RunMode::Normal),
            "desugar" => Ok(RunMode::ShowDesugaredEgglog),
            "interactive" => Ok(RunMode::Interactive),
            "no-messages" => Ok(RunMode::NoMessages),
            _ => Err(format!("Unknown run mode: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_eval() {
        #[rustfmt::skip]
        let test_cases = vec![
            vec![
                "(extract",
                "\"1",
                ")",
                "(",
                ")))",
                "\"",
                ";; )",
                ")"
            ],
            vec![
                "(extract 1) (extract",
                "2) (",
                "extract 3) (extract 4) ;;;; ("
            ],
            vec![
                "(extract \"\\\")\")"
            ]];
        for test in test_cases {
            let mut cmd_buffer = String::new();
            for (i, line) in test.iter().enumerate() {
                cmd_buffer.push_str(line);
                cmd_buffer.push('\n');
                assert_eq!(should_eval(&cmd_buffer), i == test.len() - 1);
            }
        }
    }

    #[test]
    fn test_repl() {
        let mut egraph = EGraph::default();

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "1\n");

        let input = "\n\n\n";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Interactive, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "1\n(done)\n");

        let input = "xyz";
        let mut output: Vec<u8> = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Interactive, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(error)\n");

        let missing_include = std::env::temp_dir().join(format!(
            "egglog_missing_include_{}_{}.egg",
            std::process::id(),
            "repl_test"
        ));
        let input = format!(
            "(include \"{}\")",
            missing_include.to_string_lossy().replace('\\', "/")
        );
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Interactive, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(error)\n");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(
                input.as_bytes(),
                &mut output,
                RunMode::ShowDesugaredEgglog,
                false,
            )
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "(extract 1 0)\n");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::NoMessages, false)
            .unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "");

        let input = "(extract 1)";
        let mut output = Vec::new();
        egraph
            .repl_with(input.as_bytes(), &mut output, RunMode::Normal, true)
            .unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("{}\n> 1\n> ", welcome_prompt())
        );
    }
}
