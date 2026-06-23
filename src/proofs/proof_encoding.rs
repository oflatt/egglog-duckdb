#[doc = include_str!("proof_encoding.md")]
use crate::proofs::proof_encoding_helpers::{EncodingNames, Justification};
use crate::typechecking::FuncType;
use crate::*;

// TODO refactor so that encoding state is optional on the e-graph, ProofNames not optional on EncodingState. Then we don't have to clone proof names everywhere.
#[derive(Clone)]
pub(crate) struct EncodingState {
    pub uf_parent: HashMap<String, String>,
    pub uf_function: HashMap<String, String>,
    /// Maps sort name -> proof function name (set from :internal-proof-func annotation).
    pub proof_func_parent: HashMap<String, String>,
    pub term_header_added: bool,
    // TODO this is very ugly- we should separate out a typechecking struct
    // since we didn't need an entire e-graph
    // When Some term encoding is enabled.
    pub original_typechecking: Option<Box<EGraph>>,
    pub proofs_enabled: bool,
    pub proof_testing: bool,
    pub proof_names: EncodingNames,
    /// Canonicalize-at-creation: when on, each eq-sort child of a constructor
    /// (and custom-function view) created in a rule RHS is replaced with its
    /// UF_old leader via an identity-on-miss lookup against the flat UF index
    /// `@UF_Sf` (frozen at the last completed rebuild). This makes every newly
    /// inserted view row canonical w.r.t. UF_old, which lets the FlowLog DD
    /// backend skip the `δ(constructor) ⋈ UF_old` rebuild join soundly.
    /// Default OFF; the default encoding is byte-for-byte unchanged. Gated by
    /// the `EGGLOG_CANON_AT_CREATION` environment variable.
    pub canon_at_creation: bool,
    /// Transient flag set while instrumenting a single rule's actions: true once
    /// a canonicalize-at-creation `find_UFold` lookup has been emitted into the
    /// RHS, so the generated rule must opt into `:unsafe-seminaive` (RHS
    /// function-table lookups). Reset per rule in `instrument_rule`.
    pub emitted_canon_lookup: bool,
    /// Native-UF encoding mode (gated by `--native-uf`, term-encoding +
    /// non-proof + native bridge only). When on, each eq-sort gets a
    /// `:impl displaced-union-find` UF-backed function (reusing the
    /// `@UF_Sf` name) plus an `@UFChange_S` onchange relation and a
    /// `@canon_S` find-or-self primitive, instead of the relational
    /// `@UF_S` parent table + singleparent/path_compress/uf_function_index
    /// rulesets. Union goes through the UF function; canon lookups use the
    /// `@canon_S` primitive (no `:unsafe-seminaive` needed); the
    /// per-constructor rebuild is driven by the `@UFChange_S` onchange
    /// relation. Default OFF; the relational path is byte-identical when
    /// off (duckdb/feldera/flowlog still depend on it).
    pub native_uf: bool,
    /// Maps eq-sort name -> the `@canon_S` find-or-self primitive name
    /// (only populated in native-UF mode).
    pub canon_prim: HashMap<String, String>,
    /// Maps eq-sort name -> the `@UFChange_S` onchange relation name
    /// (only populated in native-UF mode).
    pub uf_change_rel: HashMap<String, String>,
    /// Native-UF mode only: set of `@UFChange_S` relation names that already
    /// have a drain rule emitted (a sort has one rebuild fn per constructor,
    /// but only one drain rule should be emitted per onchange relation).
    pub uf_change_drained: HashSet<String>,
    /// Fast-rebuild encoding mode (`--fast-rebuild`, native-bridge only). When
    /// on, the relational rebuild rule drops the always-empty `δview ⋈ uf_old`
    /// seminaive variant (the view atom is excluded from being a delta focus),
    /// and the native-UF rebuild drops the (also empty) δview probe rule —
    /// keeping only the `view ⋈ δuf` term. Bit-exact with the full rebuild
    /// (the dropped term is empty under canonicalize-at-creation); the
    /// difference is the saved δview scan. The dataflow/SQL backends implement
    /// their own fast-rebuild via `enable_fast_rebuild()`; this flag drives
    /// only the bridge's encoding-level drop. Default OFF.
    pub fast_rebuild: bool,
    /// Maps a generated relational-rebuild rule's name -> the view function
    /// name whose atom must be excluded from seminaive delta-focus (drops the
    /// `δview ⋈ uf_old` term). Populated by `rebuilding_rules` when
    /// `fast_rebuild` is on; consumed in `EGraph::add_rule`, which resolves the
    /// view name to its backend `FunctionId` and calls
    /// `RuleBuilderOps::set_focus_exclude_table`.
    pub rebuild_view_exclude: HashMap<String, String>,
    /// Names of non-eq-sort globals (e.g. `$N` from `(let $N (bigrat 3 1))`)
    /// that the term encoder passes through as plain 0-arg key-value stores
    /// instead of building term/view tables for them. Populated in
    /// `term_encode_command`'s `Function` arm when the decl is recognized (by
    /// its `internal_let` flag) as such a global; consulted in the `Set` and
    /// expression-read paths so they emit a direct `($N)` lookup rather than a
    /// (nonexistent) view access. Keyed by name because `FuncType` (the only
    /// info available at the read sites) does not carry `internal_let`.
    pub pass_through_globals: HashSet<String>,
    /// Shared `get-size!` size snapshot for backends without an in-memory
    /// `ActionRegistry` (duckdb/flowlog/feldera — `supports_action_registry()
    /// == false`). The `get-size!` ReadPrim's normal path reads table sizes
    /// through the `ActionRegistry`/`ReadState`, which those backends lack;
    /// instead the egglog frontend refreshes this map (`name -> row_count`,
    /// from `functions` + `Backend::table_size`) before each `:until` query,
    /// and the registry-free `RegistryFreePrimWrapper` sums it (honoring the
    /// explicit `@<F>View` name filter the term encoder produces). Shared by
    /// `Arc` so the wrapper (which only gets an `ExecutionState`, not the
    /// `EGraph`) can read the frontend-maintained snapshot. Empty / unused on
    /// the bridge backend (it has a real `ActionRegistry`).
    pub get_size_snapshot: std::sync::Arc<std::sync::RwLock<HashMap<String, i64>>>,
}

impl EncodingState {
    pub(crate) fn new(symbol_gen: &mut SymbolGen) -> Self {
        Self {
            uf_parent: HashMap::default(),
            uf_function: HashMap::default(),
            proof_func_parent: HashMap::default(),
            term_header_added: false,
            original_typechecking: None,
            proofs_enabled: false,
            proof_names: EncodingNames::new(symbol_gen),
            proof_testing: false,
            // Canonicalize-at-creation is always-on for all term-encoding
            // backends in TERM mode. It is additionally gated off in PROOF mode
            // at the emission sites in `add_term_and_view` (the `@UF_Sf` lookup
            // returns an `@UFPair_Sort` in proof mode, not a bare sort).
            canon_at_creation: true,
            emitted_canon_lookup: false,
            native_uf: false,
            canon_prim: HashMap::default(),
            uf_change_rel: HashMap::default(),
            uf_change_drained: HashSet::default(),
            fast_rebuild: false,
            rebuild_view_exclude: HashMap::default(),
            pass_through_globals: HashSet::default(),
            get_size_snapshot: std::sync::Arc::new(std::sync::RwLock::new(HashMap::default())),
        }
    }
}

/// Thin wrapper around an [`EGraph`] for the term encoding
pub(crate) struct ProofInstrumentor<'a> {
    pub(crate) egraph: &'a mut EGraph,
}

impl<'a> ProofInstrumentor<'a> {
    /// Make a term state and use it to instrument the code.
    pub(crate) fn add_term_encoding(
        egraph: &'a mut EGraph,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        Self { egraph }.add_term_encoding_helper(program)
    }

    /// Mark two things as equal, adding proof if proofs are enabled.
    pub(crate) fn union(
        &mut self,
        type_name: &str,
        lhs: &str,
        rhs: &str,
        justification: &Justification,
    ) -> String {
        // Native-UF mode: union via the UF-backed function. `(set (@UF_Sf a) b
        // <edge_proof>)` calls the union-find's `union(a, b)`, which picks the
        // min leader itself and records the leader change in the onchange
        // relation. In proof mode the edge proof (a Proof term proving `a = b`)
        // is carried in the write's proof column; the leader-change callback
        // reconstructs and composes it into the onchange proof column. In term
        // mode the proof column is Unit `()`.
        if self.native_uf() {
            let uf_function_name = self.uf_function_name(type_name);
            let edge_proof = if self.egraph.proof_state.proofs_enabled {
                let to_ast_constructor = self
                    .proof_names()
                    .sort_to_ast_constructor
                    .get(type_name)
                    .unwrap();
                let rule_constructor = &self.proof_names().rule_constructor;
                let fiat_constructor = &self.proof_names().fiat_constructor;
                // Orientation: the edge proof proves `lhs = rhs` (the arguments
                // exactly as passed to the UF write). `get_proof` then reorients
                // per step (Sym for backward edges) when composing the onchange
                // proof.
                match justification {
                    Justification::Rule(rule_name, proof_list) => format!(
                        "({rule_constructor} \"{rule_name}\" {proof_list} ({to_ast_constructor} {lhs}) ({to_ast_constructor} {rhs}))"
                    ),
                    Justification::Fiat => format!(
                        "({fiat_constructor} ({to_ast_constructor} {lhs}) ({to_ast_constructor} {rhs}))"
                    ),
                    Justification::Merge(_func_name, _proof1, _proof2) => panic!(
                        "Merge functions do not include union actions, so proof should not be by merge"
                    ),
                    Justification::Proof(existing_proof) => existing_proof.clone(),
                }
            } else {
                "()".to_string()
            };
            // Term mode: `@UF_Sf` is `(S) S` → `(set (@UF_Sf a) b)`.
            // Proof mode: `@UF_Sf` is `(S S) Proof` → `(set (@UF_Sf a b) proof)`.
            return if self.egraph.proof_state.proofs_enabled {
                format!("(set ({uf_function_name} {lhs} {rhs}) {edge_proof})")
            } else {
                format!("(set ({uf_function_name} {lhs}) {rhs})")
            };
        }
        let uf_name = self.uf_name(type_name);
        let smaller = format!("(ordering-min {lhs} {rhs})");
        let larger = format!("(ordering-max {lhs} {rhs})");
        let proof = if self.egraph.proof_state.proofs_enabled {
            let to_ast_constructor = self
                .proof_names()
                .sort_to_ast_constructor
                .get(type_name)
                .unwrap();
            let rule_constructor = &self.proof_names().rule_constructor;
            let fiat_constructor = &self.proof_names().fiat_constructor;
            match justification {
                Justification::Rule(rule_name, proof_list) => format!(
                    "({rule_constructor} \"{rule_name}\" {proof_list} ({to_ast_constructor} {larger}) ({to_ast_constructor} {smaller}))"
                ),
                Justification::Fiat => format!(
                    "({fiat_constructor} ({to_ast_constructor} {larger}) ({to_ast_constructor} {smaller}))"
                ),
                Justification::Merge(_func_name, _proof1, _proof2) => panic!(
                    "Merge functions do not include union actions, so proof should not be by merge"
                ),
                Justification::Proof(existing_proof) => existing_proof.clone(),
            }
        } else {
            "()".to_string()
        };
        format!("(set ({uf_name} {larger} {smaller}) {proof})")
    }

    /// The parent table is the database representation of a union-find datastructure.
    /// When one term has two parents, those parents are unioned in the merge action.
    /// Also, we have a rule that maintains the invariant that each term points to its
    /// canonical representative.
    fn declare_sort(&mut self, sort_name: &str) -> Vec<Command> {
        // Native-UF mode: replace the relational `@UF_S` parent table +
        // `@UF_Sf` flat index + the singleparent/path_compress/uf_function_index
        // rulesets with a single `:impl displaced-union-find` UF-backed
        // function (reusing the `@UF_Sf` name) plus its onchange relation and
        // find-or-self primitive. Only ever set in non-proof term mode.
        if self.native_uf() {
            return self.declare_sort_native_uf(sort_name);
        }
        let pname = self.uf_name(sort_name);
        let uf_function_name = self.uf_function_name(sort_name);
        let fresh_name = self.egraph.parser.symbol_gen.fresh("uf_update");
        let uf_function_index_name = self.egraph.parser.symbol_gen.fresh("uf_function_index");

        let path_compress_ruleset_name = self.proof_names().path_compress_ruleset_name.clone();
        let single_parent_ruleset_name = self.proof_names().single_parent_ruleset_name.clone();
        let uf_function_index_ruleset_name =
            self.proof_names().uf_function_index_ruleset_name.clone();

        let proof_type = self.proof_type_str().to_string();

        // In proof mode, path compression composes proofs via Trans/Sym.
        // In term mode, the proof output is Unit and we just write ().
        let (path_compress_query, path_compress_action, single_parent_query, single_parent_action) =
            if self.egraph.proof_state.proofs_enabled {
                let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
                let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
                let trans = self.proof_names().eq_trans_constructor.clone();
                let sym = self.proof_names().eq_sym_constructor.clone();
                (
                    format!(
                        "(= {p1_fresh} ({pname} a b))
                        (= {p2_fresh} ({pname} b c))"
                    ),
                    format!(
                        "(delete ({pname} a b))
                       (set ({pname} a c) ({trans} {p1_fresh} {p2_fresh}))"
                    ),
                    format!(
                        "(= {p1_fresh} ({pname} a b))
                        (= {p2_fresh} ({pname} a c))"
                    ),
                    format!(
                        "(delete ({pname} a b))
                       (set ({pname} b c) ({trans} ({sym} {p1_fresh}) {p2_fresh}))"
                    ),
                )
            } else {
                // singleparent (term mode): instead of self-joining
                // pname to find pairs (a→b, a→c) and reducing them
                // pairwise (O(N²) per group, N-1 iterations), look
                // up the current representative for `a` via the
                // function-form table and redirect every non-min
                // row to it in one pass. Each group of size N is
                // collapsed in a single iteration. The action also
                // updates `uf_function_name(b)` so subsequent
                // singleparent iterations see the latest mapping
                // without waiting for uf_function_index to fire.
                //
                // Relies on `uf_function_name`'s merge being changed
                // to ordering-min (see below) so the function
                // actually holds the minimum c1 per a, not just the
                // most-recently-inserted value.
                (
                    format!("({pname} a b)\n                        ({pname} b c)"),
                    format!(
                        "(delete ({pname} a b))\n                       (set ({pname} a c) ())"
                    ),
                    format!("({pname} a b)\n                        (= c ({uf_function_name} a))"),
                    format!(
                        "(delete ({pname} a b))\n                       (set ({pname} b c) ())\n                       (set ({uf_function_name} b) c)"
                    ),
                )
            };

        // In proof mode, UF function index stores (leader, proof) pairs.
        // In term mode, it just stores the leader.
        let (uf_function_output_type, uf_pair_sort_decl, uf_index_query, uf_index_action) =
            if self.egraph.proof_state.proofs_enabled {
                let pair_sort = self.uf_pair_sort_name(sort_name);
                let proof_fresh = self.egraph.parser.symbol_gen.fresh("uf_idx_proof");
                (
                    pair_sort.clone(),
                    format!("(sort {pair_sort} (Pair {sort_name} {proof_type}))"),
                    format!("(= {proof_fresh} ({pname} a b))"),
                    format!("(set ({uf_function_name} a) (pair b {proof_fresh}))"),
                )
            } else {
                (
                    sort_name.to_string(),
                    "".to_string(),
                    format!("({pname} a b)"),
                    format!("(set ({uf_function_name} a) b)"),
                )
            };

        // `uf_function_name` merge:
        //   - proof mode: pairs (leader, proof) — `:merge new` keeps
        //     whichever assertion was last seen, which is what proof
        //     tracking expects.
        //   - term mode: plain leader ID — `:merge (ordering-min …)`
        //     so the function always holds the smallest representative
        //     and singleparent can use it as a direct lookup instead
        //     of a quadratic self-join.
        let uf_function_merge = if self.egraph.proof_state.proofs_enabled {
            ":merge new".to_string()
        } else {
            ":merge (ordering-min old new)".to_string()
        };
        // singleparent in term mode redirects via the function-form
        // table and updates it inline, so it doesn't need the
        // `(ordering-max b c) = b` selector that the original
        // pairwise rule used. Proof mode still does pairwise.
        let single_parent_filter = if self.egraph.proof_state.proofs_enabled {
            "(!= b c)\n                    (= (ordering-max b c) b)".to_string()
        } else {
            "(!= b c)".to_string()
        };
        let mut code = format!(
            "{uf_pair_sort_decl}
             (function {pname} ({sort_name} {sort_name}) {proof_type} :merge old :internal-hidden)
             (function {uf_function_name} ({sort_name}) {uf_function_output_type} {uf_function_merge} :unextractable :internal-hidden)
             ;; performs path compression, ensuring each term points to the representative
             (rule ({path_compress_query}
                    (!= b c))
                  ({path_compress_action})
                   :ruleset {path_compress_ruleset_name}
                   :name \"{fresh_name}\")
             ;; ensures each term has only one parent
             (rule ({single_parent_query}
                    {single_parent_filter})
                  ({single_parent_action})
                   :ruleset {single_parent_ruleset_name}
                   :name \"singleparent{fresh_name}\")
             ;; mirrors UF rows into a function-backed UF index for faster rebuild lookups
             (rule ({uf_index_query})
                   ({uf_index_action})
                   :ruleset {uf_function_index_ruleset_name}
                   :name \"{uf_function_index_name}\")
                   "
        );

        if self.egraph.proof_state.proofs_enabled {
            let term_proof_name = self.term_proof_name(sort_name);
            let add_to_ast_code = self.add_to_ast(sort_name);
            code = format!(
                "{add_to_ast_code}
                 (function {term_proof_name} ({sort_name}) {proof_type} :merge old :internal-hidden)
                 {code}"
            );
        }

        self.parse_program(&code)
    }

    /// Native-UF variant of `declare_sort`. Emits PR #782's UF-backed
    /// function instead of the relational union-find:
    ///
    ///   (relation @UFChange_S (S S S S S))
    ///   (function @UF_Sf (S) S :impl displaced-union-find
    ///                          :onchange @UFChange_S :canon-prim @canon_S)
    ///
    /// The `@UF_Sf` name is reused so existing references (canon-at-creation,
    /// rebuild) resolve. `@canon_S` is the find-or-self primitive (rebound to
    /// the union-find canonicalizer at backend-build time). `@UFChange_S`
    /// receives `(write_lhs write_rhs lhs_leader rhs_leader new_leader)` on
    /// each leader change.
    ///
    /// No singleparent / path_compress / uf_function_index rulesets are
    /// needed — the native union-find maintains canonicality itself.
    fn declare_sort_native_uf(&mut self, sort_name: &str) -> Vec<Command> {
        let uf_function_name = self.uf_function_name(sort_name);
        let canon_prim = self.canon_prim_name(sort_name);
        let uf_change_rel = self.uf_change_rel_name(sort_name);

        // The onchange relation desugars to a constructor over a fresh sort. Its
        // first 5 inputs are all the eq-sort being declared
        // (write_lhs write_rhs lhs_leader rhs_leader new_leader). In PROOF mode
        // it gains a trailing `Proof` column carrying the composed proof that
        // `displaced_leader = new_leader` (filled by the leader-change callback).
        //
        // The UF function is `(S) S` in term mode, but `(S S) Proof` in proof
        // mode: the union write carries the per-edge proof as the value column
        // (`(set (@UF_Sf a b) proof)` → backend row `[a, b, proof, ts]`, which
        // the provenance table consumes). `@UF_Sf` is never *read* as a function
        // in proof mode (find goes through `@canon_S`, rebuild reads the onchange
        // relation, canon-at-creation is disabled), so the schema change is safe.
        let code = if self.egraph.proof_state.proofs_enabled {
            let proof_type = self.proof_type_str().to_string();
            // In proof mode we also need the per-sort `term_proof` function and
            // the sort→AST constructor, exactly as the relational `declare_sort`
            // emits them (they're consumed by `add_term_and_view`).
            let term_proof_name = self.term_proof_name(sort_name);
            let add_to_ast_code = self.add_to_ast(sort_name);
            format!(
                "{add_to_ast_code}
                 (function {term_proof_name} ({sort_name}) {proof_type} :merge old :internal-hidden)
                 (relation {uf_change_rel} ({sort_name} {sort_name} {sort_name} {sort_name} {sort_name} {proof_type}))
                 (function {uf_function_name} ({sort_name} {sort_name}) {proof_type} :impl displaced-union-find
                           :onchange {uf_change_rel} :canon-prim {canon_prim} :internal-hidden)"
            )
        } else {
            // Term mode: the onchange relation gets a 6th `displaced` column
            // (`= max(lhs_leader, rhs_leader)`, the id that stopped being
            // canonical), written by the leader-change callback. The rebuild
            // equi-joins a view column against this *stored* column with a
            // single rule per column — half the rules of joining each of
            // `lhs_leader`/`rhs_leader` separately, and far cheaper than a
            // computed `(ordering-max ll rl)` join (which the planner cannot
            // index, forcing a full onchange×view scan).
            format!(
                "(relation {uf_change_rel} ({sort_name} {sort_name} {sort_name} {sort_name} {sort_name} {sort_name}))
                 (function {uf_function_name} ({sort_name}) {sort_name} :impl displaced-union-find
                           :onchange {uf_change_rel} :canon-prim {canon_prim} :internal-hidden)"
            )
        };
        self.parse_program(&code)
    }

    /// Rules that execute deletion and subsumption based on the tables requesting the deletion/subsumption.
    fn delete_and_subsume(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>()
            .join(" ");
        let to_delete_name = self.delete_name(&fdecl.name);
        let subsumed_name = self.subsumed_name(&fdecl.name);
        let view_name = self.view_name(&fdecl.name);
        let delete_subsume_ruleset = self.proof_names().delete_subsume_ruleset_name.clone();
        let fresh_name = self.egraph.parser.symbol_gen.fresh("delete_rule");

        format!(
            "(rule (({to_delete_name} {child_names})
                    ({view_name} {child_names} out))
                   ((delete ({view_name} {child_names} out))
                    (delete ({to_delete_name} {child_names})))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}\")
             (rule (({subsumed_name} {child_names})
                    ({view_name} {child_names} out))
                   ((subsume ({view_name} {child_names} out)))
                    :ruleset {delete_subsume_ruleset}
                    :name \"{fresh_name}_subsume\")"
        )
    }

    /// Generate rules that run a merge function for a custom function.
    /// One rule runs the merge function when two different values are present for the same children.
    /// Another rule cleans up old values, necessary because the newly merged value may be equal to one of the old values.
    fn handle_merge_fn(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        child_names: &[String],
        child_names_str: &str,
        _view_name: &str,
        rebuilding_ruleset: &str,
    ) -> String {
        let name = &fdecl.name;

        let merge_fn = &fdecl
            .merge
            .as_ref()
            .unwrap_or_else(|| panic!("Proofs don't support :no-merge"));

        let fresh_name = self.egraph.parser.symbol_gen.fresh("merge_rule");
        let cleanup_name = self.egraph.parser.symbol_gen.fresh("merge_cleanup");

        let p1_fresh = self.egraph.parser.symbol_gen.fresh("p1");
        let p2_fresh = self.egraph.parser.symbol_gen.fresh("p2");
        let view_name = self.view_name(&fdecl.name);
        let rebuilding_cleanup_ruleset = self.proof_names().rebuilding_cleanup_ruleset_name.clone();
        let proof_query = if self.egraph.proof_state.proofs_enabled {
            // View is a function with proof output; bind proof variables
            format!(
                "(= {p1_fresh} ({view_name} {child_names_str} old))
                     (= {p2_fresh} ({view_name} {child_names_str} new))
                    "
            )
        } else {
            // View is a function with Unit output; no need to bind the output
            "".to_string()
        };
        let proof_var = if self.egraph.proof_state.proofs_enabled {
            self.fresh_var()
        } else {
            "()".to_string()
        };
        let mut merge_fn_code = vec![];
        // canonicalize-at-creation may inject `find_UFold` RHS lookups into the
        // merge expression's constructor creations, which force the generated
        // merge rule to opt into :unsafe-seminaive (see `add_term_and_view`).
        self.egraph.proof_state.emitted_canon_lookup = false;
        let merge_fn_var = self.instrument_action_expr(
            merge_fn,
            &mut merge_fn_code,
            &Justification::Merge(name.clone(), p1_fresh.clone(), p2_fresh.clone()),
        );
        let merge_unsafe_opt = if self.egraph.proof_state.emitted_canon_lookup {
            ":unsafe-seminaive"
        } else {
            ""
        };
        let merge_fn_code_str = merge_fn_code.join("\n");
        let mut updated = child_names.to_vec();
        updated.push(merge_fn_var.clone());
        let term = format!("({name} {child_names_str} {merge_fn_var})");

        let rule_proof = if self.egraph.proof_state.proofs_enabled {
            let to_ast = self.fname_to_ast_name(name);
            let merge_fn_constructor = self.proof_names().merge_fn_constructor.clone();
            format!(
                "(let {proof_var}
                            ({merge_fn_constructor} \"{name}\"
                                  {p1_fresh}
                                  {p2_fresh}
                                  ({to_ast} {term})))"
            )
        } else {
            "".to_string()
        };
        let term_and_proof = self.update_view(name, &updated, &proof_var);
        let cleanup_constructor = self.egraph.parser.symbol_gen.fresh("mergecleanup");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("mergecleanupsort");
        let output_sort = fdecl.schema.output.clone();

        // The first runs the merge function adding a new row.
        // The second deletes rows with old values for the old variable, while the third deletes rows with new values for the new variable.
        format!(
            "(sort {fresh_sort})
                 (constructor {cleanup_constructor} ({output_sort} {output_sort}) {fresh_sort} :internal-hidden)
                 (rule (({view_name} {child_names_str} old)
                        ({view_name} {child_names_str} new)
                        (!= old new)
                        (= (ordering-max old new) new)
                        {proof_query})
                       (
                        {merge_fn_code_str}
                        {rule_proof}
                        {term_and_proof}
                        ({cleanup_constructor} {merge_fn_var} old)
                        ({cleanup_constructor} {merge_fn_var} new)
                       )
                        :ruleset {rebuilding_ruleset} {merge_unsafe_opt}
                        :name \"{fresh_name}\")
                
                 (rule (({cleanup_constructor} merged old)
                        ({view_name} {child_names_str} merged)
                        ({view_name} {child_names_str} old)
                        (!= merged old))
                       ((delete ({view_name} {child_names_str} old)))
                        :ruleset {rebuilding_cleanup_ruleset}
                        :name \"{cleanup_name}\")
                ",
        )
    }

    /// Generate a rule that handles congruence for constructors.
    /// When two different values are present for the same children,
    /// we union those two values together.
    fn handle_congruence(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        child_names: &[String],
        rebuilding_ruleset: &str,
    ) -> String {
        // Congruence rule
        let fresh_name = self.egraph.parser.symbol_gen.fresh("congruence_rule");
        let mut child_names_new = child_names.to_vec();
        child_names_new.push("new".to_string());
        let mut child_names_old = child_names.to_vec();
        child_names_old.push("old".to_string());
        let (query1, prf1) = self.query_view_and_get_proof(&fdecl.name, &child_names_new);
        let (query2, prf2) = self.query_view_and_get_proof(&fdecl.name, &child_names_old);
        let sym = &self.proof_names().eq_sym_constructor;
        let trans = &self.proof_names().eq_trans_constructor;

        // Proof is by transitivity. A view proof gives a proof that
        // representative r_1 = f(c_1,...,c_n).
        // We also have a proof that other eclass representative r_2 = f(c_1,...,c_n), the same term.
        // We want a proof that r1 = r2, which we get by transitivity.
        let union_code = self.union(
            &fdecl.schema.output,
            "new",
            "old",
            &Justification::Proof(format!("({trans} {prf1} ({sym} {prf2}))",)),
        );
        format!(
            "(rule ({query1}
                        {query2}
                        (!= old new)
                        (= (ordering-max old new) new))
                       ({union_code})
                        :ruleset {rebuilding_ruleset}
                        :name \"{fresh_name}\")"
        )
    }

    /// Generate rules that handle merge functions or congruence.
    /// For custom functions, we generate rules that run the merge function.
    /// For constructors, we generate congruence rules.
    fn handle_merge_or_congruence(&mut self, fdecl: &ResolvedFunctionDecl) -> String {
        let child_names = fdecl
            .schema
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| format!("c{i}_"))
            .collect::<Vec<_>>();
        let child_names_str = child_names.join(" ");
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let view_name = self.view_name(&fdecl.name);
        if fdecl.subtype == FunctionSubtype::Custom {
            // No `:merge` on a Custom function: nothing to do here.
            // `command_supports_proof_encoding` (when proofs are
            // enabled) rejects this case via `NoMergeOnNonGlobalFunction`,
            // so reaching this branch means we're in plain
            // term-encoding mode where the merge rule simply isn't
            // needed — the function is set-once or last-write-wins
            // at the relational level, with no proof artifacts to
            // thread through.
            if fdecl.merge.is_none() {
                return String::new();
            }
            self.handle_merge_fn(
                fdecl,
                &child_names,
                &child_names_str,
                &view_name,
                &rebuilding_ruleset,
            )
        } else {
            self.handle_congruence(fdecl, &child_names, &rebuilding_ruleset)
        }
    }

    /// Each function/constructor gets a term table and a view table.
    /// The term table stores underlying representative terms.
    /// The view table stores child terms and their eclass.
    /// The view table is mutated using delete, but we never delete from term tables.
    /// We re-use the original name of the function for the term table.
    fn term_and_view(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let schema = &fdecl.schema;
        let out_type = schema.output.clone();

        let name = &fdecl.name;
        let view_name = self.view_name(&fdecl.name);
        let in_sorts = ListDisplay(schema.input.clone(), " ");
        let fresh_sort = self.egraph.parser.symbol_gen.fresh("view");
        let delete_rule = self.delete_and_subsume(fdecl);
        let to_delete_name = self.delete_name(&fdecl.name);
        let subsumed_name = self.subsumed_name(&fdecl.name);
        let term_sorts = format!(
            "{in_sorts} {}",
            if fdecl.subtype == FunctionSubtype::Constructor {
                "".to_string()
            } else {
                schema.output.to_string()
            }
        );
        let view_sorts = format!("{in_sorts} {out_type}");
        let proof_constructors = self.proof_functions(fdecl, &view_sorts);

        let view_sort = if fdecl.subtype == FunctionSubtype::Constructor {
            schema.output.clone()
        } else {
            fresh_sort.clone()
        };
        let to_ast_view_sort = self.add_to_ast(&view_sort);

        if self.egraph.proof_state.proofs_enabled {
            self.egraph
                .proof_state
                .proof_names
                .fn_to_term_sort
                .insert(name.clone(), view_sort.clone());
        }
        let merge_rule = self.handle_merge_or_congruence(fdecl);
        // the term table has child_sorts as inputs
        // the view table has child_sorts + the leader term for the eclass
        // Propagate cost, unextractable, hidden, and internal_let flags from the original function
        let mut term_flags = String::new();
        if let Some(cost) = fdecl.cost {
            term_flags.push_str(&format!(" :cost {cost}"));
        }
        // View is always a function (returning Proof or Unit), with :merge old
        let proof_type = self.proof_type_str().to_string();
        let mut view_flags = String::new();
        if fdecl.unextractable {
            view_flags.push_str(" :unextractable");
        }
        if fdecl.internal_hidden {
            view_flags.push_str(" :internal-hidden");
        }
        if fdecl.internal_let {
            view_flags.push_str(" :internal-let");
        }
        let view_decl = format!(
            "(function {view_name} ({view_sorts}) {proof_type} :merge old :internal-term-constructor {name}{view_flags})"
        );
        self.parse_program(&format!(
            "
            (sort {fresh_sort})
            {to_ast_view_sort}
            (constructor {name} ({term_sorts}) {view_sort}{term_flags} :internal-hidden :unextractable)
            {view_decl}
            (constructor {to_delete_name} ({in_sorts}) {fresh_sort} :internal-hidden)
            (constructor {subsumed_name} ({in_sorts}) {fresh_sort} :internal-hidden)
            {proof_constructors}
            {merge_rule}
            {delete_rule}",
        ))
    }

    fn proof_functions(&mut self, _fdecl: &ResolvedFunctionDecl, _view_sorts: &str) -> String {
        // ViewProof is now merged into the view table as its output column
        "".to_string()
    }

    /// Rules that update the views when children change.
    fn rebuilding_rules(&mut self, fdecl: &ResolvedFunctionDecl) -> Vec<Command> {
        let types = fdecl.resolved_schema.view_types();

        // Check if there are any eq-sort columns at all; if not, no rebuild rule needed.
        if !types.iter().any(|t| t.is_eq_sort()) {
            return vec![];
        }

        // Native-UF mode: drive the rebuild off the `@UFChange_S` onchange
        // relation (the delta path). Native-UF provides the leader-change
        // deltas, so the rebuild always scopes to `view ⋈ δuf`. Under
        // canon-at-creation there is no `δview ⋈ uf_old` term, so a full
        // re-scan is never needed; this is always the onchange-delta rebuild.
        if self.native_uf() {
            return self.rebuilding_rules_native_uf(fdecl, &types);
        }

        let view_name = self.view_name(&fdecl.name);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..types.len()).map(child).collect();
        let children = format!("{}", ListDisplay(&children_vec, " "));

        // For each eq-sort column, look up its leader via the UF table.
        // For non-eq-sort columns, the leader is the same as the original.
        let mut uf_queries = vec![];
        let mut leader_vars: Vec<String> = vec![];
        let mut bool_neq_exprs = vec![];
        let mut uf_proof_vars: Vec<Option<String>> = vec![];

        for (i, ty) in types.iter().enumerate() {
            if ty.is_eq_sort() {
                let leader_var = format!("c{i}_leader_");
                let uf_function_name = self.uf_function_name(ty.name());
                let ci = child(i);

                if self.egraph.proof_state.proofs_enabled {
                    // UF function index returns a Pair(leader, proof); one lookup gives both
                    let pair_var = self.fresh_var();
                    let proof_var = format!("(pair-second {pair_var})");
                    uf_queries.push(format!(
                        "(= {pair_var} ({uf_function_name} {ci}))
                         (= {leader_var} (pair-first {pair_var}))"
                    ));
                    uf_proof_vars.push(Some(proof_var));
                } else {
                    uf_queries.push(format!("(= {leader_var} ({uf_function_name} {ci}))"));
                    uf_proof_vars.push(None);
                }

                bool_neq_exprs.push(format!("(bool-!= {ci} {leader_var})"));
                leader_vars.push(leader_var);
            } else {
                leader_vars.push(child(i));
                uf_proof_vars.push(None);
            }
        }

        let uf_query_str = uf_queries.join("\n       ");
        let or_expr = format!("(or {})", bool_neq_exprs.join("\n             "));
        let filter_query = format!("(guard {or_expr})");

        // Build the updated children: use leader_var for eq-sort columns, original for others.
        let children_updated: Vec<String> = leader_vars.clone();

        let fresh_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
        let (query_view, view_prf) = self.query_view_and_get_proof(&fdecl.name, &children_vec);

        // Build proof code if proofs are enabled.
        // We chain congruence proofs for each updated child and a transitivity proof
        // for the representative (last column) update.
        let (pf_code, pf_var) = if self.egraph.proof_state.proofs_enabled {
            let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
            let congr_constructor = self.proof_names().congr_constructor.clone();
            let sym_constructor = self.proof_names().eq_sym_constructor.clone();

            // Start with the view proof and apply congruence for each eq-sort child
            // (excluding the last column if this is a constructor, since that's the representative).
            let mut current_proof = view_prf.clone();
            let mut proof_code_parts = vec![];

            for (i, ty) in types.iter().enumerate() {
                if !ty.is_eq_sort() {
                    continue;
                }

                let uf_prf = uf_proof_vars[i].as_ref().unwrap();

                if fdecl.subtype == FunctionSubtype::Constructor && i == types.len() - 1 {
                    // Updating the representative term (last column of constructor):
                    // use transitivity with sym of the UF proof
                    let new_proof = self.fresh_var();
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                           ({eq_trans_constructor}
                              ({sym_constructor} {uf_prf})
                              {current_proof}))",
                    ));
                    current_proof = new_proof;
                } else {
                    // Updating a child via congruence
                    let new_proof = self.fresh_var();
                    proof_code_parts.push(format!(
                        "(let {new_proof}
                              ({congr_constructor} {current_proof} {i}
                                                   {uf_prf}))",
                    ));
                    current_proof = new_proof;
                }
            }

            (proof_code_parts.join("\n"), current_proof)
        } else {
            ("".to_string(), "()".to_string())
        };

        let updated_view = self.update_view(&fdecl.name, &children_updated, &pf_var);

        // `--fast-rebuild` (term mode only): drop the always-empty
        // `δview ⋈ uf_old` seminaive variant of this rule. The rebuild join is
        // `view ⋈ @UF_Sf`; its seminaive derivative is `view⋈δuf` (the real
        // re-canonicalization work) plus `δview⋈uf_old` (re-checking view rows
        // born this iteration against the unchanged UF). Under
        // canonicalize-at-creation new view rows are already canonical, so the
        // δview term is empty — but seminaive still pays to scan δview each
        // iteration. We record this rule's view function so `add_rule` can
        // exclude the view atom from being a delta focus (bridge
        // `RuleBuilderOps::set_focus_exclude_table`), keeping only `view⋈δuf`.
        // Bit-exact with the full rebuild; never in proof mode.
        if self.fast_rebuild() && !self.egraph.proof_state.proofs_enabled {
            self.egraph
                .proof_state
                .rebuild_view_exclude
                .insert(fresh_name.clone(), view_name.clone());
        }

        // Make a single rule that updates the view when any child's leader differs.
        let rule = format!(
            "(rule ({query_view}
                    {uf_query_str}
                    {filter_query}
                    )
                 (
                  {pf_code}
                  {updated_view}
                  (delete ({view_name} {children}))
                 )
                  :ruleset {} :name \"{fresh_name}\")",
            self.proof_names().rebuilding_ruleset_name
        );
        self.parse_program(&rule)
    }

    /// Native-UF variant of `rebuilding_rules`. The relational rebuild
    /// joined every view row against `@UF_Sf` and re-canonicalized any row
    /// whose child differed from its leader. Here we drive the rebuild off
    /// the `@UFChange_S` onchange relation instead.
    ///
    /// On each leader change the union-find emits an onchange row
    /// `(write_lhs write_rhs lhs_leader rhs_leader new_leader)`. With
    /// union-by-min, the *displaced* id (the leader that stopped being
    /// canonical) is `max(lhs_leader, rhs_leader)` and the surviving leader
    /// is `new_leader = min(...)`. Because every view row is canonical at
    /// insertion time (canon-at-creation) and stays canonical until one of
    /// its ids is displaced, the only newly-stale id a view row can hold is
    /// exactly that displaced id. Onchange rows are retained (a relation is
    /// append-only), so across rebuild passes every stale view row
    /// eventually matches some retained onchange row.
    ///
    /// For each eq-sort view column `j` we emit a rule that joins onchange
    /// rows against view rows holding the displaced id in column `j`, then
    /// re-canonicalizes *all* eq-sort columns via the `@canon_S` primitive
    /// (full find — so chained unions collapse in one step) and
    /// retracts the old row. This mirrors the relational rebuild's
    /// retract-old / insert-canonical behavior bit-for-bit, while only
    /// touching view rows that actually reference a changed class.
    fn rebuilding_rules_native_uf(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        types: &[ArcSort],
    ) -> Vec<Command> {
        if self.egraph.proof_state.proofs_enabled {
            return self.rebuilding_rules_native_uf_proof(fdecl, types);
        }
        let view_name = self.view_name(&fdecl.name);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..types.len()).map(child).collect();
        let children = ListDisplay(&children_vec, " ").to_string();

        // Canonicalized children: eq-sort columns wrapped in `@canon_S`,
        // non-eq-sort columns left as-is. (For non-eq columns `@canon`
        // isn't defined; they never change.)
        let children_canon: Vec<String> = types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                if ty.is_eq_sort() {
                    let canon_prim = self.canon_prim_name(ty.name());
                    format!("({canon_prim} {})", child(i))
                } else {
                    child(i)
                }
            })
            .collect();
        let updated_view = self.update_view(&fdecl.name, &children_canon, "()");

        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();

        // Guard: only fire when the row is actually stale, i.e. some eq-sort
        // child is non-canonical (`ci != (canon ci)`). Mirrors the relational
        // rebuild's `(guard (or (bool-!= ci ci_leader) ...))`. Without it a row
        // could be re-inserted unchanged, which never terminates.
        let neq_exprs: Vec<String> = types
            .iter()
            .enumerate()
            .filter(|(_, ty)| ty.is_eq_sort())
            .map(|(i, ty)| {
                let canon_prim = self.canon_prim_name(ty.name());
                format!("(bool-!= {} ({canon_prim} {}))", child(i), child(i))
            })
            .collect();
        let guard = format!("(guard (or {}))", neq_exprs.join(" "));

        // The rebuild is driven by leader changes recorded in the `@UFChange_S`
        // onchange relation. With union-by-min the surviving leader is
        // `new_leader = min(lhs_leader, rhs_leader)`; the *displaced* id (the
        // leader that stopped being canonical) is `max(lhs_leader, rhs_leader)`
        // — exactly the id a stale view row can still be holding. The
        // leader-change callback stores this displaced id as the onchange row's
        // 6th column `disp_`.
        //
        // For each eq-sort view column `j` we emit ONE rule that equi-joins the
        // view's column `j` against the *stored* displaced column. Equi-joining
        // a stored column lets the query planner index the view by column `j`
        // and probe it with `disp_`; a computed `(= cj (ordering-max ll rl))`
        // join cannot be indexed and forces a full onchange×view scan (which
        // was catastrophically slow). This halves the rebuild rules versus
        // joining `lhs_leader` and `rhs_leader` separately — the extra rule
        // there always searched the join on the *surviving* (non-displaced)
        // leader only to be rejected by the guard. The `(guard (or bool-!=))`
        // still filters to genuinely stale rows for termination.
        //
        // The action re-canonicalizes *all* eq-sort columns via the `@canon_S`
        // primitive (full find, so chained unions collapse in one step) and
        // retracts the old row, mirroring the relational rebuild's
        // retract-old / insert-canonical behavior bit-for-bit and giving it a
        // terminating fixpoint.
        let mut rules = String::new();
        for (j, ty) in types.iter().enumerate() {
            if !ty.is_eq_sort() {
                continue;
            }
            let uf_change_rel = self.uf_change_rel_name(ty.name());
            let cj = child(j);
            // Onchange row columns:
            // (write_lhs write_rhs lhs_leader rhs_leader new_leader displaced).
            let rule_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
            rules.push_str(&format!(
                "(rule (({uf_change_rel} _wl_ _wr_ _ll_ _rl_ _nl_ disp_)
                        ({view_name} {children})
                        (= {cj} disp_)
                        {guard})
                       ({updated_view}
                        (delete ({view_name} {children})))
                        :ruleset {rebuilding_ruleset} :name \"{rule_name}\")\n"
            ));

            // `--fast-rebuild` (native-UF): drop this rule's always-empty
            // `δview ⋈ uf_old` seminaive variant. The join above is
            // `@UFChange_S ⋈ view`; seminaive runs it as `δ@UFChange_S ⋈ view`
            // (the real re-canonicalization — a new leader change finds existing
            // stale view rows) PLUS `@UFChange_S_old ⋈ δview` (re-probing view
            // rows born THIS iteration against accumulated onchange rows). Under
            // canonicalize-at-creation a fresh view row holds only current
            // leaders, so it never matches an onchange `disp_` (a displaced
            // non-leader): the δview variant is always empty, yet seminaive
            // still pays to enumerate δview every iteration. Recording this
            // rule's view in `rebuild_view_exclude` makes `add_rule` exclude the
            // view atom from being a delta focus
            // (`RuleBuilderOps::set_focus_exclude_table`), keeping only
            // `δ@UFChange_S ⋈ view`. Mirrors the relational rebuild's
            // fast-rebuild gating (~:876). Plain `--native-uf` (no
            // `--fast-rebuild`) keeps the δview variant, so its per-column δview
            // enumeration is the measurable cost of NOT fast-rebuilding.
            // Bit-exact with the full rebuild; never in proof mode.
            if self.fast_rebuild() {
                self.egraph
                    .proof_state
                    .rebuild_view_exclude
                    .insert(rule_name.clone(), view_name.clone());
            }

            // Drain the onchange relation. `@UFChange_S` is a relation, so it is
            // append-only — one row per leader change, retained forever. Without
            // draining, seminaive re-joins every later iteration's *new* view
            // rows against the entire accumulated onchange history
            // (`δ@view ⋈ full@UFChange_S`), a super-linear blowup that dominated
            // `--native-uf` rebuild time.
            //
            // The drain runs in its own `@uf_change_drain` ruleset, scheduled
            // *after the rebuild saturates* (see `rebuild()`). It cannot run
            // interleaved with `@rebuilding`: congruence in `@rebuilding` issues
            // unions, whose leader-change callback writes the onchange relation
            // via `lookup_or_insert`; deleting from that same relation in the
            // same stratum corrupts its hash-cons index (panics in
            // `predict_val`). At the rebuild fixpoint, every view row is
            // canonical, so every onchange row has already been matched against
            // all view rows referencing its displaced id and is dead weight;
            // deleting them there is safe (no unconsumed row removed, no
            // concurrent union) and keeps the relation from accumulating across
            // the rebuild passes of successive `(run N)` iterations. One drain
            // rule per onchange relation.
            if self
                .egraph
                .proof_state
                .uf_change_drained
                .insert(uf_change_rel.clone())
            {
                let drain_name = self.egraph.parser.symbol_gen.fresh("uf_change_drain_rule");
                let drain_ruleset = self.proof_names().uf_change_drain_ruleset_name.clone();
                rules.push_str(&format!(
                    "(rule (({uf_change_rel} wl_ wr_ ll_ rl_ nl_ disp_))
                           ((delete ({uf_change_rel} wl_ wr_ ll_ rl_ nl_ disp_)))
                            :ruleset {drain_ruleset} :name \"{drain_name}\")\n"
                ));
            }
        }

        // FULL native-UF rebuild (`--native-uf` WITHOUT `--fast-rebuild`) keeps
        // the `δview ⋈ uf_old` term WITHOUT a separate probe rule: each
        // per-column rule above is `@UFChange_S ⋈ view`, and seminaive already
        // runs its `@UFChange_S_old ⋈ δview` variant (view rows born this
        // iteration probed against accumulated onchange rows) when the view atom
        // is not focus-excluded — which it is NOT unless `--fast-rebuild`
        // registered it in `rebuild_view_exclude` (see above). So plain `+nuf`
        // pays exactly that per-column δview enumeration — the measurable cost of
        // the FULL rebuild — and `--fast-rebuild` drops it via `focus_exclude`,
        // leaving only `δ@UFChange_S ⋈ view` (`+nuf+fastrb`). Under
        // canonicalize-at-creation a fresh view row holds only current leaders
        // and so matches no logged `disp_`, so the kept δview variant is empty:
        // bit-exact with the relational rebuild's (likewise empty) `δview ⋈
        // uf_old` term. No standalone probe rule is needed.

        self.parse_program(&rules)
    }

    /// Proof-mode variant of [`Self::rebuilding_rules_native_uf`].
    ///
    /// The onchange relation now carries a trailing `Proof` column `pf_`
    /// proving `displaced_leader = new_leader` (composed in the leader-change
    /// callback). Where the term-mode rebuild canonicalizes *all* columns to
    /// the current full leader in one firing (no proof to thread), the
    /// proof-mode rebuild canonicalizes column `j` a single displacement step
    /// to `new_leader` (`nl_`) using exactly that onchange proof, mirroring the
    /// relational rebuild's per-column `pair-second` congruence proof — chains
    /// of unions collapse across successive firings. Each firing:
    ///   * rewrites column `j` from the displaced id to `nl_`,
    ///   * threads `pf_` (proof `cj = nl_`) into the view's proof via `Congr`
    ///     (child column) or `Trans(Sym(pf_), view_proof)` (representative
    ///     column of a constructor), exactly as `rebuilding_rules` does for the
    ///     relational `@UF_Sf` index,
    ///   * retracts the stale row.
    fn rebuilding_rules_native_uf_proof(
        &mut self,
        fdecl: &ResolvedFunctionDecl,
        types: &[ArcSort],
    ) -> Vec<Command> {
        let view_name = self.view_name(&fdecl.name);
        let child = |i: usize| format!("c{i}_");
        let children_vec: Vec<String> = (0..types.len()).map(child).collect();
        let children = ListDisplay(&children_vec, " ").to_string();
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let eq_trans_constructor = self.proof_names().eq_trans_constructor.clone();
        let congr_constructor = self.proof_names().congr_constructor.clone();
        let sym_constructor = self.proof_names().eq_sym_constructor.clone();

        let mut rules = String::new();
        for (j, ty) in types.iter().enumerate() {
            if !ty.is_eq_sort() {
                continue;
            }
            let uf_change_rel = self.uf_change_rel_name(ty.name());
            let cj = child(j);
            // Onchange row columns: (write_lhs write_rhs lhs_leader rhs_leader
            // new_leader proof). We equi-join the view's column `j` against the
            // displaced leader (`ll_` or `rl_`), and read the new leader `nl_`
            // and the composed proof `pf_`.
            for leader_col in ["ll_", "rl_"] {
                let rule_name = self.egraph.parser.symbol_gen.fresh("rebuild_rule");
                // Build the updated children: column `j` becomes `nl_`, all
                // others unchanged.
                let children_updated: Vec<String> = (0..types.len())
                    .map(|i| if i == j { "nl_".to_string() } else { child(i) })
                    .collect();

                // View proof query + variable.
                let (query_view, view_prf) =
                    self.query_view_and_get_proof(&fdecl.name, &children_vec);

                // Compose the new view proof using `pf_` (proof `cj = nl_`).
                let new_proof = self.fresh_var();
                let proof_code = if fdecl.subtype == FunctionSubtype::Constructor
                    && j == types.len() - 1
                {
                    // Representative column of a constructor: transitivity with
                    // the symmetric onchange proof.
                    format!(
                        "(let {new_proof} ({eq_trans_constructor} ({sym_constructor} pf_) {view_prf}))"
                    )
                } else {
                    // Child column: congruence at index `j`.
                    format!("(let {new_proof} ({congr_constructor} {view_prf} {j} pf_))")
                };

                let updated_view = self.update_view(&fdecl.name, &children_updated, &new_proof);

                // Guard: column `j` must actually be stale w.r.t. *this* leader
                // change (`cj != nl_`); otherwise the onchange proof `pf_`
                // (proving `displaced = nl_`) would not apply and the rewrite
                // would be a no-op (non-terminating).
                rules.push_str(&format!(
                    "(rule (({uf_change_rel} _wl_ _wr_ ll_ rl_ nl_ pf_)
                            {query_view}
                            (= {cj} {leader_col})
                            (guard (bool-!= {cj} nl_)))
                           ({proof_code}
                            {updated_view}
                            (delete ({view_name} {children})))
                            :ruleset {rebuilding_ruleset} :name \"{rule_name}\")\n"
                ));
            }
        }

        self.parse_program(&rules)
    }

    /// Instrument fact replaces terms with looking up
    /// canonical versions in the view.
    /// It also needs to look up references to globals.
    /// Adds the instrumented fact to `res` and returns a proof that the fact matched.
    fn instrument_fact(
        &mut self,
        fact: &ResolvedFact,
        res: &mut Vec<String>,
        action_lookups: &mut Vec<String>,
    ) -> String {
        match fact {
            // In proof normal form, this is the only way that function calls apppear.
            // A non-eq-sort pass-through global (`(= $g var)` body equality) is
            // excluded: it has no view table (`@$gView`), so it must be read via
            // the 0-arg lookup `($g)` like the action path — handled by the
            // generic `Eq` arm below (which calls `instrument_fact_expr`, which
            // emits `($g)` for pass-through globals).
            ResolvedFact::Eq(
                _span,
                ResolvedExpr::Call(
                    _span2,
                    head @ ResolvedCall::Func(FuncType {
                        subtype: FunctionSubtype::Custom,
                        ..
                    }),
                    args,
                ),
                // TODO this could actually be arbitrary pretty easily, it's just nested functions that are hard.
                ResolvedExpr::Var(_span3, v),
            ) if !self.is_pass_through_global(head.name()) => {
                let mut new_args = vec![];
                let mut arg_proofs = vec![];
                for arg in args {
                    let (var, proof) = self.instrument_fact_expr(arg, res, action_lookups);
                    new_args.push(var);
                    arg_proofs.push(proof);
                }
                new_args.push(v.to_string());

                let view_name = self.view_name(head.name());
                let args_str = ListDisplay(new_args, " ");

                // View is always a function; query it and bind the output
                let proof_var = self.fresh_var();
                res.push(format!("(= {proof_var} ({view_name} {args_str}))"));

                if self.egraph.proof_state.proofs_enabled {
                    let mut proof = proof_var;
                    for (i, arg_proof) in arg_proofs.into_iter().enumerate() {
                        let congr = &self.proof_names().congr_constructor;
                        proof = format!(
                            "
                            ({congr} {proof} {i} {arg_proof})
                            "
                        );
                    }
                    proof
                } else {
                    "()".to_string()
                }
            }
            ResolvedFact::Eq(_span, left_expr, right_expr) => {
                let (v1, p1) = self.instrument_fact_expr(left_expr, res, action_lookups);
                let (v2, p2) = self.instrument_fact_expr(right_expr, res, action_lookups);
                res.push(format!("(= {v1} {v2})"));
                let sym = &self.proof_names().eq_sym_constructor;
                let trans = &self.proof_names().eq_trans_constructor;

                format!("({trans} ({sym} {p1}) {p2})",)
            }
            ResolvedFact::Fact(generic_expr) => {
                let (_, proof) = self.instrument_fact_expr(generic_expr, res, action_lookups);
                proof
            }
        }
    }

    /// Instruments a fact expression to use the view tables.
    /// Assumes there are no function lookups in the term.
    /// Returns a variable representing the expression and a proof that the expression was matched.
    /// Proves a ground equality t1 = t2 where t1 is the eclass representative and t2 matches `expr` syntactically.
    fn instrument_fact_expr(
        &mut self,
        expr: &ResolvedExpr,
        res: &mut Vec<String>,
        action_lookups: &mut Vec<String>,
    ) -> (String, String) {
        match expr {
            ResolvedExpr::Lit(_, lit) => {
                let proof_code = if self.egraph.proof_state.proofs_enabled {
                    let fiat_constructor = &self.proof_names().fiat_constructor;
                    let lit_sort = literal_sort(lit);
                    let to_ast = self
                        .proof_names()
                        .sort_to_ast_constructor
                        .get(lit_sort.name())
                        .unwrap();
                    format!("({fiat_constructor} ({to_ast} {lit}) ({to_ast} {lit}))")
                } else {
                    "()".to_string()
                };

                (format!("{lit}"), proof_code)
            }
            ResolvedExpr::Var(_, resolved_var) => {
                let var = &resolved_var.name;
                (
                    resolved_var.name.clone(),
                    if !self.egraph.proof_state.proofs_enabled {
                        "()".to_string()
                    } else if resolved_var.sort.is_eq_sort() {
                        let term_proof_name = self.term_proof_name(resolved_var.sort.name());
                        let fresh_proof = self.fresh_var();
                        // Every eq-sort term has its term_proof set at
                        // constructor-creation time, so this proof is guaranteed
                        // present when the rule fires. Fetch it directly in the
                        // action (the rule is then `:unsafe-seminaive`, see
                        // instrument_rule) instead of as a body join — one fewer
                        // join per eq-sort body variable. Callers that don't
                        // build a proof (run :until, check) discard these.
                        action_lookups
                            .push(format!("(let {fresh_proof} ({term_proof_name} {var}))"));
                        fresh_proof
                    } else {
                        let fiat_constructor = &self.proof_names().fiat_constructor;
                        let lit_sort = resolved_var.sort.name();
                        let to_ast = self
                            .proof_names()
                            .sort_to_ast_constructor
                            .get(lit_sort)
                            .unwrap();
                        format!("({fiat_constructor} ({to_ast} {var}) ({to_ast} {var}))")
                    },
                )
            }
            ResolvedExpr::Call(_, resolved_call, args) => {
                let mut new_args = vec![];
                // Variables and constants don't need subproofs, but constructor calls do.
                let mut arg_proofs: Vec<Option<String>> = vec![];
                for arg in args {
                    if matches!(arg, ResolvedExpr::Var(_, _) | ResolvedExpr::Lit(_, _)) {
                        new_args.push(arg.to_string());
                        arg_proofs.push(None);
                    } else {
                        let (arg_str, proof) = self.instrument_fact_expr(arg, res, action_lookups);
                        new_args.push(arg_str);
                        arg_proofs.push(Some(proof));
                    }
                }
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        // A non-eq-sort global is a plain 0-arg key-value store
                        // (see `is_non_eq_sort_global_decl`); read it back with
                        // the direct 0-arg lookup `($N)` and treat its value like
                        // a primitive constant (fiat-justified), the same as a
                        // non-eq-sort variable or literal in a fact.
                        if self.is_pass_through_global(&func_type.name) {
                            let value =
                                format!("({} {})", func_type.name, ListDisplay(&new_args, " "));
                            let proof = if self.proofs_enabled() {
                                let fiat_constructor = &self.proof_names().fiat_constructor;
                                let to_ast = self
                                    .proof_names()
                                    .sort_to_ast_constructor
                                    .get(func_type.output.name())
                                    .unwrap();
                                format!(
                                    "({fiat_constructor} ({to_ast} {value}) ({to_ast} {value}))"
                                )
                            } else {
                                "()".to_string()
                            };
                            return (value, proof);
                        }

                        assert!(
                            func_type.subtype == FunctionSubtype::Constructor,
                            "Only constructor function calls are allowed in fact expressions due to proof normal form. Got {func_type:?}",
                        );

                        let fv = self.fresh_var();
                        let view_name = self.view_name(&func_type.name);
                        let args_str = ListDisplay(new_args, " ");

                        let proof = {
                            // View is always a function; query it and bind the output
                            let view_proof_var = self.fresh_var();
                            res.push(format!(
                                "(= {view_proof_var} ({view_name} {args_str} {fv}))"
                            ));
                            if self.proofs_enabled() {
                                let mut proof = view_proof_var;
                                for (i, arg_proof) in arg_proofs.into_iter().enumerate() {
                                    if let Some(arg_proof) = arg_proof {
                                        let congr = &self.proof_names().congr_constructor;
                                        proof = format!(
                                            "
                            ({congr} {proof} {i} {arg_proof})
                            "
                                        );
                                    }
                                }
                                proof
                            } else {
                                "()".to_string()
                            }
                        };
                        (fv, proof)
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        if specialized_primitive.output().is_eq_sort() {
                            panic!(
                                "Term encoding does not support eq-sort primitive expressions in facts"
                            );
                        }
                        // `get-size!` sums the sizes of (filtered) user tables as
                        // a saturation budget. Under term encoding the user
                        // constructors are monotonic hash-cons TERM tables; their
                        // CANONICAL rows live in the `@<F>View` tables. Counting
                        // the term tables over-counts vs the normal backend (the
                        // budget then trips earlier — term != normal). Redirect
                        // `get-size!` to the canonical view tables so the size
                        // proxy is mode-invariant (see `instrument_get_size`).
                        let new_args = if specialized_primitive.name() == "get-size!" {
                            self.instrument_get_size(&new_args)
                        } else {
                            new_args
                        };
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(= {fv} ({} {}))",
                            specialized_primitive.name(),
                            ListDisplay(new_args, " ")
                        ));

                        let proof = if self.proofs_enabled() {
                            let fiat_constructor = &self.proof_names().fiat_constructor;
                            let to_ast = self
                                .proof_names()
                                .sort_to_ast_constructor
                                .get(specialized_primitive.output().name())
                                .unwrap();
                            format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))")
                        } else {
                            "()".to_string()
                        };

                        (fv.clone(), proof)
                    }
                }
            }
        }
    }

    /// Return the instrumented query and a proof that it matched.
    /// Returns `(body_facts, action_lookups, proof)`. Eq-sort variables'
    /// `term_proof` fetches are emitted into `action_lookups` as
    /// `(let p (term_proof v))` lines for the caller to splice into the
    /// rule's actions (the rule is then `:unsafe-seminaive`). Callers
    /// that don't build a proof (`run :until`, `check`) discard the
    /// lookups and the proof.
    fn instrument_facts(&mut self, facts: &[ResolvedFact]) -> (Vec<String>, Vec<String>, String) {
        let mut res = vec![];
        let mut action_lookups = vec![];
        let mut proof = vec![];

        for fact in facts.iter() {
            let f_proof = self.instrument_fact(fact, &mut res, &mut action_lookups);
            proof.push(f_proof);
        }

        (res, action_lookups, self.format_prooflist(&proof))
    }

    /// Rewrite the (string-literal) arguments of a `get-size!` call so it counts
    /// the canonical `@<F>View` tables instead of the monotonic hash-cons term
    /// tables. `get-size!` (egglog-experimental) sums the sizes of the tables
    /// named in its filter, or — when called with no args — all non-`@` tables.
    /// Under term encoding both of those count the term tables, which over-count
    /// vs the normal backend (stale/duplicate hash-cons rows; the canonical rows
    /// live in `@<F>View`). We redirect:
    ///   - no args (count all): emit the full set of view names PLUS the
    ///     non-eq-sort pass-through globals (which have no view and are kept as
    ///     plain non-`@` tables, exactly as the normal backend counts them), so
    ///     the proxy is the canonical egraph size.
    ///   - explicit `"F"` filters: map each to its view name `@<F>View` (leaving
    ///     names without a view — e.g. pass-through globals — as-is).
    /// `get-size!` (size.rs) treats an explicit filter as authoritative, so the
    /// `@`-prefixed view names are counted despite the usual `@` exclusion.
    fn instrument_get_size(&self, args: &[String]) -> Vec<String> {
        let view_of = &self.egraph.proof_state.proof_names.view_name;
        if args.is_empty() {
            // Count-all: every term-encoded function's canonical view, plus the
            // pass-through globals (no view; non-`@` tables the normal backend
            // also counts).
            let mut names: Vec<String> = view_of
                .values()
                .chain(self.egraph.proof_state.pass_through_globals.iter())
                .map(|v| format!("\"{v}\""))
                .collect();
            names.sort(); // deterministic order
            names
        } else {
            // Explicit filters: each arg is a quoted user-table name `"F"`.
            args.iter()
                .map(|quoted| {
                    let name = quoted.trim_matches('"');
                    match view_of.get(name) {
                        Some(view) => format!("\"{view}\""),
                        None => quoted.clone(),
                    }
                })
                .collect()
        }
    }

    // Actions need to be instrumented to add to the view
    // as well as to the terms tables.
    fn instrument_action(
        &mut self,
        action: &ResolvedAction,
        justification: &Justification,
    ) -> Vec<String> {
        let mut res = vec![];

        match action {
            ResolvedAction::Let(_span, v, generic_expr) => {
                let v2 = self.instrument_action_expr(generic_expr, &mut res, justification);
                res.push(format!("(let {} {})", v.name, v2));
            }
            ResolvedAction::Set(_span, h, generic_exprs, generic_expr) => {
                let mut exprs = vec![];
                for e in generic_exprs.iter().chain(std::iter::once(generic_expr)) {
                    exprs.push(self.instrument_action_expr(e, &mut res, justification));
                }

                let ResolvedCall::Func(func_type) = h else {
                    panic!(
                        "Set action on non-function, should have been prevented by typechecking"
                    );
                };

                if self.is_pass_through_global(&func_type.name) {
                    // Non-eq-sort global: store the value directly (no
                    // term/view tables exist for it). See `term_encode_command`.
                    res.push(format!(
                        "(set ({} {}) {})",
                        func_type.name,
                        ListDisplay(&exprs[..exprs.len() - 1], " "),
                        exprs[exprs.len() - 1],
                    ));
                } else {
                    let (add_code, _fv) = self.add_term_and_view(func_type, &exprs, justification);
                    res.extend(add_code);
                }
            }
            ResolvedAction::Change(_span, change, h, generic_exprs) => {
                if let ResolvedCall::Func(func_type) = h {
                    let symbol = match change {
                        Change::Delete => self.delete_name(&func_type.name),
                        Change::Subsume => self.subsumed_name(&func_type.name),
                    };
                    let children = generic_exprs
                        .iter()
                        .map(|e| self.instrument_action_expr(e, &mut res, justification))
                        .collect::<Vec<_>>();

                    res.push(format!("({symbol} {})", ListDisplay(children, " ")));
                } else {
                    panic!(
                        "Delete action on non-function, should have been prevented by typechecking"
                    );
                }
            }
            ResolvedAction::Union(_span, generic_expr, generic_expr1) => {
                let v1 = self.instrument_action_expr(generic_expr, &mut res, justification);
                let v2 = self.instrument_action_expr(generic_expr1, &mut res, justification);
                let ot = generic_expr.output_type();
                let type_name = ot.name();
                let unioned = self.union(type_name, &v1, &v2, justification);
                res.push(unioned);
            }
            ResolvedAction::Panic(..) => {
                res.push(format!("{action}"));
            }
            ResolvedAction::Expr(_span, generic_expr) => {
                self.instrument_action_expr(generic_expr, &mut res, justification);
            }
        }

        res
    }

    /// Update the view with the given arguments.
    /// The arguments include the eclass for constructors.
    /// View is always a function (returning Proof or Unit).
    fn update_view(&mut self, fname: &str, args: &[String], proof: &str) -> String {
        let view_name = self.view_name(fname);
        format!("(set ({view_name} {}) {proof})", ListDisplay(args, " "))
    }

    /// Return some code adding to the view and term tables.
    /// For constructors, `args` should not include the eclass of the resulting term (since it may not exist yet).
    /// For custom functions, `args` should include all arguments (including the output for the function).
    ///
    /// Returns a vector of strings representing code to add and a variable for the created term.
    /// We could return the term itself, but this might make the encoding blow up the code.
    fn add_term_and_view(
        &mut self,
        func_type: &FuncType,
        args: &[String],
        justification: &Justification,
    ) -> (Vec<String>, String) {
        // A fresh variable for the new term.
        let fv = self.fresh_var();
        let mut res = vec![];

        // Canonicalize-at-creation: replace every eq-sort child argument with
        // its UF_old leader, via an identity-on-miss lookup against the flat UF
        // index `@UF_Sf` (frozen at the last completed rebuild). The resulting
        // row is then canonical w.r.t. UF_old, which is what lets the FlowLog DD
        // backend skip the `δ(constructor) ⋈ UF_old` rebuild join. Always-on for
        // all term-encoding backends, but skipped in PROOF mode.
        // TODO(wave2): support canon-at-creation in proof mode — the @UF_Sortf
        // lookup returns @UFPair_Sort; project pair-first + thread pair-second
        // into the proof tree (mirror rebuilding_rules).
        let args: Vec<String> = if self.egraph.proof_state.canon_at_creation
            && !self.egraph.proof_state.proofs_enabled
        {
            args.iter()
                .enumerate()
                .map(|(i, a)| {
                    // Sort for this argument position. For constructors `args`
                    // are exactly the inputs; for custom functions the trailing
                    // arg is the output.
                    let sort = if i < func_type.input.len() {
                        Some(&func_type.input[i])
                    } else {
                        Some(&func_type.output)
                    };
                    match sort {
                        Some(s) if s.is_eq_sort() => {
                            if self.native_uf() {
                                // Native-UF mode: find-or-self via the
                                // `@canon_S` primitive. A primitive call (not
                                // a table lookup), so no :unsafe-seminaive is
                                // needed — leave `emitted_canon_lookup` clear.
                                let canon_prim = self.canon_prim_name(s.name());
                                format!("({canon_prim} {a})")
                            } else {
                                let uf_function_name = self.uf_function_name(s.name());
                                // RHS function-table lookup => rule needs
                                // :unsafe-seminaive (set in instrument_rule).
                                self.egraph.proof_state.emitted_canon_lookup = true;
                                format!("({uf_function_name} {a})")
                            }
                        }
                        _ => a.clone(),
                    }
                })
                .collect()
        } else {
            args.to_vec()
        };
        let args = args.as_slice();

        // TODO might be able to get rid of this intermediate variable in encoding
        res.push(format!(
            "(let {fv} ({} {}))",
            func_type.name,
            ListDisplay(args, " ")
        ));

        let args_with_fv = if func_type.subtype == FunctionSubtype::Constructor {
            let mut a = args.to_vec();
            // Canonicalize-at-creation: the constructor hash-cons `(name args)`
            // returns a possibly-STALE eclass (its output is not kept canonical
            // when the eclass is later unioned), so the view's representative
            // column would otherwise reference a non-canonical id. Wrap `fv` in
            // the same `@UF_Sf` identity-on-miss lookup used for the input args,
            // so the view row is born canonical in its eclass column too.
            // Always-on for term-encoding backends, but skipped in PROOF mode.
            // TODO(wave2): support canon-at-creation in proof mode — the @UF_Sortf
            // lookup returns @UFPair_Sort; project pair-first + thread pair-second
            // into the proof tree (mirror rebuilding_rules).
            if self.egraph.proof_state.canon_at_creation
                && !self.egraph.proof_state.proofs_enabled
                && func_type.output.is_eq_sort()
            {
                if self.native_uf() {
                    // Native-UF mode: find-or-self via the `@canon_S`
                    // primitive (no :unsafe-seminaive needed).
                    let canon_prim = self.canon_prim_name(func_type.output.name());
                    a.push(format!("({canon_prim} {fv})"));
                } else {
                    let uf_function_name = self.uf_function_name(func_type.output.name());
                    self.egraph.proof_state.emitted_canon_lookup = true;
                    a.push(format!("({uf_function_name} {fv})"));
                }
            } else {
                a.push(fv.clone());
            }
            a
        } else {
            args.to_vec()
        };

        let (proof_str, view_proof_var) = if self.egraph.proof_state.proofs_enabled {
            let to_ast = self.fname_to_ast_name(&func_type.name);
            let rule_constructor = &self.proof_names().rule_constructor;
            let fiat_constructor = &self.proof_names().fiat_constructor;

            let proof = match justification {
                Justification::Rule(rule_name, rule_proof) => {
                    format!(
                        "({rule_constructor} \"{rule_name}\" {rule_proof} ({to_ast} {fv}) ({to_ast} {fv}))",
                    )
                }
                Justification::Fiat => {
                    format!("({fiat_constructor} ({to_ast} {fv}) ({to_ast} {fv}))",)
                }
                Justification::Merge(fn_name, p1, p2) => {
                    let merge_constructor = &self.proof_names().merge_fn_constructor;
                    format!("({merge_constructor} \"{fn_name}\" {p1} {p2} ({to_ast} {fv}))",)
                }
                Justification::Proof(existing_proof) => existing_proof.clone(),
            };

            let proof_var = self.fresh_var();
            // add a proof for the constructor if needed
            let term_proof = if func_type.subtype == FunctionSubtype::Constructor {
                let term_proof_constructor = self.term_proof_name(func_type.output.name());
                format!("(set ({term_proof_constructor} {fv}) {proof_var})")
            } else {
                "".to_string()
            };

            (
                format!(
                    "(let {proof_var} {proof})
                     {term_proof}"
                ),
                proof_var,
            )
        } else {
            ("".to_string(), "()".to_string())
        };

        res.push(proof_str);
        res.push(self.update_view(&func_type.name, &args_with_fv, &view_proof_var));

        // add to uf table to initialize eclass for constructors
        if func_type.subtype == FunctionSubtype::Constructor {
            res.push(self.union(
                func_type.output.name(),
                &fv,
                &fv,
                &Justification::Proof(view_proof_var),
            ));
        }

        (res, fv)
    }

    /// Returns a query for (fname args) and a variable for the proof (or Unit) output.
    /// View is always a function, so we always use `(= var (view ...))` form.
    fn query_view_and_get_proof(&mut self, fname: &str, args: &[String]) -> (String, String) {
        let view_name = self.view_name(fname);
        let pf_var = self.fresh_var();
        let query = format!("(= {pf_var} ({view_name} {}))", ListDisplay(args, " "));
        (query, pf_var)
    }

    // Add to view and term tables, returning a variable for the created term.
    fn instrument_action_expr(
        &mut self,
        expr: &ResolvedExpr,
        res: &mut Vec<String>,
        proof: &Justification,
    ) -> String {
        match expr {
            ResolvedExpr::Lit(_, lit) => format!("{lit}"),
            ResolvedExpr::Var(_, resolved_var) => resolved_var.name.clone(),
            ResolvedExpr::Call(_, resolved_call, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.instrument_action_expr(arg, res, proof))
                    .collect::<Vec<_>>();
                match resolved_call {
                    ResolvedCall::Func(func_type) => {
                        if func_type.subtype == FunctionSubtype::Custom {
                            // A non-eq-sort global is a plain 0-arg key-value
                            // store with no term/view (eq-sort globals are
                            // lowered to constructors, handled below). Read it
                            // back with the direct 0-arg lookup `($N)`. The rule
                            // was marked `:unsafe-seminaive` by `remove_globals`
                            // (a RHS function lookup), which permits this read.
                            // The value flows into the parent term as an ordinary
                            // primitive child, so the parent's proof justification
                            // covers it (same as a literal).
                            if self.is_pass_through_global(&func_type.name) {
                                return format!("({} {})", func_type.name, ListDisplay(&args, " "));
                            }
                            panic!(
                                "Found a function lookup in actions, should have been prevented by typechecking"
                            );
                        }
                        let (add_code, fv) = self.add_term_and_view(func_type, &args, proof);
                        res.extend(add_code);

                        fv
                    }
                    ResolvedCall::Primitive(specialized_primitive) => {
                        let fv = self.fresh_var();
                        res.push(format!(
                            "(let {} ({} {}))",
                            fv,
                            specialized_primitive.name(),
                            ListDisplay(args, " ")
                        ));
                        fv
                    }
                }
            }
        }
    }

    /// In proof mode, rule_proof justifies the actions taken.
    fn instrument_actions(
        &mut self,
        actions: &[ResolvedAction],
        justification: &Justification,
    ) -> Vec<String> {
        let mut res = vec![];
        for action in actions {
            res.extend(self.instrument_action(action, justification));
        }
        res
    }

    /// Instrument a rule to use term encoding. This involves using the view tables in facts,
    /// adding to term and view tables in actions.
    /// When proofs are enabled we query proof tables, then build a proof for the rule in the actions.
    /// Finally, each view update also updates the proof tables.
    fn instrument_rule(&mut self, rule: &ResolvedRule) -> Vec<Command> {
        // Fetch eq-sort variables' term_proofs as action-side lookups
        // rather than body joins (see instrument_facts). Those are
        // function lookups in a RHS, so the generated rule opts into
        // `:unsafe-seminaive` (keeps delta evaluation, permits the reads).
        let (facts, action_lookups, proof_str) = self.instrument_facts(&rule.body);
        let proof_var = self.fresh_var();
        let proof = Justification::Rule(rule.name.clone(), proof_var.clone());
        // Reset the per-rule canon-lookup flag; `add_term_and_view` sets it when
        // it emits a `find_UFold` RHS lookup (canonicalize-at-creation).
        self.egraph.proof_state.emitted_canon_lookup = false;
        // The looked-up proofs feed `proof_str`, so bind them first.
        let action_lookups_str = ListDisplay(&action_lookups, "\n                    ");
        let proof_var_binding = if self.egraph.proof_state.proofs_enabled {
            format!(
                "{action_lookups_str}
                 (let {proof_var}
                          {proof_str})"
            )
        } else {
            "".to_string()
        };

        let actions = self.instrument_actions(&rule.head.0, &proof);
        // RHS `find_UFold` lookups (canonicalize-at-creation) also force
        // :unsafe-seminaive, since they are function-table lookups in actions.
        let needs_unsafe_seminaive =
            !action_lookups.is_empty() || self.egraph.proof_state.emitted_canon_lookup;
        let name = &rule.name;
        let ruleset_opt = if rule.ruleset.is_empty() {
            "".to_string()
        } else {
            format!(":ruleset {}", rule.ruleset)
        };
        let unsafe_opt = if needs_unsafe_seminaive {
            ":unsafe-seminaive"
        } else {
            ""
        };
        // Preserve the `:naive` flag. A `:naive` rule typechecks its body in the
        // Read/Full primitive contexts rather than Pure/Write (see
        // `typecheck_rule`), which is what makes a read-primitive such as
        // `get-size!` admissible — e.g. the one-shot `(run … :until (… (get-size!)))`
        // query rule (`prelude::query` / `rust_rule_full`) is built `:naive`.
        // Dropping it here re-typechecks the instrumented rule as seminaive,
        // resolving `get-size!` in `Context::Pure` where it is invalid → "Unbound
        // function get-size!".
        let naive_opt = if rule.naive { ":naive" } else { "" };
        let instrumented = format!(
            "(rule ({})
                   ({proof_var_binding}
                    {})
                    {ruleset_opt} {unsafe_opt} {naive_opt}
                    :name \"{name}\")",
            ListDisplay(facts, " "),
            ListDisplay(actions, " "),
        );
        self.parse_program(&instrumented)
    }

    /// Any schedule should be sound as long as we saturate.
    fn rebuild(&mut self) -> Schedule {
        let path_compress_ruleset = self.proof_names().path_compress_ruleset_name.clone();
        let single_parent = self.proof_names().single_parent_ruleset_name.clone();
        let uf_function_index = self.proof_names().uf_function_index_ruleset_name.clone();
        let rebuilding_cleanup_ruleset = self.proof_names().rebuilding_cleanup_ruleset_name.clone();
        let rebuilding_ruleset = self.proof_names().rebuilding_ruleset_name.clone();
        let delete_ruleset = self.proof_names().delete_subsume_ruleset_name.clone();
        // Native-UF mode: drain the append-only `@UFChange_S` onchange relation
        // after the rebuild saturates (see `rebuilding_rules_native_uf`). The
        // drain must run in its own stratum (a `(saturate ...)` step distinct
        // from `@rebuilding`): congruence in `@rebuilding` issues unions whose
        // leader-change callback writes the onchange relation, and deleting from
        // it in the same merge corrupts its hash-cons index. Draining after the
        // rebuild fixpoint is safe — every onchange row has been consumed (all
        // view rows are canonical) — and stops the relation from accumulating
        // across the rebuild passes of successive `(run N)` iterations. In
        // relational mode the drain ruleset is empty, so the schedule is
        // unchanged.
        let drain_step = if self.native_uf() {
            let drain_ruleset = self.proof_names().uf_change_drain_ruleset_name.clone();
            format!("(saturate {drain_ruleset})")
        } else {
            String::new()
        };
        self.parse_schedule(format!(
            "(seq
              (saturate
                  {rebuilding_cleanup_ruleset}
                  (saturate {single_parent})
                  (saturate {path_compress_ruleset})
                  (saturate {uf_function_index})
                  {rebuilding_ruleset})
              {drain_step}
              {delete_ruleset})"
        ))
    }

    fn instrument_schedule(&mut self, schedule: &ResolvedSchedule) -> Schedule {
        match schedule {
            ResolvedSchedule::Run(span, config) => {
                let new_run = match config.until {
                    Some(ref facts) => {
                        let (instrumented, _lookups, _proof) = self.instrument_facts(facts);
                        let instrumented_facts = self.parse_facts(&instrumented);
                        Schedule::Run(
                            span.clone(),
                            RunConfig {
                                ruleset: config.ruleset.clone(),
                                until: Some(instrumented_facts),
                            },
                        )
                    }
                    None => Schedule::Run(
                        span.clone(),
                        RunConfig {
                            ruleset: config.ruleset.clone(),
                            until: None,
                        },
                    ),
                };
                Schedule::Sequence(span.clone(), vec![new_run, self.rebuild()])
            }
            ResolvedSchedule::Sequence(span, schedules) => Schedule::Sequence(
                span.clone(),
                schedules
                    .iter()
                    .map(|s| self.instrument_schedule(s))
                    .collect(),
            ),
            ResolvedSchedule::Saturate(span, schedule) => {
                Schedule::Saturate(span.clone(), Box::new(self.instrument_schedule(schedule)))
            }
            GenericSchedule::Repeat(span, n, schedule) => Schedule::Repeat(
                span.clone(),
                *n,
                Box::new(self.instrument_schedule(schedule)),
            ),
        }
    }

    fn term_encode_command(&mut self, command: &ResolvedNCommand, res: &mut Vec<Command>) {
        log::debug!("Term encoding for {command}");
        match &command {
            ResolvedNCommand::Sort {
                span,
                name,
                presort_and_args,
                unionable,
                ..
            } => {
                let uf_name = self.uf_name(name);
                let proof_func = if self.egraph.proof_state.proofs_enabled {
                    Some(self.term_proof_name(name))
                } else {
                    None
                };
                res.push(Command::Sort {
                    span: span.clone(),
                    name: name.clone(),
                    presort_and_args: presort_and_args.clone(),
                    uf: Some(uf_name),
                    proof_func,
                    unionable: *unionable,
                });
                res.extend(self.declare_sort(name));
            }
            ResolvedNCommand::Function(fdecl) => {
                if self.is_non_eq_sort_global_decl(fdecl) {
                    // Pass a non-eq-sort global through unchanged: it is a
                    // plain 0-arg key-value store with no eclass, so it needs
                    // no term/view tables or rebuild rules. Reads use the
                    // direct 0-arg lookup `($N)` (see `instrument_action_expr`).
                    // Record the name so the `Set`/read sites (which only have a
                    // `FuncType`) recognize it via `is_pass_through_global`.
                    self.egraph
                        .proof_state
                        .pass_through_globals
                        .insert(fdecl.name.clone());
                    res.push(command.to_command().make_unresolved());
                } else {
                    res.extend(self.term_and_view(fdecl));
                    res.extend(self.rebuilding_rules(fdecl));
                }
            }
            ResolvedNCommand::NormRule { rule } => {
                res.extend(self.instrument_rule(rule));
            }
            ResolvedNCommand::CoreAction(action) => {
                let instrumented = self
                    .instrument_action(action, &Justification::Fiat)
                    .join("\n");
                res.extend(self.parse_program(&instrumented));
            }
            ResolvedNCommand::Check(span, facts) => {
                let (instrumented, _lookups, _proof) = self.instrument_facts(facts);
                res.push(Command::Check(
                    span.clone(),
                    self.parse_facts(&instrumented),
                ));
            }
            ResolvedNCommand::RunSchedule(schedule) => {
                res.push(Command::RunSchedule(self.instrument_schedule(schedule)));
            }
            ResolvedNCommand::Fail(span, cmd) => {
                self.term_encode_command(cmd, res);
                let last = res.pop().unwrap();
                res.push(Command::Fail(span.clone(), Box::new(last)));
            }
            ResolvedNCommand::Extract(span, expr, variants) => {
                // Instrument the expressions to use view tables (like actions, not facts)
                let mut action_stmts = vec![];
                let instrumented_expr =
                    self.instrument_action_expr(expr, &mut action_stmts, &Justification::Fiat);
                let instrumented_variants =
                    self.instrument_action_expr(variants, &mut action_stmts, &Justification::Fiat);

                // Add any action statements needed to set up the expressions
                for stmt in action_stmts {
                    res.extend(self.parse_program(&stmt));
                }
                // Rebuild before extract; we may have added new view rows that need canonicalization
                res.push(Command::RunSchedule(self.rebuild()));
                res.push(Command::Extract(
                    span.clone(),
                    self.parse_expr(&instrumented_expr),
                    self.parse_expr(&instrumented_variants),
                ));
            }
            ResolvedNCommand::PrintSize(span, name) => {
                // In proof mode, print the size of the view table for constructors
                let new_name = name.as_ref().map(|n| {
                    if self
                        .egraph
                        .type_info
                        .get_func_type(n)
                        .is_some_and(|f| f.subtype == FunctionSubtype::Constructor)
                    {
                        self.view_name(n)
                    } else {
                        n.clone()
                    }
                });
                res.push(Command::PrintSize(span.clone(), new_name));
            }
            ResolvedNCommand::Pop(..)
            | ResolvedNCommand::Push(..)
            | ResolvedNCommand::AddRuleset(..)
            | ResolvedNCommand::Output { .. }
            | ResolvedNCommand::Input { .. }
            | ResolvedNCommand::UnstableCombinedRuleset(..)
            | ResolvedNCommand::PrintOverallStatistics(..)
            | ResolvedNCommand::PrintFunction(..)
            | ResolvedNCommand::ProveExists(..) => {
                res.push(command.to_command().make_unresolved());
            }
            ResolvedNCommand::UserDefined(..) => {
                // Pass through as-is: term encoding has nothing to
                // instrument here — the user-defined command is
                // dispatched later by whatever registered it
                // (e.g. egglog-experimental's `run-schedule` /
                // `multi-extract`). The `command_supports_proof_encoding`
                // gate above rejects this case when proofs are
                // actually being generated.
                res.push(command.to_command().make_unresolved());
            }
        }
    }

    pub(crate) fn add_term_encoding_helper(
        &mut self,
        program: Vec<ResolvedNCommand>,
    ) -> Vec<Command> {
        let mut res = vec![];

        if !self.egraph.proof_state.term_header_added {
            res.extend(self.term_header());
            if self.egraph.proof_state.proofs_enabled {
                let proof_header = self.proof_header();
                res.extend(self.parse_program(&proof_header));
            }
            self.egraph.proof_state.term_header_added = true;
        }

        for command in program {
            self.term_encode_command(&command, &mut res);

            // run rebuilding after every command except a few
            if let ResolvedNCommand::Function(..)
            | ResolvedNCommand::NormRule { .. }
            | ResolvedNCommand::Sort { .. } = &command
            {
            } else {
                res.push(Command::RunSchedule(self.rebuild()));
            }
        }

        res
    }
}
