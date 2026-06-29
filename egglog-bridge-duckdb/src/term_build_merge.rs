//! Native TERM-BUILD custom `:merge` on DuckDB, as INLINE set-based SQL.
//!
//! The frontend lowers a term-building custom `:merge` body (e.g.
//! `(C2 (C1 old new) (C2 old new))`) to a [`MergeFn`] tree — a top-level `IfEq`
//! guard around a `Seq` of `TableInsert` / `Construct` / canon `Function` nodes
//! (`translate_term_build_merge_to_seq` / `lower_term_build_expr`). `add_table`
//! retains that tree on the view's [`crate::FunctionInfo::merge_tree`] and
//! registers the view ALL-COLUMNS keyed so two conflicting
//! `(set (@FView key) eclass)` writes coexist.
//!
//! At the iteration boundary, [`crate::EGraph::emit_term_build_merges`] resolves
//! each touched term-build view:
//!   1. CONFLICT BATCH: self-join the view (same key cols, two different eclass
//!      values) → `(key.., old_eclass, new_eclass)` rows.
//!   2. COMPILE the retained tree to a sequence of bulk SQL statements: each
//!      `Construct`/`TableInsert` mints a constructor SET-BASED with hash-consing
//!      (`COALESCE((SELECT MIN(id) … GROUP BY children), nextval(seq))`, mirroring
//!      `compile.rs`'s LetCtor hash-cons); the final node is the merged eclass.
//!   3. WRITE-BACK: delete the conflicting view rows, insert `(key.., merged)`.
//!
//! ## How this codegen generalizes (proofs later)
//!
//! The compiler walks the tree generically: `Construct`→a minting INSERT +
//! hash-cons read-back, `TableInsert`→an INSERT respecting the target view's
//! merge, `Function`(UF)→the canon UDF, `Primitive`→[`crate::compile::prim_sql`],
//! `IfEq`→a WHERE filter on the batch, `Seq`→the statement sequence returning the
//! final eclass. A proof-mode term-build merge is a `Columns([eclass, proof])`
//! whose `proof` branch is just MORE `Construct`s of proof terms — they ride the
//! exact same `Construct`→mint-INSERT mechanism, so adding proofs is adding
//! columns/statements, not a new path. (Proofs are NOT implemented here; DuckDB
//! rejects proof-mode native-merge.)

use anyhow::{Result, anyhow};
use egglog_backend_trait::MergeFn;
use egglog_numeric_id::NumericId;

use crate::{EGraph, FunctionInfo, q};

/// A compiled term-build merge: the ordered SQL statements to run per conflict
/// batch, plus the SQL expression (over batch alias `b`) for the merged eclass.
pub(crate) struct CompiledTermBuild {
    /// Bulk statements run in order against the `batch` temp table. Each is a
    /// full SQL statement (mint INSERTs, view INSERTs).
    pub stmts: Vec<String>,
    /// SQL expression (over the `batch` alias `b`) evaluating to the merged
    /// eclass for a batch row — used in the write-back INSERT.
    pub merged_expr: String,
    /// Every table the statements INSERT into (term tables + their `@<C>View`s).
    /// The runner bumps each one's watermark after running, so the seminaive
    /// skip-gate of downstream rules (which read these views) re-fires on the
    /// freshly-minted rows. De-duplicated, insertion order preserved.
    pub written_tables: Vec<String>,
}

/// Context for compiling a term-build tree to SQL.
struct Ctx<'a> {
    eg: &'a EGraph,
    /// Statements accumulated so far (mint/view INSERTs).
    stmts: Vec<String>,
    /// Tables INSERTed into (for watermark bumping). De-duplicated.
    written_tables: Vec<String>,
}

impl Ctx<'_> {
    fn record_write(&mut self, table: &str) {
        if !self.written_tables.iter().any(|t| t == table) {
            self.written_tables.push(table.to_string());
        }
    }
}

impl<'a> Ctx<'a> {
    /// Resolve a `FunctionId` to its DuckDB table name.
    fn fn_name(&self, id: egglog_backend_trait::FunctionId) -> Result<&'a str> {
        self.eg
            .backend_function_names
            .get(id.rep() as usize)
            .map(|s| s.as_str())
            .ok_or_else(|| anyhow!("term-build merge: unknown FunctionId {}", id.rep()))
    }

    fn info(&self, name: &str) -> Result<&'a FunctionInfo> {
        self.eg
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("term-build merge: table `{name}` not registered"))
    }

    /// Compile a node that EVALUATES to a value (an eclass / leaf), as a SQL
    /// scalar expression over the batch alias `b`. May push minting statements
    /// (for nested `Construct`s) as a side effect.
    ///
    /// `old`/`new` are the SQL expressions for the conflict batch's old/new
    /// eclass (already canonicalized at batch-build time); `keys` are the batch's
    /// key-column expressions (for `KeyCol(i)`).
    fn value_sql(
        &mut self,
        node: &MergeFn,
        old: &str,
        new: &str,
        keys: &[String],
    ) -> Result<String> {
        match node {
            MergeFn::Old | MergeFn::OldCol(0) => Ok(old.to_string()),
            MergeFn::New | MergeFn::NewCol(0) => Ok(new.to_string()),
            MergeFn::KeyCol(i) => keys
                .get(*i)
                .cloned()
                .ok_or_else(|| anyhow!("term-build merge: KeyCol({i}) out of range")),
            MergeFn::Const(v) => Ok((v.rep() as i64).to_string()),
            MergeFn::Primitive(ext_id, args) => {
                let op = self
                    .eg
                    .external_func_name(*ext_id)
                    .ok_or_else(|| anyhow!("term-build merge: unknown primitive {}", ext_id.rep()))?
                    .to_string();
                let arg_sqls: Vec<String> = args
                    .iter()
                    .map(|a| self.value_sql(a, old, new, keys))
                    .collect::<Result<_>>()?;
                crate::compile::prim_sql(&op, &arg_sqls, "@term_build_merge")
            }
            MergeFn::Function(fid, args) => {
                let name = self.fn_name(*fid)?;
                let info = self.info(name)?;
                if let Some(udf) = &info.native_uf_udf {
                    // Canon (find-or-self) against the per-sort UF: `udf(x)`.
                    if args.len() != 1 {
                        return Err(anyhow!(
                            "term-build merge: canon `{name}` expects 1 arg, got {}",
                            args.len()
                        ));
                    }
                    let inner = self.value_sql(&args[0], old, new, keys)?;
                    Ok(format!("{udf}({inner})"))
                } else {
                    // A `Function` against a constructor/view = a hash-cons READ:
                    // return the eclass for these children (must already be
                    // minted). Read the canonical (MIN) id from the table.
                    let arg_sqls: Vec<String> = args
                        .iter()
                        .map(|a| self.value_sql(a, old, new, keys))
                        .collect::<Result<_>>()?;
                    Ok(self.hashcons_read(name, &arg_sqls)?)
                }
            }
            MergeFn::Construct(term_fid, key_args, val_args) => {
                if !val_args.is_empty() {
                    return Err(anyhow!(
                        "term-build merge on DuckDB: Construct with value_args (proofs) \
                         is not supported (proof-mode native-merge is rejected on DuckDB)"
                    ));
                }
                let term_name = self.fn_name(*term_fid)?.to_string();
                let key_sqls: Vec<String> = key_args
                    .iter()
                    .map(|a| self.value_sql(a, old, new, keys))
                    .collect::<Result<_>>()?;
                // Mint into the TERM table, hash-consed. The view is written by
                // the matching `TableInsert` in the enclosing `Seq`; here we only
                // ensure the term row exists with the chosen id and return that id.
                self.mint_term(&term_name, &key_sqls)?;
                // The minted id, read back (now guaranteed present).
                self.hashcons_read(&term_name, &key_sqls)
            }
            other => Err(anyhow!(
                "term-build merge on DuckDB: unsupported value node {:?}",
                std::mem::discriminant(other)
            )),
        }
    }

    /// SQL scalar that reads the hash-consed (MIN) id for `children` from `table`.
    /// `(SELECT MIN(c{out}) FROM <table> WHERE c0=<k0> AND …)`. Used to recover a
    /// minted constructor's eclass per batch row.
    fn hashcons_read(&self, table: &str, children: &[String]) -> Result<String> {
        let info = self.info(table)?;
        let out_col = info.inputs_len; // term/view: id is at index inputs_len.
        if children.is_empty() {
            return Ok(format!("(SELECT MIN(c{out_col}) FROM {})", q(table)));
        }
        let conds: Vec<String> = children
            .iter()
            .enumerate()
            .map(|(i, s)| format!("c{i} = {s}"))
            .collect();
        Ok(format!(
            "(SELECT MIN(c{out_col}) FROM {} WHERE {})",
            q(table),
            conds.join(" AND ")
        ))
    }

    /// Emit a SET-BASED hash-cons mint of constructor `table` over the batch:
    /// `INSERT INTO <table> SELECT <children>, COALESCE((existing MIN id), nextval), ts`.
    /// `children` are SQL exprs over batch alias `b`. Re-mints are idempotent
    /// (the existing id is reused; `ON CONFLICT DO NOTHING` drops a duplicate).
    fn mint_term(&mut self, table: &str, children: &[String]) -> Result<()> {
        let info = self.info(table)?;
        let out_col = info.inputs_len;
        let child_cols: Vec<String> = (0..children.len()).map(|i| format!("c{i}")).collect();
        let id_expr = self.hashcons_read(table, children)?;
        // Project children + the hash-consed/fresh id, over the conflict batch.
        let sel_children: Vec<String> = children.to_vec();
        let stmt = format!(
            "INSERT INTO {tbl} ({cols}, c{out_col}, ts) \
             SELECT {sel}, COALESCE({id_expr}, nextval('__egglog_eqsort_seq')), ?2 \
             FROM batch b \
             ON CONFLICT DO NOTHING",
            tbl = q(table),
            cols = child_cols.join(", "),
            sel = sel_children.join(", "),
        );
        self.stmts.push(stmt);
        self.record_write(table);
        Ok(())
    }

    /// Emit a `TableInsert(view, [children.., eclass])`: write the view row,
    /// respecting the view's `ON CONFLICT` (a term-build view is all-cols keyed
    /// with `DO NOTHING`, so identical re-writes are idempotent and distinct
    /// (children->eclass) congruence rows coexist for `emit_inline_congruence`).
    fn table_insert(
        &mut self,
        node: &MergeFn,
        old: &str,
        new: &str,
        keys: &[String],
    ) -> Result<()> {
        let MergeFn::TableInsert(view_fid, args) = node else {
            return Err(anyhow!("term-build merge: table_insert on non-TableInsert"));
        };
        let view_name = self.fn_name(*view_fid)?.to_string();
        let info = self.info(&view_name)?;
        let arity = info.cols.len();
        if args.len() != arity {
            return Err(anyhow!(
                "term-build merge: TableInsert into `{view_name}` arity {} != {}",
                args.len(),
                arity
            ));
        }
        let arg_sqls: Vec<String> = args
            .iter()
            .map(|a| self.value_sql(a, old, new, keys))
            .collect::<Result<_>>()?;
        let cols: Vec<String> = (0..arity).map(|i| format!("c{i}")).collect();
        let conflict = crate::conflict_clause(info);
        let stmt = format!(
            "INSERT INTO {tbl} ({cols}, ts) SELECT {vals}, ?2 FROM batch b {conflict}",
            tbl = q(&view_name),
            cols = cols.join(", "),
            vals = arg_sqls.join(", "),
        );
        self.stmts.push(stmt);
        self.record_write(&view_name);
        Ok(())
    }

    /// Compile a `Seq`: run each effect (`TableInsert`) for its side effect,
    /// return the SQL expression of the LAST element (the merged eclass value).
    fn seq_sql(
        &mut self,
        items: &[MergeFn],
        old: &str,
        new: &str,
        keys: &[String],
    ) -> Result<String> {
        let (last, effects) = items
            .split_last()
            .ok_or_else(|| anyhow!("term-build merge: empty Seq"))?;
        for eff in effects {
            match eff {
                MergeFn::TableInsert(..) => self.table_insert(eff, old, new, keys)?,
                // A bare `Construct` as a Seq effect: mint its term (the value is
                // discarded). Handled generically by value_sql (which mints).
                MergeFn::Construct(..) => {
                    let _ = self.value_sql(eff, old, new, keys)?;
                }
                MergeFn::Seq(inner) => {
                    let _ = self.seq_sql(inner, old, new, keys)?;
                }
                other => {
                    return Err(anyhow!(
                        "term-build merge: unexpected Seq effect {:?}",
                        std::mem::discriminant(other)
                    ));
                }
            }
        }
        // The final element evaluates to the merged eclass.
        self.value_sql(last, old, new, keys)
    }
}

/// Compile the retained term-build `tree` for a view with `n_keys` key columns to
/// a [`CompiledTermBuild`]. The `old`/`new`/`keys` SQL exprs reference the
/// `batch` temp table's columns. Also returns the IfEq guard (a WHERE predicate
/// over the batch, filtering rows where the guard's two operands are equal — for
/// those the merge keeps `old` and mints nothing), or `None` if no guard.
pub(crate) fn compile_term_build(
    eg: &EGraph,
    tree: &MergeFn,
    n_keys: usize,
) -> Result<(CompiledTermBuild, Option<String>)> {
    // Batch column conventions (see emit_term_build_merges): key cols are
    // `b.c0..b.c{n_keys-1}`, old eclass = `b.old`, new eclass = `b.new`.
    let old = "b.old".to_string();
    let new = "b.new".to_string();
    let keys: Vec<String> = (0..n_keys).map(|i| format!("b.c{i}")).collect();

    let mut ctx = Ctx {
        eg,
        stmts: Vec::new(),
        written_tables: Vec::new(),
    };

    let (guard, body): (Option<String>, &MergeFn) = match tree {
        MergeFn::IfEq { a, b, then: _, els } => {
            // Guard: when canon(old) == canon(new) the rule-encoded merge never
            // fires (keep `then` = old, mint nothing). Filter those batch rows
            // OUT of the els-body processing via a WHERE predicate. The batch
            // already only contains old != new rows, but canon may make them
            // equal — so compute the predicate `a <> b` to keep.
            let a_sql = ctx.value_sql(a, &old, &new, &keys)?;
            let b_sql = ctx.value_sql(b, &old, &new, &keys)?;
            (Some(format!("{a_sql} <> {b_sql}")), els)
        }
        other => (None, other),
    };

    let merged_expr = match body {
        MergeFn::Seq(items) => ctx.seq_sql(items, &old, &new, &keys)?,
        // A degenerate non-Seq els (just a value): no effects.
        other => ctx.value_sql(other, &old, &new, &keys)?,
    };

    Ok((
        CompiledTermBuild {
            stmts: ctx.stmts,
            merged_expr,
            written_tables: ctx.written_tables,
        },
        guard,
    ))
}
