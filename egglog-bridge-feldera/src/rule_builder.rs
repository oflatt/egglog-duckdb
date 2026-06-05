//! `impl RuleBuilderOps` for the Feldera backend (milestone 1).
//!
//! Like the DuckDB backend's rule builder, this is an **accumulator**: each
//! `RuleBuilderOps` call appends to an in-progress [`RuleIr`] (defined in
//! `compile.rs`); `build()` registers it on the egraph and invalidates the
//! cached circuit so the next `run_rules` rebuilds with the new rule wired in.
//!
//! Milestone-1 support (PLAN §8):
//! - `new_var` / `new_var_named`: allocate body/head variables.
//! - `query_table`: 1 or 2 table body atoms (single-join).
//! - `set` / `insert`: head actions that write a row.
//!
//! Errored / unsupported (deferred to later milestones, mirroring DuckDB's
//! gating): `query_prim`, `call_external_func`, `lookup`, `subsume`, `remove`,
//! `union`. Their presence in a rule surfaces an error at `build()` time.

use anyhow::{anyhow, Result};
use egglog_backend_trait::{
    ColumnTy, ExternalFunctionId, FunctionId, PanicMsg, QueryEntry, RuleBuilderOps, RuleId, Value,
    Variable, VariableId,
};
use egglog_numeric_id::NumericId;

use crate::compile::{BodyAtom, Filter, HeadAction, RuleIr, Slot};
use crate::EGraph;

/// Accumulates a rule's body atoms and head actions, then registers them.
pub struct FelderaRuleBuilder<'a> {
    egraph: &'a mut EGraph,
    ir: RuleIr,
    /// Fresh variable counter, seeded above any caller-provided variable id.
    next_var: u32,
    /// First error hit during accumulation; surfaced at `build()`.
    deferred_err: Option<anyhow::Error>,
}

impl<'a> FelderaRuleBuilder<'a> {
    pub fn new(egraph: &'a mut EGraph, desc: &str) -> Self {
        FelderaRuleBuilder {
            egraph,
            ir: RuleIr {
                name: desc.to_string(),
                body: Vec::new(),
                filters: Vec::new(),
                head: Vec::new(),
            },
            next_var: 1 << 20, // keep builder-synthesized vars away from caller ids
            deferred_err: None,
        }
    }

    fn fresh_var(&mut self, name: Option<&str>) -> QueryEntry {
        let id = VariableId::new(self.next_var);
        self.next_var += 1;
        QueryEntry::Var(Variable {
            id,
            name: name.map(|s| s.to_string().into_boxed_str()),
        })
    }

    fn defer(&mut self, e: anyhow::Error) {
        if self.deferred_err.is_none() {
            self.deferred_err = Some(e);
        }
    }
}

impl<'a> RuleBuilderOps for FelderaRuleBuilder<'a> {
    fn new_var(&mut self, _ty: ColumnTy) -> QueryEntry {
        self.fresh_var(None)
    }

    fn new_var_named(&mut self, _ty: ColumnTy, name: &str) -> QueryEntry {
        self.fresh_var(Some(name))
    }

    fn query_table(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        is_subsumed: Option<bool>,
    ) -> Result<()> {
        if is_subsumed.is_some() {
            return Err(anyhow!(
                "Feldera backend (milestone 1): subsumption filters on body atoms \
                 are not supported"
            ));
        }
        self.ir.body.push(BodyAtom::from_entries(func, entries));
        Ok(())
    }

    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        _ret_ty: ColumnTy,
    ) -> Result<()> {
        // The only body primitive the term encoder's rebuild rulesets use is the
        // inequality guard `(!= b c)`. We recognize it by name (the frontend
        // registers primitive names via `rename_prim`) and lower it to a
        // circuit [`Filter::Ne`]. The trait's `query_prim` threads the expected
        // return value as the last entry; for a `!=` predicate the meaningful
        // operands are the first two entries.
        let name = self.egraph.external_funcs.name(func).map(str::to_string);
        match name.as_deref() {
            Some("!=") | Some("bool-!=") | Some("value-!=") => {
                if entries.len() < 2 {
                    return Err(anyhow!(
                        "Feldera backend: `!=` primitive needs two operands (got {})",
                        entries.len()
                    ));
                }
                let l = Slot::from_entry(&entries[0]);
                let r = Slot::from_entry(&entries[1]);
                self.ir.filters.push(Filter::Ne(l, r));
                Ok(())
            }
            other => Err(anyhow!(
                "Feldera backend: primitive body atom `{}` is not supported (only `!=` \
                 guards are lowered, which covers the term encoder's rebuild rulesets)",
                other.unwrap_or("<unnamed>")
            )),
        }
    }

    fn call_external_func(
        &mut self,
        _func: ExternalFunctionId,
        _args: &[QueryEntry],
        ret_ty: ColumnTy,
        _panic_msg: PanicMsg,
    ) -> QueryEntry {
        // Infallible signature: defer the error to build() and return a dummy.
        self.defer(anyhow!(
            "Feldera backend (milestone 1): external-function calls in rule heads \
             are not supported"
        ));
        QueryEntry::Const {
            val: Value::new(0),
            ty: ret_ty,
        }
    }

    fn lookup(
        &mut self,
        _func: FunctionId,
        _entries: &[QueryEntry],
        _panic_msg: PanicMsg,
    ) -> QueryEntry {
        self.defer(anyhow!(
            "Feldera backend (milestone 1): RHS function lookups are not supported"
        ));
        QueryEntry::Const {
            val: Value::new(0),
            ty: ColumnTy::Id,
        }
    }

    fn subsume(&mut self, _func: FunctionId, _entries: &[QueryEntry]) -> Result<()> {
        Err(anyhow!(
            "Feldera backend (milestone 1): subsume is not supported"
        ))
    }

    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        // `set f(k..) = v` and a relation `insert f(..)` both land here as a
        // full-row write; we store the full row uniformly.
        self.ir
            .head
            .push(HeadAction::from_entries(func, entries, false));
    }

    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        // `(delete f(k..))` — rebuild's retraction half. The term encoder's
        // path_compress / single_parent rulesets delete the stale `@uf` edge
        // before inserting the compressed one. `remove` takes only the key
        // columns; for a plain relation the whole row is the key, so the
        // entries already address the row to retract.
        self.ir
            .head
            .push(HeadAction::from_entries(func, entries, true));
    }

    fn union(&mut self, _l: QueryEntry, _r: QueryEntry) {
        self.defer(anyhow!(
            "Feldera backend (milestone 1): union is not supported (no union-find yet)"
        ));
    }

    fn panic(&mut self, message: String) {
        self.defer(anyhow!(
            "Feldera backend (milestone 1): panic action: {message}"
        ));
    }

    fn rename_prim(&mut self, id: ExternalFunctionId, name: String) {
        // Record the primitive's display name so `query_prim` can recognize
        // built-in predicates (`!=`) and lower them to circuit filters.
        self.egraph.external_funcs.set_name(id, name);
    }

    fn build(self: Box<Self>) -> Result<RuleId> {
        let this = *self;
        if let Some(e) = this.deferred_err {
            return Err(e);
        }
        let id = RuleId::new(this.egraph.rules.len() as u32);
        this.egraph.rules.push(Some(this.ir));
        this.egraph.invalidate_circuit();
        Ok(id)
    }
}
