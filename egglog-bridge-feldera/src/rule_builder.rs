//! `impl RuleBuilderOps` for the Feldera backend.
//!
//! Like the DuckDB backend's rule builder, this is an **accumulator**: each
//! `RuleBuilderOps` call appends to an in-progress [`RuleIr`] (defined in
//! `compile.rs`) in emission order; `build()` registers it on the egraph.
//!
//! Milestone 3 records the **fully general ordered IR** — table atoms,
//! primitive body atoms (guards and bindings), and head actions (`set` /
//! `delete` / `subsume` / RHS `lookup` / RHS `call_external_func` / `union` /
//! `panic`). The host interpreter (`lib.rs`) executes it as a nested-loop join
//! over the relation mirror, invoking primitives through the embedded
//! `Database`. This is what lets *real* `.egg` programs run on the Feldera
//! backend through the egglog frontend.

use anyhow::Result;
use egglog_backend_trait::{
    ColumnTy, ExternalFunctionId, FunctionId, PanicMsg, QueryEntry, RuleBuilderOps, RuleId,
    Variable, VariableId,
};
use egglog_numeric_id::NumericId;

use crate::compile::{BodyAtom, BodyOp, HeadOp, RuleIr, Slot};
use crate::EGraph;

/// Accumulates a rule's body ops and head ops, then registers them.
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
                head: Vec::new(),
            },
            next_var: 1 << 24, // keep builder-synthesized vars away from caller ids
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

    #[allow(dead_code)]
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
        _is_subsumed: Option<bool>,
    ) -> Result<()> {
        // Subsumption filters on body atoms are ignored (the backend does not
        // track subsumption yet; `supports_subsumption()` is false so the
        // frontend passes `Some(false)` — "non-subsumed only" — which is the
        // only state we ever hold anyway).
        let atom = BodyAtom::from_entries(func, entries);
        self.ir.body.push(BodyOp::Atom(atom));
        Ok(())
    }

    fn query_prim(
        &mut self,
        func: ExternalFunctionId,
        entries: &[QueryEntry],
        _ret_ty: ColumnTy,
    ) -> Result<()> {
        // The last entry is the primitive's return slot (see the bridge's
        // `query_prim`): if it is an as-yet-unbound variable it BINDS the
        // result; otherwise it is an equality guard. We record the op verbatim
        // and let the interpreter evaluate it through the embedded Database.
        if entries.is_empty() {
            return Err(anyhow::anyhow!(
                "Feldera backend: query_prim with no entries"
            ));
        }
        let args: Vec<Slot> = entries[..entries.len() - 1]
            .iter()
            .map(Slot::from_entry)
            .collect();
        let ret = Slot::from_entry(entries.last().unwrap());
        self.ir.body.push(BodyOp::Prim {
            id: func,
            args,
            ret,
        });
        Ok(())
    }

    fn call_external_func(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
        _ret_ty: ColumnTy,
        _panic_msg: PanicMsg,
    ) -> QueryEntry {
        // RHS primitive call binding a fresh result variable. We allocate a
        // synthetic result var, record a `HeadOp::Call`, and return the var so
        // later head actions can reference the result.
        let ret = self.fresh_var(None);
        let QueryEntry::Var(Variable { id, .. }) = &ret else {
            unreachable!()
        };
        let rid = id.rep();
        let slots: Vec<Slot> = args.iter().map(Slot::from_entry).collect();
        self.ir.head.push(HeadOp::Call {
            id: func,
            args: slots,
            ret: rid,
        });
        ret
    }

    fn lookup(
        &mut self,
        func: FunctionId,
        entries: &[QueryEntry],
        _panic_msg: PanicMsg,
    ) -> QueryEntry {
        // RHS function lookup binding a fresh result variable (eq-sort
        // constructor: create the row with a fresh id if absent).
        let ret = self.fresh_var(None);
        let QueryEntry::Var(Variable { id, .. }) = &ret else {
            unreachable!()
        };
        let rid = id.rep();
        let args: Vec<Slot> = entries.iter().map(Slot::from_entry).collect();
        self.ir.head.push(HeadOp::Lookup {
            func,
            args,
            ret: rid,
        });
        ret
    }

    fn subsume(&mut self, func: FunctionId, entries: &[QueryEntry]) -> Result<()> {
        let slots: Vec<Slot> = entries.iter().map(Slot::from_entry).collect();
        self.ir.head.push(HeadOp::Subsume { func, slots });
        Ok(())
    }

    fn set(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let slots: Vec<Slot> = entries.iter().map(Slot::from_entry).collect();
        self.ir.head.push(HeadOp::Set { func, slots });
    }

    fn remove(&mut self, func: FunctionId, entries: &[QueryEntry]) {
        let slots: Vec<Slot> = entries.iter().map(Slot::from_entry).collect();
        self.ir.head.push(HeadOp::Remove { func, slots });
    }

    fn union(&mut self, l: QueryEntry, r: QueryEntry) {
        self.ir.head.push(HeadOp::Union {
            l: Slot::from_entry(&l),
            r: Slot::from_entry(&r),
        });
    }

    fn panic(&mut self, message: String) {
        self.ir.head.push(HeadOp::Panic(message));
    }

    fn rename_prim(&mut self, id: ExternalFunctionId, name: String) {
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
