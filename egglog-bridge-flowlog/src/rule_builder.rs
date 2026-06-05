//! `impl RuleBuilderOps` for the FlowLog backend.
//!
//! Like the Feldera/DuckDB backends' rule builders, this is an **accumulator**:
//! each `RuleBuilderOps` call appends to an in-progress [`RuleIr`] (defined in
//! `compile.rs`) in emission order; `build()` registers it on the egraph.
//!
//! M1 records table body atoms and `set`/`insert` heads (the transitive-closure
//! step shape). The remaining ops (`query_prim`, `lookup`, `union`, `remove`,
//! `subsume`, `panic`, `call_external_func`) are accumulated into the IR for
//! later milestones but are not exercised by the M1 flowlog-driven path.

use anyhow::Result;
use egglog_backend_trait::{
    ColumnTy, ExternalFunctionId, FunctionId, PanicMsg, QueryEntry, RuleBuilderOps, RuleId,
    Variable, VariableId,
};
use egglog_numeric_id::NumericId;

use crate::compile::{BodyAtom, BodyOp, HeadOp, RuleIr, Slot};
use crate::EGraph;

/// Accumulates a rule's body ops and head ops, then registers them.
pub struct FlowlogRuleBuilder<'a> {
    egraph: &'a mut EGraph,
    ir: RuleIr,
    /// Fresh variable counter, seeded above any caller-provided variable id.
    next_var: u32,
    /// First error hit during accumulation; surfaced at `build()`.
    deferred_err: Option<anyhow::Error>,
}

impl<'a> FlowlogRuleBuilder<'a> {
    pub fn new(egraph: &'a mut EGraph, desc: &str) -> Self {
        FlowlogRuleBuilder {
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
}

impl<'a> RuleBuilderOps for FlowlogRuleBuilder<'a> {
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
        if entries.is_empty() {
            return Err(anyhow::anyhow!(
                "FlowLog backend: query_prim with no entries"
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
        Ok(id)
    }
}
