//! External-function (primitive) registry for the Feldera/DBSP backend.
//!
//! Milestone 1 is "Datalog only, primitive-light" (PLAN §8): no rule references
//! primitives yet. This module therefore provides only a *storage* registry so
//! that `register_external_func` / `new_panic` return stable ids and
//! `free_external_func` works. Actually invoking a primitive inside a rule is
//! deferred to the merges+primitives milestone (PLAN Phase 2), where primitives
//! become plain Rust closures inside `map`/`filter` (no UDF/ABI dance, unlike
//! DuckDB — PLAN §3.4).

use egglog_backend_trait::{ExternalFunction, ExternalFunctionId};
use egglog_numeric_id::NumericId;

/// Either a real registered primitive or a deferred-panic sentinel.
enum Slot {
    Func(#[allow(dead_code)] Box<dyn ExternalFunction + 'static>),
    Panic(#[allow(dead_code)] String),
    Free,
}

/// A grow-only registry of external functions, indexed by
/// [`ExternalFunctionId`]. Freed slots are tombstoned (not reused) so ids stay
/// stable for the lifetime of the egraph. Names are tracked (via the trait's
/// `rename_prim`) so the rule builder can recognize built-in predicates like
/// `!=` and lower them to circuit filters — mirroring how the DuckDB backend
/// recognizes primitives by name (`external_func_name`).
#[derive(Default)]
pub struct ExternalFuncRegistry {
    slots: Vec<Slot>,
    names: Vec<Option<String>>,
}

impl ExternalFuncRegistry {
    /// Register a primitive, returning its fresh id.
    pub fn add_func(&mut self, func: Box<dyn ExternalFunction + 'static>) -> ExternalFunctionId {
        let id = ExternalFunctionId::new(self.slots.len() as u32);
        self.slots.push(Slot::Func(func));
        self.names.push(None);
        id
    }

    /// Register a deferred-panic sentinel, returning its fresh id.
    pub fn add_panic(&mut self, message: String) -> ExternalFunctionId {
        let id = ExternalFunctionId::new(self.slots.len() as u32);
        self.slots.push(Slot::Panic(message));
        self.names.push(None);
        id
    }

    /// Record a primitive's display name (the term encoder / frontend calls
    /// this via `Backend`/`RuleBuilderOps::rename_prim`).
    pub fn set_name(&mut self, id: ExternalFunctionId, name: String) {
        if let Some(slot) = self.names.get_mut(id.rep() as usize) {
            *slot = Some(name);
        }
    }

    /// The display name of a primitive, if recorded.
    pub fn name(&self, id: ExternalFunctionId) -> Option<&str> {
        self.names.get(id.rep() as usize).and_then(|n| n.as_deref())
    }

    /// Tombstone a slot. Idempotent.
    pub fn free(&mut self, id: ExternalFunctionId) {
        if let Some(slot) = self.slots.get_mut(id.rep() as usize) {
            *slot = Slot::Free;
        }
    }

    /// The panic message for a deferred-panic id, if any. Used by tests and by
    /// a later milestone's rule lowering of `panic` actions.
    #[allow(dead_code)]
    pub fn panic_message(&self, id: ExternalFunctionId) -> Option<&str> {
        match self.slots.get(id.rep() as usize) {
            Some(Slot::Panic(m)) => Some(m.as_str()),
            _ => None,
        }
    }
}
