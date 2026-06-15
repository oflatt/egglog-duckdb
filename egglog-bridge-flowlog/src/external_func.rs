//! External-function (primitive) registry for the FlowLog backend.
//!
//! ## Milestone 1 scope
//!
//! M1's proof program (a non-recursive transitive-closure step) uses NO
//! primitives. This module exists so the [`Backend`](egglog_backend_trait::Backend)
//! surface is complete (`register_external_func` / `free_external_func` /
//! `new_panic`) and so later milestones can wire primitive evaluation through
//! the embedded `Database` the way the Feldera backend does. It is a thin
//! side-table indexed by the same [`ExternalFunctionId`] the `Database`
//! assigns, tracking names + panic sentinels. Mirrors the Feldera backend.

use egglog_backend_trait::{ExecutionState, ExternalFunction, ExternalFunctionId, Value};
use egglog_numeric_id::NumericId;

/// Either a real registered primitive, a panic sentinel, or a freed slot.
enum Slot {
    Func,
    Panic(#[allow(dead_code)] String),
    Free,
}

/// A side-table of external-function metadata, indexed by the
/// [`ExternalFunctionId`] the embedded `Database` assigned. `add_*_at` keeps
/// this `Vec` aligned with the Database's id allocation (ids advance in
/// lockstep).
#[derive(Default)]
pub struct ExternalFuncRegistry {
    slots: Vec<Slot>,
    names: Vec<Option<String>>,
}

impl ExternalFuncRegistry {
    fn ensure_len(&mut self, idx: usize) {
        while self.slots.len() <= idx {
            self.slots.push(Slot::Free);
            self.names.push(None);
        }
    }

    /// Record a primitive at the id the Database assigned.
    pub fn add_func_at(
        &mut self,
        id: ExternalFunctionId,
        _func: Box<dyn ExternalFunction + 'static>,
    ) {
        let idx = id.rep() as usize;
        self.ensure_len(idx);
        self.slots[idx] = Slot::Func;
    }

    /// Record a panic sentinel at the id the Database assigned.
    pub fn add_panic_at(&mut self, id: ExternalFunctionId, message: String) {
        let idx = id.rep() as usize;
        self.ensure_len(idx);
        self.slots[idx] = Slot::Panic(message);
    }

    /// Record a primitive's display name (`rename_prim`).
    pub fn set_name(&mut self, id: ExternalFunctionId, name: String) {
        let idx = id.rep() as usize;
        self.ensure_len(idx);
        self.names[idx] = Some(name);
    }

    /// The display name of a primitive, if recorded.
    #[allow(dead_code)]
    pub fn name(&self, id: ExternalFunctionId) -> Option<&str> {
        self.names.get(id.rep() as usize).and_then(|n| n.as_deref())
    }

    /// Tombstone a slot. Idempotent.
    pub fn free(&mut self, id: ExternalFunctionId) {
        if let Some(slot) = self.slots.get_mut(id.rep() as usize) {
            *slot = Slot::Free;
        }
    }

    /// The panic message for a deferred-panic id, if any.
    #[allow(dead_code)]
    pub fn panic_message(&self, id: ExternalFunctionId) -> Option<&str> {
        match self.slots.get(id.rep() as usize) {
            Some(Slot::Panic(m)) => Some(m.as_str()),
            _ => None,
        }
    }
}

/// Find-or-self canon-prim placeholder for the native-UF path
/// (`--native-uf --flowlog`).
///
/// PR #782's UF-backed function returns this id from `add_uf_function` as the
/// `@canon_S` find-or-self primitive. On the FlowLog backend the interpreter
/// recognizes the id (see `EGraph::native_uf_canon_prim`) and answers the call
/// host-side from the in-core `UfTable` — so this registered `ExternalFunction`
/// is never actually invoked through the `Database`. It exists only so the id
/// is a real, freeable [`ExternalFunctionId`]. Invoking it (which would mean the
/// interception was missed) returns the argument unchanged (find-on-miss = self).
#[derive(Clone)]
pub struct CanonStub;

impl ExternalFunction for CanonStub {
    fn invoke(&self, _state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        // Identity-on-miss: a find against an unrecorded id is the id itself.
        // (The native-UF interceptor normally answers this before we get here.)
        args.first().copied()
    }
}

/// A real, invokable panic sentinel registered into the Database by
/// `Backend::new_panic`. Invoking it panics with the recorded message.
#[derive(Clone)]
pub struct PanicFunc {
    message: String,
}

impl PanicFunc {
    pub fn new(message: String) -> Self {
        PanicFunc { message }
    }
}

impl ExternalFunction for PanicFunc {
    fn invoke(&self, _state: &mut ExecutionState, _args: &[Value]) -> Option<Value> {
        panic!("{}", self.message);
    }
}
