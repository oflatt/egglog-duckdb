//! Base-value pool for the Feldera/DBSP backend.
//!
//! Milestone 1 stores base values exactly the way the reference bridge does:
//! a real [`egglog_core_relations::BaseValues`] registry, wrapped in a
//! `#[repr(transparent)]` newtype that implements the dyn-friendly
//! [`BaseValuePool`] trait. This mirrors `egglog_bridge`'s `BaseValuesAsPool`
//! (see `egglog-bridge/src/backend_impl.rs`).
//!
//! PLAN §3.4 notes that DBSP rows can hold arbitrary Rust values, so a future
//! milestone may store base values inline in the `Row` representation instead
//! of interning them. For milestone 1 (Datalog-only, primitive-light) the
//! intern-handle model is simplest and gives bit-for-bit `Value` parity with
//! the reference backend.

use std::any::{Any, TypeId};

use egglog_backend_trait::{BaseValueId, BaseValuePool, Value};
use egglog_core_relations::{BaseValues, DynamicInternTable};

/// Owns the backend's [`BaseValues`] registry. Held inline on the EGraph.
#[derive(Default)]
pub struct FelderaBaseValuePool {
    inner: BaseValues,
}

impl FelderaBaseValuePool {
    /// Borrow the underlying [`BaseValues`] (used by `Backend::base_values`).
    pub fn inner(&self) -> &BaseValues {
        &self.inner
    }
}

/// `#[repr(transparent)]` view of [`BaseValues`] implementing [`BaseValuePool`].
///
/// Returned from `base_value_pool()` / `base_value_pool_mut()` via a layout-
/// preserving reference cast, identical to the bridge's trick.
#[repr(transparent)]
pub struct BaseValuesAsPool(BaseValues);

impl FelderaBaseValuePool {
    /// `&self` as `&dyn BaseValuePool`.
    pub fn as_pool(&self) -> &dyn BaseValuePool {
        // SAFETY: `BaseValuesAsPool` is `#[repr(transparent)]` over
        // `BaseValues`, so the reference cast is sound.
        let as_pool: &BaseValuesAsPool =
            unsafe { &*(&self.inner as *const BaseValues as *const BaseValuesAsPool) };
        as_pool
    }

    /// `&mut self` as `&mut dyn BaseValuePool`.
    pub fn as_pool_mut(&mut self) -> &mut dyn BaseValuePool {
        let as_pool: &mut BaseValuesAsPool =
            unsafe { &mut *(&mut self.inner as *mut BaseValues as *mut BaseValuesAsPool) };
        as_pool
    }
}

impl BaseValuePool for BaseValuesAsPool {
    fn register_type_dyn(
        &mut self,
        type_id: TypeId,
        factory: Box<dyn FnOnce() -> Box<dyn DynamicInternTable>>,
    ) -> BaseValueId {
        self.0.register_type_dyn(type_id, factory)
    }

    fn get_ty_by_type_id(&self, type_id: TypeId) -> BaseValueId {
        self.0.get_ty_by_id(type_id)
    }

    fn intern_dyn(&self, ty: BaseValueId, value: Box<dyn Any + Send + Sync>) -> Value {
        let any_ref: &dyn Any = &*value;
        self.0.intern_dyn_by_id(ty, any_ref)
    }

    fn unwrap_dyn(&self, ty: BaseValueId, val: Value) -> Box<dyn Any + Send + Sync> {
        self.0.unwrap_dyn_by_id(ty, val)
    }

    fn has_ty(&self, type_id: TypeId) -> bool {
        self.0.has_ty_by_id(type_id)
    }
}
