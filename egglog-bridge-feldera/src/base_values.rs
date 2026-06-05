//! Base-value pool for the Feldera/DBSP backend.
//!
//! The Feldera backend stores base values exactly the way the reference bridge
//! does: a real [`egglog_core_relations::BaseValues`] registry. As of Milestone
//! 3 that registry lives **inside the embedded [`Database`]** (see
//! `lib.rs`), so primitives invoked through `Database::with_execution_state`
//! observe the same interned `Value`s the frontend created — giving bit-for-bit
//! `Value` parity with the reference backend.
//!
//! This module provides the layout-preserving reference cast from
//! `&BaseValues` / `&mut BaseValues` to `&dyn BaseValuePool` (the same trick the
//! bridge uses with its `BaseValuesAsPool` newtype).
//!
//! [`Database`]: egglog_core_relations::Database

use std::any::{Any, TypeId};

use egglog_backend_trait::{BaseValueId, BaseValuePool, Value};
use egglog_core_relations::{BaseValues, DynamicInternTable};

/// `#[repr(transparent)]` view of [`BaseValues`] implementing [`BaseValuePool`].
///
/// Obtained from a `&BaseValues` (which we borrow from the embedded
/// [`egglog_core_relations::Database`]) via a layout-preserving reference cast,
/// identical to the bridge's `BaseValuesAsPool` trick.
#[repr(transparent)]
pub struct BaseValuesAsPool(BaseValues);

/// View `&BaseValues` as `&dyn BaseValuePool`.
pub fn base_values_as_pool(bv: &BaseValues) -> &dyn BaseValuePool {
    // SAFETY: `BaseValuesAsPool` is `#[repr(transparent)]` over `BaseValues`,
    // so the reference cast is sound.
    let as_pool: &BaseValuesAsPool =
        unsafe { &*(bv as *const BaseValues as *const BaseValuesAsPool) };
    as_pool
}

/// View `&mut BaseValues` as `&mut dyn BaseValuePool`.
pub fn base_values_as_pool_mut(bv: &mut BaseValues) -> &mut dyn BaseValuePool {
    // SAFETY: as above.
    let as_pool: &mut BaseValuesAsPool =
        unsafe { &mut *(bv as *mut BaseValues as *mut BaseValuesAsPool) };
    as_pool
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
