use std::{collections::HashSet, convert::TryFrom};

use egglog::{
    Core, Primitive, Read, ReadPrim, ReadState, Value,
    constraint::{AllEqualTypeConstraint, TypeConstraint},
    prelude::BaseSort,
    prelude::{I64Sort, Span, StringSort},
    sort::S,
    util::INTERNAL_SYMBOL_PREFIX,
};

#[derive(Clone)]
pub struct GetSizePrimitive;

impl Primitive for GetSizePrimitive {
    fn name(&self) -> &str {
        "get-size!"
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        AllEqualTypeConstraint::new(self.name(), span.clone())
            .with_output_sort(I64Sort.to_arcsort())
            .with_all_arguments_sort(StringSort.to_arcsort())
            .into_box()
    }
}

impl ReadPrim for GetSizePrimitive {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let filters: Option<HashSet<String>> = if args.is_empty() {
            None
        } else {
            Some(
                args.iter()
                    .map(|value| state.base_values().unwrap::<S>(*value).0)
                    .collect::<HashSet<_>>(),
            )
        };

        let total_size: usize = state
            .table_sizes()
            .into_iter()
            .filter_map(|(name, size)| {
                // An explicit filter is authoritative: count exactly the named
                // tables, even internal (`@`-prefixed) ones. Term encoding uses
                // this to point `get-size!` at the canonical `@<F>View` tables
                // (the mode-invariant egraph size) instead of the monotonic
                // hash-cons term tables — see `instrument_get_size` in
                // `proof_encoding.rs`. With no filter, internal tables are
                // excluded as before (term tables would otherwise over-count).
                match &filters {
                    Some(filter) => filter.contains(name).then_some(size),
                    None => (!name.starts_with(INTERNAL_SYMBOL_PREFIX)).then_some(size),
                }
            })
            .sum();
        let total_size = i64::try_from(total_size).ok()?;
        Some(state.base_values().get::<i64>(total_size))
    }
}
