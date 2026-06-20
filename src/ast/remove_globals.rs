//! Remove global variables from the program by translating
//! them into functions with no arguments.
//! This requires type information, so it is done after type checking.
//! Primitives are translated into functions with a primitive output.
//!
//! This is the single global-removal pass shared by all backends. The
//! `use_constructors_for_eq_sort` flag selects between two lowerings for
//! *eq-sort* globals:
//!
//! - Native backend (`false`): every global becomes a 0-arg `Custom`
//!   function whose value is `set`. References to the global become a
//!   direct `Custom` lookup `($x)`.
//! - Term/proof encoding (`true`): an eq-sort global becomes a 0-arg
//!   `Constructor` plus a `union`, which the term encoder can lift to a
//!   real 0-arg term (giving it an eclass). Non-eq-sort globals still
//!   become `Custom` functions (there is no eclass to allocate).
//!
//! In both modes, a global referenced in a rule head is lowered by
//! rewriting the reference to a 0-arg call. For a `Custom` global the
//! call is a RHS function lookup, so the rule opts into
//! `:unsafe-seminaive` (see `any_custom_global_in_head`).

use crate::*;
use crate::{core::ResolvedCall, typechecking::FuncType};
use egglog_ast::generic_ast::{GenericAction, GenericExpr, GenericRule};

struct GlobalRemover {
    /// When true, eq-sort globals are lifted to constructors (for the
    /// term/proof encoding); otherwise everything becomes a 0-arg
    /// `Custom` function (the native backend).
    use_constructors_for_eq_sort: bool,
}

/// Removes all globals from a program.
/// No top level lets are allowed after this pass,
/// nor any variable that references a global.
/// Adds new functions for global variables
/// and replaces references to globals with
/// references to the new functions.
/// e.g.
/// ```ignore
/// (let x 3)
/// (Add x x)
/// ```
/// becomes
/// ```ignore
/// (function x () i64)
/// (set (x) 3)
/// (Add (x) (x))
/// ```
///
/// If later, this global is referenced in a rule:
/// ```ignore
/// (rule ((Neg y))
///       ((Add x x)))
/// ```
/// the references in the head are rewritten to the 0-arg call:
/// ```ignore
/// (rule ((Neg y))
///       ((Add (x) (x))) :unsafe-seminaive)
/// ```
/// `(x)` is a RHS function lookup, so the rule is marked
/// `:unsafe-seminaive` to permit the read.
pub(crate) fn remove_globals(
    prog: Vec<ResolvedNCommand>,
    use_constructors_for_eq_sort: bool,
) -> Vec<ResolvedNCommand> {
    let mut remover = GlobalRemover {
        use_constructors_for_eq_sort,
    };
    prog.into_iter()
        .flat_map(|cmd| remover.remove_globals_cmd(cmd))
        .collect()
}

impl GlobalRemover {
    /// Subtype to use for a reference to a global of the given sort.
    /// Must match the decl-time lowering in `remove_globals_cmd`: under
    /// `use_constructors_for_eq_sort`, eq-sort globals are constructors and
    /// everything else is `Custom`. Decided purely from the sort so it is
    /// consistent even though each command is desugared by a fresh remover
    /// (the decl and its references may live in different commands).
    fn global_subtype(&self, sort: &ArcSort) -> FunctionSubtype {
        if self.use_constructors_for_eq_sort && sort.is_eq_sort() {
            FunctionSubtype::Constructor
        } else {
            FunctionSubtype::Custom
        }
    }

    fn resolved_var_to_call(&self, var: &ResolvedVar) -> ResolvedCall {
        assert!(
            var.is_global_ref,
            "resolved_var_to_call called on non-global var"
        );
        ResolvedCall::Func(FuncType {
            name: var.name.clone(),
            subtype: self.global_subtype(&var.sort),
            input: vec![],
            output: var.sort.clone(),
        })
    }

    /// TODO (yz) it would be better to implement replace_global_var
    /// as a function from ResolvedVar to ResolvedExpr
    /// and use it as an argument to `subst` instead of `visit_expr`,
    /// but we have not implemented `subst` for command.
    fn replace_global_vars(&self, expr: ResolvedExpr) -> ResolvedExpr {
        match expr.get_global_var() {
            Some(resolved_var) => GenericExpr::Call(
                expr.span(),
                self.resolved_var_to_call(&resolved_var),
                vec![],
            ),
            None => expr,
        }
    }

    fn remove_globals_expr(&self, expr: ResolvedExpr) -> ResolvedExpr {
        expr.visit_exprs(&mut |e| self.replace_global_vars(e))
    }

    fn remove_globals_action(&self, action: ResolvedAction) -> ResolvedAction {
        action.visit_exprs(&mut |e| self.replace_global_vars(e))
    }

    fn remove_globals_cmd(&mut self, cmd: ResolvedNCommand) -> Vec<ResolvedNCommand> {
        match cmd {
            GenericNCommand::CoreAction(action) => match action {
                GenericAction::Let(span, name, expr) => {
                    let ty = expr.output_type();
                    let body = self.remove_globals_expr(expr);

                    if self.use_constructors_for_eq_sort && ty.is_eq_sort() {
                        // Term/proof encoding: lift the eq-sort global to a
                        // 0-arg constructor and union it with its value, so
                        // the term encoder can give it a real eclass.
                        let resolved_call = ResolvedCall::Func(FuncType {
                            name: name.name.clone(),
                            subtype: FunctionSubtype::Constructor,
                            input: vec![],
                            output: ty.clone(),
                        });
                        let func_decl = ResolvedFunctionDecl {
                            name: name.name,
                            subtype: FunctionSubtype::Constructor,
                            impl_kind: FunctionImpl::Default,
                            schema: Schema {
                                input: vec![],
                                output: ty.name().to_owned(),
                            },
                            resolved_schema: resolved_call.clone(),
                            merge: None,
                            cost: None,
                            unextractable: true,
                            internal_hidden: false,
                            internal_let: true,
                            span: span.clone(),
                            term_constructor: None,
                        };
                        vec![
                            GenericNCommand::Function(func_decl),
                            GenericNCommand::CoreAction(GenericAction::Union(
                                span.clone(),
                                ResolvedExpr::Call(span, resolved_call, vec![]),
                                body,
                            )),
                        ]
                    } else {
                        // Native backend (all globals) and non-eq-sort
                        // globals under term encoding: a 0-arg `Custom`
                        // function whose value is `set`. A non-eq-sort
                        // function has no eclass/congruence, so the term
                        // encoder leaves it as a plain key-value store and
                        // reads it back with a direct 0-arg lookup.
                        let resolved_call = ResolvedCall::Func(FuncType {
                            name: name.name.clone(),
                            subtype: FunctionSubtype::Custom,
                            input: vec![],
                            output: ty.clone(),
                        });
                        let func_decl = ResolvedFunctionDecl {
                            name: name.name,
                            subtype: FunctionSubtype::Custom,
                            impl_kind: FunctionImpl::Default,
                            schema: Schema {
                                input: vec![],
                                output: ty.name().to_owned(),
                            },
                            resolved_schema: resolved_call.clone(),
                            merge: None,
                            cost: None,
                            unextractable: true,
                            internal_hidden: false,
                            internal_let: true,
                            span: span.clone(),
                            term_constructor: None,
                        };
                        vec![
                            GenericNCommand::Function(func_decl),
                            GenericNCommand::CoreAction(GenericAction::Set(
                                span,
                                resolved_call,
                                vec![],
                                body,
                            )),
                        ]
                    }
                }
                _ => vec![GenericNCommand::CoreAction(
                    self.remove_globals_action(action),
                )],
            },
            GenericNCommand::NormRule { rule } => {
                // Option B (unsafe-lookup): rewrite every global reference
                // in the body AND head to its 0-arg call. A `Custom` global
                // in the head is a RHS function lookup, so the rule must opt
                // into `:unsafe-seminaive`. (Constructor globals are 0-arg
                // terms and need no such flag.)
                let any_custom_global_in_head = {
                    let mut found = false;
                    rule.head.clone().visit_exprs(&mut |expr| {
                        if let Some(resolved_var) = expr.get_global_var()
                            && self.global_subtype(&resolved_var.sort) == FunctionSubtype::Custom
                        {
                            found = true;
                        }
                        expr
                    });
                    found
                };

                let new_rule = GenericRule {
                    span: rule.span,
                    body: rule
                        .body
                        .iter()
                        .map(|fact| {
                            fact.clone()
                                .visit_exprs(&mut |e| self.replace_global_vars(e))
                        })
                        .collect::<Vec<ResolvedFact>>(),
                    head: rule
                        .head
                        .clone()
                        .visit_exprs(&mut |e| self.replace_global_vars(e)),
                    name: rule.name.clone(),
                    ruleset: rule.ruleset.clone(),
                    unsafe_seminaive: rule.unsafe_seminaive || any_custom_global_in_head,
                    naive: rule.naive,
                    no_decomp: rule.no_decomp,
                };
                vec![GenericNCommand::NormRule { rule: new_rule }]
            }
            // Handle the corner case where a global command is wrap in (fail )
            GenericNCommand::Fail(span, cmd) => {
                let mut removed = self.remove_globals_cmd(*cmd);
                let last = removed.pop().unwrap();
                let boxed_last = Box::new(last);
                let new_command = GenericNCommand::Fail(span, boxed_last);
                removed.push(new_command);
                removed
            }
            _ => vec![cmd.visit_exprs(&mut |e| self.replace_global_vars(e))],
        }
    }
}
