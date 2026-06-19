use egglog::ast::*;
use egglog_ast::span::Span;

/// Parse-time routing for `(run-schedule <schedule>*)`.
///
/// egglog-experimental registers `run-schedule` as a user-defined command
/// (`RunExtendedSchedule`) so that schedulers (`let-scheduler` / `run-with`,
/// e.g. the back-off scheduler) can be driven from Rust. User-defined
/// commands are opaque to the proof-term encoder, so `--proofs` rejects every
/// `(run-schedule ...)` — even the basic ones that use only
/// `saturate`/`run`/`repeat`/`seq` and have an exact equivalent in core
/// egglog's proof-supported `RunSchedule` command.
///
/// This macro runs at parse time and takes precedence over the user-defined
/// routing (parser command macros are matched before the user-defined set).
/// When the schedule uses no scheduler it lowers to the core
/// [`Command::RunSchedule`] (proof-supported, identical semantics to the core
/// `run-schedule` parser). When a scheduler *is* present it reproduces the
/// original user-defined routing so `RunExtendedSchedule` runs unchanged.
pub struct RunSchedule;

impl Macro<Vec<Command>> for RunSchedule {
    fn name(&self) -> &str {
        "run-schedule"
    }

    fn parse(
        &self,
        args: &[Sexp],
        span: Span,
        parser: &mut Parser,
    ) -> Result<Vec<Command>, ParseError> {
        if args.iter().any(uses_scheduler) {
            // Scheduler in use: keep the experimental user-defined command so
            // `RunExtendedSchedule` drives it. Mirror the parser's own
            // user-defined routing (`parse_command`), which parses each arg as
            // an expression.
            let exprs = args
                .iter()
                .map(|s| parser.parse_expr(s))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(vec![Command::UserDefined(
                span,
                "run-schedule".to_owned(),
                exprs,
            )]);
        }

        // No scheduler: lower to core egglog's proof-supported RunSchedule,
        // matching the core `run-schedule` parser exactly.
        let schedules = args
            .iter()
            .map(|s| parser.parse_schedule(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![Command::RunSchedule(Schedule::Sequence(
            span, schedules,
        ))])
    }
}

/// Whether a schedule s-expression references a scheduler construct
/// (`let-scheduler` or `run-with`) anywhere in its tree. Only these require
/// the experimental `RunExtendedSchedule`; everything else
/// (`saturate`/`run`/`repeat`/`seq`) has a core equivalent.
fn uses_scheduler(sexp: &Sexp) -> bool {
    match sexp {
        Sexp::List(items, _) => {
            if let [Sexp::Atom(head, _), ..] = items.as_slice()
                && (head == "let-scheduler" || head == "run-with")
            {
                return true;
            }
            items.iter().any(uses_scheduler)
        }
        _ => false,
    }
}
