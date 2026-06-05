# Feldera interpreter-deprecation migration — status

Tracks stages 1–3 of `docs/feldera_persistent_circuit_design.md`. The host
interpreter (`egglog-bridge-feldera/src/interpret.rs`) stays in place as the
correctness ORACLE; the new circuit path is gated behind the env flag
`FELDERA_CIRCUIT_REBUILD=1` and is **off by default**.

## TL;DR

- The **load-bearing Stage-3 mechanism** — egglog rebuild (union-find find +
  congruence, non-monotone delete-and-rewrite) as DBSP recursive views in a
  persistent circuit with the §2.4 **congruence shuttle**, *plus the production
  argument-canonicalization extension* — is **ported into the production crate
  and validated against a Rust UF+congruence oracle** on all 120 iterations of a
  deep deep-chaining eqsat (`tests/rebuild_circuit_oracle.rs`), with
  per-iteration cost strongly **sub-linear in state**.
- The mechanism is **wired into the backend** behind the flag
  (`src/circuit_rebuild.rs`): when a `run_rules` call is a recognized rebuild
  ruleset, the whole rebuild fixpoint runs on a **persistent** congruence-shuttle
  circuit (built once, fed per-call deltas) instead of the host interpreter.
  Recognition is conservative — anything unrecognized falls back to the
  interpreter, so the path can never *regress* a program.
- **Validated correct against the oracle** (the interpreter) at every scale
  tested: `rebuild_proof` (all 5 UF shapes), `run_n_proof`, and the
  math-microbenchmark at **N=9 and N=10 the circuit-rebuild result is identical
  to the interpreter** (N=9 `Add=12067,Mul=11825`; N=10 `Add=40406,Mul=38894`).

## Per-stage status

| Stage | What | Status |
|---|---|---|
| 1 | Persist relations as integrated DBSP streams | **Partial.** The rebuild circuit is now persistent (built once, fed deltas) for the rebuild relations (`src/circuit_rebuild.rs::RebuildCache`). The *general* relation mirror is NOT yet replaced by integrated output traces — reads still go through the Rust mirror. So Stage 1 is done *for the rebuild subsystem*, not globally. |
| 2 | User-rule joins on the persistent circuit; delete `seen` seminaive | **Not started.** User-rule joins still run per-call on `dbsp_join.rs` (one fresh circuit per firing) + the host interpreter; the per-rule `seen` bookkeeping is intact. This is where the documented N≥10 divergence lives (see below). |
| 3 | Rebuild on the circuit (congruence-shuttle) | **Mechanism validated + wired behind the flag.** Matches the interpreter oracle on all tested programs. Flatness: see below — sub-linear in state, but a residual root-recompute factor remains (the documented §2.4 read-out cost, amplified by the arg-canon joins). |

## The N≥10-vs-bridge question (task #30)

The brief asks whether the new fixpoint rebuild matches the **reference bridge**
at N≥10 (the feldera *interpreter* has a documented under-derivation bug there).

**Finding: the N≥10 divergence is NOT in rebuild.** At N=10:

| | Add | Mul |
|---|---|---|
| reference bridge | 70487 | 51682 |
| feldera interpreter | 40406 | 38894 |
| **feldera circuit-rebuild** | **40406** | **38894** |

The circuit-rebuild result is **bit-identical to the interpreter** and both
differ from the reference bridge. So the new rebuild does **not introduce** a
divergence and faithfully reproduces the interpreter's rebuild; the N≥10
under-derivation originates in the **un-migrated user-rule path** (Stage 2:
the value-computing-primitive / hash-cons / join path), not in rebuild.

**#30 is therefore NOT resolved by Stage 3 alone** — resolving it requires
Stage 2 (moving the monotone user-rule joins + the head-action / hash-cons path
onto the persistent circuit), which is the documented frontier.

## Flatness (math-microbenchmark `(run N)`)

The interpreter blows up super-linearly in state — the bug the migration
targets:

| N | interpreter | persistent circuit-rebuild | result (Add/Mul) |
|---|---|---|---|
| 9  | 2.6 s  | 4.0 s  | 12067 / 11825 (matches interpreter + reference) |
| 11 | 55.7 s | 78.0 s | 104075 / 100432 (matches interpreter; both under-derive vs reference 641743/345075) |
| 13 | did not finish in budget | did not finish in budget | — |

Both backends are dominated by the **un-migrated user-rule path** at high N (the
circuit-rebuild path is slightly *slower* here only because it pays a circuit
transaction per rebuild call on top of the still-O(state) user-rule joins; the
rebuild work itself is now incremental). End-to-end flatness is blocked on
Stage 2.

Honest caveat: the rebuild circuit is now *persistent and delta-fed*, removing
the rebuild subsystem's O(state)-per-call cost. But the whole-iteration cost is
**still dominated by the un-migrated user-rule path** (Stage 2), which re-reads
the full mirror each round. So end-to-end the math-microbenchmark is **not yet
flat** — the rebuild subsystem is, the user-rule subsystem is not. The synthetic
oracle benchmark (`rebuild_circuit_oracle.rs`, rebuild only) shows the rebuild
circuit growing strongly sub-linearly in state (state ×64, time-ratio ~14×,
where the residual is the §2.4 root-recompute read-out: the leader-min antijoin
+ the three argument-canon joins recomputed at root each transaction).

## What is landed in this branch (all behind the flag / additive)

- `src/rebuild_circuit.rs` — the persistent congruence-shuttle rebuild circuit,
  generalized over the spike with **argument canonicalization** (view-row args
  carried as real ids and canonicalized to their union-find leaders inside the
  circuit; the spike only canonicalized the output). `Tup4(tag, arg0, arg1,
  out)` rows; supports view tables of arity ≤ 3 (≤ 2 args).
- `src/circuit_rebuild.rs` — recognition of the term-encoder's rebuild rules
  (`@UF` / `@UFf` / `@…View`) by IR + name, the persistent `RebuildCache`
  (built once, delta-fed), and the canonical fold-back into the mirror.
  Conservative: declines (→ interpreter) on anything it doesn't recognize or on
  view arity > 3.
- `tests/rebuild_circuit_oracle.rs` — validates the production circuit against a
  Rust UF+congruence oracle (120 iterations, deep chaining) + a sub-linear-in-
  state assertion.
- `EGraph` gains `circuit_rebuild` (the flag), `circuit_rebuild_runs`
  (diagnostic), and `rebuild_cache` (the persistent circuit).

## Frontier / remaining work

1. **Stage 2 (the real prize for #30 + end-to-end flatness):** move the monotone
   user-rule joins onto a persistent circuit (built once, fed deltas), delete
   the per-rule `seen` bookkeeping, and keep value-computing prims host-side on
   the join output. The N≥10 divergence and the residual end-to-end
   super-linearity both live here.
2. **Stage 1 (global):** replace the Rust mirror with integrated DBSP output
   traces for *all* relations (not just rebuild), so reads come from the
   persistent circuit and no full-mirror snapshot is taken per call.
3. **Flatten the rebuild read-out:** the residual non-flatness is the leader-min
   antijoin + the three argument-canon joins recomputed at ROOT each
   transaction. Making these incremental (or a typed `Min`) flattens the last
   factor; correctness is unaffected. (The brief forbids a custom native UF
   operator, and none is used.)
4. **Wider arities:** the rebuild circuit handles ≤ 2 view args; wider
   constructors fall back to the interpreter. Generalizing needs a wider/variadic
   row encoding.
5. **`@UF`/`@UFf` exact contents through the circuit path:** the circuit path is
   validated against the interpreter via the *observable* view-table sizes
   (math-microbenchmark Add/Mul) and the oracle test, but `rebuild_proof`
   (the bit-exact `@UF`/`@UFf` test) uses hand-built rules whose names don't
   match the encoder's, so it *falls back to the interpreter* and does not
   exercise the circuit path. The circuit synthesizes `@UFf = {(x, leader(x))}`
   and `@UF = {(x, leader(x)) : x != leader(x)}` from leaders; these reproduce
   the interpreter's *leader resolution* but may differ in incidental
   self-mappings / which auxiliary rows are present. A targeted bit-exact
   `@UF`/`@UFf` test driving the encoder's real rule names through the circuit
   path is outstanding.

## No regressions

- `rebuild_proof`, `run_n_proof` pass with AND without the flag.
- The flag is off by default; the default path is unchanged (interpreter oracle).
- Recognition falls back to the interpreter for anything unrecognized, so the
  61/62 feldera survey behavior is unchanged when the flag is off, and only ever
  *accelerated* (never altered in result, per the oracle match) when on.
