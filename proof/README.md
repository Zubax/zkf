# Float-HDL Formal Verification

This directory contains the SymbiYosys-driven equivalence proofs for certain modules under `hdl/`.
The proofs run via `nox -s formal`, which invokes [`run_proofs.py`](run_proofs.py) and renders the HTML report at
`build/float/formal/report.html`.

## How it works

For each module we write:

- An **independent combinational reference** under `refs/` — a Verilog transliteration of the
  relevant Python function in the reference model. The reference is deliberately written in a
  different style than the synthesisable RTL: single `always @(*)` blocks with blocking
  assignments, no pipeline, no shared helper modules. The intent is that a bug in the production
  RTL is unlikely to also be present in a fundamentally different implementation of the same
  spec.

- An **equivalence harness** under `harness/` — wraps the DUT and the reference with a
  single-pulse driver: assume `rst=1` at cycle 0, `rst=0` and `in_valid=1` at cycle 1, then
  `in_valid=0` from cycle 2 onward. The inputs at cycle 1 are latched into shadow registers.
  At cycle (1 + pipeline_depth) the harness asserts `out_valid` is 1 and the DUT outputs match
  the reference applied to the shadow inputs. Validity latency is asserted on every cycle.

- A **SymbiYosys flow** under `sby/` — one `.sby` file per proof, naming the parameter set,
  engine, BMC depth, and the file list.

For combinational modules the spec is small enough that the harness asserts the spec directly without a separate
reference module.

## Tool stack

| Component       | Role                             |
|-----------------|----------------------------------|
| Yosys           | RTL elaboration, SMT export      |
| SymbiYosys      | proof orchestration              |
| yosys-smtbmc    | SMT model construction           |
| Yices2          | primary SMT engine (QF_BV)       |
| Z3              | fallback SMT engine              |
| Bitwuzla        | secondary engine (multipliers)   |

Yices is the primary engine for everything: it has the smallest constant factor on the proofs in
this library and routinely beats z3/bitwuzla on instances of this size. Bitwuzla is built and
available, but the system's `yosys-smtbmc` Python driver occasionally hits a recursion limit
when emitting models through bitwuzla; on those modules we fall back to yices-only.

## Proof catalogue

Every `.sby` file under `sby/` is a primary proof and is exercised by `nox -s formal`.

| Module                  | Parameters      | Engine    | Notes |
|-------------------------|-----------------|-----------|-------|
| `zkf_abs`               | WEXP=6, WMAN=18 | yices     | spec inlined |
| `zkf_neg`               | WEXP=6, WMAN=18 | yices     | spec inlined; involution checked |
| `zkf_is_finite`         | WEXP=6, WMAN=18 | yices     | spec inlined |
| `zkf_saturate`          | WEXP=6, WMAN=18 | yices     | spec inlined; idempotence checked |
| `zkf_cmp`               | WEXP=6, WMAN=18 | yices     | references explicit case analysis |
| `zkf_sort`              | WEXP=6, WMAN=18 | yices     | multiset + ordering via cmp_ref |
| `zkf_pipe`              | W=24, N=4       | yices     | BMC depth 12 covers full propagation |
| `_zkf_pack`             | WEXP=6, WMAN=18 | yices     | at the production parameter set |
| `_zkf_div_radix4_step`  | WMAN=18         | yices     | greedy digit selection invariant |
| `zkf_mul`               | WEXP=5, WMAN=10 | yices     | one bit shy of binary16's mantissa; yices stalls indefinitely at WMAN=11 with no obvious progress past step 5; rounding heart still covered by the pack proof at full width |
| `zkf_add`               | WEXP=4, WMAN=6  | yices     | 8-stage BMC; reference uses wide-integer summation |
| `zkf_div`               | WEXP=4, WMAN=6  | yices     | wraps core+pack; reference uses wide unsigned division |

Combinational/sequential and trivial-wrapper consolidation rule applied:

- `zkf_cmp_comb` is **not** separately proved; `zkf_cmp` covers it transitively in BMC depth 3.
- `_zkf_div_core` is **not** separately proved; `zkf_div` covers it.
- `zkf_addsub` is **not** separately proved; it is a thin XOR-on-`b.sign` wrapper around `zkf_add`
  and contributes no arithmetic of its own, so the `zkf_add` proof at the same widths is
  sufficient. The `zkf_addsub` RTL is still exercised by `test_addsub.py` and by the
  `sim_properties_addsub_icarus` commutativity check.
- `_zkf_div_radix4_step` IS proved standalone at default `WMAN=18` because its correctness is
  independent of the surrounding pipeline parameters — proving it at production width narrows
  the parameter-genericity gap that the divider's reduced-width proof would otherwise leave open.

## How to run

```
nox -s formal              # all primary proofs, renders HTML report at the end
nox -s clean               # wipe build/ (incl. build/float/formal)

# Iterate on one proof:
sby -f -d build/float/formal/zkf_pack proof/sby/zkf_pack.sby
```

The report at `build/float/formal/report.html` is regenerated automatically by `run_proofs.py`;
on failure it embeds links to the SBY counter-example VCDs.

## Known limitations and design decisions

- Heavy arithmetic (`zkf_mul`, `zkf_add`, `zkf_div`) is proved at reduced widths because SBY's QF_BV instance at
  default `(WEXP=6, WMAN=18)` is currently intractable on yices in any reasonable wall-clock.
  The parameter-genericity gap is mitigated by:
  1. The full-width `_zkf_pack` proof (rounding logic shared by every arithmetic module).
  2. The full-width `_zkf_div_radix4_step` proof (digit primitive shared by every divider width).
  3. The simulation matrix in `../tb/`, which spans default widths up to binary64.
  4. The `zkf_mul` proof at near-binary16 widths `(WEXP=5, WMAN=10)`, exercising the full
     hidden-bit-product-high vs. product-low normalisation split that the reduced widths exercise
     only narrowly.

- The combinational references under `refs/` use `always @(*)` blocks with blocking assignments.
  This style is what makes the references easy to audit against the Python golden model.
