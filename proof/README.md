# Float-HDL Formal Verification

This directory contains the SymbiYosys-driven equivalence proofs for certain modules under `zkf/rtl/`.
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
  A module configured with no register stages is instead checked as combinational logic: under any `rst` and
  `in_valid`, `out_valid` follows `in_valid` and the outputs match the reference applied to the live inputs.

- A **SymbiYosys flow** under `sby/` — one `.sby` file per proof, naming the parameter set,
  engine, BMC depth, and the file list. `run_proofs.py` injects `LATENCY` for pipelined DUTs.

Latency values come from `ZkfFormat(WEXP, WMAN).model_of(<operator>)(...).timing.latency`.

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
| `zkf_finite`            | WEXP=6, WMAN=18 | yices     | spec inlined; saturation finite and idempotent; combinational and STAGE_INPUT=STAGE_OUTPUT=1 |
| `zkf_cmp`               | WEXP=6, WMAN=18 | yices     | references explicit case analysis; combinational and STAGE_INPUT=STAGE_OUTPUT=1 |
| `zkf_sort`              | WEXP=6, WMAN=18 | yices     | multiset + ordering via cmp_ref |
| `zkf_pipe`              | W=24, N=4       | yices     | BMC depth 12 covers full propagation |
| `_zkf_pack`             | WEXP=6, WMAN=18 | yices     | at the production parameter set; also with STAGE_OUTPUT=1 |
| `_zkf_pack` (biased)    | WEXP=6, WMAN=18 | yices     | EXP_IS_BIASED=1 port |
| `_zkf_pack` (sat)       | WEXP=6, WMAN=18 | yices     | SATURATE_ROUND_CARRY=1: alone, with the biased port, and with STAGE_OUTPUT=1 |
| `_zkf_pack` (narrow)    | WEXP=4, WMAN=5  | yices     | narrowest legal WEXP_UNBIASED: WEXP unbiased, WEXP+1 biased |
| `zkf_mul`               | WEXP=5, WMAN=10 | yices     | one bit shy of binary16's mantissa; yices stalls indefinitely at WMAN=11 with no obvious progress past step 5; rounding heart still covered by the pack proof at full width |
| `zkf_add`               | WEXP=4, WMAN=6  | yices     | 8-stage BMC; reference uses wide-integer summation |
| `zkf_divsqrt`           | WEXP=4, WMAN=6,7,10 | yices | free-running stimulus instead of a single pulse: any interleaving of both operations, bubbles, reset at any cycle; MODE=0/1/2, STAGE_DECODE=1 and the other knobs |
| `_zkf_divsqrt_step`     | WMAN=18,27,36,53; WW=18..88 | yices | MODE=1/2: every digit stage, maximal digit, remainder, D/D3 updates, reachable-state invariant preserved; MODE=0: the division step at any divisor, at zkf_divsqrt's and zkf_cordic's widths |
| `_zkf_divsqrt_div0`     | WMAN=18,36,53   | yices     | the divider's stage 0, folded (18, 36) and plain (18, 53) |
| `_zkf_divsqrt_root0`    | WMAN=4,5,18,27,36,53 | yices | the root's stage 0 from any radicand, establishing the invariant |
| `_zkf_divsqrt_last`     | WMAN=18,27,36,53 | yices    | the last decision with its rounding select, from the invariant, in every MODE, folded and not |

Trivial-wrapper consolidation rule applied:

- `zkf_addsub` is **not** separately proved; it is a thin XOR-on-`b.sign` wrapper around `zkf_add`
  and contributes no arithmetic of its own, so the `zkf_add` proof at the same widths is
  sufficient. The `zkf_addsub` RTL is still exercised by `test_addsub.py` and by the
  `sim_properties_addsub_icarus` commutativity check.

## How to run

```
nox -s formal              # all primary proofs, renders HTML report at the end
nox -s clean               # wipe build/ (incl. build/float/formal)

# For one proof, point --sby-dir at a scratch directory containing only that .sby.
python proof/run_proofs.py --sby-dir build/float/formal-one/sby --jobs 1 --timeout-seconds 300
```

The report at `build/float/formal/report.html` is regenerated automatically by `run_proofs.py`;
on failure it embeds links to the SBY counter-example VCDs.

## Known limitations and design decisions

- Heavy arithmetic (`zkf_mul`, `zkf_add`, `zkf_divsqrt`) is proved at reduced widths because
  SBY's QF_BV instance at default `(WEXP=6, WMAN=18)` is currently intractable on yices in any reasonable wall-clock.
  The parameter-genericity gap is mitigated by:
  1. The full-width `_zkf_pack` proof (rounding logic shared by every arithmetic module).
  2. `zkf_divsqrt`'s production-width proofs of its digit stages, both stages 0 and the last decision.
  3. The simulation matrix in `../tb/`, which spans default widths up to binary64.
  4. The `zkf_mul` proof at near-binary16 widths `(WEXP=5, WMAN=10)`, exercising the full
     hidden-bit-product-high vs. product-low normalisation split that the reduced widths exercise
     only narrowly.

- `zkf_divsqrt`'s production-width proofs are local: each part is proved from the reachable-state invariant it assumes,
  and only the end-to-end proofs (to WMAN=10) check that the parts compose, along with the wiring between them (the
  digit prefix, prefix + 1, the registered injection bits, 5 and 7 times the divisor). Its last decision also relies on
  two facts of arithmetic rather than of the circuit. No quotient or root of WMAN-bit significands lies exactly halfway
  between two WMAN-bit values: for a/b that would need a * 2^k = b * (an odd number) with more factors of 2 on the left
  than b has; for sqrt the square of the midpoint has an odd numerator below the radicand's last bit. And the
  prenormalized quotient (at most 2 - 2^(1-WMAN)) and the root (below 2 - 2^-WMAN) round below 2, so rounding never
  carries out.

- The combinational references under `refs/` use `always @(*)` blocks with blocking assignments.
  This style is what makes the references easy to audit against the Python golden model.
