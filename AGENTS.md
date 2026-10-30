# Repository Guidelines

Read the `README.md`.

Use the review-loop skill after a task is implemented before calling it done.

Written documentation shall be as lean and terse as possible.
Entities can be documented in at most one place; doc duplication is prohibited.

Never alter the git staging index unless asked; it is often used for manual partial progress tracking by humans.

## Conventions

Generated reports must be written in rich and colorful HTML format, not Markdown.

Synthesis attributes are added only where a measurement shows a significant and consistent gain;
a minor or mixed gain does not justify one, and a site whose gain turns minor is removed.
Each attribute adds maintenance and portability risk.

Less code does not always imply simpler code. Apply common sense and keep the code straightforward.

### Reset strategy

Use synchronous active-high reset for stream control only: validity flags, state-machine state, and other control
registers that define whether an output transaction is meaningful. Avoid resetting pure datapath registers whose
contents are ignored while their associated valid flag is deasserted. This keeps high-fanout reset nets out of wide
payload cones, reduces control-set pressure, and gives synthesis/place-and-route more freedom to retime and optimize
pipeline registers.

One subtle point: do not write the datapath assignment only in the reset-else branch, as it still makes data depend on
rst because the register is held during reset. A better strategy is to make datapath manipulation reset-unconditional
and only keep the control signals under rst/else.

References:

- AMD UG949, "When and Where to Use a Reset":
  <https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/When-and-Where-to-Use-a-Reset>
- Intel Hyperflex Architecture High-Performance Design Handbook, "Synchronous Resets Summary":
  <https://docs.altera.com/r/docs/683353/25.1.1/hyperflex-architecture-high-performance-design-handbook/synchronous-resets-summary?contentId=vgtR8yUs_Z5DH0ApHJFiTQ>
- Intel Hyperflex Architecture High-Performance Design Handbook, "Reset Strategies":
  <https://docs.altera.com/r/docs/683353/25.1.1/hyperflex-architecture-high-performance-design-handbook/reset-strategies?contentId=gzd92HdsL40qZGHurB0ezg>

When splitting an operator across multiple DSP tiles (e.g., product split), provide an operand-capture register
stage before the DSPs; this allows the placer to put a latch directly in front of each tile,
shielding the operator stage from the interconnect delays.

### Language

The hardware is written in Amaranth, in the most straightforward way: only its canonical, idiomatic, public API.
No private names, no part of the toolchain invoked directly, nothing that depends on a version of Amaranth or of Yosys
or on how they print, and no pinned dependency.

Everything of the package is private unless `zkf/__init__.py` explicitly exports it;
only the tests may reach private names.

Generally, public entities go above private ones in modules and classes,
except for private constants which may appear at the top where it makes sense.

A module that forwards an elaboration-time parameter down the hierarchy without actually using it should not attempt
to validate it, unless it somehow nontrivially depends on its value. This is done to reduce unnecessary coupling.

The code should not be commented except in cases where comments add information that is impossible to infer by
reading the code, in which case they must be extremely terse.

## Timing closure

Timing closure is an iterative process of hunting the next bottleneck and adding registers to break combinational paths:

- Set the desired frequency and synthesize the design.
- If f_max > f_target, exit.
- Locate the critical path and break it with a new register stage (enable an appropriate timing knob if available).
- Repeat.

This process works regardless of whether the failure to meet timings is caused by too many logic levels or long routing.
Special things to look out for:

- DSP tiles should begin and end with a register stage. If retiming has moved a register away from a DSP tile,
  it means that the adjacent hop is starving and needs a new register there, even if it's not on the critical path.
- A product is cut into a grid of multipliers only where an operand is wider than `wmultiplier`, so `wmultiplier` must
  be the DSP slice input width: a narrower one splits products that a slice takes whole. A product's count only places
  registers in it.
- Retiming is sneaky: a moved stage may cause a different path to become critical, so with retiming enabled one needs
  to evaluate the adjacent stages as well.

More pipeline stages do not necessarily improve f_max, and can cost both timing and area. Every optional stage spreads
that operator's flip-flops across more slices; on a wide, register-pressure-heavy datapath this adds routing congestion,
so a congestion-bound design gets slower as stages are added even though no logic path got longer.

A robust closure procedure that accounts for this starts lean and adds back one stage at a time:

- Disable every optional stage. Mind the load-bearing exceptions above: a product's count starts at its minimum, below
  which it is refused, and DSP tiles keep their bracketing registers.
- Read the critical path and judge, from the logic and the physics, whether it is the true bottleneck or merely a
  retiming casualty -- a path that only looks critical because a register was retimed away from the real cone. A true
  bottleneck is a recognizable deep operation (a wide barrel shift, a long carry chain, a DSP cascade); a casualty is an
  incidental cone that a stage added elsewhere will relieve.
- Add exactly one stage, at the boundary that splits the true bottleneck, and re-measure. Adding stages one at a time
  this way logic-balances a routing-dominated design without over-populating it with flip-flops.
- Repeat until f_max clears the target.
