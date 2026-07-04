# Repository Guidelines

Read the `README.md`.

Use the comment-cleanup skill before shipping any code.
Use the review-loop skill after a task is implemented before calling it done.

Written documentation shall be as lean and terse as possible.

## Conventions

Generated reports must be written in rich and colorful HTML format, not Markdown.

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

Verilog style: 4-space indentation, concise module names, snake_case files and directories, and uppercase
parameter/localparam names where practical. Keep line length at or below 120 columns.
Comment block lines should utilize the 120 column limit well, avoiding overly short lines.

In synthesizable code, prefer `case` statements over nested ternary operators unless there are contraindications.

In complex modules, it is best to avoid a large number of named nets that are only used once;
this does not help readability but rather the opposite.

Elaboration-time parameters specifying bit width have a leading `W`, e.g., `WEXP`, `WCOEF`.

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

- DSP tiles must begin and end with a register stage. If retiming has moved a register away from a DSP tile,
  it means that the adjacent hop is starving and needs a new register there, even if it's not on the critical path.
- Splitting multiplication into parallel halves (e.g., `STAGE_PRODUCT=1`) is almost never a good idea unless the
  multiplicand bitwidth exceeds the DSP slice input width.
- Retiming is sneaky: a moved stage may cause a different path to become critical, so with retiming enabled one needs
  to evaluate the adjacent stages as well.

More pipeline stages do not necessarily improve f_max, and can cost both timing and area. Every optional stage spreads
that operator's flip-flops across more slices; on a wide, register-pressure-heavy datapath this adds routing congestion,
so a congestion-bound design gets slower as stages are added even though no logic path got longer. When the critical
path is routing-dominated -- most of the delay is wire across only a few logic levels -- and adding a stage near it
makes things worse, the design is over-pipelined, not under-pipelined: strategically removing stages can raise f_max and
free flip-flops at the same time.

A robust closure procedure that accounts for this starts lean and adds back one stage at a time:

- Disable every optional stage. Mind the load-bearing exceptions above: the DSP product keeps `STAGE_PRODUCT` once the
  multiplicand exceeds the slice input width, and DSP tiles keep their bracketing registers.
- Read the critical path and judge, from the logic and the physics, whether it is the true bottleneck or merely a
  retiming casualty -- a path that only looks critical because a register was retimed away from the real cone. A true
  bottleneck is a recognizable deep operation (a wide barrel shift, a long carry chain, a DSP cascade); a casualty is an
  incidental cone that a stage added elsewhere will relieve.
- Add exactly one stage, at the boundary that splits the true bottleneck, and re-measure. Adding stages one at a time
  this way logic-balances a routing-dominated design without over-populating it with flip-flops.
- Repeat until f_max clears the target. If a newly added stage lowers f_max it was relieving congestion, not logic
  depth: back it out and split a different boundary.
