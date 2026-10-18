# TODO

Non-blocking; to go over after the rewrite is reviewed and committed. The most severe first.

- `wmultiplier` is one number, the narrower side of a rectangular multiplier: two of its slices can be one tile on the
  chip (25x18, 27x18), so a grid register may cut nothing there.
- Diamond and Vivado are not in CI; the synthesis tests run them where their tools are found.
- Mul's input-stage credit (where `stage_input` > 0 its last input stage is the product's capture) is provisional.
- The packer's sticky bit is gathered behind the normalizer's last shift, which is what binds `cordic` at 8/36
  next (14 to 18 levels); it can be gathered beside the last two levels. The detects of the later levels wait for
  the shifts ahead of them likewise, and the sum of a product's rows is two carry chains in a row.
- The kernels of WMAN 44 to 53 can start their chain a cycle earlier (measured bit-identical at the catalog's
  widths); no spec that wide is timed.
- Table generation uses the multiplier width up to degree 3 only (WMAN up to 36); the author allows assuming that no
  operand is wider than four multipliers. Shortening an operand to a multiple of the width where it passes one by up to
  three bits saves 4 to 8% of LUTs and 3 to 5 DSP48E1 on Spartan-7 at WMAN 44 and 53, and changes some tables of
  degree 3. For the optimization pass.
- The table of `cordic`'s angles becomes a memory on Yosys, which doubles the engine's LUTs at two iterations a
  cycle, and the equivalence test's proof does not finish at two or four; no spec runs at those rates.
- `FloatLayout.const` takes a bare integer as raw bits.
- Icarus in SystemVerilog mode (`-g2012`, which cocotb uses) leaves values unknown where a combinational block of
  the printed text sees no event after time zero: with the initial values, on the vectoring-only `cordic`; with a
  reset held from time zero and no transition, on the valid flags. Verilator and Icarus in Verilog-2005 mode show
  neither: a simulation artifact, left as it is.
- The default test tier takes 10 minutes on 8 workers. Simulating the C++ that `amaranth.back.cxxrtl`
  prints is 16 to 103 times faster, with a bench of its own.
