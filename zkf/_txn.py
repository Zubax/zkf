from amaranth.hdl import Module, ShapeLike, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Txn(wiring.Component):
    """
    The handshake of an operator that serves one transaction at a time. `accept` starts one; the operator answers with
    `res` under a one-cycle `res_valid`; `out` holds the answer until `out_ready` takes it, and `in_ready` rises again
    in that cycle under `overlap`, which takes the next in it, else in the next. `overlap` is for an operator whose
    `res_valid` does not depend combinationally on `accept`. `stage_output=1` registers `out`; with 0, `res` passes
    through combinationally unless it has to be held.
    """

    in_valid: Signal
    in_ready: Signal
    accept: Signal
    res_valid: Signal
    out_valid: Signal
    out_ready: Signal

    def __init__(self, shape: ShapeLike, stage_output: int, overlap: bool = False) -> None:
        self._shape = shape
        self._stage_output = stage_output
        self._overlap = overlap
        super().__init__(
            {
                "in_valid": In(1),
                "in_ready": Out(1),
                "accept": Out(1),
                "res_valid": In(1),
                "res": In(shape),
                "out_valid": Out(1),
                "out_ready": In(1),
                "out": Out(shape),
            }
        )

    def elaborate(self, platform: object) -> Module:
        m = Module()
        busy = Signal(name="busy")
        m.d.comb += self.in_ready.eq(~busy | (self.out_valid & self.out_ready) if self._overlap else ~busy)
        m.d.comb += self.accept.eq(self.in_valid & self.in_ready)
        if self._stage_output:
            valid = Signal(name="r_valid")
            out = Signal(self._shape, name="r_out", reset_less=True)
            with m.If(self.res_valid):
                m.d.sync += valid.eq(1)
                m.d.sync += out.eq(self.res)
            with m.Elif(self.out_ready):
                m.d.sync += valid.eq(0)
            m.d.comb += self.out_valid.eq(valid)
            m.d.comb += self.out.eq(out)
        else:
            pending = Signal(name="pending")
            hold = Signal(self._shape, name="hold", reset_less=True)
            stall = Signal(name="stall")
            m.d.comb += stall.eq(self.res_valid & ~self.out_ready)
            with m.If(stall):
                m.d.sync += pending.eq(1)
                m.d.sync += hold.eq(self.res)
            with m.Elif(pending & self.out_ready):
                m.d.sync += pending.eq(0)
            m.d.comb += self.out_valid.eq(self.res_valid | pending)
            with m.If(pending):
                m.d.comb += self.out.eq(hold)
            with m.Else():
                m.d.comb += self.out.eq(self.res)
        with m.If(self.accept):
            m.d.sync += busy.eq(1)
        with m.Elif(self.out_valid & self.out_ready):
            m.d.sync += busy.eq(0)
        return m
