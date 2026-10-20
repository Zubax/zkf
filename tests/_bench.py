"""
The simulation bench. A case is the inputs of one operation and the outputs it must give; the bench decides when to
present it and what happens around it. The simulator has no unknown values and starts every register at its reset
value, so what would show as an X elsewhere has to be provoked: operands are random whenever they are not being
taken, and reset comes with the pipeline full.
"""

import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from amaranth.hdl import ClockDomain, Module, Shape, ShapeCastable, ShapeLike, Value
from amaranth.lib import data, wiring
from amaranth.sim import Simulator, SimulatorContext

from zkf._value import bits_to_signed

type Values = Mapping[str, object]


@dataclass(frozen=True)
class Case:
    inputs: Values
    outputs: Values
    """An output that is not listed may be anything."""
    latency: int = 0
    """Of a handshake operator's transaction; a pipelined operator has one latency, given to the bench."""


class Mismatch(AssertionError):
    pass


def holds(shape: ShapeLike, value: object) -> bool:
    """Whether the shape holds a constant whole: Amaranth cuts an integer, and a field of a mapping, without a word."""
    if isinstance(value, Mapping):
        members = shape.members if isinstance(shape, data.StructLayout) else {}
        return set(value) == set(members) and all(holds(members[name], field) for name, field in value.items())
    if isinstance(value, int):
        cast = Shape.cast(shape)
        low, high = (-(1 << (cast.width - 1)), 1 << (cast.width - 1)) if cast.signed else (0, 1 << cast.width)
        return low <= value < high
    return True


def run_pipeline(dut: wiring.Component, latency: int, cases: Sequence[Case], seed: int = 0) -> None:
    """
    Every case goes in under `in_valid` with random gaps between cases, and its outputs must show `latency` cycles
    later under `out_valid`, which must be low in every other cycle. Once, with the pipeline as full as the cases make
    it, reset is asserted for a cycle in which a case is also offered: nothing in flight may come out afterwards, and
    what was in flight goes in again.
    """
    rng = random.Random(seed)
    bench = _Dut(dut, ("in_valid", "out_valid"), cases)
    reset_at = len(cases) // 2
    plan = [_Cycle(None, reset=True)] * 2 + [_Cycle(None)] * 3
    for index, case in enumerate(cases):
        full = reset_at - latency - 1 <= index <= reset_at
        plan += [_Cycle(None)] * (0 if full else rng.choice((0, 0, 0, 1, 2, 5)))
        plan.append(_Cycle(case, valid=True))
        if index == reset_at:
            plan.append(_Cycle(case, valid=True, reset=True))
    drain = [_Cycle(None)] * (latency + 2)

    def due(plan: Sequence[_Cycle]) -> list[Case | None]:
        want: list[Case | None] = []
        stages: deque[Case | None] = deque([None] * latency)
        for cycle in plan:
            entering = cycle.case if cycle.valid else None
            want.append(stages[-1] if latency else entering)
            if latency:
                stages.rotate(1)
                stages[0] = entering
                if cycle.reset:
                    stages = deque([None] * latency)
        return want

    shown = {id(case) for case in due(plan + drain)}
    plan += [_Cycle(case, valid=True) for case in cases if id(case) not in shown] + drain
    want = due(plan)

    # A process, not a testbench: the design then settles once a cycle and not once per input set.
    async def driver(ctx: SimulatorContext) -> None:
        with ctx.critical():
            for cycle in plan:
                ctx.set(bench.domain.rst, cycle.reset)
                ctx.set(dut.in_valid, cycle.valid)
                bench.drive(ctx, cycle.case.inputs if cycle.valid and cycle.case else bench.noise(rng))
                await ctx.tick()

    got: list[tuple[int, Sequence[object]]] = []

    async def monitor(ctx: SimulatorContext) -> None:
        ports = [getattr(dut, name) for name in bench.outputs]
        async for _, _, valid, *values in ctx.tick().sample(dut.out_valid, *ports):
            got.append((valid, values))

    bench.sim.add_process(driver)
    bench.sim.add_process(monitor)
    bench.sim.run()
    assert len(got) >= len(plan)
    for index, (expected, (valid, values)) in enumerate(zip(want, got)):
        where = f"cycle {index}"
        if bool(valid) != (expected is not None):
            raise Mismatch(f"{where}: out_valid is {valid}, {'a result' if expected else 'nothing'} is due")
        if expected is not None:
            bench.check(values, expected.outputs, where)


def run_handshake(dut: wiring.Component, cases: Sequence[Case], seed: int = 0, overlap: bool = False) -> None:
    """
    One transaction per case. It is taken in a cycle with `in_valid` and `in_ready`, and its outputs show
    `case.latency` cycles later under `out_valid`, which holds them until a cycle with `out_ready`; `in_ready` is low
    from the cycle after it is taken until it is handed over, and high in that cycle under `overlap`, else in the next.
    Some transactions are held back by the consumer, some follow the previous one with no idle cycle (in the cycle it
    is handed over under `overlap`), and some are cut by reset, while in progress or while held: nothing of them may
    come out afterwards, and one cut while in progress is given again.
    The kinds come in equal numbers, in an order that the seed shuffles.
    """
    bench = _Handshake(dut, cases, seed)
    rng = bench.rng

    async def testbench(ctx: SimulatorContext) -> None:
        await bench.reset(ctx, "start", bench.quiet)
        kinds = ("plain", "plain", "back to back", "held", "reset in progress", "reset while held")
        deck = [kinds[index % len(kinds)] for index in range(len(cases))]
        rng.shuffle(deck)
        queue = deque((f"transaction {index}", case, deck[index]) for index, case in enumerate(cases))
        taken = False  # in the cycle the previous one was handed over
        while queue:
            where, case, kind = queue.popleft()
            ctx.set(dut.out_ready, kind not in ("held", "reset while held"))
            if not taken:
                if kind != "back to back":
                    ctx.set(dut.in_valid, 0)
                    for _ in range(rng.randrange(1, 4)):
                        await bench.idle(ctx, f"{where}, idle", ready=True, expected=None)
                await bench.offer(ctx, where, case)
            taken = False
            cut = rng.randrange(1, case.latency) if kind == "reset in progress" and case.latency > 1 else None
            for cycle in range(1, case.latency):
                if cycle == cut:
                    break
                await bench.idle(ctx, f"{where}, cycle {cycle}", ready=False, expected=None)
            if cut is not None:
                await bench.reset(ctx, where, bench.quiet)
                queue.appendleft((f"{where}, again", case, "plain"))
                continue
            if kind in ("held", "reset while held"):
                for _ in range(rng.randrange(1, 5)):
                    await bench.idle(ctx, f"{where}, held", ready=False, expected=case)
                if kind == "reset while held":
                    await bench.reset(ctx, where, bench.quiet)
                    continue
                ctx.set(dut.out_ready, 1)
            if overlap and queue and queue[0][2] == "back to back":
                await bench.offer(ctx, f"{where}, result, {queue[0][0]} offered", queue[0][1], expected=case)
                taken = True
            else:
                ctx.set(dut.in_valid, 0)
                await bench.idle(ctx, f"{where}, result", ready=overlap, expected=case)
        ctx.set(dut.in_valid, 0)
        for _ in range(3):
            await bench.idle(ctx, "end", ready=True, expected=None)

    bench.sim.add_testbench(testbench)
    bench.sim.run()


def run_cuts(dut: wiring.Component, pairs: Sequence[tuple[Case, Case]], seed: int = 0, overlap: bool = False) -> None:
    """
    The first transaction of a pair, cut by reset in each cycle of its own, that of its result included, and the
    second given after each cut, once in the next cycle and once past where the first was due: the second comes out
    as it should, and nothing else does.
    """
    bench = _Handshake(dut, [case for pair in pairs for case in pair], seed)

    async def testbench(ctx: SimulatorContext) -> None:
        await bench.reset(ctx, "start", bench.quiet)
        cuts = [(pair, cut, at_once) for pair in pairs for cut in range(1, pair[0].latency + 1) for at_once in (1, 0)]
        for (first, then), cut, at_once in cuts:
            where = f"{first.inputs} cut in cycle {cut}"
            ctx.set(dut.out_ready, 0)
            await bench.offer(ctx, where, first)
            for cycle in range(1, cut):
                await bench.idle(ctx, f"{where}, cycle {cycle}", ready=False, expected=None)
            await bench.reset(ctx, where, 0 if at_once else first.latency - cut + 2)
            ctx.set(dut.out_ready, 1)
            await bench.offer(ctx, f"{where}, the next", then)
            for cycle in range(1, then.latency):
                await bench.idle(ctx, f"{where}, the next, cycle {cycle}", ready=False, expected=None)
            ctx.set(dut.in_valid, 0)
            await bench.idle(ctx, f"{where}, the next, result", ready=overlap, expected=then)
            # What is left of the first would show by the cycle it was due in.
            for _ in range(max(first.latency - cut - then.latency, 0) * at_once + 2):
                await bench.idle(ctx, f"{where}, afterwards", ready=True, expected=None)

    bench.sim.add_testbench(testbench)
    bench.sim.run()


class _Dut:
    def __init__(self, dut: wiring.Component, control: Sequence[str], cases: Sequence[Case]) -> None:
        """
        A case names every input and nothing but ports, and its values fit them: a key that names no port would be
        checked by nobody, and the simulator would cut a number that is too wide, in an expectation too. What a bundle
        must give is a value of its layout: raw bits never equal one.
        """
        self.dut = dut
        members = dut.signature.members
        self.inputs = [n for n, member in members.items() if member.flow == wiring.In and n not in control]
        self.outputs = [n for n, member in members.items() if member.flow == wiring.Out and n not in control]
        for case in cases:
            if set(case.inputs) != set(self.inputs) or not set(case.outputs) <= set(self.outputs):
                raise ValueError(f"a case of {sorted(case.inputs)} to {sorted(case.outputs)} is not of these ports")
            for name, value in [*case.inputs.items(), *case.outputs.items()]:
                shape = members[name].shape
                raw = name in case.outputs and isinstance(value, int) and isinstance(shape, ShapeCastable)
                if raw or not holds(shape, value):
                    raise ValueError(f"{name}={value!r} does not fit {shape!r}")
        top = Module()
        top.domains.sync = self.domain = ClockDomain("sync")
        top.submodules.dut = dut
        self.sim = Simulator(top)
        self.sim.add_clock(1e-6)

    def noise(self, rng: random.Random) -> dict[str, int]:
        shapes = {name: Shape.cast(self.dut.signature.members[name].shape) for name in self.inputs}
        raw = {name: rng.getrandbits(shape.width) for name, shape in shapes.items()}
        return {
            name: bits_to_signed(raw[name], shape.width) if shape.signed else raw[name]
            for name, shape in shapes.items()
        }

    def drive(self, ctx: SimulatorContext, values: Values) -> None:
        for name in self.inputs:
            port, value = getattr(self.dut, name), values[name]
            # Raw bits go to a port of any shape; a layout takes them only if it says how.
            ctx.set(Value.cast(port) if isinstance(value, int) else port, value)

    def check(self, got: Sequence[object], want: Values, where: str) -> None:
        for name, value in zip(self.outputs, got):
            if name in want and value != want[name]:
                raise Mismatch(f"{where}: {name} is {value!r}, not {want[name]!r}")


@dataclass(frozen=True)
class _Cycle:
    case: Case | None
    valid: bool = False
    reset: bool = False


class _Handshake(_Dut):
    """
    Idle inputs reach logic that the operator shares between its transactions, so they are disturbed: in the cycle
    after a transaction is taken, and then in one cycle of four, since new inputs in every cycle cost most of the
    simulator's rate. They are random bits or the inputs of a case, which are what the logic is sensitive to.
    """

    def __init__(self, dut: wiring.Component, cases: Sequence[Case], seed: int) -> None:
        super().__init__(dut, ("in_valid", "in_ready", "out_valid", "out_ready"), cases)
        self.rng = random.Random(seed)
        self.pool = [case.inputs for case in cases]
        self.ports = [getattr(dut, name) for name in self.outputs]
        self.quiet = 2 + max(case.latency for case in cases)

    def disturb(self, ctx: SimulatorContext) -> None:
        self.drive(ctx, self.noise(self.rng) if self.rng.getrandbits(1) else self.rng.choice(self.pool))

    async def step(self, ctx: SimulatorContext, where: str, ready: bool, expected: Case | None) -> None:
        got_ready, got_valid = ctx.get(self.dut.in_ready), ctx.get(self.dut.out_valid)
        if bool(got_ready) != ready:
            raise Mismatch(f"{where}: in_ready is {got_ready}")
        if bool(got_valid) != (expected is not None):
            raise Mismatch(f"{where}: out_valid is {got_valid}, {'a result' if expected else 'nothing'} is due")
        if expected is not None:
            self.check([ctx.get(port) for port in self.ports], expected.outputs, where)
        await ctx.tick()

    async def idle(self, ctx: SimulatorContext, where: str, ready: bool, expected: Case | None) -> None:
        if not self.rng.getrandbits(2):
            self.disturb(ctx)
        await self.step(ctx, where, ready, expected)

    async def offer(self, ctx: SimulatorContext, where: str, case: Case, expected: Case | None = None) -> None:
        """The cycle in which the case is taken, with `expected` due in it; the inputs are other ones in the next."""
        ctx.set(self.dut.in_valid, 1)
        self.drive(ctx, case.inputs)
        await self.step(ctx, f"{where}, offered", ready=True, expected=expected)
        ctx.set(self.dut.in_valid, self.rng.getrandbits(1))
        self.disturb(ctx)

    async def reset(self, ctx: SimulatorContext, where: str, quiet: int) -> None:
        """With a transaction offered, which must not be taken."""
        ctx.set(self.domain.rst, 1)
        ctx.set(self.dut.in_valid, 1)
        self.disturb(ctx)
        await ctx.tick()
        ctx.set(self.domain.rst, 0)
        ctx.set(self.dut.in_valid, 0)
        for _ in range(quiet):
            await self.idle(ctx, f"{where}, after reset", ready=True, expected=None)
