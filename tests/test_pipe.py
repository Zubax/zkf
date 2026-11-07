import pytest

from _bench import Case, run_pipeline
from zkf import Pipe


@pytest.mark.parametrize("n", [0, 1, 2, 9])
@pytest.mark.parametrize("w", [1, 8, 70])
def test_pipe(w: int, n: int) -> None:
    model = Pipe(w=w, n=n)
    ones = (1 << w) - 1
    values = [0, ones] + [1 << i for i in range(w)] + [(k * 0x9E3779B97F4A7C15 >> 7) & ones for k in range(40)]
    run_pipeline(model.component(), model.timing.latency, [Case({"x": v}, {"y": v}) for v in values])
