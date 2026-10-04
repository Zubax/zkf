"""ZKF (Zubax Kulibin float) engine: bit-exact reference model plus packaged RTL sources."""

from ._format import OperatorModel as OperatorModel, Timing as Timing, ZkfFormat as ZkfFormat
from ._operators import (
    AbsModel as AbsModel,
    CordicModel as CordicModel,
    CmpModel as CmpModel,
    AddModel as AddModel,
    AddSubModel as AddSubModel,
    DivModel as DivModel,
    DivsqrtModel as DivsqrtModel,
    Exp2Model as Exp2Model,
    FmaModel as FmaModel,
    FromIntModel as FromIntModel,
    Ilog2Model as Ilog2Model,
    IsFiniteModel as IsFiniteModel,
    Log2Model as Log2Model,
    MulIlog2Model as MulIlog2Model,
    MulModel as MulModel,
    NegModel as NegModel,
    PipeModel as PipeModel,
    ResizeModel as ResizeModel,
    RintModel as RintModel,
    SaturateModel as SaturateModel,
    SortModel as SortModel,
    SqrtModel as SqrtModel,
)
from ._value import (
    Atan2Result as Atan2Result,
    CmpResult as CmpResult,
    DivResult as DivResult,
    Log2Result as Log2Result,
    SinCos as SinCos,
    SqrtResult as SqrtResult,
    Zkf as Zkf,
)
from ._reference import RoundMode as RoundMode, UnsupportedFormat as UnsupportedFormat
from ._rtl import get_rtl as get_rtl

# Changing the version causes a new release to be deployed and tagged when pushed to the main branch.
__version__ = "0.8.0"
