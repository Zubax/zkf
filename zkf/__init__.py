"""ZKF (Zubax Kulibin float) engine: bit-exact reference model plus packaged RTL sources."""

from ._format import OperatorModel as OperatorModel, ZkfFormat as ZkfFormat
from ._operators import (
    AbsModel as AbsModel,
    Atan2Model as Atan2Model,
    CmpModel as CmpModel,
    AddModel as AddModel,
    AddSubModel as AddSubModel,
    DivCoreModel as DivCoreModel,
    DivModel as DivModel,
    Exp2Model as Exp2Model,
    FmaModel as FmaModel,
    FromIntModel as FromIntModel,
    IsFiniteModel as IsFiniteModel,
    Log2Model as Log2Model,
    MulIlog2ConstModel as MulIlog2ConstModel,
    MulIlog2Model as MulIlog2Model,
    MulModel as MulModel,
    NegModel as NegModel,
    PackModel as PackModel,
    PipeModel as PipeModel,
    ResizeModel as ResizeModel,
    RoundModel as RoundModel,
    SaturateModel as SaturateModel,
    SincosModel as SincosModel,
    SortModel as SortModel,
    ToIntModel as ToIntModel,
)
from ._value import (
    Atan2Result as Atan2Result,
    CmpResult as CmpResult,
    DivResult as DivResult,
    Log2Result as Log2Result,
    SinCos as SinCos,
    Zkf as Zkf,
)
from ._rtl import get_rtl as get_rtl

# Changing the version causes a new release to be deployed and tagged when pushed to the main branch.
__version__ = "0.3.0"
