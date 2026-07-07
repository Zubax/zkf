"""ZKF (Zubax Kulibin float) engine: bit-exact reference model plus packaged RTL sources."""

from ._core import (
    AbsModel as AbsModel,
    Atan2Result as Atan2Result,
    Atan2Model as Atan2Model,
    CmpResult as CmpResult,
    CmpModel as CmpModel,
    AddModel as AddModel,
    AddSubModel as AddSubModel,
    DivCoreModel as DivCoreModel,
    DivModel as DivModel,
    DivResult as DivResult,
    Exp2Model as Exp2Model,
    FmaModel as FmaModel,
    FromIntModel as FromIntModel,
    IsFiniteModel as IsFiniteModel,
    Log2Result as Log2Result,
    Log2Model as Log2Model,
    MulIlog2ConstModel as MulIlog2ConstModel,
    MulIlog2Model as MulIlog2Model,
    MulModel as MulModel,
    NegModel as NegModel,
    OperatorModel as OperatorModel,
    PackModel as PackModel,
    PipeModel as PipeModel,
    ResizeModel as ResizeModel,
    RoundModel as RoundModel,
    SaturateModel as SaturateModel,
    SinCos as SinCos,
    SincosModel as SincosModel,
    SortModel as SortModel,
    ToIntModel as ToIntModel,
    Zkf as Zkf,
    ZkfFormat as ZkfFormat,
)
from ._rtl import get_rtl as get_rtl

# Changing the version causes a new release to be deployed and tagged when pushed to the main branch.
__version__ = "0.2.0"
