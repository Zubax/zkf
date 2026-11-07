"""ZKF (Zubax Kulibin float): floating-point operators for FPGA with emulation."""

# Auxiliary and shared entities.
from ._value import UnsupportedFormat as UnsupportedFormat, Zkf as Zkf, ZkfFormat as ZkfFormat
from ._block import (
    Attrs as Attrs,
    Block as Block,
    FloatLayout as FloatLayout,
    FloatView as FloatView,
    Operator as Operator,
    Timing as Timing,
)

# Operators that are synthesized into RTL, along with their numerical reference models.
from ._absneg import AbsNeg as AbsNeg, AbsNegResult as AbsNegResult
from ._add import Add as Add
from ._cmp import Cmp as Cmp, CmpResult as CmpResult
from ._cordic import Cordic as Cordic, CordicResult as CordicResult
from ._divsqrt import Divsqrt as Divsqrt, DivsqrtResult as DivsqrtResult
from ._exp2 import Exp2 as Exp2
from ._finite import Finite as Finite, FiniteResult as FiniteResult
from ._fma import Fma as Fma
from ._from_int import FromInt as FromInt
from ._ilog2 import Ilog2 as Ilog2, Ilog2Result as Ilog2Result
from ._log2 import Log2 as Log2, Log2Result as Log2Result
from ._mul import Mul as Mul
from ._mul_ilog2 import MulIlog2 as MulIlog2
from ._pipe import Pipe as Pipe
from ._resize import Resize as Resize
from ._rint import RoundMode as RoundMode, Rint as Rint, RintResult as RintResult

# Changing the version causes a new release to be deployed and tagged when pushed to the main branch.
__version__ = "0.10.0.dev0"
