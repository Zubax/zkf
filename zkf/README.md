# `zkf` — Zubax Kulibin float golden reference model and emulator

A bit-exact Python model of the ZKF floating-point RTL operators.
Use it to verify a design that instantiates the `zkf_*` HDL modules: for any operator,
the model's result equals the RTL output bit-for-bit.

Copy this whole directory into your project and put its parent directory on `PYTHONPATH`.
`import zkf` uses only the Python standard library.
The optional `zkf.oracle` submodule imports `numpy`/`mpmath` when it is imported (`import zkf.oracle`).

Update this module in sync with RTL to ensure they don't diverge.

## Public API

The value model is re-exported from `zkf` (`__init__.py`); the oracles live in the `zkf.oracle` submodule.

`ZkfFormat(wexp, wman)` — the format and value factory:

```python
fmt = ZkfFormat(8, 24)
fmt.encode(1.5)              # round a float | int | Fraction to ZKF (RNTE); NaN -> ValueError, inf -> inf
fmt.zero(sign=0); fmt.inf(sign=0); fmt.normal(sign, exp, frac); fmt.from_int(wint, value)
fmt.pack(sign, exp, frac)    # raw non-canonical construct from field values (no range checks)
fmt.wrap(bits)               # wrap a raw packed bit pattern (== Zkf(fmt, bits))
```

`Zkf(fmt, bits)` — an immutable value with bit-exact operators.
Equality/hash are structural (same format and same bits); numeric comparison is the explicit `a.cmp(b)`.

Result records are `NamedTuple`s — unpack them or read fields by name.
`ZkfFormat` also exposes generated-table-derived operator parameters, such as polynomial degrees,
without leaking the tables themselves; this is useful for cycle latency derivation.

The optional `zkf.oracle` submodule provides the correctly-rounded transcendental oracles and the IEEE hardware-FPU
cross-checks.
