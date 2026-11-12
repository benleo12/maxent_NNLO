# Package source

Start with the [quick start](../README.md), [examples](../examples/README.md) or
[new-process guide](../docs/NEW_PROCESSES.md).

Read the current calculation in this order:

1. `basis.py`: maps and Chebyshev polynomials.
2. `constraints.py`: measurement functions and fixed-order moments.
3. `reweight.py`: validate the arrays, construct constraints, select moments and solve.
4. `solver.py`: damped Newton minimization of the convex dual.
5. `result.py`: weights and diagnostics.

`released.py` loads the shipped tables and their variations. `cli.py` provides the
installed command, while `__init__.py` keeps the public Python imports short.

The previous histogram interface lives in `legacy.py` and `maxent_match.py`.
It loads only when requested. `dy_method.py` remains a compatibility alias for
`basis.py`.
