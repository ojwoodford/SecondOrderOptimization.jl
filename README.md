# SecondOrderOptimization.jl

High-performance, zero-allocation second-order nonlinear optimization in Julia.

## Overview

`SecondOrderOptimization.jl` minimizes smooth scalar objectives $f(x)$ using second-order trust-region and damped Newton methods:

- **Iterators**: Levenberg-Marquardt (`:levenbergmarquardt`, `:levenbergmarquardt_dtd`), Powell's Dogleg (`:dogleg`), undamped Newton (`:newton`), and backtracking Gradient Descent (`:gradientdescent`).
- **Linear Solvers**: Cholesky (`:cholesky`), LDL / Bunch-Kaufman (`:ldl`, using `LDLFactorizations.jl` for sparse matrices), and QR (`:qr`), with automatic dispatch (`:auto`).
- **Variable Types & Manifolds**: Supports scalars (`Float64`), static vectors (`SVector`), dense vectors (`Vector`), heterogeneous variable vectors, and non-Euclidean manifolds via `SecondOrderOptimization.nvars` / `SecondOrderOptimization.update` or custom `retract!(to, from, step)`.
- **Zero-Allocation Reuse**: Preallocate solver buffers and factorizations via `OptimizationWorkspace`.

Looking to solve structured nonlinear least-squares factor graphs with automatic differentiation and robust kernels? See [`NLLSModels.jl`](https://github.com/ojwoodford/NLLSModels.jl), which builds on `SecondOrderOptimization.jl`.

## Quick Start

```julia
using SecondOrderOptimization, StaticArrays

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function fdf!(g, H, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    g[2] = 200.0 * (x[2] - x[1]^2)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    return f(x)
end

x0 = @SVector [-1.2, 1.0]
options = OptimizationOptions(iterator = :levenbergmarquardt, gtol = 1e-8)
x_opt, result = optimize(f, fdf!, x0, options)
```
