# SecondOrderOptimization.jl

[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/SecondOrderOptimization.jl/stable/)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/SecondOrderOptimization.jl/dev/)
[![Build Status](https://github.com/ojwoodford/SecondOrderOptimization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/SecondOrderOptimization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/SecondOrderOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/SecondOrderOptimization.jl)

High-performance, zero-allocation second-order nonlinear optimization in Julia.

## Overview

`SecondOrderOptimization.jl` minimizes smooth scalar objectives $f(x)$ using second-order trust-region and damped Newton methods:

- **Iterators**: Levenberg-Marquardt (`:levenbergmarquardt`, `:levenbergmarquardt_dtd`), Powell's Dogleg (`:dogleg`), undamped Newton (`:newton`), and backtracking Gradient Descent (`:gradientdescent`).
- **Objectives**: a cost with its gradient and Hessian (`HessianObjective`), or, for least squares, a cost with its residual and Jacobian (`JacobianObjective`), whose steps are solved without forming `JᵀJ`.
- **Linear Solvers**:
  - *Built-in*: Cholesky (`:cholesky`), LDL / Bunch-Kaufman (`:ldl`, using `LDLFactorizations.jl` for sparse matrices), and QR (`:qr`, SuiteSparse's SPQR for sparse matrices), with automatic dispatch (`:auto`).
  - *Extensions*: Iterative Krylov subspace solvers via `Krylov.jl` (`:cg` and `:minres` for a Hessian, `:lsmr` on a Jacobian) and multi-threaded sparse direct factorizations (`:pardiso` via `Pardiso.jl`).
- **Elimination**: with [`BlockLinearSystems.jl`](https://github.com/ojwoodford/BlockLinearSystems.jl) loaded, a `SchurBlockMatrix` Hessian or a `SchurBlockJacobian` Jacobian eliminates a block of the variables (by Schur complement, or by QR of the Jacobian) and solves a reduced system in the rest. The type of the prototype selects it; there is no option for it.
- **Variable Types & Manifolds**: Supports scalars (`Float64`), static vectors (`SVector`), dense vectors (`Vector`), vectors of variables of different sizes, and non-Euclidean manifolds via `nvars` / `update` from [OptimizationVariableInterface.jl](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/).
- **Zero-Allocation Reuse**: Preallocate solver buffers and factorizations via `OptimizationWorkspace`.

Looking to solve structured nonlinear least-squares factor graphs with automatic differentiation and robust kernels? See [`NLLSModel.jl`](https://github.com/ojwoodford/NLLSModel.jl), a complementary package that defines models that can be readily optimized using `SecondOrderOptimization.jl`.

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
x_opt, result = optimize(x0, HessianObjective(f, fdf!), options)
```

A run stops, converged, when the gradient is below `gtol` (default `0`, so off), the step below
`dstep`, an iteration changes the cost (up or down) by less than `reldcost` relative (default
`1e-12`) or `absdcost` absolute, or the run has stalled: the best cost has improved by less than
`stalltol` relative (default `1.5e-8`) for more than `maxstall` (3) iterations, without rising
meanwhile by more than `√stalltol`. Stalled means progress has become negligible at the best point
found, which is not proof of a minimum: a run crossing a flat valley slowly can stall too, so set
`gtol` when that matters. Check `isconverged(result)`, and test `result.termination` against the
exported `TERMINATION_*` masks (for example `result.termination & TERMINATION_MAX_FAILS != 0`) for
a specific reason.

## Least Squares from a Residual and Jacobian

For a least-squares cost, give the residual and its Jacobian instead of the gradient and Hessian.
`JacobianObjective` solves each step from the Jacobian itself, never forming `JᵀJ`, so it stays
accurate when the Jacobian is ill conditioned. The type of the Jacobian prototype selects the
linear system:

- a `Matrix{Float64}`, or the number of residuals `m`: dense QR of `J`, or StaticArrays' QR when
  `x0` has a static size and the Jacobian is small (`m·n ≤ 48`);
- a `SparseMatrixCSC{Float64, Int}`: sparse QR of `J` (SuiteSparse's SPQR), its ordering computed once;
- with BlockLinearSystems.jl, a `SchurBlockJacobian`: QR elimination of the eliminated blocks, then
  a dense or sparse reduced system in the retained variables.

`linsolver = :auto` or `:qr` factorizes; with Krylov.jl loaded, `:lsmr` solves each damped step by
LSMR on `J` instead, without factorizing. The reduced system of a `SchurBlockJacobian` also accepts
`:cholesky` and `:ldl`. Every iterator but `:gradientdescent` is supported (`:newton` is
Gauss-Newton here). An unsupported combination is an `ArgumentError` before your code is called.
The full table of supported combinations is in the
[`JacobianObjective` docstring](https://ojwoodford.github.io/SecondOrderOptimization.jl/stable/api.html#SecondOrderOptimization.JacobianObjective).

```julia
using SecondOrderOptimization

f(x) = 0.5 * ((1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2)
function resjac!(r, J, x)      # set every entry of r and every entry of J that can be non-zero
    r[1] = 1.0 - x[1]
    r[2] = 10.0 * (x[2] - x[1]^2)
    J[1, 1] = -1.0
    J[2, 1] = -20.0 * x[1]
    J[2, 2] = 10.0
    return f(x)
end
x_opt, result = optimize([-1.2, 1.0], JacobianObjective(f, resjac!, 2))
```

## Large-Scale & Iterative Solvers (Extensions)

Simply load `Krylov` or `Pardiso` to unlock optional solver backends. With a sparse Hessian
prototype, `fdf!` adds into the stored entries of `H` (which arrive zeroed):

```julia
using SecondOrderOptimization, SparseArrays
using Krylov   # unlocks :cg, :minres (and :lsmr for a JacobianObjective)

# f(x) = Σ (xᵢ - 1)² + Σ (xᵢ₊₁ - xᵢ)², with a tridiagonal Hessian
n = 1000
f(x) = sum(abs2, x .- 1) + sum(abs2, diff(x))
function fdf!(g, H, x)
    for i in 1:n
        g[i] += 2 * (x[i] - 1)
        H[i, i] += 2
    end
    for i in 1:n-1
        d = x[i+1] - x[i]
        g[i] -= 2 * d
        g[i+1] += 2 * d
        H[i, i] += 2
        H[i+1, i+1] += 2
        H[i, i+1] -= 2
        H[i+1, i] -= 2
    end
    return f(x)
end
H_sparse_prototype = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
x0 = zeros(n)

# Conjugate Gradient for large sparse positive-definite systems:
opts = OptimizationOptions(iterator = :levenbergmarquardt, linsolver = :cg)
x_opt, res = optimize(x0, HessianObjective(f, fdf!; H = H_sparse_prototype), opts)
```

With the same problem:

```julia
using Pardiso  # unlocks :pardiso (multi-threaded MKL / Panua direct solver)

opts = OptimizationOptions(iterator = :newton, linsolver = :pardiso)
x_opt, res = optimize(x0, HessianObjective(f, fdf!; H = H_sparse_prototype), opts)
```
