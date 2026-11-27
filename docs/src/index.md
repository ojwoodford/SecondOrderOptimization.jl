# SecondOrderOptimization.jl

**High-Performance Second-Order Optimization in Julia**

`SecondOrderOptimization.jl` is a lightweight, zero-allocation, modular library for solving unconstrained and manifold-constrained nonlinear optimization problems using second-order Newton-type and trust-region methods.

```math
\min_{x \in \mathcal{M}} f(x)
```

## Features

- **Zero Allocations**: Evaluates cost, gradient, and Hessian in a single pass (`fdf!(g, H, x)`), with zero dynamic heap allocations on static vectors (`SVector`) and preallocated [`OptimizationWorkspace`](@ref) caches.
- **Multiple Iteration Strategies**:
  - **Levenberg-Marquardt** (`:levenbergmarquardt` and `:levenbergmarquardt_dtd`): Dynamic damping with uniform or adaptive Marquardt diagonal scaling and speculative trial linearization reuse.
  - **Powell's Dogleg** (`:dogleg`): Trust-region interpolation between the Cauchy (steepest descent) step and the Newton step.
  - **Standard Newton** (`:newton`): Classical undamped second-order step.
  - **Your own**, through the iterator interface: [Writing Your Own Iterator](tutorials/custom_iterator.md) writes a backtracking gradient descent.
- **Flexible Manifolds**: Native support for custom variable types and manifolds, by extending `nvars` and `update` from OptimizationVariableInterface.jl (see [Manifold Optimization and Custom Variable Types](tutorials/manifold_optimization.md)).
- **Static, Dense & Sparse Systems**: Automatically selects static (`MMatrix`), dense (`Matrix`), or sparse (`SparseMatrixCSC` with zero-allocation `LDLFactorizations.jl` and `Cholesky`) linear systems.
- **Least Squares from a Jacobian**: A [`JacobianObjective`](@ref) gives the residual and its Jacobian instead of the gradient and Hessian, and each step is solved from the Jacobian without forming `JᵀJ`: dense QR, static QR for a small problem with a static size, sparse QR (SPQR) for a sparse Jacobian, or LSMR (`:lsmr`); or, for a dense Jacobian with `:cholesky` or `:ldl`, by the normal equations, `JᵀJ` formed from the Jacobian once per linearization. Its docstring lists the supported combinations (see [Least Squares from a Jacobian](tutorials/jacobian_least_squares.md)).
- **Extensible Linear Solvers**: Choose between built-in direct solvers (`:auto`, `:cholesky`, `:ldl`, `:qr`) or package extensions for multi-threaded sparse direct solves ([Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) with `:pardiso`) and zero-allocation iterative Krylov subspace solvers ([Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) with `:cg` and `:minres` for a Hessian, `:lsmr` on a Jacobian).
- **Elimination**: With [BlockLinearSystems.jl](https://github.com/ojwoodford/BlockLinearSystems.jl) loaded, a `SchurBlockHessian` or `SchurBlockJacobian` prototype eliminates a block of the variables and solves a reduced system in the rest (see [Extension Solvers](@ref)). The type of the prototype selects it; there is no option for it.
- **Rich Callbacks & Diagnostics**: Monitor optimization progress, per-iteration timing, cost reductions, trust-region radius, and allocation counts via [`OptimizationResult`](@ref) and [`CostTrajectory`](@ref).

## Quick Start

```julia
using SecondOrderOptimization
using StaticArrays

# 2D Rosenbrock function: f(x) = (1 - x_1)^2 + 100 * (x_2 - x_1^2)^2
function rosenbrock_cost(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_fdf!(g, H, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    g[2] = 200.0 * (x[2] - x[1]^2)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    return rosenbrock_cost(x)
end

# Starting point
x0 = @SVector [-1.2, 1.0]

# Optimize using Powell's Dogleg
options = OptimizationOptions(iterator = :dogleg)
x_opt, result = optimize(x0, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options)

println("Optimal solution: ", x_opt)
println("Final cost: ", result.bestcost)
println("Iterations: ", result.niterations)
```

## Threads and BLAS

Within one `optimize` or `optimize!` call, assembling the Hessian (your `fdf!`) and factorizing it run one after the other, so a multithreaded BLAS or Pardiso factorization doesn't compete with your own code. Running several optimizations at once on Julia threads (for example with `Threads.@threads` over independent problems) is different: each one's dense factorization then uses BLAS threads too, and the two thread pools oversubscribe the cores. In that case call `LinearAlgebra.BLAS.set_num_threads(1)` before starting the threads. The package never changes the BLAS thread count itself.
