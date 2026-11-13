# API Reference

## Primary Functions

```@docs
HessianObjective
optimize
optimize!
```

## Variable & Retraction Interface

```@docs
nvars
update
update!
```

## Options & Results

```@docs
OptimizationOptions
OptimizationResult
isconverged
termination_reasons
OptimizationWorkspace
OptimizationInternal
```

## Iteration Strategies

`SecondOrderOptimization.jl` supports multiple trust-region and line-search iteration strategies, passed via the `iterator` keyword of [`OptimizationOptions`](@ref):

```@docs
OptimizationIterator
newton
levenbergmarquardt
levenbergmarquardt_dtd
dogleg
gradientdescent
getiterator
NewtonData
NewtonDataRelinearize
DoglegData
LevenbergData
LevenbergDataTrial
MarquardtData
MarquardtDataTrial
GradientDescentData
```

## Linear Solvers

### Solver Strategy & Types

```@docs
OptimizationLinearSolver
auto_solver
cholesky_solver
ldl_solver
qr_solver
validatelinsolver
solve!
zero!
SolveStatus
Success
NotPositiveDefinite
Indefinite
```

### Extension Solvers

`SecondOrderOptimization.jl` integrates with optional linear solver packages via Julia 1.10+ package extensions:

- **`Krylov.jl` (via `KrylovExt`)**: Provides iterative Krylov subspace linear solvers.
  - `:cg`: Conjugate Gradients for symmetric positive-definite systems. Reports `Indefinite` on nonpositive curvature.
  - `:minres`: MINRES for symmetric indefinite systems.
  - On sparse systems, `:cg` and `:minres` are Jacobi (diagonal) preconditioned whenever the damped diagonal is positive.
  - `:krylov`: Alias for `:cg`.
  To enable: `using Krylov`.
- **`Pardiso.jl` (via `PardisoExt`)**: Provides multithreaded direct sparse factorization using Intel MKL PARDISO or Panua Pardiso.
  - `:pardiso`: Multi-threaded sparse direct solver for large sparse matrices. The symbolic analysis runs once, when the linear system is built; each solve runs only the positive-definite numerical factorization, and reports `NotPositiveDefinite` when it fails.
  To enable: `using Pardiso`.
- **`BlockLinearSystems.jl` (via `BlockLinearSystemsExt`)**: Schur-complement solvers for a `SchurBlockMatrix` Hessian.
  - `:schur`, `:dense_schur`, `:sparse_schur`.
  To enable: `using BlockLinearSystems`.

## Callbacks & Diagnostics

```@docs
CostTrajectory
storecostscallback
printoutcallback
nullcallback
```
