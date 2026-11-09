# API Reference

## Primary Functions

```@docs
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
  - `:cg`: Conjugate Gradients for symmetric positive-definite systems.
  - `:minres`: MINRES for symmetric indefinite systems.
  - `:lsmr`: LSMR for least-squares and general systems.
  - `:krylov`: Alias for `:cg`.
  To enable: `using Krylov`.
- **`Pardiso.jl` (via `PardisoExt`)**: Provides multithreaded direct sparse factorization using Intel MKL PARDISO or Panua Pardiso.
  - `:pardiso`: Multi-threaded sparse direct solver for large sparse matrices.
  To enable: `using Pardiso`.

## Callbacks & Diagnostics

```@docs
CostTrajectory
storecostscallback
printoutcallback
nullcallback
```
