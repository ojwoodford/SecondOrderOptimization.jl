# API Reference

## Primary Functions

```@docs
HessianObjective
JacobianObjective
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
- **`BlockLinearSystems.jl` (via `BlockLinearSystemsExt`)**: Schur-complement elimination for a `SchurBlockMatrix` Hessian.
  The type of `H` selects the linear system, including elimination for a `SchurBlockMatrix`: there is no solver symbol or option for it.
  `linsolver` then names the solver for the reduced system: `:auto` or `:cholesky` (Cholesky, falling back to Bunch-Kaufman) or `:ldl` (Bunch-Kaufman); any other is an `ArgumentError` when the system is built.
  To enable: `using BlockLinearSystems`.

## Linear System Interface

The iterators drive any [`AbstractLinearSystem`](@ref) through the functions in its docstring. A new
system type, or a new kind of Hessian storage for the Schur systems, can be checked against that
contract with [`check_linear_system`](@ref), which needs a method of `conformance_matrix` and
`conformance_fill!` for the storage it uses.

A tall system meets the same interface without forming its Hessian. For a
[`JacobianObjective`](@ref), [`SecondOrderOptimization.DenseTallLinearSystem`](@ref) holds `J` and
`r`, and `A = JᵀJ` is implicit: `getA` is a [`SecondOrderOptimization.TallHessian`](@ref) wrapping
`J` and `r`, on which `fast_bAb(A, v)` is `‖J v‖²` and the diagonal is the squared column norms.
`b = Jᵀr` is the gradient. The iterators call `fdf!(b, A, x)` as for any system; for a
`JacobianObjective` that `fdf!` is an internal wrapper which calls `resjac!(r, J, x)` and forms
`b`. `solve!` factors `J` by QR, never `JᵀJ`, and solves the same damped normal equations
`(A + λD) x = b` as any other system, with `D` the identity or the Marquardt scaling (in the units
of `diag(JᵀJ)`). The column norms of `J` are computed once per linearization and serve both
`max_diag` and the tall system's `updatescaling!` method. Every iterator but `:gradientdescent`
runs on it unchanged; `:newton` is then Gauss-Newton.

```@docs
AbstractLinearSystem
SecondOrderOptimization.DenseTallLinearSystem
SecondOrderOptimization.TallHessian
usesstaticsystem
MAX_STATIC_SYSTEM
check_linear_system
SecondOrderOptimization.conformance_matrix
SecondOrderOptimization.conformance_fill!
SecondOrderOptimization.conformance_fillb!
SecondOrderOptimization.conformance_traits
```

## Callbacks & Diagnostics

```@docs
CostTrajectory
storecostscallback
printoutcallback
nullcallback
```
