# API Reference

## Primary Functions

```@docs
HessianObjective
JacobianObjective
optimize
optimize!
```

## Options & Results

```@docs
OptimizationOptions
OptimizationResult
isconverged
OptimizationWorkspace
```

Why a run stopped is in `result.termination`, a bitfield. Test it against the exported masks, for
example `result.termination & TERMINATION_GRAD_TOL != 0`, and use [`isconverged`](@ref) for whether
the run converged:

| Mask | Set when |
|:---|:---|
| `TERMINATION_REL_DCOST`, `TERMINATION_ABS_DCOST` | an iteration changed the cost, up or down, by less than `reldcost` (relative) or `absdcost` |
| `TERMINATION_STEP_SIZE` | the step was below `dstep` |
| `TERMINATION_GRAD_TOL` | the gradient was below `gtol` |
| `TERMINATION_STALLED` | the best cost improved by less than `stalltol` relative for more than `maxstall` iterations, at least one of them stayed within that band, and none rose by more than `sqrt(stalltol)` (converged: progress has become negligible at the best point, which is not proof of a minimum) |
| `TERMINATION_MAX_FAILS` | more than `maxstall` iterations without a significant improvement, and the cost did not settle: every one rose beyond the stall band, or one rose by more than `sqrt(stalltol)` |
| `TERMINATION_MAX_ITERS`, `TERMINATION_MAX_TIME` | `maxiters` or `maxtime` was reached |
| `TERMINATION_COST_INF`, `TERMINATION_COST_NAN` | the cost was not finite |
| `TERMINATION_STEP_INF`, `TERMINATION_STEP_NAN` | the step was not finite |
| `TERMINATION_GRAD_INF`, `TERMINATION_GRAD_NAN` | the gradient held an infinite value, or a `NaN` |
| `TERMINATION_START_NOT_FINITE` | the cost or the gradient at the start was not finite |
| `TERMINATION_LINSOLVER_FAILED` | the linear solver failed |

A callback's own flags are in `result.termination >> 16`.

```@docs
TERMINATION_REL_DCOST
TERMINATION_ABS_DCOST
TERMINATION_STEP_SIZE
TERMINATION_GRAD_TOL
TERMINATION_STALLED
TERMINATION_MAX_FAILS
TERMINATION_MAX_ITERS
TERMINATION_MAX_TIME
TERMINATION_COST_INF
TERMINATION_COST_NAN
TERMINATION_STEP_INF
TERMINATION_STEP_NAN
TERMINATION_GRAD_INF
TERMINATION_GRAD_NAN
TERMINATION_START_NOT_FINITE
TERMINATION_LINSOLVER_FAILED
```

## Iterators and Linear Solvers

An iterator is chosen by the `iterator` keyword of [`OptimizationOptions`](@ref), as a symbol: one of the four
built in, `:newton`, `:levenbergmarquardt`, `:levenbergmarquardt_dtd` or `:dogleg`, or one you define (gradient
descent is the example of [Writing Your Own Iterator](tutorials/custom_iterator.md)). A
linear solver is chosen by the `linsolver` keyword, also as a symbol: `:auto`, `:cholesky`, `:ldl`
or `:qr`, and those the extensions below add.

The linear systems and their solvers are those of
[DampedLinearSolve.jl](https://ojwoodford.github.io/DampedLinearSolve.jl/dev/), on which this package is built: its
documentation states the linear-system contract the iterators use, the solvers each system takes and
how `:auto` chooses, and how to implement the contract for your own storage. `OptimizationOptions`
validates `linsolver` by asking it.

```@docs
IteratorInterface
```

### Extension Solvers

Loading these packages adds solvers, through DampedLinearSolve's package extensions:

- **`Krylov.jl`**: Provides iterative Krylov subspace linear solvers. `:cg` is the Krylov solver for a Hessian.
  - `:cg`: Conjugate Gradients for symmetric positive-definite systems. Reports the system as indefinite on nonpositive curvature.
  - `:minres`: MINRES for symmetric indefinite systems.
  - On sparse systems, `:cg` and `:minres` are Jacobi (diagonal) preconditioned whenever the damped diagonal is positive.
  - `:lsmr`: LSMR on the Jacobian of a [`JacobianObjective`](@ref) (dense or sparse), with the damping as its regularization parameter. Rejected for a `HessianObjective`, where it would square the condition number of the normal equations again.
  To enable: `using Krylov`.
- **`Pardiso.jl`**: Provides multithreaded direct sparse factorization using Intel MKL PARDISO or Panua Pardiso.
  - `:pardiso`: Multi-threaded sparse direct solver for large sparse matrices. The symbolic analysis runs once, when the linear system is built; each solve runs only the positive-definite numerical factorization, and reports the system as not positive definite when it fails.
  To enable: `using Pardiso`.
- **`BlockLinearSystems.jl`** (which implements DampedLinearSolve's linear-system contract for its own systems): Schur-complement elimination for a `SchurBlockHessian` prototype.
  The type of `H` selects the linear system, including elimination for a `SchurBlockHessian`: there is no solver symbol or option for it.
  `linsolver` then names the solver for the reduced system: `:auto` or `:cholesky` (Cholesky, falling back to Bunch-Kaufman) or `:ldl` (Bunch-Kaufman); any other is an `ArgumentError` when the system is built.
  On the Jacobian path, a `SchurBlockJacobian{Float64, RType}` prototype for a [`JacobianObjective`](@ref) selects QR elimination of the eliminated blocks; `RType` (dense `Matrix{Float64}` or `SparseMatrixCSC{Float64, Int}`) is the storage of the reduced system, and `linsolver` names its solver (see the [`JacobianObjective`](@ref) docstring). With Krylov.jl also loaded (BlockLinearSystems' `DampedLinearSolveKrylovExt`), `:lsmr` solves the reduced problem by LSMR on the reduced Jacobian.
  The elimination kernels run threaded over the structure's chunks only when Polyester.jl is also loaded (BlockLinearSystems' `PolyesterExt`); otherwise they run serially. A warm run allocates nothing, except that on Julia 1.10 each threaded kernel call may allocate a little (up to 2 048 bytes per thread).
  To enable: `using BlockLinearSystems`.

## Callbacks & Diagnostics

```@docs
CostTrajectory
storecostscallback
printoutcallback
nullcallback
```
