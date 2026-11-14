"""
    OptimizationOptions(; kwargs...)

Stores configuration options and stopping criteria for second-order optimization.

# Keyword Arguments
- `maxiters::Int = 100`: Maximum number of outer iterations (linearization evaluations).
- `reldcost::Float64 = 1e-12`: Minimum relative change in cost in one iteration required to
  continue: the run stops, converged (`TERMINATION_REL_DCOST`), when an iteration changes the
  cost by less than `reldcost * abs(bestcost)`, a rise as well as a fall. It is kept tight, since
  a single iteration with a tiny change is not convergence (a run crawling across a flat valley
  has many); the slower test is `stalltol`.
- `absdcost::Float64 = 1e-15`: Minimum absolute change in cost, in either direction, in one
  iteration required to continue (converged, `TERMINATION_ABS_DCOST`).
- `dstep::Float64 = 1e-15`: Minimum infinity-norm of parameter step vector to continue.
- `gtol::Float64 = 0.0`: Gradient infinity-norm threshold for termination.
- `maxstall::Int = 3`: Maximum number of consecutive iterations in which the best cost does not
  improve significantly (by more than `stalltol` relative, counted from the last significant
  improvement; a rise in cost never improves it). After more than that the run stops: stalled
  (`TERMINATION_STALLED`, converged) if the cost has settled: at least one of those iterations
  left the cost within `stalltol` (relative) above the cost at the last significant
  improvement, and none raised it by more than `sqrt(stalltol)` (relative; the bound is computed
  from `stalltol` when the options are created). Otherwise `TERMINATION_MAX_FAILS` (a failure:
  the cost kept rising, or jumped).
- `stalltol::Float64 = 1.5e-8`: Relative improvement in the best cost that counts as significant
  for `maxstall`; about `sqrt(eps(Float64))`, the change in cost that rounding can produce near a
  minimum computed with cancellation. With `stalltol = 0`, `maxstall` counts consecutive
  iterations that raise the cost, as it did before `stalltol` existed. A stalled termination says
  that progress has become negligible at the best point found. It cannot tell a minimum from a
  flat valley crossed slowly: whether the step was held back by damping or a trust radius is
  known only to the iterator. Check the gradient (`gtol`) when that matters.
- `maxtime::Float64 = 30.0`: Maximum wall-clock optimization runtime in seconds.
- `iterator::Union{Symbol, OptimizationIterator} = :levenbergmarquardt`: Iterator algorithm (`:newton`, `:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`, `:gradientdescent`).
- `linsolver::Union{Symbol, OptimizationLinearSolver} = :auto`: Linear solver method (`:auto`, `:cholesky`, `:ldl`, `:qr`, or `:cg`, `:minres`, `:krylov` via `Krylov.jl`, or `:pardiso` via `Pardiso.jl`).
- `callbackchangesstate::Union{Bool, StaticBool} = false`: Set `true` if callback modifies problem state.

# Example
```julia
opts = OptimizationOptions(
    iterator = :dogleg,
    linsolver = :cholesky,
    maxiters = 50,
    gtol = 1e-6
)
```
"""
OptimizationOptions

"""
    OptimizationResult

Holds summary statistics and results from an optimization run.

# Fields
- `startcost::Float64`: Cost prior to optimization.
- `bestcost::Float64`: Final minimum cost achieved.
- `timetotal::UInt64`: Total optimization time in nanoseconds.
- `timeinit::UInt64`: Initialization time in nanoseconds.
- `timecost::UInt64`: Time spent evaluating cost function `f`.
- `timegradient::UInt64`: Time spent computing gradients and Hessians `fdf!`.
- `timesolver::UInt64`: Time spent solving linear systems.
- `niterations::Int`: Total outer iterations performed.
- `costcomputations::Int`: Total cost evaluations performed.
- `gradientcomputations::Int`: Total gradient/Hessian evaluations performed.
- `linearsolves::Int`: Total linear solves performed.
- `termination::Int`: Bitmask indicating termination reasons.
- `iterator::OptimizationIterator`: Algorithm used for optimization.
- `initallocations::Int`: Heap allocations during setup.
- `initallocated::Int`: Bytes allocated during setup.
- `optimizeallocations::Int`: Heap allocations during optimization loop.
- `optimizeallocated::Int`: Bytes allocated during optimization loop.

# Example
```julia
x_opt, result = optimize(x0, HessianObjective(f, fdf!))
println("Final cost: ", result.bestcost)
println("Iterations: ", result.niterations)
```
"""
OptimizationResult

"""
    OptimizationWorkspace()
    OptimizationWorkspace(x0, obj::HessianObjective, options=OptimizationOptions(); startstats=Stats())
    OptimizationWorkspace(x0, obj::JacobianObjective, options=OptimizationOptions(); startstats=Stats())

Preallocated cache for linear systems, solver factorizations, and trial variable buffers,
enabling zero-allocation repeated optimizations.

The cache is reused when the variable type, dimension and block layout, the options type and the
objective's structure all match: for a [`HessianObjective`](@ref) the Hessian's type and structure,
for a [`JacobianObjective`](@ref) the number of residuals `m`. A workspace built for one objective
kind is never reused for the other. Otherwise it is rebuilt (and, for a workspace whose type
parameters can't hold the new data, emptied, with a one-time warning). A reused workspace keeps
the Hessian matrix of the call that built it: see *The Hessian prototype* in
[`HessianObjective`](@ref). A workspace for a [`JacobianObjective`](@ref) instead owns its
Jacobian storage (a copy of the prototype, or zeros for an integer `m`), so a prototype matrix is
never written when a workspace is used, and a different prototype matrix of the same size does
not rebuild the workspace: at the start of each run, every cached Jacobian (the trial system's
too) is set to the contents of the prototype handed in (or to zeros, for an integer `m`), so the
run starts exactly as a fresh one would. `resjac!` then writes into the workspace's storage.

# Example
```julia
obj = HessianObjective(f, fdf!)
ws = OptimizationWorkspace(x0, obj, options)
for trial in 1:10
    x_opt, res = optimize(x0, obj, options; workspace=ws)
end
```
"""
OptimizationWorkspace

"""
    OptimizationInternal(linsystem, x0, startstats=Stats())

Internal state container holding current, trial (`varnext`), and best (`varbest`) variable buffers,
linear system `linsystem`, iteration count `iternum`, and timing/allocation statistics during optimization.

# Example
```julia
function my_callback(cost, x_trial, data::OptimizationInternal, iteratedata)
    println("Iter ", data.iternum, ": bestcost = ", data.bestcost, ", trial cost = ", cost)
    return (cost, 0)
end
```
"""
OptimizationInternal

"""
    OptimizationIterator

Enum indicating the optimization iterator algorithm:
- [`newton`](@ref): Standard Newton step without damping.
- [`levenbergmarquardt`](@ref): Levenberg-Marquardt with diagonal uniform scaling (`lambda * I`).
- [`levenbergmarquardt_dtd`](@ref): Marquardt scaling using running maximum diagonal entries (`lambda * D'D`).
- [`dogleg`](@ref): Powell's dogleg trust-region method interpolating Cauchy and Newton steps.
- [`gradientdescent`](@ref): Backtracking gradient descent.

# Example
```julia
opts = OptimizationOptions(iterator = levenbergmarquardt_dtd)
```
"""
OptimizationIterator

"""
    newton

[`OptimizationIterator`](@ref) constant selecting the undamped Newton method.
"""
newton

"""
    levenbergmarquardt

[`OptimizationIterator`](@ref) constant selecting Levenberg-Marquardt with uniform diagonal damping (`lambda * I`).
"""
levenbergmarquardt

"""
    levenbergmarquardt_dtd

[`OptimizationIterator`](@ref) constant selecting Levenberg-Marquardt with adaptive diagonal scaling (`lambda * D'D`).
"""
levenbergmarquardt_dtd

"""
    dogleg

[`OptimizationIterator`](@ref) constant selecting Powell's dogleg trust-region method.
"""
dogleg

"""
    gradientdescent

[`OptimizationIterator`](@ref) constant selecting backtracking gradient descent.
"""
gradientdescent

"""
    getiterator(options::OptimizationOptions) -> OptimizationIterator
    getiterator(s::Symbol) -> OptimizationIterator

Returns the [`OptimizationIterator`](@ref) enum value corresponding to `options.iterator` or symbol `s`.

# Example
```julia
it = getiterator(OptimizationOptions(iterator = :dogleg)) # dogleg
```
"""
getiterator

"""
    NewtonData

Iterator state for undamped Newton optimization when `callbackchangesstate` is `true`.
"""
NewtonData

"""
    NewtonDataRelinearize

Iterator state for undamped Newton optimization when `callbackchangesstate` is `false`,
relinearizing immediately at each trial point.
"""
NewtonDataRelinearize

"""
    DoglegData

Iterator state for Powell's dogleg trust-region optimization (Nocedal & Wright 2006 framework),
storing the active trust-region radius `trustradius`, maximum radius bound `maxtrustradius`,
and scratch buffers for the Cauchy and Newton steps.
"""
DoglegData

"""
    LevenbergData

Iterator state for Levenberg-Marquardt (`lambda * I`) when `callbackchangesstate` is `true`.
"""
LevenbergData

"""
    LevenbergDataTrial

Iterator state for Levenberg-Marquardt (`lambda * I`) when `callbackchangesstate` is `false`,
reusing trial linearizations on accepted first steps.
"""
LevenbergDataTrial

"""
    MarquardtData

Iterator state for Levenberg-Marquardt (`lambda * D'D`) when `callbackchangesstate` is `true`.
"""
MarquardtData

"""
    MarquardtDataTrial

Iterator state for Levenberg-Marquardt (`lambda * D'D`) when `callbackchangesstate` is `false`,
reusing trial linearizations on accepted first steps.
"""
MarquardtDataTrial

"""
    GradientDescentData

Iterator state for backtracking gradient descent, storing the current `stepsize`.
"""
GradientDescentData

"""
    OptimizationLinearSolver

Enum specifying linear solver strategy:
- [`auto_solver`](@ref): Cholesky for dense/static matrices, LDL for sparse matrices.
- [`cholesky_solver`](@ref): Cholesky factorization.
- [`ldl_solver`](@ref): LDL / Bunch-Kaufman factorization.
- [`qr_solver`](@ref): QR factorization.

Additional solver symbols can be passed as Symbols when package extensions are loaded:
- `:cg`, `:minres`, `:krylov`: Iterative solvers provided by `Krylov.jl` via the `KrylovExt` extension.
- `:pardiso`: Multithreaded direct sparse solver provided by `Pardiso.jl` via the `PardisoExt` extension.

# Example
```julia
opts = OptimizationOptions(linsolver = :ldl)
```
"""
OptimizationLinearSolver

"""
    auto_solver

[`OptimizationLinearSolver`](@ref) constant selecting automatic solver dispatch (Cholesky for dense/static, LDL for sparse).
"""
auto_solver

"""
    cholesky_solver

[`OptimizationLinearSolver`](@ref) constant selecting Cholesky factorization.
"""
cholesky_solver

"""
    ldl_solver

[`OptimizationLinearSolver`](@ref) constant selecting LDL / Bunch-Kaufman factorization.
"""
ldl_solver

"""
    qr_solver

[`OptimizationLinearSolver`](@ref) constant selecting QR factorization.
"""
qr_solver

"""
    SolveStatus

Enum returned by [`solve!`](@ref) indicating the status of the linear system solution:
- [`Success`](@ref): The linear system was solved successfully.
- [`NotPositiveDefinite`](@ref): The matrix was found not to be positive definite during Cholesky or LDL factorization.
- [`Indefinite`](@ref): The matrix was detected to be indefinite or negative curvature was encountered during solving.
"""
SolveStatus

"""
    Success

[`SolveStatus`](@ref) indicating that the linear system was solved successfully.
"""
Success

"""
    NotPositiveDefinite

[`SolveStatus`](@ref) indicating that the matrix failed a positive-definiteness test (e.g. non-positive pivot in Cholesky).
"""
NotPositiveDefinite

"""
    Indefinite

[`SolveStatus`](@ref) indicating that the matrix was detected to be indefinite or negative curvature was encountered.
"""
Indefinite

"""
    CostTrajectory()

Storage container for per-iteration costs, wall-clock timestamps, and step vectors.
Used with [`storecostscallback`](@ref).

# Example
```julia
trajectory = CostTrajectory()
x_opt, res = optimize(x0, HessianObjective(f, fdf!); callback=storecostscallback(trajectory))
```
"""
CostTrajectory

"""
    optimize!(x0, obj::HessianObjective, options=OptimizationOptions(); callback=nullcallback, workspace=nothing, startstats=Stats()) -> result
    optimize!(x0, obj::JacobianObjective, options=OptimizationOptions(); callback=nullcallback, workspace=nothing, startstats=Stats()) -> result

Minimizes the objective `obj` starting from `x0`, updating `x0` in place, and returns the
[`OptimizationResult`](@ref). On return `x0` holds the best point found.

`x0` must be a variable that is updated in place (`ismutablevar(x0)` from
OptimizationVariableInterface: e.g. a `Vector`, an `MVector`, or a `Vector` of static blocks).
For a `Number`, an `SVector` or any other immutable variable it throws an `ArgumentError`: use
[`optimize`](@ref), which returns the solution.

# Arguments
- `x0`: Starting point, overwritten with the solution.
- `obj`: The objective. A [`HessianObjective`](@ref) gives the cost, its gradient and Hessian, and
  the Hessian prototype. A [`JacobianObjective`](@ref) gives the cost, the residual and its
  Jacobian, and the Jacobian prototype or number of residuals; it supports every iterator but
  `:gradientdescent` (`:newton` is then Gauss-Newton) and `linsolver = :auto` or `:qr` so far,
  and anything else is an `ArgumentError` before `f` or `resjac!` is called.
- `options::OptimizationOptions`: Optimization settings and stopping criteria.

# Keyword Arguments
- `callback = nullcallback`: Callback invoked per iteration: `callback(cost, x_trial, data, iteratedata) -> (cost, terminate_flags)`.
- `workspace = nothing`: Optional [`OptimizationWorkspace`](@ref) for memory reuse.
- `startstats = Stats()`: Time and allocation counters at the start, for the timings in the result.
"""
optimize!

"""
    optimize(x0, obj::HessianObjective, options=OptimizationOptions(); callback=nullcallback, workspace=nothing, startstats=Stats()) -> (x, result)
    optimize(x0, obj::JacobianObjective, options=OptimizationOptions(); callback=nullcallback, workspace=nothing, startstats=Stats()) -> (x, result)

Minimizes the objective `obj` starting from `x0`, and returns the best point found and the
[`OptimizationResult`](@ref). `x0` is not modified: the optimization runs on `copyvars(x0)`.
Works for any variable type, including immutable ones (`Number`, `StaticVector`, custom
manifold types); for a mutable `x0`, [`optimize!`](@ref) avoids the copy.

Takes the same arguments as [`optimize!`](@ref), including either objective type.

# Example
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
x_opt, res = optimize(x0, HessianObjective(f, fdf!))

# The same problem as two residuals, r = [1 - x₁, 10(x₂ - x₁²)], with cost ½‖r‖²
function resjac!(r, J, x)
    r[1] = 1.0 - x[1]
    r[2] = 10.0 * (x[2] - x[1]^2)
    J[1, 1] = -1.0
    J[2, 1] = -20.0 * x[1]
    J[2, 2] = 10.0
    return 0.5 * (r[1]^2 + r[2]^2)
end
x_opt, res = optimize(x0, JacobianObjective(x -> 0.5 * f(x), resjac!, 2))
```
"""
optimize

"""
    zero!(ls)

Prepares the linear system `ls` for a fresh linearization: zeroes the gradient vector and the
Hessian matrix that `fdf!` accumulates into. For the tall system of a
[`JacobianObjective`](@ref), whose `resjac!` sets `J` and `r` rather than adding to them, it
only invalidates the cached factorization and column norms, and zeroes the gradient.

# Example
```julia
ls = SecondOrderOptimization.DenseLinearSystem(3)
zero!(ls)
```
"""
zero!

"""
    solve!(ls, solver=static(:auto)) -> SolveStatus
    solve!(x, A, b) -> SolveStatus

Solves the linear system `A * x = b` in-place using the specified `solver` and returns the [`SolveStatus`](@ref) enum (`Success`, `NotPositiveDefinite`, `Indefinite`).

# Supported Solver Identifiers
- Core solvers: `:auto`, `:cholesky`, `:ldl`, `:qr` (or enum constants [`auto_solver`](@ref), [`cholesky_solver`](@ref), [`ldl_solver`](@ref), [`qr_solver`](@ref)).
- Extension solvers (when package is loaded):
  - `using Krylov`: `:cg` (Conjugate Gradient), `:minres` (MINRES), `:krylov` (defaults to `:cg`). On sparse systems, `:cg` and `:minres` are Jacobi (diagonal) preconditioned whenever the damped diagonal is positive.
  - `using Pardiso`: `:pardiso` (Intel MKL PARDISO or Panua Pardiso multi-threaded direct sparse solver).

Note: On `StaticLinearSystem`, `:qr` executes static LU decomposition via `\`, and `:ldl` uses Cholesky, falling back to LU when the matrix is not positive definite (the solve is then reported as `Indefinite`).

# Example
```julia
x = zeros(2)
A = [2.0 0.0; 0.0 4.0]
b = [4.0, 8.0]
solve!(x, A, b)
```
"""
solve!

"""
    validatelinsolver(solver::Union{Symbol, StaticSymbol, OptimizationLinearSolver}) -> Bool

Returns `true` if `solver` is a supported linear solver backend, or `false` otherwise.
Extension solvers (`:cg`, `:minres`, `:krylov`, `:pardiso`) register validation methods when their respective packages (`Krylov.jl` or `Pardiso.jl`) are loaded.
"""
validatelinsolver

"""
    nullcallback(cost, x_trial, data, iteratedata) -> (cost, 0)

Default no-op callback incurring zero runtime overhead.

# Example
```julia
x_opt, res = optimize(x0, HessianObjective(f, fdf!); callback=nullcallback)
```
"""
nullcallback

"""
    printoutcallback(cost, x_trial, data, iteratedata) -> (cost, 0)

Callback that prints iteration count, cost, cost change, step norm, and trust-region radius to stdout.

# Example
```julia
x_opt, res = optimize(x0, HessianObjective(f, fdf!); callback=printoutcallback)
```
"""
printoutcallback

"""
    storecostscallback(store) -> Function

Constructs a callback function that logs iteration metrics into `store`,
where `store` is either a `Vector{Float64}` or a [`CostTrajectory`](@ref).

# Example
```julia
costs = Float64[]
x_opt, res = optimize(x0, HessianObjective(f, fdf!); callback=storecostscallback(costs))
```
"""
storecostscallback

"""
    AbstractLinearSystem

Supertype of the linear systems the iterators solve: the Hessian `A` and gradient `b` that `fdf!`
fills, the step `x`, and the damping to apply. The built-in systems are `StaticLinearSystem`,
`DenseLinearSystem`, `SparseLinearSystem` and [`DenseTallLinearSystem`](@ref) (for a
[`JacobianObjective`](@ref): `A = JᵀJ` held implicitly as `J` and `r`, `b = Jᵀr`, and the
iterators' `fdf!` a wrapper that calls `resjac!` and forms `b`), and, with BlockLinearSystems.jl loaded,
`DenseSchurLinearSystem` and `SparseSchurLinearSystem`. A new system type must provide the functions
below (those not exported are extended as `SecondOrderOptimization.name`), and should pass
[`check_linear_system`](@ref).

# Linear system interface

| Function | Contract |
|:---|:---|
| `makelinearsystem(x0, H, linsolver)` | Builds the system from the Hessian prototype `H` (see below for which type each `H` gives). `linsolver` is used to preallocate a factorization or Krylov workspace; a sparse system must then be solved with that same solver. |
| `getA(ls)`, `getb(ls)`, `getx(ls)` | `A` is the Hessian storage and `b` the gradient vector that `fdf!(b, A, x)` fills; `x` is the vector `solve!` writes. `b` and `x` have length `n`. The iterators read and negate `x`, and dogleg writes its own step into it. `A` must support `fast_bAb(A, v)` for `v` of the type of `b`. |
| `zero!(ls)` | Prepares the system for a fresh linearization, keeping its structure (pattern, workspaces): storage that `fdf!` accumulates into (`A`, `b`) is zeroed. A tall system only invalidates its factorization and column norms (and zeroes `b`), since `resjac!` overwrites `J` and `r`. |
| `dampen!(ls, λ)`, `dampen!(ls, λ, s)` | Records damping without changing `A`: `solve!` then solves with `A + λI`, or `A + λ Diagonal(s)` (`s` in the units of `diag(A)`, not squared). Each call replaces the damping set before, including the scaling; `dampen!(ls, λ, nothing)` is `dampen!(ls, λ)`, and `λ = 0` is undamped. |
| `solve!(ls, linsolver) -> SolveStatus` | Solves the damped system `A x = b` into `getx(ls)` (the step is `-x`; the iterators negate it). Returns `Success` when `x` solves it; `NotPositiveDefinite` when the factorization failed (`x` is then unspecified); `Indefinite` when it solved but found `A` indefinite (from `:ldl`, and CG's nonpositive curvature). The iterators treat anything but `Success` as failure: Newton shifts the diagonal, Levenberg-Marquardt raises `λ`, and dogleg shifts the diagonal as Newton does, taking the Cauchy step only if every shift fails. Leaves `A` and `b` exactly as written, damped or not, whether the solve succeeds or fails. `linsolver` may be a `Symbol`, a `StaticSymbol` or an `OptimizationLinearSolver`. |
| `copy_structure(ls)` | A new system with the same structure and its own `A` and `b` (contents unspecified). It may share `x` and factorization workspaces with `ls`, provided each `solve!` is self-contained, so that solving one never changes the other's next solution. |
| `sharex(ls, from)` | `ls`, or a copy of it sharing its `A` and `b`, whose `getx` is `getx(from)`. |
| `SecondOrderOptimization.max_diag(ls)` | The largest absolute diagonal entry of `A`; `NaN` if a diagonal entry is `NaN` (as `max` propagates it). |
| `SecondOrderOptimization.dampingfloor(ls)` | The smallest damping the solver resolves, relative to the diagonal: `eps(Float64)` by default (Cholesky of `A + λD`), `eps(Float64)^2` for the tall system (QR of `J`). Levenberg-Marquardt floors its damping at `min(dampingfloor(ls) * max_diag(ls), 1e-12)` for `:levenbergmarquardt` (1e-12 if `max_diag` is not finite or is zero), and at `min(dampingfloor(ls), 1e-12)` for `:levenbergmarquardt_dtd`, whose `λ` multiplies the scaled diagonal: the floor can be lower than the former constant 1e-12, never higher. |
| `SecondOrderOptimization.updatescaling!(lmd, ls)` | For `:levenbergmarquardt_dtd`: raises `lmd.scaling` to the running maximum of `abs.(diag(A))`, with a relative floor. The generic method reads `A[i, i]`; a system whose `A` can't be indexed needs its own method (the sparse, Schur and tall systems have one). |
| `SecondOrderOptimization.refresh_diag_cache!(ls)` | Called by the iterators after `fdf!` writes `A`. A no-op by default; no built-in system needs it for correctness. |
| `quadratic_model_change(ls, dx)` | `dx' A dx / 2 + b' dx`, undamped. The generic method uses `fast_bAb(getA(ls), dx)` and `fast_dot(getb(ls), dx)`. |

Which system `makelinearsystem` builds:
- `H = nothing`: a static system when `nvars(x0)` is a static size (a `Number`, a `StaticVector`,
  a tuple of them) of at most [`MAX_STATIC_SYSTEM`](@ref) (16); otherwise a dense system of size
  `nvars(x0)`. [`usesstaticsystem`](@ref)`(x0)` says which.
- `H::MMatrix`: a static system that writes into `H`; any other `StaticMatrix`: a static system of
  its own.
- `H::Matrix{Float64}`: a dense system that writes into `H`; any other `AbstractMatrix`: a dense
  system on a `Float64` copy.
- `H::SparseMatrixCSC{Float64}`: a sparse system that writes into `H`, or into a copy with explicit
  zeros added if a diagonal entry is missing.
- `H::SchurBlockMatrix` (BlockLinearSystems.jl): a Schur-complement system, dense or sparse
  according to the storage of `H.R`; `linsolver` (`:auto`, `:cholesky` or `:ldl`) names the solver
  for the reduced system.
"""
AbstractLinearSystem
