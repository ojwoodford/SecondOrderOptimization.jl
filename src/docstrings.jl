"""
    OptimizationOptions(; kwargs...)

Stores configuration options and stopping criteria for second-order optimization.

# Keyword Arguments
- `maxiters::Int = 100`: Maximum number of outer iterations (linearization evaluations).
- `reldcost::Float64 = 1e-12`: Minimum relative change in cost in one iteration required to
  continue: the run stops, converged (`TERMINATION_REL_DCOST`), when an iteration changes the
  cost by less than `reldcost * abs(bestcost)`, a rise as well as a fall. It is kept tight, since
  a single iteration with a tiny change is not convergence (a run crawling across a flat valley
  has many); the slower test is `stalltol`. Not tested after a step the iterator restricted (a
  truncated or shifted dogleg step, a shifted Newton solve, an LM step whose damping outweighs
  the curvature along it), which is small because of the safeguard, not because of convergence.
- `absdcost::Float64 = 1e-15`: Minimum absolute change in cost, in either direction, in one
  iteration required to continue (converged, `TERMINATION_ABS_DCOST`).
- `dstep::Float64 = 1e-15`: Minimum infinity-norm of parameter step vector to continue. Like
  `reldcost`, not tested after a restricted step.
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
- `iterator::Symbol = :levenbergmarquardt`: Iterator algorithm (`:newton`, `:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`, `:gradientdescent`).
- `linsolver::Symbol = :auto`: Linear solver method: `:auto`, `:cholesky`, `:ldl` or `:qr`; `:cg` or `:minres` (Krylov solvers for a Hessian) and `:lsmr` (for a [`JacobianObjective`](@ref) only), via `Krylov.jl`; or `:pardiso`, via `Pardiso.jl`. Which of these an objective accepts is set by its prototype: see [`HessianObjective`](@ref) and the table of supported combinations in [`JacobianObjective`](@ref) (a dense Jacobian takes `:auto`, `:qr`, `:cholesky`, `:ldl` and `:lsmr`; a sparse one `:auto`, `:qr` and `:lsmr`).
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
- `iterator::Symbol`: The iterator used, as selected by the `iterator` option (for example `:dogleg`).
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
- `:cg`, `:minres`: Iterative Krylov solvers for a Hessian, provided by `Krylov.jl` via the `KrylovExt` extension.
- `:lsmr`: LSMR on the Jacobian of a [`JacobianObjective`](@ref), provided by `Krylov.jl` via the `KrylovExt` extension.
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
  Jacobian, and the Jacobian prototype or number of residuals; it supports every iterator
  (`:newton` is then Gauss-Newton) and the `linsolver` values its docstring lists for each
  prototype, and anything else is an `ArgumentError` before `f` or `resjac!` is called.
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


# Termination masks: bits of `OptimizationResult.termination`

"""
    TERMINATION_REL_DCOST

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when an iteration changed the cost, up or down, by less than `reldcost` relative to the best cost, after a step the iterator did not restrict (see `reldcost` in [`OptimizationOptions`](@ref)). A convergence criterion.
Test it with `result.termination & TERMINATION_REL_DCOST != 0`; see [`isconverged`](@ref).
"""
TERMINATION_REL_DCOST

"""
    TERMINATION_ABS_DCOST

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when an iteration changed the cost, up or down, by less than `absdcost`. A convergence criterion.
Test it with `result.termination & TERMINATION_ABS_DCOST != 0`; see [`isconverged`](@ref).
"""
TERMINATION_ABS_DCOST

"""
    TERMINATION_STEP_SIZE

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the step was below `dstep` (infinity norm) and the iterator did not restrict it (see `reldcost` in [`OptimizationOptions`](@ref)). A convergence criterion.
Test it with `result.termination & TERMINATION_STEP_SIZE != 0`; see [`isconverged`](@ref).
"""
TERMINATION_STEP_SIZE

"""
    TERMINATION_GRAD_TOL

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the gradient was below `gtol` (infinity norm). A convergence criterion.
Test it with `result.termination & TERMINATION_GRAD_TOL != 0`; see [`isconverged`](@ref).
"""
TERMINATION_GRAD_TOL

"""
    TERMINATION_STALLED

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the best cost improved by less than `stalltol` relative for more than `maxstall` iterations, at least one of them stayed within that band, and none rose by more than `sqrt(stalltol)`. A convergence criterion: progress has become negligible at the best point, which is not proof of a minimum.
Test it with `result.termination & TERMINATION_STALLED != 0`; see [`isconverged`](@ref).
"""
TERMINATION_STALLED

"""
    TERMINATION_MAX_FAILS

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when more than `maxstall` iterations passed without a significant improvement, and the cost did not settle: every one rose beyond the stall band, or one rose by more than `sqrt(stalltol)`. A limit, not convergence.
Test it with `result.termination & TERMINATION_MAX_FAILS != 0`; see [`isconverged`](@ref).
"""
TERMINATION_MAX_FAILS

"""
    TERMINATION_MAX_ITERS

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when `maxiters` iterations were reached. A limit, not convergence.
Test it with `result.termination & TERMINATION_MAX_ITERS != 0`; see [`isconverged`](@ref).
"""
TERMINATION_MAX_ITERS

"""
    TERMINATION_MAX_TIME

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when `maxtime` seconds were exceeded. A limit, not convergence.
Test it with `result.termination & TERMINATION_MAX_TIME != 0`; see [`isconverged`](@ref).
"""
TERMINATION_MAX_TIME

"""
    TERMINATION_COST_INF

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the cost was infinite. A failure.
Test it with `result.termination & TERMINATION_COST_INF != 0`; see [`isconverged`](@ref).
"""
TERMINATION_COST_INF

"""
    TERMINATION_COST_NAN

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the cost was `NaN`. A failure.
Test it with `result.termination & TERMINATION_COST_NAN != 0`; see [`isconverged`](@ref).
"""
TERMINATION_COST_NAN

"""
    TERMINATION_STEP_INF

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the step contained an infinite value. A failure.
Test it with `result.termination & TERMINATION_STEP_INF != 0`; see [`isconverged`](@ref).
"""
TERMINATION_STEP_INF

"""
    TERMINATION_STEP_NAN

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the step contained a `NaN`. A failure.
Test it with `result.termination & TERMINATION_STEP_NAN != 0`; see [`isconverged`](@ref).
"""
TERMINATION_STEP_NAN

"""
    TERMINATION_GRAD_INF

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the gradient contained an infinite value
(and no `NaN`). A failure. Test it with `result.termination & TERMINATION_GRAD_INF != 0`; see [`isconverged`](@ref).
"""
TERMINATION_GRAD_INF

"""
    TERMINATION_GRAD_NAN

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the gradient contained a `NaN`. A failure.
Test it with `result.termination & TERMINATION_GRAD_NAN != 0`; see [`isconverged`](@ref).
"""
TERMINATION_GRAD_NAN

"""
    TERMINATION_START_NOT_FINITE

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the cost or the gradient was not finite at the start point. A failure.
Test it with `result.termination & TERMINATION_START_NOT_FINITE != 0`; see [`isconverged`](@ref).
"""
TERMINATION_START_NOT_FINITE

"""
    TERMINATION_LINSOLVER_FAILED

Bit mask of [`OptimizationResult`](@ref)`.termination`, set when the linear solver failed. A failure.
Test it with `result.termination & TERMINATION_LINSOLVER_FAILED != 0`; see [`isconverged`](@ref).
"""
TERMINATION_LINSOLVER_FAILED
