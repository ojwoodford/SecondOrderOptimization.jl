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
- `iterator::Symbol = :levenbergmarquardt`: Iterator algorithm: one of the four built in (`:newton`, `:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`), or one defined through [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface) (the manual's tutorial writes a gradient descent, `:gradientdescent`).
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

Iterator state for undamped Newton optimization: whether the last step needed a diagonal shift, and, when
`callbackchangesstate` is `false` (Newton then linearizes at each trial point at once, since it always moves there),
the cost at the trial point.
"""
NewtonData

"""
    DoglegData

Iterator state for Powell's dogleg trust-region optimization (Nocedal & Wright 2006 framework),
storing the active trust-region radius `trustradius`, maximum radius bound `maxtrustradius`,
scratch buffers for the Cauchy and Newton steps, and a `TrialLinearization` (with a spare system when
`callbackchangesstate` is `false`).
"""
DoglegData

"""
    LevenbergData

Iterator state for Levenberg-Marquardt (`lambda * I`): the damping, and a `TrialLinearization` (with a spare system,
for reusing the linearization at an accepted first trial, when `callbackchangesstate` is `false`).
"""
LevenbergData

"""
    MarquardtData

Iterator state for Levenberg-Marquardt (`lambda * D'D`): the damping, the scaling `D'D` and its work vector, and a
`TrialLinearization` (with a spare system, for reusing the linearization at an accepted first trial, when
`callbackchangesstate` is `false`).
"""
MarquardtData

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

# The iterator interface (`SecondOrderOptimization.IteratorInterface`): the methods an iterator defines

"""
    iteratorname(::StaticSymbol{:name}) -> String

The display name of the iterator selected by `iterator = :name` (`show(result)` prints it). Required: defining it
registers the symbol, so that `OptimizationOptions(iterator = :name)` accepts it. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
iteratorname

"""
    construct(::StaticSymbol{:name}, ls, callbackchangesstate::StaticBool) -> itdata

Returns the data of the iterator `:name` for a run on the linear system `ls`: the state it keeps between
iterations. Required, one method per iterator. It may call [`similarstep`](@ref)`(ls)` for work vectors, and
`TrialLinearization(ls, callbackchangesstate)`, and nothing else of `ls`. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
construct

"""
    reset!(itdata)

Restores the iterator's data to the state [`construct`](@ref) returned, so that a run with a reused
[`OptimizationWorkspace`](@ref) repeats a first run. Required. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
reset!

"""
    iterate!(itdata, data, f, fdf!, options) -> Float64

One iteration. Sets the trial point ([`settrial!`](@ref)) and leaves the step in [`getstep`](@ref)`(data)`, and returns
the cost at the trial point; or [`LINSOLVER_FAILED`](@ref) when no solve succeeded; or `NaN` when no finite step could be
formed. `f` is the objective's cost function and `fdf!` its linearization, to be handed to the functions that evaluate
them ([`trialcost!`](@ref), [`linearizetrial!`](@ref)); `options` is the run's [`OptimizationOptions`](@ref), of which
an iterator reads `options.dstep` and `options.reldcost`, and hands `options` itself to [`solvestep!`](@ref).

The loop moves to the returned point even if its cost is higher, keeping the best point itself; to reject a step, retry
inside `iterate!`. Required. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
iterate!

"""
    restrictedstep(itdata) -> Bool

Whether the step last returned was held back by the iterator's safeguard (damping, a trust radius, a shifted solve)
rather than being its model's minimiser. After an accepted restricted step the loop does not test `reldcost` or
`dstep`, since the step is small because of the safeguard. Optional; the default is `false`. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
restrictedstep

"""
    triallinearization(itdata) -> Union{TrialLinearization, Nothing}

The iterator's [`TrialLinearization`](@ref), if it holds one, so that the default [`linearize!`](@ref) can swap in the
linearization it made at an accepted trial point, and so that a run with a reused workspace can prepare its spare
system. Optional; the default is `nothing`. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
triallinearization

"""
    linearize!(itdata, data, fdf!) -> Float64

Linearizes the objective at the current point, the point the loop moved to, and returns the cost there. Optional:
the default swaps in a trial linearization the iterator has ready (see `triallinearization`), and otherwise
evaluates `fdf!` at the current point. Override it only to do something else, for example to return a cost the
iterator has already computed by [`linearizetrial!`](@ref). Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
linearize!

# The functions an iterator calls on the loop's state

"""
    solvestep!(data, options) -> Bool
    solvestep!(data, options, λ)
    solvestep!(data, options, λ, scaling)

Solves the linear system for the step with the run's linear solver, `options.linsolver`. The two-argument form solves
the system as it is currently damped; the others first set the damping to `λ` (a multiple of the identity), or to `λ`
times the diagonal `scaling` (Marquardt's form), which stays set until it is changed. On success writes the step,
the negated solution, into [`getstep`](@ref)`(data)` and returns `true`; otherwise returns `false`, and the step's
contents are undefined. Each call counts as one linear solve in the result, and is timed. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
solvestep!

"""
    undamp!(data)

Sets the system's damping to zero, so that [`predictedchange`](@ref) and [`curvature`](@ref) use the undamped model.
The loop undamps after each iteration. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
undamp!

"""
    systemisfinite(data) -> Bool

Whether every entry of the system's matrix is finite. A failed solve of a non-finite matrix cannot be repaired by
damping, so an iterator stops retrying. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
systemisfinite

"""
    systemeltype(data) -> Type

The element type of the linear system's matrix (`Float64` or `Float32`). The step, the gradient and the costs are
`Float64` whatever it is. A method of DampedLinearSolve's `systemeltype`, re-exported by
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
systemeltype(::OptimizationInternal)

"""
    dampingfloor(data) -> Float64

The linear solver's smallest useful relative damping: a positive-definite matrix damped by
`dampingfloor(data) * maxdiagonal(data)` solves. A method of DampedLinearSolve's `dampingfloor`, re-exported by
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
dampingfloor(::OptimizationInternal)

"""
    getgradient(data) -> AbstractVector{Float64}

The gradient at the current point. Read-only for an iterator. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
getgradient

"""
    getstep(data) -> AbstractVector{Float64}

The step storage, which [`solvestep!`](@ref) writes. An iterator leaves the step it took here: the loop reads it after
[`iterate!`](@ref). Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
getstep

"""
    similarstep(ls) -> AbstractVector{Float64}

A new work vector like the step, for [`construct`](@ref) to allocate the iterator's data. Its contents are undefined.
Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
similarstep

"""
    predictedchange(data, s) -> Float64

The change in cost the quadratic model predicts for the step `s`, `gᵀs + ½ sᵀHs`, on the undamped system (call
[`undamp!`](@ref) first if the system is damped). Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
predictedchange

"""
    curvature(data, v) -> Float64

The curvature of the model along `v`, `vᵀHv` (`‖Jv‖²` for a Jacobian system), undamped. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
curvature

"""
    maxdiagonal(data) -> Float64

The largest absolute diagonal entry of the system's matrix (of `JᵀJ` for a Jacobian system), or `NaN` if a diagonal
entry is `NaN`. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
maxdiagonal

"""
    getdiagonal!(out, data) -> out

Writes the undamped diagonal of the system's matrix (the squared column norms of `J` for a Jacobian system) into
`out`, a vector like the step, and returns it. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
getdiagonal!

"""
    getbestcost(data) -> Float64

The lowest cost the run has reached. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
getbestcost

"""
    settrial!(data)
    settrial!(data, s)

Sets the trial point to the current point updated by the step `s`, [`getstep`](@ref)`(data)` by default. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
settrial!

"""
    trialcost!(data, f) -> Float64
    trialcost!(tl::TrialLinearization, data, f, fdf!, likely::Bool) -> Float64

The cost at the trial point. The first form evaluates `f`, counted and timed as a cost evaluation. The second, for an
iterator that holds a [`TrialLinearization`](@ref) `tl`: if `likely` (the iterator expects to accept the trial point)
and `tl` holds a spare system, the cost is computed by `fdf!`, linearizing at the trial point into the spare (counted
and timed as a gradient evaluation), which is then ready for the next [`linearize!`](@ref); otherwise by `f`, as the
first form. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
trialcost!

"""
    TrialLinearization(ls, callbackchangesstate::StaticBool)

A component for an iterator's data that saves an evaluation: when the iterator expects its trial point to be
accepted it linearizes there ([`trialcost!`](@ref) with `likely = true`), into a spare system, while computing the
cost; the default [`linearize!`](@ref) then swaps that system in instead of linearizing again. Built in
[`construct`](@ref), from its arguments. With `callbackchangesstate` true it holds no spare (the callback may change
the point or the cost after the trial) and always evaluates `f`. The iterator returns it from
[`triallinearization`](@ref), resets it in its [`reset!`](@ref) (`reset!(tl)`), and discards a trial it will not
accept with [`discardtrial!`](@ref). Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
TrialLinearization

"""
    discardtrial!(tl::TrialLinearization)

Marks the spare linearization as not ready: for a trial the iterator rejects, after a failed solve, and at the start
of [`iterate!`](@ref). Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
discardtrial!

"""
    linearizetrial!(data, fdf!) -> Float64

Linearizes the objective at the trial point into the current system, in place, and returns the cost there. Counted
and timed as a gradient evaluation. For an iterator that always moves to its trial point: it then returns that cost
from its own [`linearize!`](@ref) method. Part of [`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
linearizetrial!

"""
    LINSOLVER_FAILED

The value [`iterate!`](@ref) returns when no linear solve succeeded: a `NaN` with its own bit pattern, which the loop
recognises (compared with `===`) and stops on, with `TERMINATION_LINSOLVER_FAILED`. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
LINSOLVER_FAILED
