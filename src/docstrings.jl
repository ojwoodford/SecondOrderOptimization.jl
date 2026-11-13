"""
    OptimizationOptions(; kwargs...)

Stores configuration options and stopping criteria for second-order optimization.

# Keyword Arguments
- `maxiters::Int = 100`: Maximum number of outer iterations (linearization evaluations).
- `reldcost::Float64 = 1e-15`: Minimum relative decrease in cost required to continue.
- `absdcost::Float64 = 1e-15`: Minimum absolute decrease in cost required to continue.
- `dstep::Float64 = 1e-15`: Minimum infinity-norm of parameter step vector to continue.
- `gtol::Float64 = 0.0`: Gradient infinity-norm threshold for termination.
- `maxfails::Int = 3`: Maximum consecutive failed steps (cost increases) before termination.
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

Preallocated cache for linear systems, solver factorizations, and trial variable buffers,
enabling zero-allocation repeated optimizations.

The cache is reused when the variable type, dimension and block layout, the options type and the
Hessian's type and structure all match; otherwise it is rebuilt (and, for a workspace whose type
parameters can't hold the new data, emptied, with a one-time warning). A reused workspace keeps
the Hessian matrix of the call that built it: see *The Hessian prototype* in [`HessianObjective`](@ref).

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

Minimizes the objective `obj` starting from `x0`, updating `x0` in place, and returns the
[`OptimizationResult`](@ref). On return `x0` holds the best point found.

`x0` must be a variable that is updated in place (`ismutablevar(x0)` from
OptimizationVariableInterface: e.g. a `Vector`, an `MVector`, or a `Vector` of static blocks).
For a `Number`, an `SVector` or any other immutable variable it throws an `ArgumentError`: use
[`optimize`](@ref), which returns the solution.

# Arguments
- `x0`: Starting point, overwritten with the solution.
- `obj::HessianObjective`: The cost, its gradient and Hessian, and the Hessian prototype; see [`HessianObjective`](@ref).
- `options::OptimizationOptions`: Optimization settings and stopping criteria.

# Keyword Arguments
- `callback = nullcallback`: Callback invoked per iteration: `callback(cost, x_trial, data, iteratedata) -> (cost, terminate_flags)`.
- `workspace = nothing`: Optional [`OptimizationWorkspace`](@ref) for memory reuse.
- `startstats = Stats()`: Time and allocation counters at the start, for the timings in the result.
"""
optimize!

"""
    optimize(x0, obj::HessianObjective, options=OptimizationOptions(); callback=nullcallback, workspace=nothing, startstats=Stats()) -> (x, result)

Minimizes the objective `obj` starting from `x0`, and returns the best point found and the
[`OptimizationResult`](@ref). `x0` is not modified: the optimization runs on `copyvars(x0)`.
Works for any variable type, including immutable ones (`Number`, `StaticVector`, custom
manifold types); for a mutable `x0`, [`optimize!`](@ref) avoids the copy.

Takes the same arguments as [`optimize!`](@ref).

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
```
"""
optimize

"""
    zero!(ls)

Zeroes the gradient vector and Hessian matrix in linear system `ls`.

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
