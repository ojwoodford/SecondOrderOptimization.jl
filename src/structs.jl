using SparseArrays, Static
import Printf.@printf

@enum OptimizationIterator newton levenbergmarquardt dogleg gradientdescent levenbergmarquardt_dtd

struct OptimizationOptions{S, L, B <: StaticBool}
    reldcost::Float64           # Minimum relative reduction in cost required to avoid termination
    absdcost::Float64           # Minimum absolute reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    gtol::Float64               # Minimum L-infinity norm of the gradient required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than current best before termination
    maxiters::Int               # Maximum number of outer iterations
    maxtime::UInt64             # Maximum optimization time allowed, in nanoseconds
    iterator::StaticSymbol{S}   # Static Symbol of the inner iterator (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
    linsolver::StaticSymbol{L}  # Static Symbol of the linear solver (:auto, :cholesky, :ldl, :qr)
    callbackchangesstate::B     # True if callback can modify problem state/cost

    function OptimizationOptions(reldcost, absdcost, dstep, gtol, maxfails, maxiters, maxtime, iterator::StaticSymbol{S}, linsolver::StaticSymbol{L}, callbackchangesstate::B) where {S, L, B <: StaticBool}
        return new{S, L, B}(reldcost, absdcost, dstep, gtol, maxfails, maxiters, maxtime, iterator, linsolver, callbackchangesstate)
    end
end

makestaticbool(x::Union{Bool, StaticBool}) = static(x)
makestaticbool(::Type{True}) = True()
makestaticbool(::Type{False}) = False()

# Termination bitmask constants
const TERMINATION_COST_INF          = 1 << 0
const TERMINATION_COST_NAN          = 1 << 1
const TERMINATION_REL_DCOST         = 1 << 2
const TERMINATION_ABS_DCOST         = 1 << 3
const TERMINATION_STEP_INF          = 1 << 4
const TERMINATION_STEP_NAN          = 1 << 5
const TERMINATION_STEP_SIZE         = 1 << 6
const TERMINATION_MAX_FAILS         = 1 << 7
const TERMINATION_MAX_ITERS         = 1 << 8
const TERMINATION_MAX_TIME          = 1 << 9
const TERMINATION_COST_DISCREPANCY  = 1 << 10
const TERMINATION_START_NOT_FINITE  = 1 << 11
const TERMINATION_GRAD_NOT_FINITE   = 1 << 12
const TERMINATION_GRAD_TOL          = 1 << 13
const TERMINATION_LINSOLVER_FAILED  = 1 << 14

const SUCCESS_CONVERGENCE_MASK = TERMINATION_REL_DCOST | TERMINATION_ABS_DCOST | TERMINATION_STEP_SIZE | TERMINATION_GRAD_TOL
const FAILURE_TERMINATION_MASK = TERMINATION_COST_INF | TERMINATION_COST_NAN | TERMINATION_STEP_INF | TERMINATION_STEP_NAN | TERMINATION_MAX_FAILS | TERMINATION_MAX_ITERS | TERMINATION_MAX_TIME | TERMINATION_COST_DISCREPANCY | TERMINATION_START_NOT_FINITE | TERMINATION_GRAD_NOT_FINITE | TERMINATION_LINSOLVER_FAILED

const LINSOLVER_FAILED = reinterpret(Float64, 0x7ff8000000000001)

function OptimizationOptions(;
    maxiters=100,
    reldcost=1.e-15,
    absdcost=1.e-15,
    dstep=1.e-15,
    gtol=0.0,
    maxfails=3,
    maxtime=30.0,
    iterator=:levenbergmarquardt,
    linsolver=:auto,
    callback=nothing,
    iteratordata=nothing,
    callbackchangesstate=False()
)
    isnothing(callback) || throw(ArgumentError("Callbacks should be passed directly to optimize, not to the options struct."))
    isnothing(iteratordata) || throw(ArgumentError("Iteratordata should not be passed to the options struct."))
    iter_sym = Symbol(iterator)
    getiterator(iter_sym) # validates iterator symbol
    linsolv_sym = linsolver_symbol(linsolver)
    checklinsolver(linsolv_sym) # validates linear solver symbol
    maxtime_ns = isfinite(maxtime) ? UInt64(round(max(maxtime, 0.0) * 1e9)) : typemax(UInt64)
    return OptimizationOptions(
        Float64(reldcost), Float64(absdcost), Float64(dstep), Float64(gtol),
        Int(maxfails), Int(maxiters), maxtime_ns,
        static(iter_sym), static(linsolv_sym),
        makestaticbool(callbackchangesstate)
    )
end

getiterator(options::OptimizationOptions) = getiterator(options.iterator)
getiterator(s::StaticSymbol) = getiterator(dynamic(s))
function getiterator(s::Symbol)::OptimizationIterator
    s === :newton && return newton
    s === :levenbergmarquardt && return levenbergmarquardt
    s === :levenbergmarquardt_dtd && return levenbergmarquardt_dtd
    s === :dogleg && return dogleg
    s === :gradientdescent && return gradientdescent
    throw(ArgumentError("Unknown iterator: $s"))
end

struct OptimizationResult
    # Costs
    startcost::Float64                    # The function cost prior to minimization
    bestcost::Float64                     # The lowest function cost achieved
    # Times (nanoseconds)
    timetotal::UInt64                     # Total time taken to run the optimization
    timeinit::UInt64                      # Time to initialize the internal data structures
    timecost::UInt64                      # Time spent computing the cost
    timegradient::UInt64                  # Time spent computing gradients and Hessians
    timesolver::UInt64                    # Time spent solving the linear problems
    # Counts
    niterations::Int                      # Number of outer optimization iterations performed
    costcomputations::Int                 # Number of cost computations performed
    gradientcomputations::Int             # Number of gradient/Hessian computations performed
    linearsolves::Int                     # Number of linear solves performed
    # Termination reason
    termination::Int                      # Bitmask indicating which termination criteria were met
    # Iterator type
    iterator::OptimizationIterator        # The iterator that was used
    # Allocations
    initallocations::Int                  # Number of separate allocations during initialization
    initallocated::Int                    # Total bytes allocated during initialization
    optimizeallocations::Int              # Number of separate allocations during optimization
    optimizeallocated::Int                # Total bytes allocated during optimization
end

OptimizationResult(data, termination, iterator::OptimizationIterator) = OptimizationResult(
    data.startcost,
    data.bestcost,
    data.optimize.time_ns - data.start.time_ns,
    data.init.time_ns - data.start.time_ns,
    data.costs.time_ns,
    data.gradients.time_ns,
    data.solves.time_ns,
    data.iternum,
    data.costs.count,
    data.gradients.count,
    data.solves.count,
    termination,
    iterator,
    data.init.num_allocs - data.start.num_allocs,
    data.init.bytes_allocd - data.start.bytes_allocd,
    data.optimize.num_allocs - data.init.num_allocs,
    data.optimize.bytes_allocd - data.init.bytes_allocd
)
OptimizationResult(data, termination, iterator_sym::Symbol) = OptimizationResult(data, termination, getiterator(iterator_sym))

Base.:+(a::OptimizationResult, b::OptimizationResult) = OptimizationResult(
    a.startcost + b.startcost,
    a.bestcost + b.bestcost,
    a.timetotal + b.timetotal,
    a.timeinit + b.timeinit,
    a.timecost + b.timecost,
    a.timegradient + b.timegradient,
    a.timesolver + b.timesolver,
    a.niterations + b.niterations,
    a.costcomputations + b.costcomputations,
    a.gradientcomputations + b.gradientcomputations,
    a.linearsolves + b.linearsolves,
    a.termination | b.termination,
    a.iterator,
    a.initallocations + b.initallocations,
    a.initallocated + b.initallocated,
    a.optimizeallocations + b.optimizeallocations,
    a.optimizeallocated + b.optimizeallocated
)

"""
    isconverged(x::OptimizationResult) -> Bool

Returns `true` if the optimization terminated due to satisfying at least one convergence criterion
(`gtol`, `reldcost`, `absdcost`, or `dstep`) without encountering failure conditions (such as non-finite
costs, NaN steps, max outer iterations, max runtime, or cost calculation discrepancies).
"""
function isconverged(x::OptimizationResult)
    t = x.termination
    return ((t & TERMINATION_GRAD_TOL) != 0) || (((t & SUCCESS_CONVERGENCE_MASK) != 0) && ((t & FAILURE_TERMINATION_MASK) == 0))
end

function Base.show(io::IO, x::OptimizationResult)
    optimtime = x.timetotal - x.timeinit
    timedoptim = max(x.timecost + x.timegradient + x.timesolver, optimtime)
    cost_change_pct = (x.startcost != 0.0 && isfinite(x.startcost) && isfinite(x.bestcost)) ?
        100.0 * (x.startcost - x.bestcost) / abs(x.startcost) : 0.0
    @printf(io, "SecondOrderOptimization optimization using %s took %s and %d iterations to change the cost from %e to %e (%.2f%% change), requiring:\n",
        String(x.iterator),
        timestring(x.timetotal), x.niterations, x.startcost, x.bestcost, cost_change_pct)
    @printf(io, "   %s for initialization (%.2f%% of total time, with %d allocations totalling %s), and\n",
        timestring(x.timeinit), 100.0 * x.timeinit / x.timetotal, x.initallocations, bytesstring(x.initallocated))
    @printf(io, "   %s for optimization (%.2f%% of total time, with %d allocations totalling %s), of which:\n",
        timestring(optimtime), 100.0 * optimtime / x.timetotal, x.optimizeallocations, bytesstring(x.optimizeallocated))
    @printf(io, "        %d cost computations accounted for %.2f%% of the time,\n",
        x.costcomputations, 100.0 * x.timecost / timedoptim)
    @printf(io, "        %d gradient computations accounted for %.2f%%, and\n",
        x.gradientcomputations, 100.0 * x.timegradient / timedoptim)
    @printf(io, "        %d linear solver computations accounted for %.2f%%.\n",
        x.linearsolves, 100.0 * x.timesolver / timedoptim)
    if 0 != x.termination;            println(io, "Reason(s) for termination:"); end
    if 0 != x.termination & TERMINATION_COST_INF;         println(io, "   Cost is infinite."); end
    if 0 != x.termination & TERMINATION_COST_NAN;         println(io, "   Cost is NaN."); end
    if 0 != x.termination & TERMINATION_REL_DCOST;        println(io, "   Relative decrease in cost below threshold."); end
    if 0 != x.termination & TERMINATION_ABS_DCOST;        println(io, "   Absolute decrease in cost below threshold."); end
    if 0 != x.termination & TERMINATION_STEP_INF;         println(io, "   Step contains an infinite value."); end
    if 0 != x.termination & TERMINATION_STEP_NAN;         println(io, "   Step contains a NaN."); end
    if 0 != x.termination & TERMINATION_STEP_SIZE;        println(io, "   Step size below threshold."); end
    if 0 != x.termination & TERMINATION_MAX_FAILS;        println(io, "   Too many consecutive iterations increasing the cost."); end
    if 0 != x.termination & TERMINATION_MAX_ITERS;        println(io, "   Maximum number of outer iterations reached."); end
    if 0 != x.termination & TERMINATION_MAX_TIME;         println(io, "   Maximum allowed computation time exceeded."); end
    if 0 != x.termination & TERMINATION_COST_DISCREPANCY; println(io, "   Internal discrepancy in cost computations."); end
    if 0 != x.termination & TERMINATION_START_NOT_FINITE; println(io, "   Cost or gradient not finite at the start point."); end
    if 0 != x.termination & TERMINATION_GRAD_NOT_FINITE;  println(io, "   Gradient not finite."); end
    if 0 != x.termination & TERMINATION_GRAD_TOL;         println(io, "   Gradient norm below threshold."); end
    if 0 != x.termination & TERMINATION_LINSOLVER_FAILED; println(io, "   Linear solver failed to solve the linear system."); end
    userflags = x.termination >> 16
    if 0 != userflags; println(io, "   Terminated by user-defined callback, with flags: ", string(userflags, base=2)); end
end

mutable struct OptimizationInternal{LSType, VarType}
    # Costs
    startcost::Float64
    bestcost::Float64
    # Times (nanoseconds)
    start::Stats
    init::Stats
    optimize::Stats
    # Stats for key computations
    costs::TimeCounter
    gradients::TimeCounter
    solves::TimeCounter
    iternum::Int
    linsystem::LSType
    variables::VarType
    varnext::VarType
    varbest::VarType
    varbestvalid::Bool

    function OptimizationInternal(linsystem::LSType, x0::VarType, startstats=Stats()) where {LSType, VarType}
        varnext = copyvars(x0)
        return new{LSType, VarType}(
            0.0, 0.0, startstats, Stats(0), Stats(0),
            TimeCounter(), TimeCounter(), TimeCounter(), 0,
            linsystem, x0, varnext, x0, false
        )
    end
end

function reset!(data::OptimizationInternal, x0, startstats)
    data.startcost = 0.0
    data.bestcost = 0.0
    data.start = startstats
    data.init = Stats(0)
    data.optimize = Stats(0)
    data.costs = TimeCounter()
    data.gradients = TimeCounter()
    data.solves = TimeCounter()
    data.iternum = 0
    if x0 === data.varnext
        data.variables, data.varnext = data.varnext, data.variables
    elseif data.varbestvalid && x0 === data.varbest
        data.variables, data.varbest = data.varbest, data.variables
    else
        data.variables = x0
    end
    data.varnext = syncvars!(data.varnext, x0)
    if data.varbestvalid
        data.varbest = syncvars!(data.varbest, x0)
    else
        data.varbest = x0
    end
    return data
end
