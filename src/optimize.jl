const AnyObjective = Union{HessianObjective, JacobianObjective}

function optimize!(
    x0,
    obj::AnyObjective,
    options::OptimizationOptions = OptimizationOptions();
    callback::CB = nullcallback,
    workspace::WS = nothing,
    startstats::Stats = Stats()
) where {CB, WS}
    ismutablevar(x0) || throw(ArgumentError("optimize! updates x0 in place, but a $(typeof(x0)) cannot be updated in place. Use `x, result = optimize(x0, obj, options)` instead."))
    _, result = optimize_internal!(x0, obj, options, callback, workspace, startstats)
    return result
end

function optimize(
    x0,
    obj::AnyObjective,
    options::OptimizationOptions = OptimizationOptions();
    callback::CB = nullcallback,
    workspace::WS = nothing,
    startstats::Stats = Stats()
) where {CB, WS}
    return optimize_internal!(copyvars(x0), obj, options, callback, workspace, startstats)
end

# What differs between the objective kinds: the checks at entry, the checks before a linear
# system is built, the linear system, the structure the workspace is keyed on, and the function
# the iterators call to linearize, `fdf!(b, A, x)`
# LSMR works on a Jacobian. On the normal equations it would work with HᵀH, squaring the
# condition number a second time (M10), so a HessianObjective rejects it.
function checkentry(x, obj::HessianObjective, options::OptimizationOptions)
    dynamic(options.linsolver) === :lsmr && throw(ArgumentError("linsolver = :lsmr is only for a JacobianObjective: it solves the least-squares problem in J. On the normal equations of a HessianObjective it would square their condition number again; use :cg or :minres there."))
    return nothing
end
checkentry(x, obj::JacobianObjective, options::OptimizationOptions) = checkobjective(x, obj, options)
checkbuild(x, obj::HessianObjective) = checkhessian(obj.H, dynamic(nvars(x)))
checkbuild(x, obj::JacobianObjective) = nothing
# The linear system for the variables `x`: the optimizer decides its size and whether it is static
# (`systemsize`), and the prototype its storage
objective_linearsystem(x, obj::HessianObjective, linsolver) = hessian_linearsystem(x, obj.H, linsolver)
objective_linearsystem(x, obj::JacobianObjective, linsolver) = tall_linearsystem(obj.J, systemsize(x), linsolver)
hessian_linearsystem(x, ::Nothing, linsolver) = makelinearsystem(systemsize(x), linsolver)
hessian_linearsystem(x, H, linsolver) = makelinearsystem(H, linsolver)
objective_structure(obj::HessianObjective) = obj.H
objective_structure(obj::JacobianObjective) = jacobian_structure(obj.J)
jacobian_structure(J::Union{Integer, Matrix}) = JacobianStructureKey(nresiduals_of(J))
jacobian_structure(J::SparseMatrixCSC) = SparseJacobianPattern(J)
jacobian_structure(J) = JacobianPrototypeKey(structurekey(J))
linearizer(obj::HessianObjective) = obj.fdf!
linearizer(obj::JacobianObjective) = TallLinearize(obj.resjac!)

# How the iterators and the outer loop drive a tall system unchanged: they call
# `fdf!(getb(ls), getA(ls), x)`, and this wrapper turns that into `resjac!(r, J, x)` on the storage's
# J and r, then has the storage form g = Jᵀr and invalidate what it computed from the old J and r
struct TallLinearize{RJ}
    resjac!::RJ
end
function (t::TallLinearize)(g::AbstractVector, A, x)
    cost = t.resjac!(getr(A), getJ(A), x)
    gradient!(g, A)
    markstale!(A)
    return cost
end

# What a run with a reused workspace must set up before it starts: for a JacobianObjective, the
# Jacobian storage (see `initjacobian!`)
initlinearization!(cached, obj::HessianObjective) = nothing
function initlinearization!(cached::Tuple, obj::JacobianObjective)
    data, iteratedata = cached
    initjacobian!(data.linsystem, obj.J)
    initjacobian!(trialsystem(iteratedata), obj.J)
    return nothing
end

# The start of a run with a reused workspace. Every J the run uses (the main system's, and the
# trial system's if the iterator has one) must start as the J a fresh run would: the prototype as
# handed, or zeros for an integer m. A cached J holds the previous run's entries, so it is set
# from the prototype (never written itself). r needs nothing: resjac! sets every entry.
initjacobian!(::Nothing, J) = nothing
initjacobian!(::Nothing, ::Integer) = nothing
function initjacobian!(ls, ::Integer)
    A = getA(ls)
    fill!(getJ(A), 0.0)
    markstale!(A)
    return ls
end
function initjacobian!(ls, P)
    A = getA(ls)
    J = getJ(A)
    J === P || copyprototype!(J, P)
    markstale!(A)
    return ls
end
# A reused workspace's J has the prototype's pattern (the workspace key checks it), so for a sparse J
# only the values are set
copyprototype!(J, P) = copyto!(J, P)
copyprototype!(J::SparseMatrixCSC, P::SparseMatrixCSC) = copyto!(nonzeros(J), nonzeros(P))

# The iterator's spare linear system, for reusing the trial linearization, or `nothing`. An
# iterator data type that holds one must have a method, so that whatever prepares a run's systems
# finds it.
trialsystem(iteratedata) = nothing
trialsystem(lmd::LevenbergDataTrial) = lmd.trial
trialsystem(lmd::MarquardtDataTrial) = lmd.trial
trialsystem(dd::DoglegData) = dd.trial          # nothing when callbackchangesstate = true
# The system a workspace caches. For a JacobianObjective it owns its Jacobian storage, a copy of the
# prototype: were it the prototype itself, a later run with another prototype would set this one's
# contents, and a run handed this prototype again would start from the other's.
workspace_linearsystem(x, obj::HessianObjective, linsolver) = objective_linearsystem(x, obj, linsolver)
workspace_linearsystem(x, obj::JacobianObjective, linsolver) = tall_linearsystem(workspace_prototype(obj.J), systemsize(x), linsolver)
workspace_prototype(J) = copy(J)
workspace_prototype(m::Integer) = m

# Runs on `x` (updated in place when it is mutable) and returns `(x, result)`
function optimize_internal!(x, obj::AnyObjective, options::OptimizationOptions, callback::CB, workspace::WS, startstats::Stats) where {CB, WS}
    checkentry(x, obj, options)
    structure = objective_structure(obj)
    if !isnothing(workspace) && workspacevalid(workspace, x, options, structure)
        cached = something(workspace.data)
        initlinearization!(cached, obj)
        return run_cached_optimize!(cached, x, obj.f, linearizer(obj), options, callback, startstats)
    else
        checkbuild(x, obj)
        ls = isnothing(workspace) ? objective_linearsystem(x, obj, options.linsolver) : workspace_linearsystem(x, obj, options.linsolver)
        data = OptimizationInternal(ls, x, startstats)
        iteratedata = construct(options.iterator, ls, options.callbackchangesstate)
        if !isnothing(workspace)
            storeworkspace!(workspace, cachekey(x, options, structure), (data, iteratedata))
        end
        return run_optimize!(data, iteratedata, x, obj.f, linearizer(obj), options, callback)
    end
end

function run_cached_optimize!(cached::Tuple{D, ID}, x, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB, startstats::Stats) where {D, ID, F, FDF, CB}
    data, iteratedata = cached
    reset!(data, x, startstats)
    reset!(iteratedata)
    return run_optimize!(data, iteratedata, x, f, fdf!, options, callback)
end

function run_optimize!(data::D, iteratedata::ID, x, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB) where {D, ID, F, FDF, CB}
    data.init = Stats()
    converged = optimizeloop!(data, iteratedata, f, fdf!, options, callback)
    data.optimize = Stats()
    if data.variables !== x
        x = syncvars!(x, data.variables)
        if data.varnext === x
            data.varnext = data.variables
        end
        if data.varbestvalid && data.varbest === x
            data.varbest = data.variables
        end
        data.variables = x
    end
    return x, OptimizationResult(data, converged, dynamic(options.iterator))
end

# A callback returned a cost other than the one it was given, which only a callback that changes the
# state may do. (Out of line, so that the loop does not build the message.)
@noinline callbackchangedcost(given, returned) = throw(ArgumentError("The callback changed the cost (from $given to $returned), but callbackchangesstate is false. A callback that changes the variables or the cost needs OptimizationOptions(callbackchangesstate = true)."))

@inline function optimizeloop!(data::D, iteratedata::ID, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB) where {D, ID, F, FDF, CB}
    stoptime = typemax(UInt64) - options.maxtime <= Base.time_ns() ? typemax(UInt64) : Base.time_ns() + options.maxtime
    stalls = 0       # consecutive iterations without a significant improvement of the best cost
    rose = false    # whether the previous iteration raised the cost (for the best-point bookkeeping)

    cost = computelinearization!(data, fdf!, data.variables)
    data.startcost = cost
    data.bestcost = cost
    refcost = cost  # the best cost when `stalls` was last reset
    lowsince = Inf  # the lowest cost of the iterations since then (excluding the one that reset it)
    highsince = -Inf # and the highest

    converged = 0
    converged |= isinf(cost)                                         << 0
    converged |= isnan(cost)                                         << 1
    maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
    converged |= isinf(maxgrad)                                      << 11
    converged |= isnan(maxgrad)                                      << 12
    converged |= (converged != 0)                                    << 10
    converged |= (data.iternum >= options.maxiters)                  << 8
    converged |= (maxgrad < options.gtol)                            << 13

    while converged == 0
        data.iternum += 1
        cost_ = iterate!(iteratedata, data, f, fdf!, options)
        if cost_ === LINSOLVER_FAILED
            converged |= TERMINATION_LINSOLVER_FAILED
            break
        end
        cost, terminate = callback(cost_, data.varnext, data, iteratedata)
        dynamic(options.callbackchangesstate) || isequal(cost, cost_) || callbackchangedcost(cost_, cost)

        dcost = data.bestcost - cost
        if dcost >= 0
            data.bestcost = cost
            rose = false
            # A significant improvement resets the stall counter. With stalltol = 0 every iteration
            # that does not raise the cost resets it (refcost then always equals bestcost), so
            # maxstall counts consecutive rises, as before stalltol existed.
            if cost <= refcost - options.stalltol * abs(refcost)
                refcost = cost
                lowsince = Inf
                highsince = -Inf
                stalls = 0
            else
                stalls += 1
                lowsince = min(lowsince, cost)
                highsince = max(highsince, cost)
            end
        else
            dcost = -dcost # Avoid bouncing around a local minimum
            stalls += 1
            lowsince = min(lowsince, cost)
            highsince = max(highsince, cost)
            if !rose     # the first rise since the best cost: keep the best point
                rose = true
                if !data.varbestvalid
                    data.varbest = copyvars(data.variables)
                    data.varbestvalid = true
                else
                    data.variables, data.varbest = data.varbest, data.variables
                end
            end
        end

        data.variables, data.varnext = data.varnext, data.variables

        maxstep = maximum(abs, getx(data.linsystem))::Float64
        converged |= isinf(cost)                                         << 0
        converged |= isnan(cost)                                         << 1
        converged |= (dcost < abs(data.bestcost) * options.reldcost)     << 2
        converged |= (dcost < options.absdcost)                          << 3
        converged |= isinf(maxstep)                                      << 4
        converged |= isnan(maxstep)                                      << 5
        converged |= (maxstep < options.dstep)                           << 6
        if stalls > options.maxstall
            # Stalled (converged) if the cost has settled: some iteration since the last significant
            # improvement stayed within the stall band above refcost, and none rose by more than
            # stallrise = sqrt(stalltol). Otherwise too many rises (failure). The lowest such cost, not
            # the last, decides settling, so that a run bouncing at its minimum by about the band's
            # width is not classified by its last bounce; the highest keeps a run whose cost jumps
            # (divergence after one insignificant improvement) from being called converged. With
            # stalltol = 0 every iteration counted is a rise, so this is max_fails, as before.
            settled = lowsince <= (refcost + options.stalltol * abs(refcost)) && highsince <= (refcost + options.stallrise * abs(refcost))
            converged |= settled ? TERMINATION_STALLED : TERMINATION_MAX_FAILS
        end
        converged |= (data.iternum >= options.maxiters)                  << 8
        converged |= (Base.time_ns() > stoptime)                         << 9
        converged |= terminate                                           << 16
        if converged != 0
            break
        end

        linearize!(iteratedata, data, fdf!)
        maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
        converged |= isinf(maxgrad)                                      << 11
        converged |= isnan(maxgrad)                                      << 12
        converged |= (maxgrad < options.gtol)                            << 13
    end

    if !(data.bestcost >= cost)
        data.variables, data.varbest = data.varbest, data.variables
    end
    return converged
end
