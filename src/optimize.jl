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
checkentry(x, obj::HessianObjective, options::OptimizationOptions) = nothing
checkentry(x, obj::JacobianObjective, options::OptimizationOptions) = checkobjective(x, obj, options)
checkbuild(x, obj::HessianObjective) = checkhessian(x, obj.H)
checkbuild(x, obj::JacobianObjective) = nothing
objective_linearsystem(x, obj::HessianObjective, linsolver) = makelinearsystem(x, obj.H, linsolver)
objective_linearsystem(x, obj::JacobianObjective, linsolver) = makelinearsystem(x, obj, linsolver)
objective_structure(obj::HessianObjective) = obj.H
objective_structure(obj::JacobianObjective) = JacobianStructureKey(nresiduals(obj))
linearizer(obj::HessianObjective) = obj.fdf!
linearizer(obj::JacobianObjective) = TallLinearize(obj.resjac!)

# Runs on `x` (updated in place when it is mutable) and returns `(x, result)`
function optimize_internal!(x, obj::AnyObjective, options::OptimizationOptions, callback::CB, workspace::WS, startstats::Stats) where {CB, WS}
    checkentry(x, obj, options)
    structure = objective_structure(obj)
    if !isnothing(workspace) && workspacevalid(workspace, x, options, structure)
        return run_cached_optimize!(something(workspace.data), x, obj.f, linearizer(obj), options, callback, startstats)
    else
        checkbuild(x, obj)
        ls = objective_linearsystem(x, obj, options.linsolver)
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
    return x, OptimizationResult(data, converged, getiterator(options))
end

@inline function optimizeloop!(data::D, iteratedata::ID, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB) where {D, ID, F, FDF, CB}
    stoptime = typemax(UInt64) - options.maxtime <= Base.time_ns() ? typemax(UInt64) : Base.time_ns() + options.maxtime
    fails = 0
    converged = 0

    cost = computelinearization!(data, fdf!, data.variables)
    data.startcost = cost
    data.bestcost = cost

    converged |= isinf(cost)                                         << 0
    converged |= isnan(cost)                                         << 1
    maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
    converged |= !isfinite(maxgrad)                                  << 12
    converged |= (converged != 0)                                    << 11
    converged |= (maxgrad < options.gtol)                            << 13

    while converged == 0
        data.iternum += 1
        cost_ = iterate!(iteratedata, data, f, fdf!, options)
        if cost_ === LINSOLVER_FAILED
            converged |= TERMINATION_LINSOLVER_FAILED
            break
        end
        cost, terminate = callback(cost_, data.varnext, data, iteratedata)
        @assert(dynamic(options.callbackchangesstate) || isequal(cost, cost_), "callback expected to not change the cost")

        dcost = data.bestcost - cost
        if dcost >= 0
            data.bestcost = cost
            fails = 0
        else
            fails += 1
            if fails == 1
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
        if dcost >= 0
            converged |= (dcost < abs(data.bestcost) * options.reldcost) << 2
            converged |= (dcost < options.absdcost)                      << 3
        end
        converged |= isinf(maxstep)                                      << 4
        converged |= isnan(maxstep)                                      << 5
        converged |= (maxstep < options.dstep)                           << 6
        converged |= (fails > options.maxfails)                          << 7
        converged |= (data.iternum >= options.maxiters)                  << 8
        converged |= (Base.time_ns() > stoptime)                         << 9
        converged |= terminate                                           << 16
        if converged != 0
            break
        end

        cost_ = linearize!(iteratedata, data, fdf!)
        converged |= !isapprox(cost_, cost; rtol=1.0e-10, atol=1.0e-25)  << 10
        maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
        converged |= !isfinite(maxgrad)                                  << 12
        converged |= (maxgrad < options.gtol)                            << 13
    end

    if !(data.bestcost >= cost)
        data.variables, data.varbest = data.varbest, data.variables
    end
    return converged
end
