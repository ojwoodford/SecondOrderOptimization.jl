function optimize!(
    f::F,
    fdf!::FDF,
    x,
    options::OptimizationOptions = OptimizationOptions();
    H::HMat = nothing,
    callback::CB = nullcallback,
    workspace::WS = nothing,
    startstats::Stats = Stats()
) where {F, FDF, HMat, CB, WS}
    if !isnothing(workspace) && workspacevalid(workspace, x, options, H)
        return run_cached_optimize!(something(workspace.data), x, f, fdf!, options, callback, startstats)
    else
        ls = makelinearsystem(x, H, options.linsolver)
        data = OptimizationInternal(ls, x, startstats)
        iteratedata = construct(options.iterator, ls, options.callbackchangesstate)
        if !isnothing(workspace)
            storeworkspace!(workspace, cachekey(x, options, H), (data, iteratedata))
        end
        return run_optimize!(data, iteratedata, f, fdf!, options, callback, x)
    end
end

function optimize(
    f::F,
    fdf!::FDF,
    x0,
    options::OptimizationOptions = OptimizationOptions();
    kwargs...
) where {F, FDF}
    return optimize!(f, fdf!, copyvars(x0), options; kwargs...)
end

function run_cached_optimize!(cached::Tuple{D, ID}, x, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB, startstats::Stats) where {D, ID, F, FDF, CB}
    data, iteratedata = cached
    reset!(data, x, startstats)
    reset!(iteratedata)
    return run_optimize!(data, iteratedata, f, fdf!, options, callback, x)
end

function run_optimize!(data::D, iteratedata::ID, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB, x=data.variables) where {D, ID, F, FDF, CB}
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

current_grad(iteratedata::LevMarDataTrial, data) = iteratedata.trialready ? getgrad(iteratedata.trial) : nothing
current_grad(::NewtonDataRelinearize, data) = getgrad(data.linsystem)
current_grad(::Any, data) = nothing

@inline function optimizeloop!(data::D, iteratedata::ID, f::F, fdf!::FDF, options::OptimizationOptions, callback::CB) where {D, ID, F, FDF, CB}
    stoptime = typemax(UInt64) - options.maxtime <= Base.time_ns() ? typemax(UInt64) : Base.time_ns() + options.maxtime
    fails = 0
    converged = 0

    cost = computelinearization!(data, fdf!, data.variables)
    data.startcost = cost
    data.bestcost = cost

    if !isfinite(cost) || !all(isfinite, getgrad(data.linsystem))
        converged |= isinf(cost) << 0
        converged |= isnan(cost) << 1
        converged |= 1 << 11
        return converged
    end

    if options.gtol > 0.0
        maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
        if maxgrad < options.gtol
            return 1 << 12
        end
    end

    while true
        data.iternum += 1
        cost_ = iterate!(iteratedata, data, f, fdf!, options)
        if cost_ === LINSOLVER_FAILED
            converged |= TERMINATION_LINSOLVER_FAILED
            break
        end
        cost, terminate = callback(cost_, data.varnext, data, iteratedata)
        @assert(dynamic(options.callbackchangesstate) || cost == cost_, "callback expected to not change the cost")

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
        converged |= isinf(cost)                                     << 0
        converged |= isnan(cost)                                     << 1
        if dcost >= 0
            converged |= (dcost < abs(data.bestcost) * options.reldcost) << 2
            converged |= (dcost < options.absdcost)                      << 3
        end
        converged |= isinf(maxstep)                                  << 4
        converged |= isnan(maxstep)                                  << 5
        converged |= (maxstep < options.dstep)                       << 6
        converged |= (fails > options.maxfails)                      << 7
        converged |= (data.iternum >= options.maxiters)              << 8
        converged |= (Base.time_ns() > stoptime)                     << 9
        converged |= terminate                                       << 16

        if options.gtol > 0.0
            grad = current_grad(iteratedata, data)
            if !isnothing(grad)
                maxgrad = maximum(abs, grad)::Float64
                converged |= (maxgrad < options.gtol)                << 12
            end
        end

        if converged != 0
            break
        end

        cost_ = linearize!(iteratedata, data, fdf!)
        if !isapprox(cost_, cost)
            converged |= 1 << 10
            break
        end
        if options.gtol > 0.0
            maxgrad = maximum(abs, getgrad(data.linsystem))::Float64
            if maxgrad < options.gtol
                converged |= 1 << 12
                break
            end
        end
    end

    if !(data.bestcost >= cost)
        data.variables, data.varbest = data.varbest, data.variables
    end
    return converged
end
