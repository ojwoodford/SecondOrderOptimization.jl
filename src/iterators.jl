using SparseArrays, Static, LinearAlgebra

negate!(x) = @.(x = -x)

function Base.String(iterator::OptimizationIterator)
    if iterator == newton
        return "Newton"
    elseif iterator == levenbergmarquardt
        return "Levenberg-Marquardt"
    elseif iterator == levenbergmarquardt_dtd
        return "Levenberg-Marquardt (D'D scaling)"
    elseif iterator == dogleg
        return "Dogleg"
    elseif iterator == gradientdescent
        return "gradient descent"
    else
        return "unknown iterator"
    end
end

function computelinearization!(data, fdf!, x)
    zero!(data.linsystem)
    cost = 0.0
    data.gradients += @elapsed_ns cost = fdf!(getb(data.linsystem), getA(data.linsystem), x)
    refresh_diag_cache!(data.linsystem)
    return cost
end

# Default linearization is at data.variables
linearize!(::Any, data, fdf!) = computelinearization!(data, fdf!, data.variables)

# Newton optimization
struct NewtonData
end
construct(::StaticSymbol{:newton}, ls, ::True) = NewtonData()
reset!(::NewtonData) = nothing

@inline function iterate!(::NewtonData, data, f, fdf!, options::OptimizationOptions)::Float64
    gethessian(data.linsystem)
    status = Success
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        if status == Success
        else
            mu = max(max_diag(data.linsystem) * 1e-4, 1e-6)
            shift_iters = 30
            while status != Success && shift_iters > 0
                shift_iters -= 1
                dampen!(data.linsystem, mu)
                mu *= 4.0
                status = solve!(data.linsystem, options.linsolver)
            end
            dampen!(data.linsystem, 0.0)
        end
    end
    if status == Success
        if !all(isfinite, getx(data.linsystem))
            return NaN
        end
        negate!(getx(data.linsystem))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        return cost_
    end
    return LINSOLVER_FAILED
end

mutable struct NewtonDataRelinearize
    cost::Float64
end
construct(::StaticSymbol{:newton}, ls, ::False) = NewtonDataRelinearize(0.0)
reset!(::NewtonDataRelinearize) = nothing

@inline function iterate!(nd::NewtonDataRelinearize, data, f, fdf!, options::OptimizationOptions)::Float64
    gethessian(data.linsystem)
    status = Success
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        if status == Success
        else
            mu = max(max_diag(data.linsystem) * 1e-4, 1e-6)
            shift_iters = 30
            while status != Success && shift_iters > 0
                shift_iters -= 1
                dampen!(data.linsystem, mu)
                mu *= 4.0
                status = solve!(data.linsystem, options.linsolver)
            end
            dampen!(data.linsystem, 0.0)
        end
    end
    if status == Success
        if !all(isfinite, getx(data.linsystem))
            return NaN
        end
        negate!(getx(data.linsystem))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        zero!(data.linsystem)
        data.gradients += @elapsed_ns nd.cost = fdf!(getb(data.linsystem), getA(data.linsystem), data.varnext)
        refresh_diag_cache!(data.linsystem)
        return nd.cost
    end
    return LINSOLVER_FAILED
end

linearize!(nd::NewtonDataRelinearize, data, fdf!) = nd.cost

# Dogleg optimization
mutable struct DoglegData{T,L}
    trustradius::Float64
    maxtrustradius::Float64
    cauchy::T
    newton::T
    trial::L              # nothing, or a spare linear system for reusing the trial linearization
    trialcost::Float64
    trialready::Bool
end

DoglegData(trustradius::Float64, maxtrustradius::Float64, cauchy::T, newton::T) where {T} =
    DoglegData(trustradius, maxtrustradius, cauchy, newton, nothing, NaN, false)
DoglegData(trustradius::Float64, cauchy::T, newton::T) where {T} =
    DoglegData(trustradius, 1e10, cauchy, newton)
function DoglegData(ls, maxtrustradius::Float64=1e10; trial=nothing)
    x = getx(ls)
    return DoglegData(0.0, maxtrustradius, similar(x), similar(x), trial, NaN, false)
end

construct(::StaticSymbol{:dogleg}, ls, ::True) = DoglegData(ls)
construct(::StaticSymbol{:dogleg}, ls, ::False) = DoglegData(ls; trial=copy_structure(ls))
reset!(dd::DoglegData) = (dd.trustradius = 0.0; dd.trialready = false; dd.trialcost = NaN)

@inline function evaltrial!(dd::DoglegData{T,Nothing}, data, f, fdf!, ::Bool) where {T}
    data.costs += @elapsed_ns cost_ = f(data.varnext)
    return cost_
end
@inline function evaltrial!(dd::DoglegData, data, f, fdf!, usefdf::Bool)
    if !usefdf
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        return cost_
    end
    dd.trial = sharex(dd.trial, data.linsystem)
    zero!(dd.trial)
    data.gradients += @elapsed_ns cost_ = fdf!(getb(dd.trial), getA(dd.trial), data.varnext)
    refresh_diag_cache!(dd.trial)
    dd.trialready = true
    dd.trialcost = cost_
    return cost_
end

linearize!(::DoglegData{T,Nothing}, data, fdf!) where {T} = computelinearization!(data, fdf!, data.variables)
function linearize!(dd::DoglegData, data, fdf!)
    if !dd.trialready
        return computelinearization!(data, fdf!, data.variables)
    end
    dd.trialready = false
    trial = sharex(dd.trial, data.linsystem)
    dd.trial = data.linsystem
    data.linsystem = trial
    return dd.trialcost
end

@inline function iterate!(dd::DoglegData, data, f, fdf!, options::OptimizationOptions)::Float64
    dd.trialready = false
    hessian, gradient = gethessgrad(data.linsystem)

    gHg = fast_bAb(hessian, gradient)
    if !isfinite(gHg)
        return NaN
    end
    gnorm2 = sqnorm(gradient)
    if gnorm2 == 0
        # Stationary point: zero step (outer loop terminates on dstep/reldcost)
        getx(data.linsystem) .= 0.0
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        return data.bestcost
    end
    gnorm = sqrt(gnorm2)

    # Cauchy point: exact model minimiser along -g, exists only for positive curvature
    cauchy_a = 0.0
    cauchy_norm = Inf
    if gHg > 0
        cauchy_a = gnorm2 / gHg
        cauchy_norm = cauchy_a * gnorm
        if isfinite(cauchy_norm)
            dd.cauchy .= (-cauchy_a) .* gradient
        end
    end
    cauchy_ok = isfinite(cauchy_norm)

    # Newton step, with the same shifted-retry fallback as NewtonData
    newton_norm = Inf
    newton_pred = 0.0
    newton_valid = false
    status = Success
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        if status != Success
            md = max_diag(data.linsystem)
            mu = isfinite(md) ? max(md * 1e-4, 1e-6) : 1e-6
            shifts = 20
            while status != Success && shifts > 0
                shifts -= 1
                dampen!(data.linsystem, mu)
                mu *= 4.0
                status = solve!(data.linsystem, options.linsolver)
            end
            dampen!(data.linsystem, 0.0)   # hessian must be undamped below
        end
    end
    if status == Success
        negate!(getx(data.linsystem))
        dd.newton .= getx(data.linsystem)
        newton_norm = fast_norm(dd.newton)
        newton_dot_g = fast_dot(gradient, dd.newton)
        if isfinite(newton_norm) && newton_dot_g < 0
            # Validity = positive predicted model reduction under the true (undamped) H
            newton_pred = -newton_dot_g - 0.5 * fast_bAb(hessian, dd.newton)
            newton_valid = isfinite(newton_pred) && newton_pred > 0
        end
    end

    # Scale-free initial radius
    if !(dd.trustradius > 0) || !isfinite(dd.trustradius)
        if newton_valid
            r = newton_norm
        elseif cauchy_ok
            r = cauchy_norm
        else
            md = max_diag(data.linsystem)
            r = (isfinite(md) && md > 0) ? gnorm / md : 1.0
        end
        dd.trustradius = min(r, dd.maxtrustradius)
    end

    x = getx(data.linsystem)
    cost_ = data.bestcost
    noise = 10 * eps(Float64) * abs(data.bestcost)
    inner_iters = 0
    while true
        inner_iters += 1
        Δ = dd.trustradius
        step_norm = Δ
        pred_red = NaN

        if newton_valid && newton_norm <= Δ
            # Full Newton step inside the region
            x .= dd.newton
            step_norm = newton_norm
            pred_red = newton_pred
        elseif !cauchy_ok || cauchy_norm >= Δ
            # Truncated gradient step (also covers negative curvature)
            s = Δ / gnorm
            x .= (-s) .* gradient
            pred_red = s * (gnorm2 - 0.5 * s * gHg)
        elseif newton_valid
            # Dogleg: cauchy_norm < Δ < newton_norm
            x .= dd.newton .- dd.cauchy
            aw = sqnorm(x)
            c0 = fast_dot(dd.cauchy, x)
            trsq = Δ * Δ - cauchy_norm * cauchy_norm
            tau = 0.0
            if aw > 0 && isfinite(trsq) && trsq > 0
                disc = sqrt(c0 * c0 + aw * trsq)
                tau = c0 <= 0 ? (-c0 + disc) / aw : trsq / (c0 + disc)
                tau = clamp(tau, 0.0, 1.0)
            end
            x .*= tau
            x .+= dd.cauchy
            if tau == 0.0
                step_norm = cauchy_norm
            end
            pred_red = -fast_dot(gradient, x) - 0.5 * fast_bAb(hessian, x)
        else
            # Newton unavailable: full Cauchy step
            x .= dd.cauchy
            step_norm = cauchy_norm
            pred_red = 0.5 * cauchy_a * gnorm2
        end

        step_sz = maximum(abs, x)::Float64
        if !isfinite(step_sz)
            return NaN
        end

        data.varnext = update!(data.varnext, data.variables, x)
        # On the first trial, linearize at the trial point too if the step is likely to be accepted
        usefdf = inner_iters == 1 && pred_red >= abs(data.bestcost) * options.reldcost
        cost_ = evaltrial!(dd, data, f, fdf!, usefdf)

        # Noise-floored reduction ratio (Fletcher): stable when both reductions ~ eps*cost
        act_red = data.bestcost - cost_
        denom = pred_red + noise
        rho = (isfinite(act_red) && isfinite(denom) && denom > 0) ? (act_red + noise) / denom : -Inf

        if act_red >= 0 && rho > 1e-4
            if rho < 0.25
                dd.trustradius = 0.25 * step_norm
            elseif rho > 0.75 && step_norm >= 0.9 * Δ
                dd.trustradius = min(2.0 * Δ, dd.maxtrustradius)
            end
            return cost_
        end

        # Rejected
        dd.trialready = false
        dd.trustradius = 0.25 * step_norm
        if !(step_sz >= options.dstep) || !(dd.trustradius >= floatmin(Float64)) || inner_iters > 100
            return cost_
        end
    end
end

printoutcallback(cost, x_trial, data, iteratedata::DoglegData) = printoutcallback(cost, data, iteratedata.trustradius)

# Levenberg-Marquardt optimization
abstract type LevMarData end
abstract type LevMarDataTrial <: LevMarData end

mutable struct LevenbergData <: LevMarData
    lambda::Float64
end
construct(::StaticSymbol{:levenbergmarquardt}, ls, ::True) = LevenbergData(0.0)
reset!(lmd::LevenbergData) = (lmd.lambda = 0.0)

mutable struct LevenbergDataTrial{L} <: LevMarDataTrial
    lambda::Float64
    trial::L
    trialcost::Float64
    trialready::Bool
end
construct(::StaticSymbol{:levenbergmarquardt}, ls, ::False) = LevenbergDataTrial(0.0, copy_structure(ls), NaN, false)
reset!(lmd::LevenbergDataTrial) = (lmd.trialready = false; lmd.trialcost = NaN; lmd.lambda = 0.0)

const AnyLevenbergData = Union{LevenbergData, LevenbergDataTrial}

function updatescaling!(lmd::AnyLevenbergData, ls)
    if lmd.lambda == 0
        md = max_diag(ls)
        lmd.lambda = isfinite(md) ? max(md * 1e-6, 1e-12) : 1e-6
    end
end

dampen!(ls::AbstractLinearSystem, ::AnyLevenbergData, lambda) = dampen!(ls, max(0.0, lambda))
dampen!(ls::AbstractLinearSystem, lmd::AnyLevenbergData) = dampen!(ls, lmd.lambda)

mutable struct MarquardtData{T} <: LevMarData
    lambda::Float64
    scaling::T
end
construct(::StaticSymbol{:levenbergmarquardt_dtd}, ls, ::True) = MarquardtData(0.0, zero(getx(ls)))
reset!(lmd::MarquardtData) = (fill!(lmd.scaling, 0.0); lmd.lambda = 0.0)

mutable struct MarquardtDataTrial{T, L} <: LevMarDataTrial
    lambda::Float64
    scaling::T
    trial::L
    trialcost::Float64
    trialready::Bool
end
construct(::StaticSymbol{:levenbergmarquardt_dtd}, ls, ::False) = MarquardtDataTrial(0.0, zero(getx(ls)), copy_structure(ls), NaN, false)
reset!(lmd::MarquardtDataTrial) = (fill!(lmd.scaling, 0.0); lmd.trialready = false; lmd.trialcost = NaN; lmd.lambda = 0.0)

const AnyMarquardtData = Union{MarquardtData, MarquardtDataTrial}

function updatescaling!(lmd::AnyMarquardtData, ls::SparseLinearSystem)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(lmd.scaling))
    for (i, idx) in enumerate(ls.diag_inds)
        val = abs(ls.A.nzval[idx])
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], isfinite(val) ? val : 1.0)
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

function updatescaling!(lmd::AnyMarquardtData, ls)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(lmd.scaling))
    A = gethessian(ls)
    for i in axes(A, 1)
        val = abs(A[i, i])
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], isfinite(val) ? val : 1.0)
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

dampen!(ls::AbstractLinearSystem, lmd::AnyMarquardtData, lambda) = dampen!(ls, max(0.0, lambda), lmd.scaling)
dampen!(ls::AbstractLinearSystem, lmd::AnyMarquardtData) = dampen!(ls, lmd.lambda, lmd.scaling)

dampingnorm(::AnyLevenbergData, x) = sqnorm(x)
function dampingnorm(lmd::AnyMarquardtData, x)
    total = 0.0
    scaling = lmd.scaling
    @inbounds @fastmath @simd for i in eachindex(x)
        total += scaling[i] * x[i] * x[i]
    end
    return total
end

function linearize!(lmd::LevMarDataTrial, data, fdf!)
    if !lmd.trialready
        return computelinearization!(data, fdf!, data.variables)
    end
    lmd.trialready = false
    trial = sharex(lmd.trial, data.linsystem)
    lmd.trial = data.linsystem
    data.linsystem = trial
    return lmd.trialcost
end

# Multiplier for the damping after an accepted step, from the actual and the predicted change
# in cost (Nielsen's update). The predicted change is not negative when the Hessian is
# indefinite, in which case the step is treated as low quality. The factor is clamped so a
# tiny or wrong-signed prediction can't make the damping explode.
function lambda_update_factor(actual::Float64, predicted::Float64)
    predicted < 0 || return 2.0
    q = actual / predicted
    q >= 0.983 && return 0.1
    return clamp(1.0 - (2.0 * q - 1.0)^3, 0.1, 2.0)
end

@inline function iterate!(levmardata::LevMarData, data, f, fdf!, options::OptimizationOptions)::Float64
    @assert levmardata.lambda >= 0.0
    hessian, gradient = gethessgrad(data.linsystem)
    updatescaling!(levmardata, data.linsystem)
    mu = 2.0
    inner_iters = 0
    while true
        inner_iters += 1
        status = Success
        data.solves += @elapsed_ns begin
            while true
                dampen!(data.linsystem, levmardata)
                status = solve!(data.linsystem, options.linsolver)
                if status == Success
                    negate!(getx(data.linsystem))
                    break
                end
                levmardata.lambda *= mu
                mu *= 2.0
                if levmardata.lambda > 1e16 || inner_iters > 100
                    dampen!(data.linsystem, 0.0)
                    break
                end
            end
        end
        if status != Success
            return LINSOLVER_FAILED
        end
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if !isfinite(step_sz)
            dampen!(data.linsystem, 0.0)
            return NaN
        end
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                predicted = 0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem))
                levmardata.lambda = max(levmardata.lambda * lambda_update_factor(cost_ - data.bestcost, predicted), 1e-12)
            end
            return cost_
        end
        levmardata.lambda *= mu
        mu *= 2.0
    end
end

@inline function iterate!(levmardata::LevMarDataTrial, data, f, fdf!, options::OptimizationOptions)::Float64
    @assert levmardata.lambda >= 0.0
    hessian, gradient = gethessgrad(data.linsystem)
    updatescaling!(levmardata, data.linsystem)
    mu = 2.0
    levmardata.trialready = false
    firsttrial = true
    inner_iters = 0
    while true
        inner_iters += 1
        status = Success
        data.solves += @elapsed_ns begin
            while true
                dampen!(data.linsystem, levmardata)
                status = solve!(data.linsystem, options.linsolver)
                if status == Success
                    negate!(getx(data.linsystem))
                    break
                end
                levmardata.trialready = false
                levmardata.lambda *= mu
                mu *= 2.0
                if levmardata.lambda > 1e16 || inner_iters > 100
                    dampen!(data.linsystem, 0.0)
                    break
                end
            end
        end
        if status != Success
            return LINSOLVER_FAILED
        end
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if !isfinite(step_sz)
            dampen!(data.linsystem, 0.0)
            return NaN
        end
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        step = getx(data.linsystem)
        if firsttrial && 0.5 * (levmardata.lambda * dampingnorm(levmardata, step) - fast_dot(gradient, step)) >= abs(data.bestcost) * options.reldcost
            levmardata.trial = sharex(levmardata.trial, data.linsystem)
            zero!(levmardata.trial)
            data.gradients += @elapsed_ns cost_ = fdf!(getb(levmardata.trial), getA(levmardata.trial), data.varnext)
            refresh_diag_cache!(levmardata.trial)
            levmardata.trialready = true
            levmardata.trialcost = cost_
        else
            data.costs += @elapsed_ns cost_ = f(data.varnext)
        end
        firsttrial = false
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                predicted = 0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem))
                levmardata.lambda = max(levmardata.lambda * lambda_update_factor(cost_ - data.bestcost, predicted), 1e-12)
            end
            return cost_
        end
        levmardata.trialready = false
        levmardata.lambda *= mu
        mu *= 2.0
    end
end

printoutcallback(cost, x_trial, data, iteratedata::LevMarData) = printoutcallback(cost, data, 1.0 / iteratedata.lambda)

# Gradient descent optimization
mutable struct GradientDescentData
    stepsize::Float64
end
construct(::StaticSymbol{:gradientdescent}, ls, ::Any) = GradientDescentData(1.0)
construct(s::StaticSymbol, ls) = construct(s, ls, False())
reset!(gdd::GradientDescentData) = (gdd.stepsize = 1.0)

@inline function iterate!(gddata::GradientDescentData, data, f, fdf!, options::OptimizationOptions)::Float64
    gradient = getgrad(data.linsystem)
    if !all(isfinite, gradient)
        return NaN
    end
    getx(data.linsystem) .= -gradient .* gddata.stepsize
    data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
    data.costs += @elapsed_ns costc = f(data.varnext)
    inner_iters = 0
    while !(costc <= data.bestcost)
        inner_iters += 1
        if inner_iters > 100 || !(gddata.stepsize >= floatmin(Float64)) || !isfinite(gddata.stepsize)
            return costc
        end
        if isfinite(costc)
            coststep = fast_dot(getx(data.linsystem), gradient)
            costdiff = data.bestcost + coststep - costc
            if costdiff != 0.0
                gddata.stepsize *= 0.5 * coststep / costdiff
            else
                gddata.stepsize *= 0.5
            end
        else
            gddata.stepsize *= 0.5
        end
        getx(data.linsystem) .= -gradient .* gddata.stepsize
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns costc = f(data.varnext)
    end
    gddata.stepsize *= 2.0
    return costc
end

printoutcallback(cost, x_trial, data, iteratedata::GradientDescentData) = printoutcallback(cost, data, iteratedata.stepsize)
