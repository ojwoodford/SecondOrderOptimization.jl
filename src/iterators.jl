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
mutable struct DoglegData{T}
    trustradius::Float64
    maxtrustradius::Float64
    cauchy::T
    newton::T

    function DoglegData(trustradius::Float64, maxtrustradius::Float64, cauchy::T, newton::T) where {T}
        return new{T}(trustradius, maxtrustradius, cauchy, newton)
    end
    function DoglegData(trustradius::Float64, cauchy::T, newton::T) where {T}
        return new{T}(trustradius, 1e10, cauchy, newton)
    end
    function DoglegData(ls, maxtrustradius::Float64=1e10)
        x = getx(ls)
        return new{typeof(x)}(0.0, maxtrustradius, similar(x), similar(x))
    end
end
construct(::StaticSymbol{:dogleg}, ls, ::Any) = DoglegData(ls)
reset!(dd::DoglegData) = (dd.trustradius = 0.0)

@inline function iterate!(doglegdata::DoglegData, data, f, fdf!, options::OptimizationOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    gnorm2 = sqnorm(gradient)
    gnorm = isfinite(gnorm2) ? sqrt(gnorm2) : fast_norm(gradient)
    if !(gnorm > 0)
        fill!(getx(data.linsystem), zero(eltype(gradient)))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        return cost_
    end

    # Compute Cauchy point info & curvature along gradient
    gHg = fast_bAb(hessian, gradient)
    has_pos_curv = isfinite(gHg) && (gHg > 0)
    a = Inf
    alpha = Inf
    alpha2 = Inf
    if has_pos_curv
        a = gnorm2 / gHg
        alpha = a * gnorm
        alpha2 = alpha * alpha
        doglegdata.cauchy .= (-a) .* gradient
    end

    # 1. Newton-First Fast-Path: Solve H * delta_N = -g via options.linsolver
    newton_norm = Inf
    newton_valid = false
    if has_pos_curv
        status = Success
        data.solves += @elapsed_ns begin
            status = solve!(data.linsystem, options.linsolver)
        end
        if status == Success
            negate!(getx(data.linsystem))
            doglegdata.newton .= getx(data.linsystem)
            newton_norm = fast_norm(doglegdata.newton)
            newton_dot_g = fast_dot(gradient, doglegdata.newton)
            if isfinite(newton_norm) && (newton_dot_g < 0)
                newton_valid = true
                alpha = min(alpha, newton_norm)
                alpha2 = alpha * alpha
            end
        end
    end

    # Initialize trust radius if needed (Delta_0 = max(1.0, ||delta_N||_2))
    if doglegdata.trustradius <= 0.0 || !isfinite(doglegdata.trustradius)
        if newton_valid
            doglegdata.trustradius = max(1.0, newton_norm)
        elseif has_pos_curv && isfinite(alpha)
            doglegdata.trustradius = max(1.0, alpha)
        else
            doglegdata.trustradius = isfinite(gnorm) ? max(1.0, gnorm) : 1.0
        end
        doglegdata.trustradius = min(doglegdata.trustradius, doglegdata.maxtrustradius)
    end

    # 2 & 3. Inner dogleg trial step and Nocedal & Wright radius adaptation
    cost_ = data.bestcost
    inner_iters = 0
    while true
        inner_iters += 1
        step_norm = doglegdata.trustradius

        if newton_valid && newton_norm <= doglegdata.trustradius
            # Newton step is inside the trust region: take it directly
            getx(data.linsystem) .= doglegdata.newton
            step_norm = newton_norm
        elseif !has_pos_curv || !isfinite(alpha) || (alpha >= doglegdata.trustradius)
            # Cauchy point is on/outside trust boundary or negative curvature: truncated gradient step
            scale = -step_norm / gnorm
            getx(data.linsystem) .= scale .* gradient
        elseif newton_valid
            # Dogleg boundary intersection: alpha < trustradius < newton_norm
            getx(data.linsystem) .= doglegdata.newton .- doglegdata.cauchy
            aw = sqnorm(getx(data.linsystem))
            c0 = fast_dot(doglegdata.cauchy, getx(data.linsystem))
            trsq = step_norm * step_norm - alpha2
            if aw > 0 && isfinite(trsq) && trsq > 0
                disc = sqrt(c0 * c0 + aw * trsq)
                tau = c0 <= 0 ? (-c0 + disc) / aw : trsq / (c0 + disc)
                tau = clamp(tau, 0.0, 1.0)
            else
                tau = 0.0
            end
            getx(data.linsystem) .*= tau
            getx(data.linsystem) .+= doglegdata.cauchy
        else
            # Newton step unavailable / indefinite: Cauchy step
            getx(data.linsystem) .= doglegdata.cauchy
            step_norm = alpha
        end

        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)

        # Evaluate actual reduction vs predicted reduction: Delta m = -g'*delta - 0.5 * delta'*H*delta
        act_red = data.bestcost - cost_
        delta_dot_g = fast_dot(gradient, getx(data.linsystem))
        delta_H_delta = fast_bAb(hessian, getx(data.linsystem))
        pred_red = -delta_dot_g - 0.5 * delta_H_delta

        rho = (isfinite(act_red) && isfinite(pred_red) && pred_red > 0) ? (act_red / pred_red) : -Inf

        if rho > 1e-4
            # Step accepted: adapt radius for next outer iteration
            if rho > 0.75 && step_norm >= 0.9 * doglegdata.trustradius
                doglegdata.trustradius = min(2.0 * doglegdata.trustradius, doglegdata.maxtrustradius)
            elseif rho < 0.25
                doglegdata.trustradius = 0.25 * doglegdata.trustradius
            end
            return cost_
        else
            # Step rejected: shrink radius and retry
            doglegdata.trustradius = 0.25 * min(doglegdata.trustradius, step_norm)
            step_sz = maximum(abs, getx(data.linsystem))::Float64
            if !(step_sz >= options.dstep) || !(doglegdata.trustradius >= floatmin(Float64)) || !isfinite(doglegdata.trustradius) || inner_iters > 100
                return cost_
            end
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
        lmd.lambda = max(max_diag(ls) * 1e-6, 1e-12)
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
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(ls.A.nzval[idx]))
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
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(A[i, i]))
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
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || !isfinite(step_sz) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem)))
                levmardata.lambda *= stepquality < 0.983 ? 1.0 - (2.0 * stepquality - 1.0)^3 : 0.1
                levmardata.lambda = max(levmardata.lambda, 1e-12)
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
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || !isfinite(step_sz) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem)))
                levmardata.lambda *= stepquality < 0.983 ? 1.0 - (2.0 * stepquality - 1.0)^3 : 0.1
                levmardata.lambda = max(levmardata.lambda, 1e-12)
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
