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
    data.solves += @elapsed_ns negate!(solve!(data.linsystem, options.linsolver))
    data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
    data.costs += @elapsed_ns cost_ = f(data.varnext)
    return cost_
end

mutable struct NewtonDataRelinearize
    cost::Float64
end
construct(::StaticSymbol{:newton}, ls, ::False) = NewtonDataRelinearize(0.0)
reset!(::NewtonDataRelinearize) = nothing

@inline function iterate!(nd::NewtonDataRelinearize, data, f, fdf!, options::OptimizationOptions)::Float64
    gethessian(data.linsystem)
    data.solves += @elapsed_ns negate!(solve!(data.linsystem, options.linsolver))
    data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
    zero!(data.linsystem)
    data.gradients += @elapsed_ns nd.cost = fdf!(getb(data.linsystem), getA(data.linsystem), data.varnext)
    return nd.cost
end

linearize!(nd::NewtonDataRelinearize, data, fdf!) = nd.cost

# Dogleg optimization
mutable struct DoglegData{T}
    trustradius::Float64
    cauchy::T
    newton::T

    function DoglegData(ls)
        return new{typeof(getx(ls))}(0.0, similar(getx(ls)), similar(getx(ls)))
    end
end
construct(::StaticSymbol{:dogleg}, ls, ::Any) = DoglegData(ls)
reset!(dd::DoglegData) = (dd.trustradius = 0.0)

@inline function iterate!(doglegdata::DoglegData, data, f, fdf!, options::OptimizationOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    gnorm2 = sqnorm(gradient)
    gnorm = isfinite(gnorm2) ? sqrt(gnorm2) : norm(gradient)
    if !(gnorm > 0)
        fill!(getx(data.linsystem), zero(eltype(gradient)))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        return cost_
    end
    beta = 0.0
    a = Inf
    alpha = Inf
    alpha2 = Inf
    gHg = 0.0
    newton_dot_g = 0.0
    data.solves += @elapsed_ns begin
        gHg = fast_bAb(hessian, gradient)
        if gHg > 0
            a = gnorm2 / gHg
            alpha = a * gnorm
        end
        alpha2 = alpha * alpha
        if isfinite(alpha2)
            doglegdata.cauchy .= (-a) .* gradient
            if doglegdata.trustradius == 0
                doglegdata.trustradius = alpha
            end
        else
            alpha = Inf
            alpha2 = Inf
            if doglegdata.trustradius == 0
                doglegdata.trustradius = isfinite(gnorm) ? gnorm : 1.0
            end
        end
        if alpha < doglegdata.trustradius
            negate!(solve!(data.linsystem, options.linsolver))
            doglegdata.newton .= getx(data.linsystem)
            beta = fast_norm(doglegdata.newton)
            newton_dot_g = fast_dot(gradient, doglegdata.newton)
            if !isfinite(beta) || !(newton_dot_g < 0)
                doglegdata.trustradius = alpha
            end
        end
    end
    basecost = max(-0.5 * newton_dot_g, floatmin(Float64))
    curv = isfinite(gHg) && isfinite(gnorm2) ? gHg / gnorm2 : 0.0
    cost_ = data.bestcost
    inner_iters = 0
    while true
        inner_iters += 1
        if !(alpha < doglegdata.trustradius)
            step_norm = doglegdata.trustradius
            scale = -step_norm / gnorm
            getx(data.linsystem) .= scale .* gradient
            linear_approx = step_norm * (gnorm - 0.5 * step_norm * curv)
        else
            if beta <= doglegdata.trustradius
                step_norm = beta
                getx(data.linsystem) .= doglegdata.newton
                linear_approx = basecost
            else
                step_norm = doglegdata.trustradius
                getx(data.linsystem) .= doglegdata.newton .- doglegdata.cauchy
                sq_leg = sqnorm(getx(data.linsystem))
                c = fast_dot(doglegdata.cauchy, getx(data.linsystem))
                trsq = step_norm * step_norm - alpha2
                if sq_leg > 0 && isfinite(trsq)
                    disc = sqrt(c * c + sq_leg * trsq)
                    step = c <= 0 ? (-c + disc) / sq_leg : trsq / (c + disc)
                else
                    step = 0.0
                end
                getx(data.linsystem) .*= step
                getx(data.linsystem) .+= doglegdata.cauchy
                linear_approx = 0.5 * (a * (1 - step)^2 * gnorm2) + step * (2 - step) * basecost
            end
        end
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        mu = (data.bestcost - cost_) / linear_approx
        if cost_ < data.bestcost && mu > 0.375
            doglegdata.trustradius = max(doglegdata.trustradius, 3 * step_norm)
        elseif !(cost_ <= data.bestcost && mu >= 0.125)
            doglegdata.trustradius = 0.5 * step_norm
        end
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || !(doglegdata.trustradius >= floatmin(Float64)) || !isfinite(doglegdata.trustradius) || inner_iters > 100
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
        lmd.lambda = max(max_diag(ls) * 1e-6, 1e-12)
    end
end

dampen!(ls, ::AnyLevenbergData, dlambda) = uniformscaling!(ls, dlambda)

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

dampen!(ls, lmd::AnyMarquardtData, dlambda) = damp_marquardt!(ls, lmd.scaling, dlambda)

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
    lastlambda = 0.0
    mu = 2.0
    inner_iters = 0
    while true
        inner_iters += 1
        dampen!(data.linsystem, levmardata, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        data.solves += @elapsed_ns negate!(solve!(data.linsystem, options.linsolver))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || !isfinite(step_sz) || inner_iters > 100
            dampen!(data.linsystem, levmardata, -lastlambda)
            stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem)))
            levmardata.lambda *= stepquality < 0.983 ? 1.0 - (2.0 * stepquality - 1.0)^3 : 0.1
            levmardata.lambda = max(levmardata.lambda, 1e-12)
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
    lastlambda = 0.0
    mu = 2.0
    levmardata.trialready = false
    firsttrial = true
    inner_iters = 0
    while true
        inner_iters += 1
        dampen!(data.linsystem, levmardata, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        data.solves += @elapsed_ns negate!(solve!(data.linsystem, options.linsolver))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        step = getx(data.linsystem)
        if firsttrial && 0.5 * (levmardata.lambda * dampingnorm(levmardata, step) - fast_dot(gradient, step)) >= abs(data.bestcost) * options.reldcost
            levmardata.trial = sharex(levmardata.trial, data.linsystem)
            zero!(levmardata.trial)
            data.gradients += @elapsed_ns cost_ = fdf!(getb(levmardata.trial), getA(levmardata.trial), data.varnext)
            levmardata.trialready = true
            levmardata.trialcost = cost_
        else
            data.costs += @elapsed_ns cost_ = f(data.varnext)
        end
        firsttrial = false
        step_sz = maximum(abs, getx(data.linsystem))::Float64
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || !isfinite(step_sz) || inner_iters > 100
            dampen!(data.linsystem, levmardata, -lastlambda)
            stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem)))
            levmardata.lambda *= stepquality < 0.983 ? 1.0 - (2.0 * stepquality - 1.0)^3 : 0.1
            levmardata.lambda = max(levmardata.lambda, 1e-12)
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
