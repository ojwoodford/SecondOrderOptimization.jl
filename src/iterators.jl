negate!(x) = @.(x = -x)

# The iterators' names for the linear system's parts: the gradient is b, the Hessian A
getgrad(ls) = getb(ls)
gethessian(ls) = getA(ls)
gethessgrad(ls) = gethessian(ls), getgrad(ls)

# An iterator exists if it has a display name: each iterator defines `iteratorname` beside its
# `construct`, `reset!` and `iterate!`. The names accepted are found from those methods.
iteratorname(::StaticSymbol) = nothing
iteratorname(s::Symbol) = iteratorname(static(s))

function knowniterators()
    syms = Symbol[]
    for m in methods(iteratorname)
        T = Base.unwrap_unionall(m.sig).parameters[2]
        T isa DataType && T <: StaticSymbol && T.parameters[1] isa Symbol && push!(syms, T.parameters[1])
    end
    return sort!(syms)
end

function validateiterator(s::Symbol)
    isnothing(iteratorname(s)) || return nothing
    throw(ArgumentError("Unknown iterator: :$s. The iterators are " * join((":$x" for x in knowniterators()), ", ", " and ") * "."))
end

function computelinearization!(data, fdf!, x)
    zero!(data.linsystem)
    cost = 0.0
    data.gradients += @elapsed_ns cost = fdf!(getb(data.linsystem), getA(data.linsystem), x)
    return cost
end

# The first retry when an undamped Newton solve fails: a shift of eps·max_diag, eps that of the system's element type
# (at least the solver's `dampingfloor`, at which a positive-definite matrix solves). It cures a failure that came
# from rounding alone in an ill-conditioned matrix that is positive definite, without the larger
# shift below making the step negligible. If it fails, the matrix is not positive definite, and the
# shifted retry below runs as before.
function probe_shift!(data, options)
    md = maxdiagonal(data)
    (isfinite(md) && md > 0) || return false
    return solvestep!(data, options, max(dampingfloor(data), Float64(eps(systemeltype(data)))) * md)
end

# First diagonal shift tried when an undamped Newton solve fails. `max_diag` is NaN if the
# diagonal holds a NaN, and `dampen!` ignores a NaN shift, so fall back to a fixed value.
function initial_shift(data)
    md = maxdiagonal(data)
    return isfinite(md) ? max(md * 1e-4, 1e-6) : 1e-6
end

# The Newton step, shared by Newton and dogleg: the undamped solve, and if it fails the probe and then up to
# `maxshifts` growing diagonal shifts (none for a non-finite matrix, which damping cannot repair), leaving the system
# undamped. Returns whether a solve succeeded and whether the undamped one failed.
@inline function newtonstep!(data, options, maxshifts::Int)
    solvestep!(data, options) && return true, false
    ok = probe_shift!(data, options)
    mu = initial_shift(data)
    shifts = systemisfinite(data) ? maxshifts : 0
    while !ok && shifts > 0
        shifts -= 1
        ok = solvestep!(data, options, mu)
        mu *= 4.0
    end
    undamp!(data)
    return ok, true
end

# Whether the step the iterator last returned was restricted: shortened or damped by the
# iterator's safeguard rather than taken as its model's minimiser. The loop does not read a small
# step or a small change in cost after an accepted restricted step as convergence. Each iterator
# records it for the step it returns; an iterator that records nothing is never restricted.
restrictedstep(iteratedata) = false   # COV_EXCL_LINE: runs (test/edgepaths.jl), but a constant-result method is called without running its body, so coverage never counts it

# Newton optimization: a step is restricted when its solve needed a diagonal shift. With callbackchangesstate = false
# Newton always moves to its trial point, so it linearizes there at once (`linearizetrial!`) and its `linearize!`
# returns that cost; otherwise it evaluates the cost alone.
mutable struct NewtonData{R <: StaticBool}
    cost::Float64         # the cost at the last trial point, when relinearizing
    restricted::Bool
    relinearize::R
end
iteratorname(::StaticSymbol{:newton}) = "Newton"
construct(::StaticSymbol{:newton}, ls, callbackchangesstate::StaticBool) = NewtonData(0.0, false, !callbackchangesstate)
reset!(nd::NewtonData) = (nd.restricted = false; nothing)
restrictedstep(nd::NewtonData) = nd.restricted

@inline function iterate!(nd::NewtonData, data, f, fdf!, options::OptimizationOptions)::Float64
    ok, nd.restricted = newtonstep!(data, options, 30)
    ok || return LINSOLVER_FAILED
    settrial!(data)
    if dynamic(nd.relinearize)
        nd.cost = linearizetrial!(data, fdf!)
        return nd.cost
    end
    return trialcost!(data, f)
end

linearize!(nd::NewtonData{True}, data, fdf!) = nd.cost

# Dogleg optimization
mutable struct DoglegData{T,L}
    trustradius::Float64
    maxtrustradius::Float64
    cauchy::T
    newton::T
    trial::L              # nothing, or a spare linear system for reusing the trial linearization
    trialcost::Float64
    trialready::Bool
    restricted::Bool      # the step returned was not the full Newton point of an unshifted solve
end

function DoglegData(ls, maxtrustradius::Float64=1e10; trial=nothing)
    x = getx(ls)
    return DoglegData(0.0, maxtrustradius, similar(x), similar(x), trial, NaN, false, false)
end

iteratorname(::StaticSymbol{:dogleg}) = "Dogleg"
construct(::StaticSymbol{:dogleg}, ls, ::True) = DoglegData(ls)
construct(::StaticSymbol{:dogleg}, ls, ::False) = DoglegData(ls; trial=copy_structure(ls))
reset!(dd::DoglegData) = (dd.trustradius = 0.0; dd.trialready = false; dd.trialcost = NaN; dd.restricted = false)
restrictedstep(dd::DoglegData) = dd.restricted

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
    dd.restricted = false
    hessian, gradient = gethessgrad(data.linsystem)

    gHg = quadform(hessian, gradient)
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

    # Newton step, with the same shifted-retry fallback as Newton
    newton_norm = Inf
    newton_pred = 0.0
    newton_valid = false
    ok, shifted = newtonstep!(data, options, 20)   # leaves the system undamped, for the model below
    if ok
        dd.newton .= getx(data.linsystem)
        newton_norm = fast_norm(dd.newton)
        newton_dot_g = fast_dot(gradient, dd.newton)
        if isfinite(newton_norm) && newton_dot_g < 0
            # Validity = positive predicted model reduction under the true (undamped) H
            newton_pred = -quadratic_model_change(data.linsystem, dd.newton)
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
        fullnewton = false

        if newton_valid && newton_norm <= Δ
            # Full Newton step inside the region
            fullnewton = true
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
            # tau > 0, so the step ends on the boundary (step_norm = Δ): cauchy_norm < Δ ≤ maxtrustradius keeps trsq
            # finite and positive, and aw * trsq could only overflow for a Newton step some 1e144 long
            x .*= tau
            x .+= dd.cauchy
            pred_red = -quadratic_model_change(data.linsystem, x)
        else
            # Newton unavailable: full Cauchy step
            x .= dd.cauchy
            step_norm = cauchy_norm
            pred_red = 0.5 * cauchy_a * gnorm2
        end

        step_sz = maximum(abs, x)::Float64   # finite: every step above is built from finite parts

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
            dd.restricted = !(fullnewton && !shifted)
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

mutable struct LevenbergData{L} <: LevMarData
    lambda::Float64
    tl::TrialLinearization{L}
    restricted::Bool      # the damping term exceeded the curvature term along the accepted step
end
iteratorname(::StaticSymbol{:levenbergmarquardt}) = "Levenberg-Marquardt"
construct(::StaticSymbol{:levenbergmarquardt}, ls, callbackchangesstate::StaticBool) = LevenbergData(0.0, TrialLinearization(ls, callbackchangesstate), false)
reset!(lmd::LevenbergData) = (reset!(lmd.tl); lmd.lambda = 0.0; lmd.restricted = false)

# The smallest damping applied, from the linear system's `dampingfloor`, relative to the
# diagonal. Levenberg's λ multiplies I, so the floor is scaled by the largest diagonal entry
# (falling back to the damping floor when that is not finite or is zero); Marquardt's λ multiplies the scaled
# diagonal already. Both are capped at the damping floor `eps(T)^(3/4)`, T the system's element type (1.8e-12 for
# Float64, where it was the constant 1e-12; 6.4e-6 for Float32; a starting value, rules.md 26): a higher floor (eps
# times a large diagonal) holds back the weak directions of an ill-conditioned problem even where Cholesky still
# resolves a smaller damping, and failed solves are caught by the retry loop.
dampingcap(data) = Float64(eps(systemeltype(data)))^(3 / 4)
function lambdafloor(::LevenbergData, data)
    md = maxdiagonal(data)
    cap = dampingcap(data)
    return (isfinite(md) && md > 0) ? min(dampingfloor(data) * md, cap) : cap
end

function updatescaling!(lmd::LevenbergData, data)
    if lmd.lambda == 0
        md = maxdiagonal(data)
        lmd.lambda = isfinite(md) ? max(md * 1e-6, lambdafloor(lmd, data)) : 1e-6
    end
end

# The damped solve, at the iterator's damping
dampedsolve!(lmd::LevenbergData, data, options) = solvestep!(data, options, lmd.lambda)

mutable struct MarquardtData{T, L} <: LevMarData
    lambda::Float64
    scaling::T
    diag::T               # the undamped diagonal, written by `getdiagonal!` for `updatescaling!`
    tl::TrialLinearization{L}
    restricted::Bool
end
iteratorname(::StaticSymbol{:levenbergmarquardt_dtd}) = "Levenberg-Marquardt (D'D scaling)"
construct(::StaticSymbol{:levenbergmarquardt_dtd}, ls, callbackchangesstate::StaticBool) =
    MarquardtData(0.0, fill!(similarstep(ls), 0.0), similarstep(ls), TrialLinearization(ls, callbackchangesstate), false)
reset!(lmd::MarquardtData) = (fill!(lmd.scaling, 0.0); reset!(lmd.tl); lmd.lambda = 0.0; lmd.restricted = false)

# Marquardt scaling from the undamped diagonal of A (the squared column norms of J for a Jacobian system), which
# the system writes (`getdiag!`): a running maximum of its absolute values, a non-finite entry counting as 1,
# then a floor of 1e-8 times the largest scale (and at least 1e-12). The scale is in the units of diag(A), as
# `dampen!(ls, λ, s)` expects. The relative floor is about the precision of the matrix: 1e-8 for Float64, and
# eps(Float32)^(1/2) ≈ 3.5e-4 for Float32 (the rule of single-precision.md; a starting value).
function updatescaling!(lmd::MarquardtData, data)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(lmd.scaling))
    diag = getdiagonal!(lmd.diag, data)
    for i in eachindex(diag)
        val = abs(diag[i])
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], isfinite(val) ? val : 1.0)
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, max(m * scalingfloor(systemeltype(data)), 1e-12))
end
scalingfloor(::Type{Float64}) = 1e-8
scalingfloor(::Type{T}) where T = Float64(eps(T))^(1 / 2)

lambdafloor(::MarquardtData, data) = min(dampingfloor(data), dampingcap(data))

dampedsolve!(lmd::MarquardtData, data, options) = solvestep!(data, options, lmd.lambda, lmd.scaling)

dampingnorm(::LevenbergData, x) = sqnorm(x)
function dampingnorm(lmd::MarquardtData, x)
    total = 0.0
    scaling = lmd.scaling
    @inbounds @fastmath @simd for i in eachindex(x)
        total += scaling[i] * x[i] * x[i]
    end
    return total
end

restrictedstep(lmd::LevMarData) = lmd.restricted
triallinearization(lmd::LevMarData) = lmd.tl

# Whether the damping term exceeds the curvature term along an accepted step s, λ·sᵀDs > sᵀHs.
# sᵀHs = 2·(predicted − gᵀs), from the undamped predicted change already computed, so this costs
# one dot product. A non-positive sᵀHs with λ > 0 is restricted.
lm_restricted(lmd, data, step, predicted) = lmd.lambda * dampingnorm(lmd, step) > 2 * (predicted - fast_dot(getgradient(data), step))

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
    levmardata.restricted = false
    gradient = getgradient(data)
    updatescaling!(levmardata, data)
    mu = 2.0
    discardtrial!(levmardata.tl)
    firsttrial = true
    inner_iters = 0
    while true
        inner_iters += 1
        ok = false
        while true
            ok = dampedsolve!(levmardata, data, options)
            ok && break
            discardtrial!(levmardata.tl)
            levmardata.lambda *= mu
            mu *= 2.0
            # Damping cannot repair a non-finite matrix: give up at once
            if levmardata.lambda > 1e16 || !systemisfinite(data)
                undamp!(data)
                break
            end
        end
        if !ok
            return LINSOLVER_FAILED
        end
        step = getstep(data)
        step_sz = maximum(abs, step)::Float64   # finite: the solve succeeded
        settrial!(data)
        # On the first trial, linearize at the trial point too if the step is likely to be accepted
        likely = firsttrial && 0.5 * (levmardata.lambda * dampingnorm(levmardata, step) - fast_dot(gradient, step)) >= abs(getbestcost(data)) * options.reldcost
        cost_ = trialcost!(levmardata.tl, data, f, fdf!, likely)
        firsttrial = false
        if cost_ <= getbestcost(data) || !(step_sz >= options.dstep) || inner_iters > 100 || levmardata.lambda > 1e16
            undamp!(data)
            if cost_ <= getbestcost(data)
                predicted = predictedchange(data, step)
                levmardata.restricted = lm_restricted(levmardata, data, step, predicted)
                levmardata.lambda = max(levmardata.lambda * lambda_update_factor(cost_ - getbestcost(data), predicted), lambdafloor(levmardata, data))
            end
            return cost_
        end
        discardtrial!(levmardata.tl)
        levmardata.lambda *= mu
        mu *= 2.0
    end
end

printoutcallback(cost, x_trial, data, iteratedata::LevMarData) = printoutcallback(cost, data, 1.0 / iteratedata.lambda)

# Gradient descent optimization
mutable struct GradientDescentData
    stepsize::Float64
end
iteratorname(::StaticSymbol{:gradientdescent}) = "gradient descent"
construct(::StaticSymbol{:gradientdescent}, ls, ::Any) = GradientDescentData(1.0)
reset!(gdd::GradientDescentData) = (gdd.stepsize = 1.0)

@inline function iterate!(gddata::GradientDescentData, data, f, fdf!, options::OptimizationOptions)::Float64
    gradient = getgrad(data.linsystem)   # finite: the loop stops on a non-finite gradient before iterating
    getx(data.linsystem) .= .-gradient .* gddata.stepsize
    data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
    data.costs += @elapsed_ns costc = f(data.varnext)
    inner_iters = 0
    while !(costc <= data.bestcost)
        inner_iters += 1
        if inner_iters > 100 || !(gddata.stepsize >= floatmin(Float64)) || !isfinite(gddata.stepsize)
            return costc
        end
        if isfinite(costc)
            # Minimiser of the quadratic through the cost and slope at 0 and the cost at the trial
            # step, as a fraction of the step, safeguarded to [0.1, 0.5]. The slope term is not
            # positive and the trial cost is above the best, so `costdiff` is negative, never zero.
            coststep = fast_dot(getx(data.linsystem), gradient)
            costdiff = data.bestcost + coststep - costc
            gddata.stepsize *= clamp(0.5 * coststep / costdiff, 0.1, 0.5)
        else
            gddata.stepsize *= 0.5
        end
        getx(data.linsystem) .= .-gradient .* gddata.stepsize
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns costc = f(data.varnext)
    end
    gddata.stepsize *= 2.0
    return costc
end

printoutcallback(cost, x_trial, data, iteratedata::GradientDescentData) = printoutcallback(cost, data, iteratedata.stepsize)
