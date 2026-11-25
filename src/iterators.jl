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
function probe_shift!(ls, linsolver)
    md = max_diag(ls)
    (isfinite(md) && md > 0) || return NotPositiveDefinite
    dampen!(ls, max(dampingfloor(ls), Float64(eps(systemeltype(ls)))) * md)
    return solve!(ls, linsolver)
end

# First diagonal shift tried when an undamped Newton solve fails. `max_diag` is NaN if the
# diagonal holds a NaN, and `dampen!` ignores a NaN shift, so fall back to a fixed value.
function initial_shift(ls)
    md = max_diag(ls)
    return isfinite(md) ? max(md * 1e-4, 1e-6) : 1e-6
end

# Whether the step the iterator last returned was restricted: shortened or damped by the
# iterator's safeguard rather than taken as its model's minimiser. The loop does not read a small
# step or a small change in cost after an accepted restricted step as convergence. Each iterator
# records it for the step it returns; an iterator that records nothing is never restricted.
restrictedstep(iteratedata) = false   # COV_EXCL_LINE: runs (test/edgepaths.jl), but a constant-result method is called without running its body, so coverage never counts it

# Newton optimization: a step is restricted when its solve needed a diagonal shift
mutable struct NewtonData
    restricted::Bool
end
iteratorname(::StaticSymbol{:newton}) = "Newton"
construct(::StaticSymbol{:newton}, ls, ::True) = NewtonData(false)
reset!(nd::NewtonData) = (nd.restricted = false; nothing)
restrictedstep(nd::NewtonData) = nd.restricted

@inline function iterate!(nd::NewtonData, data, f, fdf!, options::OptimizationOptions)::Float64
    status = Success
    nd.restricted = false
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        if status == Success
        else
            nd.restricted = true
            status = probe_shift!(data.linsystem, options.linsolver)
            mu = initial_shift(data.linsystem)
            shift_iters = isfinitesystem(data.linsystem) ? 30 : 0   # damping cannot repair a non-finite matrix
            while status != Success && shift_iters > 0
                shift_iters -= 1
                dampen!(data.linsystem, mu)
                mu *= 4.0
                status = solve!(data.linsystem, options.linsolver)
            end
            dampen!(data.linsystem, 0.0)
        end
    end
    if status == Success   # a successful solve's x is finite (DampedLinearSolve's contract)
        negate!(getx(data.linsystem))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        return cost_
    end
    return LINSOLVER_FAILED
end

mutable struct NewtonDataRelinearize
    cost::Float64
    restricted::Bool
end
construct(::StaticSymbol{:newton}, ls, ::False) = NewtonDataRelinearize(0.0, false)
reset!(nd::NewtonDataRelinearize) = (nd.restricted = false; nothing)
restrictedstep(nd::NewtonDataRelinearize) = nd.restricted

@inline function iterate!(nd::NewtonDataRelinearize, data, f, fdf!, options::OptimizationOptions)::Float64
    status = Success
    nd.restricted = false
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        if status == Success
        else
            nd.restricted = true
            status = probe_shift!(data.linsystem, options.linsolver)
            mu = initial_shift(data.linsystem)
            shift_iters = isfinitesystem(data.linsystem) ? 30 : 0   # damping cannot repair a non-finite matrix
            while status != Success && shift_iters > 0
                shift_iters -= 1
                dampen!(data.linsystem, mu)
                mu *= 4.0
                status = solve!(data.linsystem, options.linsolver)
            end
            dampen!(data.linsystem, 0.0)
        end
    end
    if status == Success   # a successful solve's x is finite (DampedLinearSolve's contract)
        negate!(getx(data.linsystem))
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        zero!(data.linsystem)
        data.gradients += @elapsed_ns nd.cost = fdf!(getb(data.linsystem), getA(data.linsystem), data.varnext)
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

    # Newton step, with the same shifted-retry fallback as NewtonData
    newton_norm = Inf
    newton_pred = 0.0
    newton_valid = false
    status = Success
    data.solves += @elapsed_ns begin
        status = solve!(data.linsystem, options.linsolver)
        shifted = status != Success
        if shifted
            status = probe_shift!(data.linsystem, options.linsolver)
            mu = initial_shift(data.linsystem)
            shifts = isfinitesystem(data.linsystem) ? 20 : 0
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
abstract type LevMarDataTrial <: LevMarData end

mutable struct LevenbergData <: LevMarData
    lambda::Float64
    restricted::Bool      # the damping term exceeded the curvature term along the accepted step
end
LevenbergData(lambda::Float64) = LevenbergData(lambda, false)
iteratorname(::StaticSymbol{:levenbergmarquardt}) = "Levenberg-Marquardt"
construct(::StaticSymbol{:levenbergmarquardt}, ls, ::True) = LevenbergData(0.0)
reset!(lmd::LevenbergData) = (lmd.lambda = 0.0; lmd.restricted = false)

mutable struct LevenbergDataTrial{L} <: LevMarDataTrial
    lambda::Float64
    trial::L
    trialcost::Float64
    trialready::Bool
    restricted::Bool
end
LevenbergDataTrial(lambda::Float64, trial, trialcost::Float64, trialready::Bool) = LevenbergDataTrial(lambda, trial, trialcost, trialready, false)
construct(::StaticSymbol{:levenbergmarquardt}, ls, ::False) = LevenbergDataTrial(0.0, copy_structure(ls), NaN, false)
reset!(lmd::LevenbergDataTrial) = (lmd.trialready = false; lmd.trialcost = NaN; lmd.lambda = 0.0; lmd.restricted = false)

const AnyLevenbergData = Union{LevenbergData, LevenbergDataTrial}

# The smallest damping applied, from the linear system's `dampingfloor`, relative to the
# diagonal. Levenberg's λ multiplies I, so the floor is scaled by the largest diagonal entry
# (falling back to the damping floor when that is not finite or is zero); Marquardt's λ multiplies the scaled
# diagonal already. Both are capped at the damping floor `eps(T)^(3/4)`, T the system's element type (1.8e-12 for
# Float64, where it was the constant 1e-12; 6.4e-6 for Float32; a starting value, rules.md 26): a higher floor (eps
# times a large diagonal) holds back the weak directions of an ill-conditioned problem even where Cholesky still
# resolves a smaller damping, and failed solves are caught by the retry loop.
dampingcap(ls) = Float64(eps(systemeltype(ls)))^(3 / 4)
function lambdafloor(::AnyLevenbergData, ls)
    md = max_diag(ls)
    cap = dampingcap(ls)
    return (isfinite(md) && md > 0) ? min(dampingfloor(ls) * md, cap) : cap
end

function updatescaling!(lmd::AnyLevenbergData, ls)
    if lmd.lambda == 0
        md = max_diag(ls)
        lmd.lambda = isfinite(md) ? max(md * 1e-6, lambdafloor(lmd, ls)) : 1e-6
    end
end

dampen!(ls::AbstractLinearSystem, lmd::AnyLevenbergData) = dampen!(ls, lmd.lambda)

mutable struct MarquardtData{T} <: LevMarData
    lambda::Float64
    scaling::T
    diag::T               # the undamped diagonal, written by `getdiag!` for `updatescaling!`
    restricted::Bool
end
MarquardtData(lambda::Float64, scaling) = MarquardtData(lambda, scaling, similar(scaling), false)
iteratorname(::StaticSymbol{:levenbergmarquardt_dtd}) = "Levenberg-Marquardt (D'D scaling)"
construct(::StaticSymbol{:levenbergmarquardt_dtd}, ls, ::True) = MarquardtData(0.0, zero(getx(ls)))
reset!(lmd::MarquardtData) = (fill!(lmd.scaling, 0.0); lmd.lambda = 0.0; lmd.restricted = false)

mutable struct MarquardtDataTrial{T, L} <: LevMarDataTrial
    lambda::Float64
    scaling::T
    diag::T               # the undamped diagonal, written by `getdiag!` for `updatescaling!`
    trial::L
    trialcost::Float64
    trialready::Bool
    restricted::Bool
end
MarquardtDataTrial(lambda::Float64, scaling, trial, trialcost::Float64, trialready::Bool) =
    MarquardtDataTrial(lambda, scaling, similar(scaling), trial, trialcost, trialready, false)
construct(::StaticSymbol{:levenbergmarquardt_dtd}, ls, ::False) = MarquardtDataTrial(0.0, zero(getx(ls)), copy_structure(ls), NaN, false)
reset!(lmd::MarquardtDataTrial) = (fill!(lmd.scaling, 0.0); lmd.trialready = false; lmd.trialcost = NaN; lmd.lambda = 0.0; lmd.restricted = false)

const AnyMarquardtData = Union{MarquardtData, MarquardtDataTrial}

# Marquardt scaling from the undamped diagonal of A (the squared column norms of J for a Jacobian system), which
# the system writes (`getdiag!`): a running maximum of its absolute values, a non-finite entry counting as 1,
# then a floor of 1e-8 times the largest scale (and at least 1e-12). The scale is in the units of diag(A), as
# `dampen!(ls, λ, s)` expects. The relative floor is about the precision of the matrix: 1e-8 for Float64, and
# eps(Float32)^(1/2) ≈ 3.5e-4 for Float32 (the rule of single-precision.md; a starting value).
function updatescaling!(lmd::AnyMarquardtData, ls)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(lmd.scaling))
    diag = getdiag!(lmd.diag, ls)
    for i in eachindex(diag)
        val = abs(diag[i])
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], isfinite(val) ? val : 1.0)
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, max(m * scalingfloor(systemeltype(ls)), 1e-12))
end
scalingfloor(::Type{Float64}) = 1e-8
scalingfloor(::Type{T}) where T = Float64(eps(T))^(1 / 2)

lambdafloor(::AnyMarquardtData, ls) = min(dampingfloor(ls), dampingcap(ls))

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

restrictedstep(lmd::LevMarData) = lmd.restricted

# Whether the damping term exceeds the curvature term along an accepted step s, λ·sᵀDs > sᵀHs.
# sᵀHs = 2·(predicted − gᵀs), from the undamped predicted change already computed, so this costs
# one dot product. A non-positive sᵀHs with λ > 0 is restricted.
lm_restricted(lmd, ls, step, predicted) = lmd.lambda * dampingnorm(lmd, step) > 2 * (predicted - fast_dot(getgrad(ls), step))

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
    levmardata.restricted = false
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
                # Damping cannot repair a non-finite matrix: give up at once
                if levmardata.lambda > 1e16 || !isfinitesystem(data.linsystem)
                    dampen!(data.linsystem, 0.0)
                    break
                end
            end
        end
        if status != Success
            return LINSOLVER_FAILED
        end
        step_sz = maximum(abs, getx(data.linsystem))::Float64   # finite: the solve succeeded
        data.varnext = update!(data.varnext, data.variables, getx(data.linsystem))
        data.costs += @elapsed_ns cost_ = f(data.varnext)
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                predicted = quadratic_model_change(data.linsystem, getx(data.linsystem))
                levmardata.restricted = lm_restricted(levmardata, data.linsystem, getx(data.linsystem), predicted)
                levmardata.lambda = max(levmardata.lambda * lambda_update_factor(cost_ - data.bestcost, predicted), lambdafloor(levmardata, data.linsystem))
            end
            return cost_
        end
        levmardata.lambda *= mu
        mu *= 2.0
    end
end

@inline function iterate!(levmardata::LevMarDataTrial, data, f, fdf!, options::OptimizationOptions)::Float64
    @assert levmardata.lambda >= 0.0
    levmardata.restricted = false
    gradient = getgrad(data.linsystem)
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
                # Damping cannot repair a non-finite matrix: give up at once
                if levmardata.lambda > 1e16 || !isfinitesystem(data.linsystem)
                    dampen!(data.linsystem, 0.0)
                    break
                end
            end
        end
        if status != Success
            return LINSOLVER_FAILED
        end
        step_sz = maximum(abs, getx(data.linsystem))::Float64   # finite: the solve succeeded
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
        if cost_ <= data.bestcost || !(step_sz >= options.dstep) || inner_iters > 100 || levmardata.lambda > 1e16
            dampen!(data.linsystem, 0.0)
            if cost_ <= data.bestcost
                predicted = quadratic_model_change(data.linsystem, getx(data.linsystem))
                levmardata.restricted = lm_restricted(levmardata, data.linsystem, getx(data.linsystem), predicted)
                levmardata.lambda = max(levmardata.lambda * lambda_update_factor(cost_ - data.bestcost, predicted), lambdafloor(levmardata, data.linsystem))
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
