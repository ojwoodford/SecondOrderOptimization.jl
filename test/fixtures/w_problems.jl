# Both objectives of a least-squares problem: the HessianObjective forms the normal equations
# from the same J
function w_objectives(resid, jac, m, n)
    f(x) = 0.5 * sum(abs2, resid(x))
    resjac!(r, J, x) = (r .= resid(x); J .= jac(x); 0.5 * sum(abs2, r))
    fdf!(g, H, x) = (r = resid(x); J = jac(x); g .= J' * r; H .= J' * J; 0.5 * sum(abs2, r))
    return JacobianObjective(f, resjac!, m), HessianObjective(f, fdf!; H = zeros(n, n))
end

# The iterator's own state after each iteration: the damping, the trust radius, or nothing
w_state(itd) = hasproperty(itd, :lambda) ? itd.lambda : hasproperty(itd, :trustradius) ? itd.trustradius : 0.0

# A callback recording the cost, the iterator's state and the variables after each iteration
function w_trace()
    costs, states, xs = Float64[], Float64[], Vector{Float64}[]
    cb = (cost, x, data, itd) -> (push!(costs, cost); push!(states, w_state(itd)); push!(xs, collect(Float64, x)); (cost, 0))
    return costs, states, xs, cb
end

w_opts(it, extra...) = OptimizationOptions(; iterator=it, extra...)

# The well-conditioned problems of the J tests: Rosenbrock as two residuals; an exponential fit
# with a nonzero residual; n = 30. The J tests' n = 30 problem, M tanh.(x) - y, has no finite
# minimizer (some variables run off to infinity, where sech² makes JᵀJ singular), so here the
# nonlinearity is the monotone x + sin(x)/2, which keeps J well conditioned and the minimizer finite.
function w_problems()
    t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
    rng = Xoshiro(11); M = randn(rng, 45, 30) ./ sqrt(45) + I(45)[:, 1:30]; yy = randn(rng, 45)
    return (
        ("Rosenbrock", x -> [1 - x[1], 10 * (x[2] - x[1]^2)], x -> [-1.0 0.0; -20x[1] 10.0], 2, [-1.2, 1.0]),
        ("exponential fit", x -> @.(x[1] * exp(x[2] * t) - y), x -> hcat(exp.(x[2] .* t), x[1] .* t .* exp.(x[2] .* t)), 25, [1.0, 0.0]),
        ("n = 30", x -> M * (x .+ 0.5 .* sin.(x)) .- yy, x -> M * Diagonal(1 .+ 0.5 .* cos.(x)), 45, zeros(30)),
    )
end

# The degree-13 polynomial fit of the J tests (cond(J) ≈ 4e9), scaled so that the iterators'
# absolute floors on the damping are far below the smallest squared singular value
function w_illconditioned()
    tt = collect(range(0, 1; length=60))
    V = 1e4 * hcat((tt .^ k for k in 0:13)...)
    yv = 1e4 * (@. sin(3tt) + 0.01 * cos(17tt))
    return V, yv
end

# A two-residual Rosenbrock counting the calls to f and resjac!
const WCALLS = Ref(0)
wrosen_cost(x) = (WCALLS[] += 1; 0.5 * ((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2))
function wrosen_resjac!(r, J, x)
    WCALLS[] += 1
    r[1] = 1 - x[1]
    r[2] = 10 * (x[2] - x[1]^2)
    J[1, 1] = -1.0
    J[2, 1] = -20 * x[1]
    J[2, 2] = 10.0
    return 0.5 * (r[1]^2 + r[2]^2)
end

# A dogleg callback recording, for each iteration, the kind of step taken (the Gauss-Newton step,
# a step along the gradient, or the interpolated dogleg step), the number of trial points
# evaluated (the points in `pts`, which the objective records, other than the iteration's start;
# emptied here), the trust radius before and after, the step norm, and the reduction ratio as the
# iterator computes it
function w_doglegtrace(pts::Vector{Vector{Float64}})
    rec = NamedTuple[]
    prevΔ = Ref(NaN)
    cb = function (cost, x, data, dd)
        ls = data.linsystem
        step = collect(getx(ls)); g = collect(SOO.getgrad(ls))
        along = norm(step - dot(step, g) / dot(g, g) * g) <= 1e-12 * norm(step)
        kind = step == dd.newton ? :gaussnewton : along ? :gradient : :dogleg
        noise = 10eps() * abs(data.bestcost)
        rho = (data.bestcost - cost + noise) / (-quadratic_model_change(ls, step) + noise)
        ntrials = count(!=(collect(Float64, data.variables)), pts)
        push!(rec, (; kind, ntrials, before=prevΔ[], after=dd.trustradius, stepnorm=norm(step), rho, newton=copy(dd.newton), grad=g))
        prevΔ[] = dd.trustradius
        empty!(pts)
        return (cost, 0)
    end
    return rec, cb
end

# Allocation of a warm optimize! with a workspace, measured in a function (Julia 1.10 boxes
# variables captured at testset top level)
function w_allocs(x, obj, opts, ws)
    x .= (-1.2, 1.0)
    return @allocated optimize!(x, obj, opts; workspace=ws)
end
