# The conformance check of an iterator (IteratorInterface's `check_iterator`): runs the iterator on problems defined
# here, whose `fdf!` (or `resjac!`) returns exactly `f(x)`, and checks the iterator contract. It is not called by
# `optimize`, and does not depend on Test.

# The requirements, numbered as `check_iterator`'s docstring numbers them
const CONFORMANCE_REQUIREMENTS = (
    "the value iterate! returns is a Float64, and when it is a cost it is the cost f at the trial point",
    "the trial point is the current point updated by getstep(data)",
    "restrictedstep(itdata) is a Bool",
    "the system is undamped when the callback runs",
    "a run on a problem whose Hessian is singular, or holds an Inf, ends without an exception",
    "a run on a problem whose cost is NaN away from the start ends without an exception",
    "a second run on the same OptimizationWorkspace allocates nothing and repeats the first run's costs",
    "the cost and gradient counts are the calls to f and fdf!",
)

@noinline function conformance_error(name, k, problem, where_, detail = "")
    msg = "check_iterator(:$name): requirement $k ($(CONFORMANCE_REQUIREMENTS[k])) is violated on $problem" * (isempty(where_) ? "" : " $where_")
    isempty(detail) || (msg *= ": " * detail)
    error(msg)
end

# A function that counts its calls
struct ConformanceCounted{F}
    f::F
    n::Base.RefValue{Int}
end
ConformanceCounted(f) = ConformanceCounted(f, Ref(0))
(c::ConformanceCounted)(args...) = (c.n[] += 1; c.f(args...))

# The problems. Each `fdf!` and `resjac!` returns exactly `f(x)`, computed by the same function.
conf_quadratic_f(x) = 0.5 * (4 * x[1]^2 + 2 * x[1] * x[2] + 3 * x[2]^2 + 2 * x[2] * x[3] + 2 * x[3]^2) - (x[1] + 2 * x[2] + 3 * x[3])
function conf_quadratic_fdf!(g, H, x)
    g[1] = 4 * x[1] + x[2] - 1
    g[2] = x[1] + 3 * x[2] + x[3] - 2
    g[3] = x[2] + 2 * x[3] - 3
    H[1, 1] = 4.0; H[1, 2] = 1.0; H[1, 3] = 0.0
    H[2, 1] = 1.0; H[2, 2] = 3.0; H[2, 3] = 1.0
    H[3, 1] = 0.0; H[3, 2] = 1.0; H[3, 3] = 2.0
    return conf_quadratic_f(x)
end
conf_rosenbrock_f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function conf_rosenbrock_fdf!(g, H, x)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = H[2, 1] = -400 * x[1]
    H[2, 2] = 200.0
    return conf_rosenbrock_f(x)
end
# The least-squares fit r_i = x₁ exp(x₂ t_i) − y_i, six points
const CONF_T = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
const CONF_Y = (2.0, 1.5, 1.2, 0.9, 0.75, 0.6)
conf_residual(x, i) = x[1] * exp(x[2] * CONF_T[i]) - CONF_Y[i]
function conf_fit_f(x)
    s = 0.0
    for i in eachindex(CONF_T)
        s += conf_residual(x, i)^2
    end
    return 0.5 * s
end
function conf_fit_resjac!(r, J, x)
    for i in eachindex(CONF_T)
        e = exp(x[2] * CONF_T[i])
        r[i] = conf_residual(x, i)
        J[i, 1] = e
        J[i, 2] = x[1] * CONF_T[i] * e
    end
    return conf_fit_f(x)
end
# The same fit through its normal equations (the Gauss-Newton Hessian)
function conf_fit_fdf!(g, H, x)
    g[1] = g[2] = 0.0
    H[1, 1] = H[1, 2] = H[2, 1] = H[2, 2] = 0.0
    for i in eachindex(CONF_T)
        e = exp(x[2] * CONF_T[i])
        r = conf_residual(x, i)
        j1, j2 = e, x[1] * CONF_T[i] * e
        g[1] += j1 * r; g[2] += j2 * r
        H[1, 1] += j1 * j1; H[1, 2] += j1 * j2; H[2, 2] += j2 * j2
    end
    H[2, 1] = H[1, 2]
    return conf_fit_f(x)
end
# Requirement 5: a singular Hessian, f = (x₁ + x₂ − 1)², and a Hessian holding an Inf
conf_singular_f(x) = (x[1] + x[2] - 1)^2
function conf_singular_fdf!(g, H, x)
    g[1] = g[2] = 2 * (x[1] + x[2] - 1)
    H[1, 1] = H[1, 2] = H[2, 1] = H[2, 2] = 2.0
    return conf_singular_f(x)
end
conf_inf_f(x) = x[1]^2 + x[2]^2
function conf_inf_fdf!(g, H, x)
    g[1] = 2 * x[1]; g[2] = 2 * x[2]
    H[1, 1] = Inf; H[1, 2] = H[2, 1] = 0.0; H[2, 2] = 2.0
    return conf_inf_f(x)
end
# Requirement 6: the cost is NaN for x₁ > 0.5, and the minimum of the finite part is beyond it
conf_nan_f(x) = x[1] > 0.5 ? NaN : 0.5 * ((x[1] - 3)^2 + x[2]^2)
function conf_nan_fdf!(g, H, x)
    c = conf_nan_f(x)
    g[1] = isnan(c) ? NaN : x[1] - 3; g[2] = isnan(c) ? NaN : x[2]
    H[1, 1] = isnan(c) ? NaN : 1.0; H[1, 2] = H[2, 1] = 0.0; H[2, 2] = 1.0
    return c
end

# The problems of the per-iteration requirements, for a system kind (:static or :dense): (name, x0, f, objective)
function conformance_problems(kind::Symbol)
    st = kind === :static
    v(x...) = st ? SVector(x...) : [x...]
    return (
        ("the convex quadratic ($kind)", v(1.0, 1.0, 1.0), conf_quadratic_f, conf_quadratic_fdf!, nothing),
        ("Rosenbrock ($kind)", v(-1.2, 1.0), conf_rosenbrock_f, conf_rosenbrock_fdf!, nothing),
        ("the least-squares fit as a HessianObjective ($kind)", v(1.0, 0.0), conf_fit_f, conf_fit_fdf!, nothing),
        ("the least-squares fit as a JacobianObjective ($kind)", v(1.0, 0.0), conf_fit_f, nothing, conf_fit_resjac!),
    )
end
conf_objective(f, fdf!, ::Nothing, x0) = HessianObjective(f, fdf!; H = x0 isa StaticArray ? nothing : zeros(length(x0), length(x0)))
conf_objective(f, ::Nothing, resjac!, x0) = JacobianObjective(f, resjac!, x0 isa StaticArray ? length(CONF_T) : zeros(length(CONF_T), length(x0)))

# The callback of requirements 1 to 4: checks each iteration, and records the costs
struct ConformanceCallback{F}
    f::F                 # the problem's f, uncounted
    name::Symbol
    problem::String
    costs::Vector{Float64}
end
function (cb::ConformanceCallback)(cost, x, data, itdata)
    it = data.iternum
    where_ = "at iteration $it"
    cost isa Float64 || conformance_error(cb.name, 1, cb.problem, where_, "returned a $(typeof(cost))")
    isnan(cost) || isequal(cost, cb.f(x)) || conformance_error(cb.name, 1, cb.problem, where_, "returned $cost, but f at the trial point is $(cb.f(x))")
    isequal(x, update(data.variables, getstep(data))) || conformance_error(cb.name, 2, cb.problem, where_)
    restrictedstep(itdata) isa Bool || conformance_error(cb.name, 3, cb.problem, where_, "it is a $(typeof(restrictedstep(itdata)))")
    push!(cb.costs, cost)
    return cost, 0
end

# A callback that only records the costs, into storage reserved beforehand (for the allocation of requirement 7)
struct ConformanceRecord
    costs::Vector{Float64}
end
(cb::ConformanceRecord)(cost, x, data, itdata) = (push!(cb.costs, cost); (cost, 0))

conf_run!(x, obj, opts, cb, ws) = ismutablevar(x) ? (optimize!(x, obj, opts; callback = cb, workspace = ws); x) : first(optimize(x, obj, opts; callback = cb, workspace = ws))
conf_allocated(x, obj, opts, cb, ws) = @allocated conf_run!(x, obj, opts, cb, ws)

"""
    check_iterator(name::Symbol) -> true

Checks that the iterator registered as `name` meets the iterator contract, and returns `true`; otherwise throws an
`ErrorException` whose message names the first requirement violated, the problem, and the iteration. It runs the
iterator on problems of its own whose `fdf!` (and `resjac!`) returns exactly `f(x)`: a convex quadratic, Rosenbrock,
and a small least-squares fit as both a [`HessianObjective`](@ref) and a [`JacobianObjective`](@ref), each with a
static and a dense system, and with `callbackchangesstate` false and true. At every iteration it checks that

1. the value [`iterate!`](@ref) returns is a `Float64`, and, when it is a cost (not `NaN`), that it is `isequal` to
   `f` at the trial point;
2. the trial point is `isequal` to the current point updated by [`getstep`](@ref)`(data)`;
3. [`restrictedstep`](@ref)`(itdata)` is a `Bool`.

Requirement 4, that the system is undamped when the callback runs, is not checked: the loop undamps the system after
`iterate!` returns, before the callback, so no iterator can break it. For each run it checks that

5. on a problem whose Hessian is singular, and on one whose Hessian holds an `Inf`, the run ends without an exception
   (a step, `LINSOLVER_FAILED` or `NaN` are all conforming);
6. on a problem whose cost is `NaN` away from the start, the run ends without an exception;
7. a second run on the same [`OptimizationWorkspace`](@ref) allocates nothing and repeats the first run's costs
   exactly (which tests [`reset!`](@ref));
8. the result's cost and gradient counts are the numbers of calls to `f` and to `fdf!` (or `resjac!`), so that the
   iterator evaluates them only through [`trialcost!`](@ref) and [`linearizetrial!`](@ref). (The count of linear
   solves cannot be observed from outside the system, and is not checked.)

It is not called by `optimize`; call it in your iterator's tests. Part of
[`IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface).
"""
function check_iterator(name::Symbol)
    validateiterator(name)
    for kind in (:static, :dense), (pname, x0, f, fdf!, resjac!) in conformance_problems(kind), ccs in (false, true)
        problem = "$pname with callbackchangesstate = $ccs"
        opts = OptimizationOptions(; iterator = name, callbackchangesstate = ccs, maxiters = 50)
        # Requirements 1 to 3 and 8: a run with the checking callback, and counted functions
        cf = ConformanceCounted(f)
        cd = ConformanceCounted(something(fdf!, resjac!))
        obj = isnothing(fdf!) ? conf_objective(cf, nothing, cd, x0) : conf_objective(cf, cd, nothing, x0)
        cb = ConformanceCallback(f, name, problem, Float64[])
        _, result = optimize(copy(x0), obj, opts; callback = cb)
        result.costcomputations == cf.n[] || conformance_error(name, 8, problem, "at the end of the run",
            "$(result.costcomputations) cost computations counted, f called $(cf.n[]) times")
        result.gradientcomputations == cd.n[] || conformance_error(name, 8, problem, "at the end of the run",
            "$(result.gradientcomputations) gradient computations counted, $(isnothing(fdf!) ? "resjac!" : "fdf!") called $(cd.n[]) times")
        # Requirement 7: two runs on one workspace, the second warm
        obj = conf_objective(f, fdf!, resjac!, x0)
        ws = OptimizationWorkspace(copy(x0), obj, opts)
        first_ = ConformanceRecord(sizehint!(Float64[], opts.maxiters + 1))
        conf_run!(copy(x0), obj, opts, first_, ws)
        second = ConformanceRecord(sizehint!(Float64[], opts.maxiters + 1))
        x = copy(x0)
        bytes = conf_allocated(x, obj, opts, second, ws)
        isequal(second.costs, first_.costs) || conformance_error(name, 7, problem, "on the second run",
            "its costs differ from the first run's from iteration $(something(findfirst(k -> k > length(second.costs) || k > length(first_.costs) || !isequal(second.costs[k], first_.costs[k]), 1:max(length(first_.costs), length(second.costs))), 0))")
        bytes == 0 || conformance_error(name, 7, problem, "on the second run", "it allocated $bytes bytes")
    end
    # Requirements 5 and 6: runs that must end without an exception
    for kind in (:static, :dense), ccs in (false, true),
            (k, pname, f, fdf!) in ((5, "the problem whose Hessian is singular", conf_singular_f, conf_singular_fdf!),
                                    (5, "the problem whose Hessian holds an Inf", conf_inf_f, conf_inf_fdf!),
                                    (6, "the problem whose cost is NaN away from the start", conf_nan_f, conf_nan_fdf!))
        problem = "$pname ($kind) with callbackchangesstate = $ccs"
        x0 = kind === :static ? SVector(0.0, 0.0) : [0.0, 0.0]
        opts = OptimizationOptions(; iterator = name, callbackchangesstate = ccs, maxiters = 50)
        try
            optimize(x0, conf_objective(f, fdf!, nothing, x0), opts)
        catch err
            conformance_error(name, k, problem, "", "the run threw $(sprint(showerror, err))")
        end
    end
    return true
end
