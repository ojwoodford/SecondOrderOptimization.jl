using SecondOrderOptimization, StaticArrays, LinearAlgebra, Static, Random, Test
const SOO = SecondOrderOptimization
using SecondOrderOptimization: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!

# W: the iterators other than :levenbergmarquardt on a JacobianObjective (the dense tall system),
# against the normal equations built from the same J

# The iterators widened to in section W, each tested in both variants
const W_ITERATORS = (:levenbergmarquardt_dtd, :dogleg, :newton)

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

@testset "jacobianiterators.jl" begin
    @testset "Each iterator against the normal equations: :$it, callbackchangesstate = $ccs" for it in W_ITERATORS, ccs in (false, true)
        for (name, resid, jac, m, x0) in w_problems()
            objJ, objH = w_objectives(resid, jac, m, length(x0))
            cJ, sJ, _, cbJ = w_trace()
            cH, sH, _, cbH = w_trace()
            _, resJ = optimize(x0, objJ, w_opts(it, :callbackchangesstate => ccs); callback=cbJ)
            _, resH = optimize(x0, objH, w_opts(it, :callbackchangesstate => ccs, :linsolver => :cholesky); callback=cbH)
            if it === :newton
                # undamped Newton rejects no step, so at the minimum the cost's rounding noise
                # counts as failures, and it may stop on max_fails (in both formulations alike)
                @test all(r -> r in (:rel_dcost, :abs_dcost, :step_size, :max_fails, :grad_tol), termination_reasons(resJ))
                @test all(r -> r in (:rel_dcost, :abs_dcost, :step_size, :max_fails, :grad_tol), termination_reasons(resH))
            else
                @test isconverged(resJ) && isconverged(resH)
            end
            # the cost at every iteration both runs share, which may end a few iterations apart
            # once the decreases are at the level of rounding
            k = min(length(cJ), length(cH))
            @test k >= 2
            @test all(isapprox.(cJ[1:k], cH[1:k]; rtol=1e-8, atol=1e-8 * abs(resH.startcost) * eps()))
            @test abs(length(cJ) - length(cH)) <= 3
            # the damping or trust radius, over the leading iterations whose relative decrease is
            # above 1e-6 (rounding in the actual decrease compounds through the update after that)
            prev = [resJ.startcost; cJ[1:k-1]]
            lead = something(findfirst(prev .- cH[1:k] .<= 1e-6 .* abs.(cH[1:k])), k + 1) - 1
            it === :newton || @test lead >= 1     # Newton has no damping or radius to compare
            @test all(isapprox.(sJ[1:lead], sH[1:lead]; rtol=1e-8))
            @test resJ.bestcost ≈ resH.bestcost rtol=1e-8 atol=1e-20
        end
    end

    @testset "Both variants of :$it give the same iterates" for it in W_ITERATORS
        for (name, resid, jac, m, x0) in w_problems()
            objJ, _ = w_objectives(resid, jac, m, length(x0))
            c1, s1, x1, cb1 = w_trace()
            c2, s2, x2, cb2 = w_trace()
            _, r1 = optimize(x0, objJ, w_opts(it, :callbackchangesstate => false); callback=cb1)
            _, r2 = optimize(x0, objJ, w_opts(it, :callbackchangesstate => true); callback=cb2)
            @test c1 == c2 && s1 == s2 && x1 == x2
            @test r1.niterations == r2.niterations && r1.bestcost == r2.bestcost
        end
    end

    @testset "Marquardt scaling: columns of J differing by 1e6 in norm" begin
        # Rosenbrock with its second variable in units of 1e-6, so the second column of J is 1e6
        # times smaller in norm than the first. Uniform damping swamps that column.
        d = [1.0, 1e-6]
        resid(x) = [1 - x[1] * d[1], 10 * (x[2] * d[2] - (x[1] * d[1])^2)]
        jac(x) = [-d[1] 0.0; -20 * x[1] * d[1]^2 10 * d[2]]
        x0 = [-1.2, 1.0] ./ d
        objJ, objH = w_objectives(resid, jac, 2, 2)
        J0 = jac(x0)
        @test norm(J0[:, 1]) / norm(J0[:, 2]) > 1e6
        for (obj, ls) in ((objJ, :auto), (objH, :cholesky))
            _, rL = optimize(x0, obj, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=ls, maxiters=500))
            _, rM = optimize(x0, obj, OptimizationOptions(iterator=:levenbergmarquardt_dtd, linsolver=ls, maxiters=500))
            @test isconverged(rM) && rM.bestcost < 1e-20
            @test rM.niterations < 50
            @test 5 * rM.niterations < rL.niterations
        end
    end

    @testset "Marquardt scaling: invariant to a diagonal rescaling of the variables" begin
        # x = D y: the Jacobian in y is J D, its column norms scale by D, and the scaled damping
        # λ·diag(DᵀJᵀJD) moves with them, so the iterates in y are D⁻¹ times those in x
        # a monotone nonlinearity of each variable, so the minimizer is unique
        rng = Xoshiro(13); M = randn(rng, 40, 8) ./ sqrt(40) + I(40)[:, 1:8]; yy = randn(rng, 40)
        resid(x) = M * (x .+ 0.5 .* sin.(x)) .- yy
        jac(x) = M * Diagonal(1 .+ 0.5 .* cos.(x))
        D = exp10.(range(-1.5, 1.5; length=8))
        objx, _ = w_objectives(resid, jac, 40, 8)
        objy, _ = w_objectives(y -> resid(D .* y), y -> jac(D .* y) * Diagonal(D), 40, 8)
        x0 = fill(0.1, 8)
        for ccs in (false, true)
            cx, sx, xs, cbx = w_trace()
            cy, sy, ys, cby = w_trace()
            optimize(x0, objx, w_opts(:levenbergmarquardt_dtd, :callbackchangesstate => ccs); callback=cbx)
            optimize(x0 ./ D, objy, w_opts(:levenbergmarquardt_dtd, :callbackchangesstate => ccs); callback=cby)
            k = min(length(cx), length(cy))
            @test k >= 4
            @test all(isapprox.(cx[1:k], cy[1:k]; rtol=1e-10))
            @test all(isapprox(D .* ys[i], xs[i]; rtol=1e-8) for i in 1:k)
            @test all(isapprox.(sx[1:4], sy[1:4]; rtol=1e-8))
        end
        # uniform damping is not invariant: the rescaled run takes another path
        cx, _, _, cbx = w_trace()
        cy, _, _, cby = w_trace()
        optimize(x0, objx, w_opts(:levenbergmarquardt); callback=cbx)
        optimize(x0 ./ D, objy, w_opts(:levenbergmarquardt); callback=cby)
        @test !isapprox(cx[1], cy[1]; rtol=1e-6)
    end

    @testset "Marquardt scaling: the rule, from the cached column norms" begin
        J = [1.0 0.0 3.0; 2.0 1e-5 0.0; 0.0 0.0 4.0]
        ls = SOO.DenseTallLinearSystem(3, 3)
        SOO.TallLinearize((r, Jw, x) -> (r .= 1.0; Jw .= J; 0.0))(getb(ls), getA(ls), nothing)
        ws = getA(ls).ws
        @test ws.normsstale && ws.nnorms == 0
        # the generic rule on diag(JᵀJ): a running maximum, floored at 1e-8 of the largest
        lmd = MarquardtData(0.0, zeros(3))
        SOO.updatescaling!(lmd, ls)
        @test lmd.lambda == 1e-6
        @test lmd.scaling == [5.0, max(1e-10, 25.0 * 1e-8), 25.0]
        @test SOO.max_diag(ls) == 25.0
        # the column norms were computed once for this linearization, and shared by max_diag
        @test ws.nnorms == 1 && !ws.normsstale
        # against the generic method on the normal equations
        lmdH = MarquardtData(0.0, zeros(3))
        SOO.updatescaling!(lmdH, SOO.DenseLinearSystem{Float64}(J' * J, J' * ones(3), zeros(3)))
        @test lmdH.scaling == lmd.scaling
        # a new linearization recomputes them, and the scaling keeps its running maximum
        SOO.TallLinearize((r, Jw, x) -> (r .= 1.0; Jw .= 0.5 .* J; Jw[2, 2] = 1.0; 0.0))(getb(ls), getA(ls), nothing)
        SOO.updatescaling!(lmd, ls)
        @test ws.nnorms == 2
        @test lmd.scaling == [5.0, 1.0, 25.0]
        @test lmd.lambda == 1e-6
        # zero! marks them stale; a non-finite column counts as 1
        zero!(ls)
        @test ws.normsstale
        getA(ls).J .= J; getA(ls).J[1, 1] = NaN; SOO.markstale!(getA(ls))
        lmd = MarquardtDataTrial(0.0, zeros(3), nothing, NaN, false)
        SOO.updatescaling!(lmd, ls)
        @test lmd.scaling == [1.0, 25.0 * 1e-8, 25.0]
        @test isnan(SOO.max_diag(ls))
    end

    @testset "Dogleg: each kind of step, and the trust radius" begin
        pts = Vector{Float64}[]
        resid(x) = [1 - x[1], 10 * (x[2] - x[1]^2)]
        jac(x) = [-1.0 0.0; -20x[1] 10.0]
        f(x) = (push!(pts, collect(x)); 0.5 * sum(abs2, resid(x)))
        resjac!(r, J, x) = (push!(pts, collect(x)); r .= resid(x); J .= jac(x); 0.5 * sum(abs2, r))
        fdf!(g, H, x) = (push!(pts, collect(x)); r = resid(x); J = jac(x); g .= J' * r; H .= J' * J; 0.5 * sum(abs2, r))
        for (obj, ls) in ((JacobianObjective(f, resjac!, 2), :auto), (HessianObjective(f, fdf!; H=zeros(2, 2)), :cholesky)), ccs in (false, true)
            rec, cb = w_doglegtrace(pts)
            empty!(pts)
            optimize!([-1.2, 1.0], obj, w_opts(:dogleg, :linsolver => ls, :callbackchangesstate => ccs); callback=cb)
            kinds = [r.kind for r in rec]
            @test count(==(:gaussnewton), kinds) >= 1
            @test count(==(:gradient), kinds) >= 1
            @test count(==(:dogleg), kinds) >= 1
            later = rec[2:end]
            # after a rejected trial the radius shrinks: the accepted step is at most a quarter of
            # the radius the iteration started with
            rejected = filter(r -> r.ntrials > 1, later)
            @test !isempty(rejected)
            @test all(r.stepnorm <= 0.25 * r.before * (1 + 1e-12) for r in rejected)
            # after a good step that reached the boundary the radius doubles; after a poor one it is
            # a quarter of the step
            good = filter(r -> r.ntrials == 1 && r.rho > 0.75 && r.stepnorm >= 0.9 * r.before, later)
            @test !isempty(good)
            @test all(r.after == 2 * r.before for r in good)
            poor = filter(r -> r.ntrials == 1 && r.rho < 0.25, later)
            @test all(r.after == 0.25 * r.stepnorm for r in poor)
        end
    end

    @testset "Dogleg: rank-deficient J takes the shifted Gauss-Newton step, not the Cauchy step" begin
        # The undamped solve returns NotPositiveDefinite; dogleg then retries with a diagonal shift
        # (as :newton does), so its Gauss-Newton step is the shifted one, and the Cauchy step is the
        # fallback only if every shift fails
        rng = Xoshiro(5)
        a, c = randn(rng, 15), randn(rng, 15); y = randn(rng, 15)
        objJ, _ = w_objectives(x -> @.(a * (x[1] + x[2]) + c * x[3] - y), x -> hcat(a, a, c), 15, 3)
        ls = SOO.makelinearsystem(zeros(3), objJ, :auto)
        zero!(ls); SOO.linearizer(objJ)(getb(ls), getA(ls), zeros(3)); dampen!(ls, 0.0)
        @test solve!(ls, :auto) == NotPositiveDefinite
        for ccs in (false, true)
            rec, cb = w_doglegtrace(Vector{Float64}[])
            x, res = optimize(zeros(3), objJ, w_opts(:dogleg, :callbackchangesstate => ccs); callback=cb)
            @test rec[1].kind == :gaussnewton
            # the shifted step is not along the gradient (the Cauchy direction)
            @test norm(rec[1].newton - dot(rec[1].newton, rec[1].grad) / sum(abs2, rec[1].grad) * rec[1].grad) > 1e-3 * norm(rec[1].newton)
            @test res.bestcost < res.startcost && isconverged(res)
        end
    end

    @testset "The tall system tolerates x being overwritten between solves" begin
        # dogleg writes its own step into getx(ls) between solves, and its trial system shares x
        rng = Xoshiro(23)
        J = randn(rng, 9, 4); r = randn(rng, 9)
        ls = SOO.DenseTallLinearSystem(copy(J))
        getA(ls).r .= r; mul!(getb(ls), J', r); SOO.markstale!(getA(ls))
        for λ in (0.0, 0.3)
            dampen!(ls, λ)
            @test solve!(ls, :auto) == Success
            x1 = copy(getx(ls))
            getx(ls) .= NaN
            @test solve!(ls, :auto) == Success && getx(ls) == x1
            getx(ls) .= 1e300
            @test solve!(ls, :auto) == Success && getx(ls) == x1
        end
        trial = sharex(copy_structure(ls), ls)
        @test getx(trial) === getx(ls)
        getA(trial).J .= 2 .* J; getA(trial).r .= r; mul!(getb(trial), 2 .* J', r); SOO.markstale!(getA(trial))
        dampen!(trial, 0.0); solve!(trial, :auto)
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == Success && getx(ls) ≈ (J' * J) \ (J' * r)
    end

    @testset "Newton (Gauss-Newton): a linear problem in one iteration" begin
        rng = Xoshiro(21)
        A = randn(rng, 12, 4); b = randn(rng, 12)
        obj = JacobianObjective(x -> 0.5 * sum(abs2, A * x - b), (r, J, x) -> (r .= A * x - b; J .= A; 0.5 * sum(abs2, r)), 12)
        for x0 in (zeros(4), 1e3 * randn(rng, 4)), ccs in (false, true)
            _, _, xs, cb = w_trace()
            x, res = optimize(x0, obj, w_opts(:newton, :callbackchangesstate => ccs); callback=cb)
            @test xs[1] ≈ A \ b rtol=1e-11
            @test x ≈ A \ b rtol=1e-11
            @test :linsolver_failed ∉ termination_reasons(res)
        end
    end

    @testset "Newton's diagonal shift on the tall system, and dampen!(ls, 0.0) restoring it" begin
        rng = Xoshiro(29)
        # full rank: undamped, shifted, then undamped again gives the identical x
        J = randn(rng, 10, 4); r = randn(rng, 10)
        ls = SOO.DenseTallLinearSystem(copy(J))
        getA(ls).r .= r; mul!(getb(ls), J', r); SOO.markstale!(getA(ls))
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == Success
        x0 = copy(getx(ls))
        mu = SOO.initial_shift(ls)
        @test mu ≈ max(maximum(sum(abs2, J; dims=1)) * 1e-4, 1e-6) rtol=1e-14
        dampen!(ls, mu)
        @test solve!(ls, :auto) == Success && getx(ls) ≈ (J' * J + mu * I) \ (J' * r)
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == Success && getx(ls) == x0
        # rank deficient: the undamped solve fails, the shift succeeds, and dampen!(ls, 0.0) is
        # undamped (failing) again
        J[:, 4] = J[:, 2]
        ls = SOO.DenseTallLinearSystem(copy(J))
        getA(ls).r .= r; mul!(getb(ls), J', r); SOO.markstale!(getA(ls))
        @test solve!(ls, :auto) == NotPositiveDefinite
        dampen!(ls, SOO.initial_shift(ls))
        @test solve!(ls, :auto) == Success
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == NotPositiveDefinite
        @test getA(ls).J == J && getA(ls).r == r
    end

    @testset "Scaled damped solve against (JᵀJ + λ Diagonal(s)) \\ Jᵀr" begin
        rng = Xoshiro(17)
        for (m, n) in ((20, 5), (6, 6), (3, 5))
            J = randn(rng, m, n) * Diagonal(exp10.(range(-2, 2; length=n)))
            r = randn(rng, m)
            ls = SOO.DenseTallLinearSystem(copy(J))
            getA(ls).r .= r; mul!(getb(ls), J', r); SOO.markstale!(getA(ls))
            Jb, rb = big.(J), big.(r)
            for λ in (1e-8, 1e-3, 1.0, 1e3), s in (vec(sum(abs2, J; dims=1)), exp10.(range(-6, 6; length=n)), fill(2.0, n))
                dampen!(ls, λ, s)
                @test solve!(ls, :auto) == Success
                xref = Float64.((Jb' * Jb + λ * Diagonal(big.(s))) \ (Jb' * rb))
                @test getx(ls) ≈ xref rtol=1e-8
                # s is in the units of diag(JᵀJ), not squared
                λ >= 1 && @test !isapprox(getx(ls), Float64.((Jb' * Jb + λ * Diagonal(big.(s) .^ 2)) \ (Jb' * rb)); rtol=1e-3)
            end
            @test getA(ls).J == J && getA(ls).r == r
        end
    end

    @testset "check_linear_system, with a non-uniform scaled damping" begin
        # the checker's dampen!(ls, λ, s) case uses s = 1 + i/n
        for (m, n) in ((6, 6), (11, 6), (4, 6))
            @test check_linear_system(s -> SOO.DenseTallLinearSystem(m, n); n, linsolvers=(:auto, :qr)) === nothing
        end
    end

    @testset "Rank-deficient J (a duplicated column) through :$it" for it in W_ITERATORS, ccs in (false, true)
        rng = Xoshiro(5)
        a, c = randn(rng, 15), randn(rng, 15); y = randn(rng, 15)
        objJ, _ = w_objectives(x -> @.(a * (x[1] + x[2]) + c * x[3] - y), x -> hcat(a, a, c), 15, 3)
        costs, _, _, cb = w_trace()
        x, res = optimize(zeros(3), objJ, w_opts(it, :callbackchangesstate => ccs); callback=cb)
        costmin = 0.5 * sum(abs2, hcat(a, c) * (hcat(a, c) \ y) - y)
        @test :linsolver_failed ∉ termination_reasons(res)
        @test isconverged(res)
        @test res.bestcost ≈ costmin rtol=1e-10
        @test costs[1] < res.startcost
    end

    @testset "m < n through :$it" for it in W_ITERATORS, ccs in (false, true)
        rng = Xoshiro(9)
        M = randn(rng, 3, 5); y = 0.3 * randn(rng, 3)
        objJ, _ = w_objectives(x -> M * tanh.(x) - y, x -> M * Diagonal(sech.(x) .^ 2), 3, 5)
        x, res = optimize(zeros(5), objJ, w_opts(it, :callbackchangesstate => ccs))
        @test :linsolver_failed ∉ termination_reasons(res)
        @test isconverged(res) && res.bestcost < 1e-20
    end

    @testset "Non-finite input through :$it" for it in W_ITERATORS, ccs in (false, true)
        # at the start
        objnan = JacobianObjective(x -> 1.0, (r, J, x) -> (r .= 1.0; J .= NaN; 1.0), 2)
        _, res = optimize([0.0, 0.0], objnan, w_opts(it, :callbackchangesstate => ccs))
        @test :grad_not_finite in termination_reasons(res) && :start_not_finite in termination_reasons(res)
        @test res.niterations == 0
        # after the first step: J holds an Inf once x[1] > 0.5
        f(x) = 0.5 * sum(abs2, x .- 1)
        objlate = JacobianObjective(f, (r, J, x) -> (r .= x .- 1; J .= I(2); x[1] > 0.5 && (J[1, 1] = Inf); f(x)), 2)
        _, res = optimize([0.0, 0.0], objlate, w_opts(it, :callbackchangesstate => ccs))
        @test :grad_not_finite in termination_reasons(res)
        @test res.niterations >= 1 && !isconverged(res)
        # a NaN cost everywhere but the start: every trial is rejected, and the run stops on it
        fnan(x) = x == [0.0, 0.0] ? f(x) : NaN
        objcost = JacobianObjective(fnan, (r, J, x) -> (r .= x .- 1; J .= I(2); fnan(x)), 2)
        x, res = optimize([0.0, 0.0], objcost, w_opts(it, :callbackchangesstate => ccs))
        @test :cost_nan in termination_reasons(res) && !isconverged(res)
        @test x == [0.0, 0.0] && res.bestcost == f(x)
    end

    @testset "Workspace reuse and no allocation per iteration: :$it" for it in W_ITERATORS, ccs in (false, true)
        for obj in (JacobianObjective(wrosen_cost, wrosen_resjac!, 2), JacobianObjective(wrosen_cost, wrosen_resjac!, zeros(2, 2)))
            opts = w_opts(it, :callbackchangesstate => ccs, :gtol => 1e-10)
            ws = OptimizationWorkspace([-1.2, 1.0], obj, opts)
            x = [-1.2, 1.0]
            w_allocs(x, obj, opts, ws)
            @test w_allocs(x, obj, opts, ws) == 0
            @test x ≈ [1.0, 1.0]
            @inferred optimize!([-1.2, 1.0], obj, opts; workspace=ws)
            @inferred optimize([-1.2, 1.0], obj, opts)
            # From (-1.2, 1) the first Gauss-Newton step raises the cost, and the outer loop then
            # copies the best variables once (not per iteration); from (0.9, 0.8) it descends
            res = optimize!(it === :newton ? [0.9, 0.8] : [-1.2, 1.0], obj, opts)
            @test res.optimizeallocations == 0 && isconverged(res)
            @test res.niterations >= (it === :newton ? 2 : 6)     # Gauss-Newton solves Rosenbrock's residuals in two steps
        end
    end

    @testset "Ill-conditioned problem: excess cost over the least-squares minimum, :$it" for it in W_ITERATORS
        V, yv = w_illconditioned()
        objJ, objH = w_objectives(x -> V * x - yv, x -> V, 60, 14)
        exactcost(x) = Float64(0.5 * sum(abs2, big.(V) * big.(x) - big.(yv)))
        costmin = exactcost(big.(V) \ big.(yv))
        xJ, resJ = optimize(zeros(14), objJ, w_opts(it, :maxiters => 500))
        xH, resH = optimize(zeros(14), objH, w_opts(it, :maxiters => 500, :linsolver => :cholesky))
        exJ = (exactcost(xJ) - costmin) / costmin
        exH = (exactcost(xH) - costmin) / costmin
        @info "Ill-conditioned fit, iterator = :$it (relative excess cost over the least-squares minimum $costmin)" tall_excess=exJ tall_iterations=resJ.niterations tall_termination=join(termination_reasons(resJ), ",") normal_excess=exH normal_iterations=resH.niterations normal_termination=join(termination_reasons(resH), ",")
        @test :linsolver_failed ∉ termination_reasons(resJ)
        @test resJ.bestcost < resJ.startcost && resH.bestcost < resH.startcost
        # Marquardt scaling floors the damping at λ ≥ 1e-12 times diag(JᵀJ), and cond(J)² ≈ 1.5e19,
        # so the damping holds back the small singular directions in both formulations alike (it is
        # invariant to the variables' scale, so no rescaling of the problem avoids it)
        if it === :levenbergmarquardt_dtd
            @test exJ > 1e-3 && exH > 1e-3
        end
        # The Gauss-Newton step of dogleg and Newton is undamped: from QR it reaches the minimum
        # in a few iterations; from Cholesky of the normal equations it does not. Measured on
        # Julia 1.13 / 1.10: dogleg tall 1.8e-14 in 8 iterations / 4.2e-15 in 6, normal 6.5e4 at
        # maxiters / 1.8e-3 in 396; Newton tall 1.3e-14 in 7 / 2.5e-15 in 8, normal 6.5e4 / 2.2e-4,
        # both at maxiters. (Newton stops on max_fails at the minimum: it rejects no step, and the
        # cost's rounding noise counts as failures.)
        if it in (:dogleg, :newton)
            @test exJ <= 1e-12
            @test exH > 1e-6
            @test resJ.niterations < resH.niterations
        end
    end

    @testset ":gradientdescent is rejected before f or resjac! is called" begin
        obj = JacobianObjective(wrosen_cost, wrosen_resjac!, 2)
        WCALLS[] = 0
        opts = OptimizationOptions(iterator=:gradientdescent)
        @test_throws ArgumentError optimize([-1.2, 1.0], obj, opts)
        @test_throws ArgumentError optimize!([-1.2, 1.0], obj, opts)
        @test_throws ArgumentError OptimizationWorkspace([-1.2, 1.0], obj, opts)
        @test WCALLS[] == 0
    end
end

# Z1: the optimizer does not zero r and J before each resjac! call; resjac! sets them

# Every iterator accepted with a JacobianObjective
const Z_ITERATORS = (:levenbergmarquardt, W_ITERATORS...)

# A tall system whose zero! forgets to invalidate the factorization, for the checker
struct LazyZeroTall <: AbstractLinearSystem
    inner::SOO.DenseTallLinearSystem
end
SOO.getA(w::LazyZeroTall) = getA(w.inner)
SOO.getb(w::LazyZeroTall) = getb(w.inner)
SOO.getx(w::LazyZeroTall) = getx(w.inner)
SOO.solve!(w::LazyZeroTall, s::StaticSymbol) = solve!(w.inner, s)
SOO.dampen!(w::LazyZeroTall, λ::Real) = (dampen!(w.inner, λ); w)
SOO.dampen!(w::LazyZeroTall, λ::Real, s) = (dampen!(w.inner, λ, s); w)
SOO.copy_structure(w::LazyZeroTall) = LazyZeroTall(copy_structure(w.inner))
SOO.sharex(w::LazyZeroTall, from) = LazyZeroTall(sharex(w.inner, from.inner))
SOO.zero!(w::LazyZeroTall) = (fill!(getb(w.inner), 0.0); w)          # broken: the old QR is kept
SOO.max_diag(w::LazyZeroTall) = SOO.max_diag(w.inner)
SOO.conformance_fillb!(w::LazyZeroTall, b::AbstractVector) = (SOO.conformance_fillb!(w.inner, b); w)

# A banded least-squares problem: residual i couples x[i] and x[i+1], and a last residual couples
# x[1] and x[n]. `banded!` writes only the entries of J that can be non-zero; `full!` writes all.
function z_banded(n)
    y = [0.3 * sin(1.7i) for i in 1:n+1]
    resid(x) = [[x[i] + 0.5 * sin(x[i+1]) - y[i] for i in 1:n-1]; x[n] - y[n]; 0.3 * x[1] * x[n] - y[n+1]]
    function jac(x)
        J = zeros(n + 1, n)
        for i in 1:n-1
            J[i, i] = 1.0; J[i, i+1] = 0.5 * cos(x[i+1])
        end
        J[n, n] = 1.0
        J[n+1, 1] = 0.3 * x[n]; J[n+1, n] = 0.3 * x[1]
        return J
    end
    f(x) = 0.5 * sum(abs2, resid(x))
    function banded!(r, J, x)
        r .= resid(x)
        for i in 1:n-1
            J[i, i] = 1.0; J[i, i+1] = 0.5 * cos(x[i+1])
        end
        J[n, n] = 1.0
        J[n+1, 1] = 0.3 * x[n]; J[n+1, n] = 0.3 * x[1]
        return f(x)
    end
    full!(r, J, x) = (r .= resid(x); J .= jac(x); f(x))
    return f, banded!, full!, jac
end

@testset "Z1: resjac! sets r and J; the optimizer does not zero them" begin
    @testset "A resjac! that writes only the band of J: :$it, callbackchangesstate = $ccs" for it in Z_ITERATORS, ccs in (false, true)
        n = 6
        f, banded!, full!, _ = z_banded(n)
        x0 = fill(0.4, n)
        opts = w_opts(it, :callbackchangesstate => ccs)
        cR, _, xR, cbR = w_trace()
        _, resR = optimize(x0, JacobianObjective(f, full!, n + 1), opts; callback=cbR)
        @test isconverged(resR) || (it === :newton && :max_fails in termination_reasons(resR))
        @test resR.bestcost < 0.1 * resR.startcost      # a nonzero-residual minimum (n + 1 residuals)
        # the banded writer gives the identical run: with an integer m, and with a prototype of
        # zeros (Q1: a prototype is used with the values it holds)
        for J in (n + 1, zeros(n + 1, n))
            c, _, xs, cb = w_trace()
            _, res = optimize(x0, JacobianObjective(f, banded!, J), opts; callback=cb)
            @test c == cR && xs == xR
            @test res.bestcost == resR.bestcost && res.niterations == resR.niterations
        end
        # the trial system's J starts as a copy of the main one (copy_structure copies, not similar)
        P = fill(7.0, n + 1, n)
        ls = SOO.makelinearsystem(x0, JacobianObjective(f, banded!, P), :auto)
        @test getA(ls).J === P && P == fill(7.0, n + 1, n)
        itd = SOO.construct(static(it), ls, static(ccs))
        if hasfield(typeof(itd), :trial) && !isnothing(itd.trial)
            @test getA(itd.trial).J == P && getA(itd.trial).J !== P
        end
        @test getA(copy_structure(ls)).J == P
    end

    @testset "The optimizer does not write J or r: :$it, callbackchangesstate = $ccs" for it in Z_ITERATORS, ccs in (false, true)
        # record J and r on leaving resjac!, and compare with them on the next entry with the same
        # storage (the trial variants alternate between two systems)
        last = IdDict{Any, Tuple{Matrix{Float64}, Vector{Float64}}}()
        nfirst, ncompared, nmismatch, nnotzero = Ref(0), Ref(0), Ref(0), Ref(0)
        function recording!(r, J, x)
            if haskey(last, J)
                ncompared[] += 1
                (J == last[J][1] && r == last[J][2]) || (nmismatch[] += 1)
            else
                nfirst[] += 1
                all(iszero, J) || (nnotzero[] += 1)     # r is not initialized (Q1): resjac! sets all of it
            end
            cost = wrosen_resjac!(r, J, x)
            last[J] = (copy(J), copy(r))
            return cost
        end
        obj = JacobianObjective(wrosen_cost, recording!, 2)
        opts = w_opts(it, :callbackchangesstate => ccs)
        _, res = optimize([-1.2, 1.0], obj, opts)
        @test ncompared[] >= 2 && nmismatch[] == 0 && nnotzero[] == 0      # Newton solves it in two steps
        @test nfirst[] == (ccs || it === :newton ? 1 : 2)
        # with rejected steps: more cost evaluations than iterations, or (Newton) a cost increase
        @test res.costcomputations + res.gradientcomputations > res.niterations + 1 || it === :newton
        # a reused workspace: each run starts every system's J from zero (integer m), then never writes it
        ws = OptimizationWorkspace([-1.2, 1.0], obj, opts)
        for _ in 1:2
            empty!(last)
            nfirst[] = ncompared[] = nmismatch[] = nnotzero[] = 0
            optimize!([-1.2, 1.0], obj, opts; workspace=ws)
            @test ncompared[] >= 2 && nmismatch[] == 0 && nnotzero[] == 0
        end
    end

    @testset "A reused workspace does not carry J's entries from another objective: :$it" for it in Z_ITERATORS, ccs in (false, true)
        # objA writes J[1, 2]; objB has J[1, 2] = 0 at every x and never writes it
        objA = JacobianObjective(x -> 0.5 * ((x[1] + 3x[2] - 1)^2 + (x[2] - 2)^2),
                                 (r, J, x) -> (r[1] = x[1] + 3x[2] - 1; r[2] = x[2] - 2; J[1, 1] = 1.0; J[1, 2] = 3.0; J[2, 2] = 1.0; 0.5 * sum(abs2, r)), 2)
        objB = JacobianObjective(x -> 0.5 * ((x[1] - 1)^2 + (x[2] - 2)^2),
                                 (r, J, x) -> (r[1] = x[1] - 1; r[2] = x[2] - 2; J[1, 1] = 1.0; J[2, 2] = 1.0; 0.5 * sum(abs2, r)), 2)
        opts = w_opts(it, :callbackchangesstate => ccs)
        ws = OptimizationWorkspace([0.0, 0.0], objA, opts)
        optimize([0.0, 0.0], objA, opts; workspace=ws)
        xw, rw = optimize([0.0, 0.0], objB, opts; workspace=ws)
        xf, rf = optimize([0.0, 0.0], objB, opts)
        @test SOO.workspacevalid(ws, [0.0, 0.0], opts, SOO.objective_structure(objB))
        @test xw == xf && rw.niterations == rf.niterations
        @test xw ≈ [1.0, 2.0]
        # and zero allocation per run still holds with the per-run zeroing
        w_allocs(zeros(2), objB, opts, ws)
        @test w_allocs(zeros(2), objB, opts, ws) == 0
    end

    @testset "After zero!, a solve uses the newly written J, as do the column norms" begin
        rng = Xoshiro(31)
        J1, r1 = randn(rng, 8, 3), randn(rng, 8)
        J2, r2 = 3.0 .* randn(rng, 8, 3), randn(rng, 8)
        ls = SOO.DenseTallLinearSystem(8, 3)
        SOO.TallLinearize((r, J, x) -> (r .= r1; J .= J1; 0.0))(getb(ls), getA(ls), nothing)
        dampen!(ls, 0.0); @test solve!(ls, :auto) == Success
        @test getx(ls) ≈ J1 \ r1
        lmd = MarquardtData(0.0, zeros(3)); SOO.updatescaling!(lmd, ls)
        @test SOO.max_diag(ls) ≈ maximum(sum(abs2, J1; dims=1))
        nf, nn = getA(ls).ws.nfactor, getA(ls).ws.nnorms
        # zero! keeps J and r, but whatever is written next is what the system solves with
        zero!(ls)
        @test getA(ls).J == J1 && getA(ls).r == r1 && all(iszero, getb(ls))
        getA(ls).J .= J2; getA(ls).r .= r2; mul!(getb(ls), J2', r2)   # written directly, without markstale!
        @test SOO.max_diag(ls) ≈ maximum(sum(abs2, J2; dims=1))
        lmd2 = MarquardtData(0.0, zeros(3)); SOO.updatescaling!(lmd2, ls)
        @test lmd2.scaling ≈ max.(vec(sum(abs2, J2; dims=1)), 1e-8 * maximum(sum(abs2, J2; dims=1)))
        for λ in (0.0, 0.5)
            dampen!(ls, λ)
            @test solve!(ls, :auto) == Success
            @test getx(ls) ≈ (J2' * J2 + λ * I) \ (J2' * r2)
        end
        dampen!(ls, 0.5, lmd2.scaling)
        @test solve!(ls, :auto) == Success && getx(ls) ≈ (J2' * J2 + 0.5 * Diagonal(lmd2.scaling)) \ (J2' * r2)
        @test getA(ls).ws.nfactor == nf + 1 && getA(ls).ws.nnorms == nn + 1
        # the checker catches a zero! that keeps the old factorization
        @test check_linear_system(s -> SOO.DenseTallLinearSystem(9, 4); n=4, linsolvers=(:auto,)) === nothing
        err = try check_linear_system(s -> LazyZeroTall(SOO.DenseTallLinearSystem(9, 4)); n=4, linsolvers=(:auto,)); nothing catch e; e end
        @test err isa ErrorException && occursin("zero! invalidates what was computed from the old A", err.msg)
    end

    @testset "A linearization with NaN entries is fully replaced by the next one" begin
        rng = Xoshiro(37)
        J, r = randn(rng, 7, 3), randn(rng, 7)
        Jnan = copy(J); Jnan[2, 2] = NaN; Jnan[5, 1] = NaN
        rnan = copy(r); rnan[4] = NaN
        for ls in (SOO.DenseTallLinearSystem(7, 3), copy_structure(SOO.DenseTallLinearSystem(7, 3)))
            lin(Jw, rw) = (zero!(ls); SOO.TallLinearize((r_, J_, x) -> (r_ .= rw; J_ .= Jw; 0.0))(getb(ls), getA(ls), nothing))
            lin(Jnan, rnan)
            @test solve!(ls, :auto) != Success && isnan(SOO.max_diag(ls)) && !all(isfinite, getb(ls))
            lin(J, r)
            @test all(isfinite, getA(ls).J) && all(isfinite, getA(ls).r) && all(isfinite, getb(ls))
            @test SOO.max_diag(ls) ≈ maximum(sum(abs2, J; dims=1))
            dampen!(ls, 0.0)
            @test solve!(ls, :auto) == Success && getx(ls) ≈ J \ r
            lmd = MarquardtData(0.0, zeros(3)); SOO.updatescaling!(lmd, ls)
            @test all(isfinite, lmd.scaling)
        end
        # through optimize: the trial system linearized at a point where J and r are NaN (the
        # cost is NaN there too, so the step is rejected) is then fully rewritten and reused
        f(x) = x[1] > 0.6 ? NaN : 0.5 * ((x[1] - 0.5)^2 + 10 * (x[2] - x[1]^2)^2)
        function nan_resjac!(r, J, x)
            if x[1] > 0.6
                r .= NaN; J .= NaN
            else
                r[1] = x[1] - 0.5; r[2] = sqrt(10) * (x[2] - x[1]^2)
                J[1, 1] = 1.0; J[1, 2] = 0.0; J[2, 1] = -2sqrt(10) * x[1]; J[2, 2] = sqrt(10)
            end
            return f(x)
        end
        for it in Z_ITERATORS
            x, res = optimize([-1.5, 2.0], JacobianObjective(f, nan_resjac!, 2), w_opts(it))
            @test :grad_not_finite ∉ termination_reasons(res)
            @test isfinite(res.bestcost) && res.bestcost < res.startcost
        end
    end
end

# Q1: a Matrix prototype is used as handed; the optimizer never writes J

# A problem whose J has an entry resjac! never writes: J[1, 2], which the caller sets in the
# prototype. `partial!` leaves it; `explicit!` writes it with the same value.
function q_partial(c12)
    function partial!(r, J, x)       # writes all of r and every entry of J but J[1, 2]
        r[1] = x[1] + c12 * x[2] - 1 + 0.1 * sin(x[1])
        r[2] = x[2] - 2 + 0.2 * x[1]^2
        r[3] = x[1] * x[2] - 1.5
        J[1, 1] = 1 + 0.1 * cos(x[1])
        J[2, 1] = 0.4 * x[1]; J[2, 2] = 1.0
        J[3, 1] = x[2]; J[3, 2] = x[1]
        return 0.5 * (r[1]^2 + r[2]^2 + r[3]^2)
    end
    explicit!(r, J, x) = (cost = partial!(r, J, x); J[1, 2] = c12; cost)
    f(x) = 0.5 * ((x[1] + c12 * x[2] - 1 + 0.1 * sin(x[1]))^2 + (x[2] - 2 + 0.2 * x[1]^2)^2 + (x[1] * x[2] - 1.5)^2)
    return f, partial!, explicit!
end
q_prototype(c12) = (P = zeros(3, 2); P[1, 2] = c12; P)

@testset "Q1: a Matrix prototype is used as handed" begin
    @testset "Unwritten prototype entries are part of J: :$it, callbackchangesstate = $ccs" for it in Z_ITERATORS, ccs in (false, true)
        c12 = 0.7
        f, partial!, explicit! = q_partial(c12)
        opts = w_opts(it, :callbackchangesstate => ccs)
        x0 = [0.2, 0.3]
        cR, _, xR, cbR = w_trace()
        _, resR = optimize(x0, JacobianObjective(f, explicit!, 3), opts; callback=cbR)
        @test resR.bestcost < resR.startcost
        P = q_prototype(c12)
        c, _, xs, cb = w_trace()
        _, res = optimize(x0, JacobianObjective(f, partial!, P), opts; callback=cb)
        @test c == cR && xs == xR
        # the prototype's unwritten entry is bit-identical afterwards
        @test P[1, 2] === c12
        # a second run with a reused workspace, built from this prototype. A workspace owns its
        # Jacobian storage, so P is not written at all from here on
        Pbefore = copy(P)
        ws = OptimizationWorkspace(x0, JacobianObjective(f, partial!, P), opts)
        @test getA(ws.data[1].linsystem).J !== P
        for _ in 1:2
            c, _, xs, cb = w_trace()
            optimize(x0, JacobianObjective(f, partial!, P), opts; workspace=ws, callback=cb)
            @test c == cR && xs == xR
        end
        @test P[1, 2] === c12
        # the same workspace with another prototype matrix, holding another constant: the run is
        # that of a fresh run with the new prototype, and the new prototype is not written
        f2, partial2!, explicit2! = q_partial(-0.4)
        c2R, _, x2R, cb2R = w_trace()
        optimize(x0, JacobianObjective(f2, explicit2!, 3), opts; callback=cb2R)
        P2 = q_prototype(-0.4)
        P2copy = copy(P2)
        @test SOO.workspacevalid(ws, x0, opts, SOO.objective_structure(JacobianObjective(f2, partial2!, P2)))
        c, _, xs, cb = w_trace()
        optimize(x0, JacobianObjective(f2, partial2!, P2), opts; workspace=ws, callback=cb)
        @test c == c2R && xs == x2R
        @test P2 == P2copy
        # and back to the first prototype on the same workspace
        c, _, xs, cb = w_trace()
        optimize(x0, JacobianObjective(f, partial!, P), opts; workspace=ws, callback=cb)
        @test c == cR && xs == xR
        @test P[1, 2] === c12 && P == Pbefore
        # zero allocation per run still holds with the per-run copy
        w_allocs(zeros(2), JacobianObjective(f2, partial2!, P2), opts, ws)
        @test w_allocs(zeros(2), JacobianObjective(f2, partial2!, P2), opts, ws) == 0
    end

    @testset "Integer m: J starts as zeros, on the first run and on a reused workspace: :$it, callbackchangesstate = $ccs" for it in Z_ITERATORS, ccs in (false, true)
        seen = IdDict{Any, Bool}()
        nnotzero = Ref(0)
        function check12!(r, J, x)
            if !haskey(seen, J)
                seen[J] = true
                J[1, 2] == 0.0 || (nnotzero[] += 1)
            end
            return q_partial(0.0)[2](r, J, x)
        end
        writer!(r, J, x) = (cost = q_partial(0.0)[2](r, J, x); J[1, 2] = 5.0; cost)   # leaves J[1, 2] = 5
        f = q_partial(0.0)[1]
        opts = w_opts(it, :callbackchangesstate => ccs)
        optimize([0.2, 0.3], JacobianObjective(f, check12!, 3), opts)
        @test nnotzero[] == 0 && length(seen) >= 1
        ws = OptimizationWorkspace([0.2, 0.3], JacobianObjective(f, writer!, 3), opts)
        optimize([0.2, 0.3], JacobianObjective(f, writer!, 3), opts; workspace=ws)
        empty!(seen)
        optimize([0.2, 0.3], JacobianObjective(f, check12!, 3), opts; workspace=ws)
        @test nnotzero[] == 0 && length(seen) == (ccs || it === :newton ? 1 : 2)
    end
end
