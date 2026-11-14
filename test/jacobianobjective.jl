using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Static, Random, Test
const SOO = SecondOrderOptimization
using SecondOrderOptimization: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!

# J1, J4: JacobianObjective, its validation, and the optimize / optimize! entry points

# A two-residual Rosenbrock, counting the calls to f and resjac!
const JCALLS = Ref(0)
jrosen_cost(x) = (JCALLS[] += 1; 0.5 * ((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2))
function jrosen_resjac!(r, J, x)
    JCALLS[] += 1
    r[1] = 1 - x[1]
    r[2] = 10 * (x[2] - x[1]^2)
    J[1, 1] = -1.0
    J[2, 1] = -20 * x[1]
    J[2, 2] = 10.0
    return 0.5 * (r[1]^2 + r[2]^2)
end

@testset "jacobianobjective.jl" begin
    @testset "JacobianObjective" begin
        obj = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        @test obj.f === jrosen_cost && obj.resjac! === jrosen_resjac! && obj.J === 2
        J = zeros(2, 2)
        @test JacobianObjective(jrosen_cost, jrosen_resjac!, J).J === J
        @test SOO.nresiduals(obj) == 2 && SOO.nresiduals(JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(5, 2))) == 5
    end

    @testset "Unsupported Jacobian prototypes are rejected when the objective is built" begin
        for (J, needle) in ((spzeros(2, 2), "sparse"), (zeros(Float32, 2, 2), "Float64"), (zeros(Int, 2, 2), "Float64"),
                            (zeros(MMatrix{2, 2}), "not supported"), (view(zeros(3, 2), 1:2, :), "not supported"),
                            (Diagonal(ones(2)), "not supported"), (nothing, "not supported"), (2.0, "not supported"),
                            (true, "not supported"), (0, "at least 1"), (-3, "at least 1"), (zeros(0, 2), "no rows"))
            err = try JacobianObjective(jrosen_cost, jrosen_resjac!, J); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin(needle, err.msg)
            @test occursin("Matrix{Float64}", err.msg) && occursin(":levenbergmarquardt", err.msg)
        end
    end

    @testset "Validation at entry: iterator, linsolver and the prototype's columns" begin
        obj = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        err = try SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(iterator=:gradientdescent)); nothing catch e; e end
        @test err isa ArgumentError && occursin(":gradientdescent", err.msg) && occursin("gains nothing", err.msg)
        @test occursin(":levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg or :newton", err.msg)
        err = try SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(iterator=gradientdescent)); nothing catch e; e end
        @test err isa ArgumentError && occursin(":gradientdescent", err.msg)
        # the other iterators are accepted, as symbols and as OptimizationIterator values
        for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton, levenbergmarquardt, levenbergmarquardt_dtd, dogleg, newton)
            @test SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(iterator=it)) === nothing
        end
        for s in (:cholesky, :ldl)
            err = try SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(linsolver=s)); nothing catch e; e end
            @test err isa ArgumentError && occursin(":$s", err.msg) && occursin(":auto or :qr", err.msg)
        end
        for s in (:auto, :qr, auto_solver, qr_solver)
            @test SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(linsolver=s)) === nothing
        end
        err = try SOO.checkobjective([0.0, 0.0, 0.0], JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 2)), OptimizationOptions()); nothing catch e; e end
        @test err isa DimensionMismatch && occursin("3 columns", err.msg)
        # the integer form has no columns to check, and any m >= 1 is accepted
        @test SOO.checkobjective([0.0, 0.0, 0.0], JacobianObjective(jrosen_cost, jrosen_resjac!, 1), OptimizationOptions()) === nothing
        # no user function was called by any of this
        @test JCALLS[] == 0
    end

    @testset "Every rejection throws before f or resjac! is called" begin
        obj = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        bad = Any[(obj, OptimizationOptions(iterator=it), ArgumentError) for it in (:gradientdescent, gradientdescent)]
        append!(bad, [(obj, OptimizationOptions(linsolver=s), ArgumentError) for s in (:cholesky, :ldl, cholesky_solver, ldl_solver)])
        push!(bad, (JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 3)), OptimizationOptions(), DimensionMismatch))
        for (o, opts, E) in bad
            JCALLS[] = 0
            @test_throws E optimize([-1.2, 1.0], o, opts)
            @test_throws E optimize!([-1.2, 1.0], o, opts)
            @test_throws E optimize([-1.2, 1.0], o, opts; workspace=OptimizationWorkspace())
            @test_throws E OptimizationWorkspace([-1.2, 1.0], o, opts)
            @test JCALLS[] == 0
        end
    end
end

# Both objectives of a least-squares problem, from allocating residual and Jacobian functions: the
# HessianObjective forms the normal equations from the same J
function lsq_objectives(resid, jac, m, n)
    f(x) = 0.5 * sum(abs2, resid(x))
    resjac!(r, J, x) = (r .= resid(x); J .= jac(x); 0.5 * sum(abs2, r))
    fdf!(g, H, x) = (r = resid(x); J = jac(x); g .= J' * r; H .= J' * J; 0.5 * sum(abs2, r))
    return JacobianObjective(f, resjac!, m), HessianObjective(f, fdf!; H = zeros(n, n))
end

# A callback recording the cost and the damping of each iteration
function lambdatrace()
    costs, lambdas = Float64[], Float64[]
    return costs, lambdas, (cost, x, data, itd) -> (push!(costs, cost); push!(lambdas, itd.lambda); (cost, 0))
end

# A rotation in SO(3), updated on the right by the exponential of the step
struct Rot3
    R::SMatrix{3, 3, Float64, 9}
end
skew3(w) = @SMatrix [0.0 -w[3] w[2]; w[3] 0.0 -w[1]; -w[2] w[1] 0.0]
function expso3(w)
    θ = norm(w)
    K = skew3(w)
    θ < 1e-12 && return SMatrix{3, 3}(1.0I) + K
    return SMatrix{3, 3}(1.0I) + (sin(θ) / θ) * K + ((1 - cos(θ)) / θ^2) * (K * K)
end
SOO.nvars(::Rot3) = static(3)
SOO.update(x::Rot3, dx, start=1) = Rot3(x.R * expso3(SVector(dx[start], dx[start + 1], dx[start + 2])))

@testset "optimize with a JacobianObjective" begin
    @testset "Linear problem: one undamped step lands on A \\ b" begin
        rng = Xoshiro(21)
        A = randn(rng, 12, 4); b = randn(rng, 12)
        obj = JacobianObjective(x -> 0.5 * sum(abs2, A * x - b), (r, J, x) -> (r .= A * x - b; J .= A; 0.5 * sum(abs2, r)), 12)
        for x0 in (zeros(4), randn(rng, 4), 1e3 * randn(rng, 4))
            ls = SOO.makelinearsystem(x0, obj, :auto)
            zero!(ls)
            SOO.linearizer(obj)(getb(ls), getA(ls), x0)
            dampen!(ls, 0.0)
            @test solve!(ls, :auto) == Success
            @test x0 - getx(ls) ≈ A \ b rtol=1e-11
        end
        x, res = optimize(randn(rng, 4), obj, OptimizationOptions(gtol=1e-10))
        @test isconverged(res)
        @test x ≈ A \ b rtol=1e-9
    end

    @testset "Agreement with the normal equations on well-conditioned problems" begin
        # Rosenbrock as two residuals; an exponential fit with a nonzero residual; n = 30
        t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
        rng = Xoshiro(11); M = randn(rng, 45, 30) ./ sqrt(45) + I(45)[:, 1:30]; yy = randn(rng, 45)
        problems = (
            ("Rosenbrock", x -> [1 - x[1], 10 * (x[2] - x[1]^2)], x -> [-1.0 0.0; -20x[1] 10.0], 2, [-1.2, 1.0]),
            ("exponential fit", x -> @.(x[1] * exp(x[2] * t) - y), x -> hcat(exp.(x[2] .* t), x[1] .* t .* exp.(x[2] .* t)), 25, [1.0, 0.0]),
            ("n = 30", x -> M * tanh.(x) .- yy, x -> M * Diagonal(sech.(x) .^ 2), 45, zeros(30)),
        )
        for (name, resid, jac, m, x0) in problems
            objJ, objH = lsq_objectives(resid, jac, m, length(x0))
            cJ, lJ, cbJ = lambdatrace()
            cH, lH, cbH = lambdatrace()
            xJ, resJ = optimize(x0, objJ, OptimizationOptions(); callback=cbJ)
            xH, resH = optimize(x0, objH, OptimizationOptions(linsolver=:cholesky); callback=cbH)
            @test isconverged(resJ) && isconverged(resH)
            # The cost at every iteration both runs share. The runs may end a few iterations apart,
            # once the cost decreases are at the level of rounding.
            k = min(length(cJ), length(cH))
            @test k >= min(length(cJ), length(cH), 6)
            @test all(isapprox.(cJ[1:k], cH[1:k]; rtol=1e-8, atol=1e-8 * abs(resH.startcost) * eps()))
            @test abs(length(cJ) - length(cH)) <= 3
            # The damping follows the ratio of the actual to the predicted decrease. The actual
            # decrease carries a relative rounding error of eps·cost/decrease, which the damping
            # update compounds from one iteration to the next, so compare the damping over the
            # leading iterations whose relative decrease is above 1e-6
            prev = [resJ.startcost; cJ[1:k-1]]
            lead = something(findfirst(prev .- cH[1:k] .<= 1e-6 .* abs.(cH[1:k])), k + 1) - 1
            @test lead >= 2
            @test all(isapprox.(lJ[1:lead], lH[1:lead]; rtol=1e-8))
            @test resJ.bestcost ≈ resH.bestcost rtol=1e-8 atol=1e-20
        end
    end

    @testset "Ill-conditioned problem: the tall path reaches the least-squares minimum" begin
        # A degree-13 polynomial fit (cond(J) ≈ 4e9), scaled so that the iterator's floor on the
        # damping (1e-12) is far below the smallest squared singular value
        tt = collect(range(0, 1; length=60))
        V = 1e4 * hcat((tt .^ k for k in 0:13)...)
        yv = 1e4 * (@. sin(3tt) + 0.01 * cos(17tt))
        @test 1e9 < cond(V) < 1e11
        objJ, objH = lsq_objectives(x -> V * x - yv, x -> V, 60, 14)
        opts = OptimizationOptions(maxiters=500)
        xJ, resJ = optimize(zeros(14), objJ, opts)
        xH, resH = optimize(zeros(14), objH, OptimizationOptions(maxiters=500, linsolver=:cholesky))
        # Costs in extended precision: in Float64 the cost of an iterate with coefficients this
        # large is rounded at about 1e-8 relative
        exactcost(x) = Float64(0.5 * sum(abs2, big.(V) * big.(x) - big.(yv)))
        costmin = exactcost(big.(V) \ big.(yv))
        gJ = norm(V' * (V * xJ - yv), Inf)
        gH = norm(V' * (V * xH - yv), Inf)
        @info "Ill-conditioned fit (cond(J) = $(round(cond(V), sigdigits=2)), least-squares minimum $costmin)" tall_cost=resJ.bestcost tall_excess=exactcost(xJ)-costmin tall_gradnorm=gJ tall_iterations=resJ.niterations normal_cost=resH.bestcost normal_excess=exactcost(xH)-costmin normal_gradnorm=gH normal_iterations=resH.niterations
        @test resJ.bestcost <= resH.bestcost
        @test exactcost(xJ) - costmin <= 1e-12 * costmin
        @test exactcost(xH) - costmin > 1e-6 * costmin
        @test resJ.niterations < resH.niterations
        # Compared on cost only. The gradient norm (as the optimizer computes it, in Float64) is at
        # the level of its own rounding here: the gradient at the exact minimizer, rounded to Float64,
        # is of the same order, so gJ < gH (asserted before Q5) depends on the platform and on the
        # last few iterations; both norms are in the @info record above.
    end

    @testset "The cost is not derived from r: reweighted residuals" begin
        # A line fit with outliers under the Cauchy cost ½Σ log(1 + rᵢ²); resjac! writes the
        # reweighted residual √ρ'·rᵢ and Jacobian, whose ½‖r‖² is not the cost but whose Jᵀr is its gradient
        t = collect(range(-1, 1; length=30)); y = @. 0.5 + 2t + 0.02 * sin(9t)
        y[[4, 17, 25]] .+= (8.0, -6.0, 10.0)
        raw(x) = @. x[1] + x[2] * t - y
        cauchy(x) = 0.5 * sum(log1p, raw(x) .^ 2)
        function cauchy_resjac!(r, J, x)
            r0 = raw(x)
            w = @. sqrt(1 / (1 + r0^2))
            @. r = w * r0
            @. J[:, 1] = w
            @. J[:, 2] = w * t
            return cauchy(x)
        end
        x, res = optimize([0.0, 0.0], JacobianObjective(cauchy, cauchy_resjac!, 30), OptimizationOptions(gtol=1e-10))
        @test isconverged(res)
        @test :cost_discrepancy ∉ termination_reasons(res)
        r = zeros(30); J = zeros(30, 2)
        @test cauchy_resjac!(r, J, x) != 0.5 * sum(abs2, r)
        @test norm(J' * r, Inf) < 1e-10
        @test x ≈ [0.5, 2.0] atol=0.05      # the outliers are down-weighted
        @test res.bestcost ≈ cauchy(x)
    end

    @testset "Rank-deficient Jacobian: a duplicated column" begin
        rng = Xoshiro(5)
        a, c = randn(rng, 15), randn(rng, 15); y = randn(rng, 15)
        resid(x) = @. a * (x[1] + x[2]) + c * x[3] - y
        jac(x) = hcat(a, a, c)
        objJ, _ = lsq_objectives(resid, jac, 15, 3)
        ls = SOO.makelinearsystem(zeros(3), objJ, :auto)
        zero!(ls); SOO.linearizer(objJ)(getb(ls), getA(ls), zeros(3)); dampen!(ls, 0.0)
        @test solve!(ls, :auto) == NotPositiveDefinite
        x, res = optimize(zeros(3), objJ, OptimizationOptions())
        costmin = 0.5 * sum(abs2, hcat(a, c) * (hcat(a, c) \ y) - y)
        @test res.bestcost ≈ costmin rtol=1e-10
        @test isconverged(res)
    end

    @testset "m < n and m == n" begin
        rng = Xoshiro(9)
        M = randn(rng, 3, 5); y = 0.3 * randn(rng, 3)
        objJ, _ = lsq_objectives(x -> M * tanh.(x) - y, x -> M * Diagonal(sech.(x) .^ 2), 3, 5)
        x, res = optimize(zeros(5), objJ, OptimizationOptions())
        @test isconverged(res) && res.bestcost < 1e-20
        cvec = [1.0, -2.0, 3.0, 0.5]
        objJ, _ = lsq_objectives(x -> x .^ 3 .+ x .- cvec, x -> Diagonal(3 .* x .^ 2 .+ 1) * Matrix(1.0I, 4, 4), 4, 4)
        x, res = optimize(zeros(4), objJ, OptimizationOptions())
        @test isconverged(res) && res.bestcost < 1e-20
        @test x .^ 3 .+ x ≈ cvec
    end

    @testset "An integer m and a Matrix prototype give identical results" begin
        for opts in (OptimizationOptions(), OptimizationOptions(callbackchangesstate=true, linsolver=:qr))
            x1, r1 = optimize([-1.2, 1.0], JacobianObjective(jrosen_cost, jrosen_resjac!, 2), opts)
            x2, r2 = optimize([-1.2, 1.0], JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 2)), opts)
            @test x1 == x2
            @test (r1.startcost, r1.bestcost, r1.niterations, r1.costcomputations, r1.gradientcomputations, r1.linearsolves, r1.termination) ==
                  (r2.startcost, r2.bestcost, r2.niterations, r2.costcomputations, r2.gradientcomputations, r2.linearsolves, r2.termination)
        end
        # the prototype is the matrix resjac! writes into
        J = zeros(2, 2)
        optimize([-1.2, 1.0], JacobianObjective(jrosen_cost, jrosen_resjac!, J), OptimizationOptions())
        @test J != zeros(2, 2)
    end

    @testset "Variables" begin
        opts = OptimizationOptions(gtol=1e-10)
        objJ = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        # Vector, in place
        x = [-1.2, 1.0]
        @test isconverged(optimize!(x, objJ, opts)) && x ≈ [1.0, 1.0]
        # SVector and Float64 through optimize; optimize! refuses them
        xs, res = optimize(SVector(-1.2, 1.0), objJ, opts)
        @test xs isa SVector{2, Float64} && xs ≈ [1.0, 1.0] && isconverged(res)
        @test_throws ArgumentError optimize!(SVector(-1.2, 1.0), objJ, opts)
        obj1 = JacobianObjective(x -> 0.5 * (x^2 - 2)^2, (r, J, x) -> (r[1] = x^2 - 2; J[1, 1] = 2x; 0.5 * r[1]^2), 1)
        x1, res = optimize(1.0, obj1, opts)
        @test x1 isa Float64 && x1 ≈ sqrt(2) && isconverged(res)
        @test_throws ArgumentError optimize!(1.0, obj1, opts)
        # a Vector of static blocks: two points pulled towards targets, coupled by a distance
        tgt = (SVector(1.0, 2.0), SVector(-1.0, 0.5))
        blockres(x) = [x[1] - tgt[1]; x[2] - tgt[2]; norm(x[1] - x[2]) - 2.0]
        function blockjac(x)
            J = zeros(5, 4)
            J[1:2, 1:2] = I(2); J[3:4, 3:4] = I(2)
            d = x[1] - x[2]; u = d / norm(d)
            J[5, 1:2] = u; J[5, 3:4] = -u
            return J
        end
        objB = JacobianObjective(x -> 0.5 * sum(abs2, blockres(x)), (r, J, x) -> (r .= blockres(x); J .= blockjac(x); 0.5 * sum(abs2, r)), 5)
        xb = [SVector(0.0, 0.0), SVector(0.5, 0.0)]
        res = optimize!(xb, objB, opts)
        @test isconverged(res)
        @test norm(blockjac(xb)' * blockres(xb), Inf) < 1e-8
        # a rotation, updated through the exponential map: the Jacobian is with respect to the
        # tangent step of `update`, here d(R exp([w]×) p)/dw = -R [p]×
        P = [SVector(1.0, 0.0, 0.0), SVector(0.0, 2.0, 0.0), SVector(0.0, 0.0, 1.0), SVector(1.0, 1.0, 1.0)]
        Rtrue = expso3(SVector(0.4, -0.3, 0.9))
        Q = [Rtrue * p for p in P]
        rotres(x::Rot3) = reduce(vcat, [x.R * p - q for (p, q) in zip(P, Q)])
        function rot_resjac!(r, J, x::Rot3)
            for (i, p) in enumerate(P)
                r[3i-2:3i] = x.R * P[i] - Q[i]
                J[3i-2:3i, :] = -x.R * skew3(p)
            end
            return 0.5 * sum(abs2, r)
        end
        x0 = Rot3(expso3(SVector(0.1, 0.2, -0.1)))
        r = zeros(12); J = zeros(12, 3)
        rot_resjac!(r, J, x0)
        h = 1e-7
        Jfd = reduce(hcat, [(rotres(SOO.update(x0, h * I(3)[:, k])) - rotres(x0)) / h for k in 1:3])
        @test J ≈ Jfd rtol=1e-5
        xr, res = optimize(x0, JacobianObjective(x -> 0.5 * sum(abs2, rotres(x)), rot_resjac!, 12), opts)
        @test xr isa Rot3 && isconverged(res)
        @test xr.R ≈ Rtrue atol=1e-10
    end

    @testset "Both Levenberg variants" begin
        objJ = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        x_trial, res_trial = optimize([-1.2, 1.0], objJ, OptimizationOptions(gtol=1e-10))
        x_plain, res_plain = optimize([-1.2, 1.0], objJ, OptimizationOptions(gtol=1e-10, callbackchangesstate=true))
        @test isconverged(res_trial) && isconverged(res_plain)
        @test x_trial ≈ [1.0, 1.0] && x_plain ≈ [1.0, 1.0]
        # the trial variant reuses the trial linearization, so it needs fewer cost evaluations
        @test res_trial.costcomputations < res_plain.costcomputations
        @test res_trial.niterations == res_plain.niterations
        @test SOO.construct(static(:levenbergmarquardt), SOO.makelinearsystem([0.0, 0.0], objJ, :auto), False()) isa LevenbergDataTrial
    end

    @testset "Non-finite residuals or Jacobians stop the run" begin
        # at the start
        objnan = JacobianObjective(x -> 1.0, (r, J, x) -> (r .= 1.0; J .= NaN; 1.0), 2)
        _, res = optimize([0.0, 0.0], objnan, OptimizationOptions())
        @test :grad_not_finite in termination_reasons(res) && :start_not_finite in termination_reasons(res)
        @test res.niterations == 0
        # after the first accepted step: J holds an Inf once x[1] > 0.5
        f(x) = 0.5 * sum(abs2, x .- 1)
        objlate = JacobianObjective(f, (r, J, x) -> (r .= x .- 1; J .= I(2); x[1] > 0.5 && (J[1, 1] = Inf); f(x)), 2)
        _, res = optimize([0.0, 0.0], objlate, OptimizationOptions())
        @test :grad_not_finite in termination_reasons(res)
        @test res.niterations >= 1 && !isconverged(res)
    end

    @testset "Workspace" begin
        opts = OptimizationOptions()
        objJ = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        ws = OptimizationWorkspace([-1.2, 1.0], objJ, opts)
        @test !isempty(ws)
        x = [-1.2, 1.0]
        allocs(x, o, w) = (x .= (-1.2, 1.0); @allocated optimize!(x, o, opts; workspace=w))
        allocs(x, objJ, ws)
        @test allocs(x, objJ, ws) == 0
        @test x ≈ [1.0, 1.0]
        # the same with a Matrix prototype and the other Levenberg variant
        opts2 = OptimizationOptions(callbackchangesstate=true)
        objP = JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 2))
        ws2 = OptimizationWorkspace([-1.2, 1.0], objP, opts2)
        allocs2(x, o, w) = (x .= (-1.2, 1.0); @allocated optimize!(x, o, opts2; workspace=w))
        allocs2(x, objP, ws2)
        @test allocs2(x, objP, ws2) == 0
        # keyed on the objective kind and on m
        key = SOO.objective_structure
        @test SOO.workspacevalid(ws, [0.0, 0.0], opts, key(objJ))
        @test SOO.workspacevalid(ws, [0.0, 0.0], opts, key(objP))
        @test !SOO.workspacevalid(ws, [0.0, 0.0], opts, key(JacobianObjective(jrosen_cost, jrosen_resjac!, 3)))
        objH = HessianObjective(x -> 0.5 * sum(abs2, x), (g, H, x) -> (g .= x; H .= I(2); 0.5 * sum(abs2, x)))
        @test !SOO.workspacevalid(ws, [0.0, 0.0], opts, key(objH))
        wsH = OptimizationWorkspace([1.0, 1.0], objH, opts)
        @test !SOO.workspacevalid(wsH, [0.0, 0.0], opts, key(objJ))
        @test !SOO.workspacevalid(OptimizationWorkspace([1.0, 1.0], HessianObjective(objH.f, objH.fdf!; H=zeros(2, 2)), opts), [0.0, 0.0], opts, key(objJ))
        # a workspace for another m is rebuilt (and, being of the same type, kept); one for the
        # other kind is of another type, so is emptied
        obj3 = JacobianObjective(x -> 0.5 * sum(abs2, [x[1] - 1, x[2] - 2, x[1] + x[2] - 3]),
                                 (r, J, x) -> (r .= [x[1] - 1, x[2] - 2, x[1] + x[2] - 3]; J .= [1.0 0.0; 0.0 1.0; 1.0 1.0]; 0.5 * sum(abs2, r)), 3)
        x3, _ = optimize([0.0, 0.0], obj3, opts; workspace=ws)
        @test x3 ≈ [1.0, 2.0]
        @test SOO.workspacevalid(ws, [0.0, 0.0], opts, key(obj3)) && !SOO.workspacevalid(ws, [0.0, 0.0], opts, key(objJ))
        @test size(getA(ws.data[1].linsystem).J) == (3, 2)
        xh, _ = @test_logs (:warn, r"emptied") optimize([3.0, 3.0], objH, opts; workspace=OptimizationWorkspace([-1.2, 1.0], objJ, opts))
        @test xh ≈ [0.0, 0.0] atol=1e-8
        xj, _ = @test_logs (:warn, r"emptied") optimize([-1.2, 1.0], objJ, opts; workspace=OptimizationWorkspace([3.0, 3.0], objH, opts))
        @test xj ≈ [1.0, 1.0]
        # an adaptive workspace switches between them
        wsa = OptimizationWorkspace()
        @test optimize([-1.2, 1.0], objJ, opts; workspace=wsa)[1] ≈ [1.0, 1.0]
        @test optimize([0.0, 0.0], obj3, opts; workspace=wsa)[1] ≈ [1.0, 2.0]
        @test optimize([3.0, 3.0], objH, opts; workspace=wsa)[1] ≈ [0.0, 0.0] atol=1e-8
    end

    @testset "Type stability and no allocation per iteration" begin
        opts = OptimizationOptions()
        objJ = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        objP = JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 2))
        for obj in (objJ, objP)
            @inferred optimize([-1.2, 1.0], obj, opts)
            @inferred optimize!([-1.2, 1.0], obj, opts)
            @inferred optimize(SVector(-1.2, 1.0), obj, opts)
            ws = OptimizationWorkspace([-1.2, 1.0], obj, opts)
            @inferred optimize!([-1.2, 1.0], obj, opts; workspace=ws)
            res = optimize!([-1.2, 1.0], obj, opts)
            @test res.optimizeallocations == 0 && res.niterations > 5
        end
    end

    @testset "Termination and callbacks" begin
        objJ = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        # gtol stops on ‖Jᵀr‖∞
        x, res = optimize([-1.2, 1.0], objJ, OptimizationOptions(gtol=1e-6, reldcost=0.0, absdcost=0.0, dstep=0.0))
        @test termination_reasons(res) == [:grad_tol]
        r = zeros(2); J = zeros(2, 2); jrosen_resjac!(r, J, x)
        @test norm(J' * r, Inf) < 1e-6
        _, res = optimize([-1.2, 1.0], objJ, OptimizationOptions(gtol=1e3))
        @test termination_reasons(res) == [:grad_tol] && res.niterations == 0
        # storecostscallback records the trajectory
        traj = CostTrajectory()
        _, res = optimize([-1.2, 1.0], objJ, OptimizationOptions(); callback=storecostscallback(traj))
        @test length(traj.costs) == res.niterations && traj.costs[end] == res.bestcost && length(traj.steps[1]) == 2
        costs = Float64[]
        optimize([-1.2, 1.0], objJ, OptimizationOptions(); callback=storecostscallback(costs))
        @test costs == traj.costs
        # maxiters and maxtime
        _, res = optimize([-1.2, 1.0], objJ, OptimizationOptions(maxiters=3))
        @test :max_iters in termination_reasons(res) && res.niterations == 3
        _, res = optimize([-1.2, 1.0], objJ, OptimizationOptions(maxtime=0.0))
        @test :max_time in termination_reasons(res) && res.niterations == 1
        # the time in resjac! is counted under gradients
        @test res.gradientcomputations >= 1 && res.timegradient > 0
    end
end
