using SecondOrderOptimization, SparseArrays, StaticArrays, Static, LinearAlgebra, Test
include("fixtures/load.jl"); loadfixtures("gradientdescent.jl")   # :gradientdescent, the tutorial's user-defined iterator
const SOO = SecondOrderOptimization

# The less travelled paths of the iterators, callbacks and workspace: each testset names the branch it reaches

@testset "printoutcallback prints a row per iteration, for every iterator" begin
    cost(x) = 0.5 * sum(abs2, x .- 1)
    fdf!(g, H, x) = (g .= x .- 1; H .= Matrix(1.0I, 2, 2); cost(x))
    for it in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
        out = mktemp() do path, io
            redirect_stdout(io) do
                optimize([0.0, 0.0], HessianObjective(cost, fdf!), OptimizationOptions(iterator = it, maxiters = 2); callback = printoutcallback)
            end
            flush(io)
            read(path, String)
        end
        lines = split(chomp(out), '\n')
        @test startswith(lines[1], "iter      cost      cost_change    |step|")
        @test endswith(lines[1], it in (:newton, :gradientdescent) ? "|step|" : "tr_radius")   # a user-defined iterator prints the four standard columns
        @test length(lines) >= 3   # the header, iteration 0 and at least one iteration
    end
end

@testset "empty!(::CostTrajectory)" begin
    ct = CostTrajectory()
    cost(x) = 0.5 * sum(abs2, x .- 1)
    fdf!(g, H, x) = (g .= x .- 1; H .= Matrix(1.0I, 2, 2); cost(x))
    optimize([0.0, 0.0], HessianObjective(cost, fdf!), OptimizationOptions(maxiters = 3); callback = storecostscallback(ct))
    @test !isempty(ct.costs)
    @test empty!(ct) === ct
    @test isempty(ct.costs) && isempty(ct.times_ns) && isempty(ct.steps)
end

@testset "Workspace keys: hashes agree with ==, and an empty workspace matches nothing" begin
    H = sparse([4.0 1; 1 3])
    k1 = SOO.h_structure_key(H)
    k2 = SOO.h_structure_key(copy(H))
    @test k1 == k2 && hash(k1) == hash(k2)
    @test hash(k1) != hash(SOO.h_structure_key(sparse([4.0 0; 0 3])))
    @test hash(SOO.JacobianPrototypeKey(3)) == hash(SOO.JacobianPrototypeKey(3))
    @test hash(SOO.JacobianPrototypeKey(3)) != hash(SOO.JacobianPrototypeKey(4))
    @test !SOO.workspacevalid(OptimizationWorkspace(), [0.0], OptimizationOptions(), nothing)
end

@testset "An iterator that records nothing never restricts its step" begin
    # Behind a barrier, so that the call is made rather than folded to `false` where it is compiled
    @test SOO.restrictedstep(Base.inferencebarrier(GradientDescentData(1.0))) === false   # the fixture's type
end

@testset "reset! with x0 the workspace's best point swaps it in" begin
    cost(x) = 0.5 * sum(abs2, x .- 1)
    fdf!(g, H, x) = (g .= x .- 1; H .= Matrix(1.0I, 2, 2); cost(x))
    ws = OptimizationWorkspace([0.0, 0.0], HessianObjective(cost, fdf!))
    data = something(ws.data)[1]
    best = [0.5, 0.5]
    oldvariables = data.variables
    data.varbest = best
    data.varbestvalid = true
    SOO.reset!(data, best, SOO.Stats())
    @test data.variables === best
    @test data.varbest === oldvariables
    @test data.varbest == best   # synced to x0
end

# The retries of a failed solve with callbackchangesstate = true (the iterators without a trial system): one
# Hessian whose damping is given up when it grows too large, and one that damping cannot repair
@testset "A failed solve without a trial system ends the run" begin
    cost(x) = x[1]^2 + x[2]^2
    function unsolvable!(g, H, x)
        g .= 2 .* x
        H .= [0.0 1e20; 1e20 0.0]
        return cost(x)
    end
    function nonfinite!(g, H, x)
        g .= 2 .* x
        H .= [2.0 0.0; 0.0 NaN]
        return cost(x)
    end
    for fdf! in (unsolvable!, nonfinite!), it in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd)
        opts = OptimizationOptions(iterator = it, linsolver = :cholesky, callbackchangesstate = true, maxiters = 10)
        x, res = optimize([1.0, 1.0], HessianObjective(cost, fdf!), opts)
        @test res.termination == TERMINATION_LINSOLVER_FAILED
        @test x == [1.0, 1.0]
    end
end

@testset "Dogleg: a non-finite gᵀHg ends the run" begin
    cost(x) = sum(abs2, x)
    fdf!(g, H, x) = (g .= 2 .* x; H .= [Inf 0.0; 0.0 2.0]; cost(x))
    for ccs in (true, false)
        x, res = optimize([1.0, 1.0], HessianObjective(cost, fdf!), OptimizationOptions(iterator = :dogleg, callbackchangesstate = ccs))
        @test res.niterations == 1
        @test (res.termination & TERMINATION_COST_NAN) != 0
        @test x == [1.0, 1.0]
    end
end

# The first trust radius is the Newton step's length when that step is valid; without it, the Cauchy step's;
# without either, |g| over the largest diagonal entry (1 when that is not positive). This H is too indefinite
# for the shifted retries of the Newton solve, so there is no Newton step.
@testset "Dogleg: the first trust radius without a valid Newton step" begin
    H = [0.0 1e20; 1e20 0.0]
    for (g0, r0) in (([1.0, 1.0], sqrt(2) * 1e-20),   # gᵀHg = 2e20 > 0: the Cauchy step's length, |g|³/gᵀHg
                     ([1.0, -1.0], 1.0))              # gᵀHg < 0, and max(diag(H)) = 0
        cost(x) = 0.5 * dot(x, H, x) + dot(g0, x)
        fdf!(g, Hm, x) = (g .= H * x .+ g0; Hm .= H; cost(x))
        radius = Ref(NaN)
        function firstradius(cost_, x, data, iteratedata)
            data.iternum == 1 && (radius[] = iteratedata.trustradius)
            return cost_, 0
        end
        x, res = optimize([0.0, 0.0], HessianObjective(cost, fdf!), OptimizationOptions(iterator = :dogleg, maxiters = 1); callback = firstradius)
        # The first step was taken from the first radius, and then the radius was scaled from it
        @test res.bestcost < 0
        @test any(r -> radius[] ≈ r, (r0, 2r0, 0.25r0))
    end
end

@testset "Gradient descent: a line search that finds no decrease stops" begin
    # The linearization reports a cost of 0 and every trial point costs 1, however short the step (even one
    # that rounds away), so the search gives up after 100 trials
    x0 = [1.0, 1.0]
    ntrials = Ref(0)
    cost(x) = (ntrials[] += 1; 1.0)
    fdf!(g, H, x) = (g .= 1.0; 0.0)
    x, res = optimize(x0, HessianObjective(cost, fdf!), OptimizationOptions(iterator = :gradientdescent, maxiters = 1))
    @test ntrials[] == 101   # the first trial and 100 shortened ones
    @test res.bestcost == 0.0
    @test x == x0
end
