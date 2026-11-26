using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Test, Static
include("fixtures/load.jl"); loadfixtures("gradientdescent.jl")   # :gradientdescent, the tutorial's user-defined iterator

function nanwall_cost(x)
    val = x[1]
    return val > 0.5 ? NaN : 0.5 * (val - 3.0)^2
end

function nanwall_fdf!(g, H, x)
    val = x[1]
    if val > 0.5
        g[1] = NaN
        H[1, 1] = NaN
        return NaN
    end
    g[1] = val - 3.0
    H[1, 1] = 1.0
    return 0.5 * (val - 3.0)^2
end

function infwall_cost(x)
    val = x[1]
    return val > 0.5 ? Inf : 0.5 * (val - 3.0)^2
end

function infwall_fdf!(g, H, x)
    val = x[1]
    if val > 0.5
        g[1] = Inf
        H[1, 1] = Inf
        return Inf
    end
    g[1] = val - 3.0
    H[1, 1] = 1.0
    return 0.5 * (val - 3.0)^2
end

# Concave region (H <= 0) with +Inf wall at x > 0.5: f(x) = -0.5 * (x + 1)^2
function concave_infwall_cost(x)
    val = x[1]
    return val > 0.5 ? Inf : -0.5 * (val + 1.0)^2
end

function concave_infwall_fdf!(g, H, x)
    val = x[1]
    if val > 0.5
        g[1] = Inf
        H[1, 1] = Inf
        return Inf
    end
    g[1] = -(val + 1.0)
    H[1, 1] = -1.0
    return -0.5 * (val + 1.0)^2
end

# Zero-Hessian linear cost with +Inf wall at x > 0.5: f(x) = -2x
function linear_infwall_cost(x)
    val = x[1]
    return val > 0.5 ? Inf : -2.0 * val
end

function linear_infwall_fdf!(g, H, x)
    val = x[1]
    if val > 0.5
        g[1] = Inf
        H[1, 1] = Inf
        return Inf
    end
    g[1] = -2.0
    H[1, 1] = 0.0
    return -2.0 * val
end

@testset "nantrial.jl" begin
    # Test rejection of NaN trial steps
    for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
        opts = OptimizationOptions(iterator=iterator, maxiters=50)
        x0 = @SVector [0.0]
        x_opt, res = optimize(x0, HessianObjective(nanwall_cost, nanwall_fdf!), opts)
        @test 0.4 < x_opt[1] <= 0.5
        @test isfinite(res.bestcost) && res.bestcost < 0.5 * 9.0
        @test res.bestcost == nanwall_cost(x_opt)
    end

    # Test rejection of +Inf trial steps via trust-region / step-size shrinking
    for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
        opts = OptimizationOptions(iterator=iterator, maxiters=50)
        x0 = @SVector [0.0]
        x_opt, res = optimize(x0, HessianObjective(infwall_cost, infwall_fdf!), opts)
        @test 0.4 < x_opt[1] <= 0.5
        @test isfinite(res.bestcost) && res.bestcost < 0.5 * 9.0
        @test res.bestcost == infwall_cost(x_opt)
    end

    # Dogleg with negative curvature (g'Hg < 0) and zero curvature (g'Hg == 0) hitting +Inf wall
    opts_dl = OptimizationOptions(iterator=:dogleg, linsolver=:ldl, maxiters=50)
    x_conc, res_conc = optimize(@SVector([0.0]), HessianObjective(concave_infwall_cost, concave_infwall_fdf!), opts_dl)
    @test 0.4 < x_conc[1] <= 0.5
    @test res_conc.bestcost == concave_infwall_cost(x_conc)

    x_lin, res_lin = optimize(@SVector([0.0]), HessianObjective(linear_infwall_cost, linear_infwall_fdf!), opts_dl)
    @test 0.4 < x_lin[1] <= 0.5
    @test res_lin.bestcost == linear_infwall_cost(x_lin)

    # Test non-finite start point
    x_bad = @SVector [1.0] # In NaN wall region
    opts = OptimizationOptions()
    x_opt, res = optimize(x_bad, HessianObjective(nanwall_cost, nanwall_fdf!), opts)
    @test (res.termination & TERMINATION_START_NOT_FINITE) != 0
    @test (res.termination & TERMINATION_COST_NAN) != 0

    # Test non-finite linearization at an accepted point (e.g. acos(min(x, 1)) where gradient becomes NaN for x >= 1)
    nanlinearization_cost(x) = 0.5 * (acos(min(x[1], 1.0))^2 + (x[1] - 2.0)^2)
    function nanlinearization_fdf!(g, H, x)
        r1 = acos(min(x[1], 1.0))
        r2 = x[1] - 2.0
        dr1 = x[1] >= 1.0 ? NaN : -1.0 / sqrt(1.0 - x[1]^2)
        dr2 = 1.0
        g[1] = r1 * dr1 + r2 * dr2
        H[1, 1] = dr1^2 + dr2^2
        return 0.5 * (r1^2 + r2^2)
    end
    for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton, :gradientdescent)
        opts = OptimizationOptions(iterator=iterator, maxtime=2.0)
        x_opt, res = optimize([0.0], HessianObjective(nanlinearization_cost, nanlinearization_fdf!), opts)
        @test (res.termination & (TERMINATION_STEP_NAN | TERMINATION_COST_NAN | TERMINATION_GRAD_NAN)) != 0
        @test isfinite(res.bestcost) && res.bestcost < nanlinearization_cost([0.0])
        @test res.bestcost == nanlinearization_cost(x_opt)
    end
end

# AS4: a Hessian that damping cannot repair (a NaN at every evaluation) ends the run at its first failed solve,
# for every iterator and storage, with x0 returned and the linear solver's failure reported
@testset "A non-finite Hessian stops the solve retries at once (AS4)" begin
    base = [4.0 1 0 0; 1 3 1 0; 0 1 2 0.5; 0 0 0.5 1]
    bv = [1.0, 2, 3, 4]
    cost(x) = 0.5 * dot(x, base, x) - dot(bv, x)
    function fdf!(g, H, x)
        g .= base * x .- bv
        H isa SparseMatrixCSC ? (H.nzval .= vec(base)) : (H .= base)
        H[2, 2] = NaN
        return cost(x)
    end
    for (storage, H, x0, solvers) in (("static", nothing, @SVector(zeros(4)), (:auto, :qr)),
                                      ("dense", zeros(4, 4), zeros(4), (:auto, :cholesky, :ldl, :qr)),
                                      ("sparse", sparse(ones(4, 4)), zeros(4), (:auto, :cholesky, :ldl, :qr))),
        s in solvers, it in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd)
        @testset "$storage, $s, $it" begin
            x, res = optimize(x0, HessianObjective(cost, fdf!; H), OptimizationOptions(; iterator = it, linsolver = s))
            @test res.niterations == 1
            @test res.termination == TERMINATION_LINSOLVER_FAILED
            @test x == x0
        end
        # Dogleg's curvature gᵀHg is NaN, so it forms no step and returns NO_STEP: the loop stops before the callback
        # and the step tests, with the same termination every run (it read the step storage before, uninitialised
        # memory for a static system, and set a step bit at random)
        @testset "$storage, $s, dogleg" begin
            for _ in 1:3
                calls = Ref(0)
                cb = (cost, x, data, itdata) -> (calls[] += 1; (cost, false))
                x, res = optimize(x0, HessianObjective(cost, fdf!; H), OptimizationOptions(; iterator = :dogleg, linsolver = s); callback = cb)
                @test res.niterations == 1
                @test res.termination == TERMINATION_COST_NAN
                @test calls[] == 0
                @test x == x0
            end
        end
    end
end

# An infinite gradient and a NaN gradient set different bits, at the start and after a step
@testset "TERMINATION_GRAD_INF and TERMINATION_GRAD_NAN" begin
    for (bad, bit, other) in ((Inf, TERMINATION_GRAD_INF, TERMINATION_GRAD_NAN), (NaN, TERMINATION_GRAD_NAN, TERMINATION_GRAD_INF))
        cost(x) = 0.5 * sum(abs2, x .- 1)
        # at the start
        fdf0!(g, H, x) = (g .= x .- 1; g[1] = bad; H .= 0; H[1, 1] = H[2, 2] = 1.0; cost(x))
        _, res = optimize([0.0, 0.0], HessianObjective(cost, fdf0!; H = zeros(2, 2)), OptimizationOptions())
        @test res.niterations == 0
        @test res.termination & bit != 0 && res.termination & TERMINATION_START_NOT_FINITE != 0
        @test res.termination & other == 0
        # after the first step, which lands at x[1] = 1
        fdf1!(g, H, x) = (g .= x .- 1; x[1] > 0.5 && (g[2] = bad); H .= 0; H[1, 1] = H[2, 2] = 1.0; cost(x))
        _, res = optimize([0.0, 0.0], HessianObjective(cost, fdf1!; H = zeros(2, 2)), OptimizationOptions(; iterator = :newton))
        @test res.niterations >= 1
        @test res.termination & bit != 0 && res.termination & TERMINATION_START_NOT_FINITE == 0
        @test res.termination & other == 0
        @test !isconverged(res)
    end
end
