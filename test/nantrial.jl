using SecondOrderOptimization, StaticArrays, Test, Static

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
    @test (res.termination & (1 << 11)) != 0
    @test (res.termination & (1 << 1)) != 0

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
        @test (res.termination & (TERMINATION_STEP_NAN | TERMINATION_COST_NAN | TERMINATION_GRAD_NOT_FINITE)) != 0
        @test isfinite(res.bestcost) && res.bestcost < nanlinearization_cost([0.0])
        @test res.bestcost == nanlinearization_cost(x_opt)
    end
end
