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
        x_opt, res = optimize(nanwall_cost, nanwall_fdf!, x0, opts)
        @test 0.4 < x_opt[1] <= 0.5
        @test isfinite(res.bestcost) && res.bestcost < 0.5 * 9.0
        @test res.bestcost == nanwall_cost(x_opt)
    end

    # Test rejection of +Inf trial steps via trust-region / step-size shrinking
    for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
        opts = OptimizationOptions(iterator=iterator, maxiters=50)
        x0 = @SVector [0.0]
        x_opt, res = optimize(infwall_cost, infwall_fdf!, x0, opts)
        @test 0.4 < x_opt[1] <= 0.5
        @test isfinite(res.bestcost) && res.bestcost < 0.5 * 9.0
        @test res.bestcost == infwall_cost(x_opt)
    end

    # Dogleg with negative curvature (g'Hg < 0) and zero curvature (g'Hg == 0) hitting +Inf wall
    opts_dl = OptimizationOptions(iterator=:dogleg, linsolver=:ldl, maxiters=50)
    x_conc, res_conc = optimize(concave_infwall_cost, concave_infwall_fdf!, @SVector([0.0]), opts_dl)
    @test 0.4 < x_conc[1] <= 0.5
    @test res_conc.bestcost == concave_infwall_cost(x_conc)

    x_lin, res_lin = optimize(linear_infwall_cost, linear_infwall_fdf!, @SVector([0.0]), opts_dl)
    @test 0.4 < x_lin[1] <= 0.5
    @test res_lin.bestcost == linear_infwall_cost(x_lin)

    # Test non-finite start point
    x_bad = @SVector [1.0] # In NaN wall region
    opts = OptimizationOptions()
    x_opt, res = optimize(nanwall_cost, nanwall_fdf!, x_bad, opts)
    @test (res.termination & (1 << 11)) != 0
    @test (res.termination & (1 << 1)) != 0
end
