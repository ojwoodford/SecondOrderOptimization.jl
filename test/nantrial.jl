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

@testset "nantrial.jl" begin
    # Test rejection of NaN trial steps
    for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg)
        opts = OptimizationOptions(iterator=iterator, maxiters=50)
        x0 = @SVector [0.0]
        x_opt, res = optimize(nanwall_cost, nanwall_fdf!, x0, opts)
        @test 0.4 < x_opt[1] <= 0.5
        @test isfinite(res.bestcost) && res.bestcost < 0.5 * 9.0
        @test res.bestcost == nanwall_cost(x_opt)
    end

    # Test non-finite start point
    x_bad = @SVector [1.0] # In NaN wall region
    opts = OptimizationOptions()
    x_opt, res = optimize(nanwall_cost, nanwall_fdf!, x_bad, opts)
    @test (res.termination & (1 << 11)) != 0
    @test (res.termination & (1 << 1)) != 0
end
