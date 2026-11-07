using SecondOrderOptimization, StaticArrays, LinearAlgebra, Test

@testset "general_objectives.jl" begin
    # 1. Nonzero minimum / offset cost in Dogleg and all second-order iterators
    offset_cost(x) = 1000.0 + (x[1] - 1.0)^2 + 2.0 * (x[2] - 3.0)^2
    function offset_fdf!(g, H, x)
        g[1] = 2.0 * (x[1] - 1.0)
        g[2] = 4.0 * (x[2] - 3.0)
        H[1, 1] = 2.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 4.0
        return offset_cost(x)
    end

    x0 = @SVector [10.0, -5.0]
    for it in (:dogleg, :levenbergmarquardt, :levenbergmarquardt_dtd, :newton)
        opts = OptimizationOptions(iterator=it, gtol=1e-8)
        x_opt, res = optimize(offset_cost, offset_fdf!, x0, opts)
        @test isapprox(x_opt, [1.0, 3.0]; atol=1e-5)
        @test isapprox(res.bestcost, 1000.0; atol=1e-5)
        @test isconverged(res)
    end

    # 2. Negative cost minimization
    neg_cost(x) = (x[1] - 2.0)^2 + 2.0 * (x[2] + 1.0)^2 - 50.0
    function neg_fdf!(g, H, x)
        g[1] = 2.0 * (x[1] - 2.0)
        g[2] = 4.0 * (x[2] + 1.0)
        H[1, 1] = 2.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 4.0
        return neg_cost(x)
    end

    x0_neg = @SVector [0.1, 0.3]
    for it in (:dogleg, :levenbergmarquardt, :levenbergmarquardt_dtd, :newton, :gradientdescent)
        opts = OptimizationOptions(iterator=it, gtol=1e-8, maxiters=200)
        x_opt, res = optimize(neg_cost, neg_fdf!, x0_neg, opts)
        @test isapprox(x_opt, [2.0, -1.0]; atol=1e-4)
        @test isapprox(res.bestcost, -50.0; atol=1e-4)
        @test isconverged(res)
    end

    # 3. maxtime = Inf and zero-time
    opts_inf = OptimizationOptions(maxtime=Inf)
    @test opts_inf.maxtime == typemax(UInt64)
    x_opt, res_inf = optimize(offset_cost, offset_fdf!, x0, opts_inf)
    @test isconverged(res_inf)

    # 4. Zero / flat Hessian does not get stuck in infinite loop
    zero_h_cost(x) = abs2(x[1] - 1.0)
    function zero_h_fdf!(g, H, x)
        g[1] = 2.0 * (x[1] - 1.0)
        H[1, 1] = 0.0 # flat Hessian
        return zero_h_cost(x)
    end
    opts_zh = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=5, dstep=1e-3)
    x_opt_zh, res_zh = optimize(zero_h_cost, zero_h_fdf!, @SVector[5.0], opts_zh)
    @test res_zh.bestcost < zero_h_cost(@SVector[5.0])

    # 5. Invalid options throw ArgumentError
    @test_throws ArgumentError OptimizationOptions(iterator=:invalid_iter)
    @test_throws ArgumentError OptimizationOptions(linsolver=:invalid_solver)
    @test_throws ArgumentError OptimizationOptions(callback=(cost, args...)->(cost, 0))

    # 6. Dense Linear System Zero Allocation on repeat run with workspace
    quad_dense_cost(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    function quad_dense_fdf!(g, H, x)
        g[1] = 2.0 * (x[1] - 1.0)
        g[2] = 2.0 * (x[2] - 2.0)
        H[1, 1] = 2.0
        H[1, 2] = 0.0
        H[2, 1] = 0.0
        H[2, 2] = 2.0
        return quad_dense_cost(x)
    end
    x0_dense = [0.0, 0.0]
    opts_dense = OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky)
    ws_dense = OptimizationWorkspace(x0_dense, opts_dense)
    x_buf = copy(x0_dense)
    optimize!(quad_dense_cost, quad_dense_fdf!, x_buf, opts_dense; workspace=ws_dense)
    allocs = @allocated optimize!(quad_dense_cost, quad_dense_fdf!, x_buf, opts_dense; workspace=ws_dense)
    @test allocs == 0

    # 7. CostTrajectory stores step trajectory
    ct = CostTrajectory()
    x_opt, res = optimize(offset_cost, offset_fdf!, x0, OptimizationOptions(iterator=:newton); callback=storecostscallback(ct))
    @test length(ct.trajectory) > 0
    @test length(ct.trajectory) == length(ct.costs)
    @test all(x -> length(x) == 2, ct.trajectory)
end
