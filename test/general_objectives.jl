using SecondOrderOptimization, StaticArrays, LinearAlgebra, SparseArrays, Test

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
    # M5: a huge finite maxtime saturates instead of throwing InexactError
    for t in (1e11, 1e30, floatmax(Float64))
        @test OptimizationOptions(maxtime=t).maxtime == typemax(UInt64)
    end
    @test OptimizationOptions(maxtime=1e9).maxtime == UInt64(1e18)
    @test OptimizationOptions(maxtime=-1.0).maxtime == 0
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

    # 5. B5: Levenberg-Marquardt damping update stays bounded when the predicted change is
    # not negative (indefinite Hessian)
    luf = SecondOrderOptimization.lambda_update_factor
    @test luf(-1.0, 0.0) == 2.0             # zero prediction: was a factor of Inf
    @test luf(-1.0, 1e-20) == 2.0           # wrong-signed prediction
    @test luf(-1.0, 0.1) == 2.0             # q = -10: was a factor of 9262
    @test luf(1.0, -0.1) == 2.0             # q = -10 with a negative prediction
    @test luf(-0.5, -1.0) == 1.0            # q = 0.5
    @test luf(-1.0, -1.0) == 0.1            # q = 1, very good step
    @test luf(0.0, -1.0) == 2.0             # q = 0
    for q in range(-5, 2; length=101)
        @test 0.1 <= luf(-q, -1.0) <= 2.0
    end

    # Double well in x, sinusoid in y: indefinite Hessian near the start. :qr returns a step
    # even when the damped Hessian is indefinite, so the predicted change can be positive.
    dw_cost(x) = -x[1]^2 + 0.25 * x[1]^4 + 1 - cos(3 * x[2])
    function dw_fdf!(g, H, x)
        g[1] = -2 * x[1] + x[1]^3
        g[2] = 3 * sin(3 * x[2])
        H .= 0
        H[1, 1] = -2 + 3 * x[1]^2
        H[2, 2] = 9 * cos(3 * x[2])
        return dw_cost(x)
    end
    for solver in (:qr, :auto), it in (:levenbergmarquardt, :levenbergmarquardt_dtd), x0 in ([0.1, 0.4], [0.05, 0.9], [1e-3, 1.0])
        lambdas = Float64[]
        lmcb(cost, xt, data, itd) = (push!(lambdas, itd.lambda); (cost, 0))
        x_dw, res_dw = optimize(dw_cost, dw_fdf!, copy(x0), OptimizationOptions(iterator=it, linsolver=solver, gtol=1e-8); H=zeros(2, 2), callback=lmcb)
        @test all(isfinite, lambdas)
        @test maximum(lambdas) < 1e8
        @test isconverged(res_dw)
        if solver === :qr
            # :qr ignores indefiniteness, so it may stop at the saddle x[1] = 0
            @test abs(x_dw[1]) < 1e-6 || isapprox(abs(x_dw[1]), sqrt(2); atol=1e-6)
        else
            @test isapprox(abs(x_dw[1]), sqrt(2); atol=1e-6)   # reached a minimum in x
        end
        @test abs(sin(3 * x_dw[2])) < 1e-6 && cos(3 * x_dw[2]) > 0
    end

    # M7: the Hessian prototype must match the number of variables
    sq(x) = sum(abs2, x)
    sq_fdf!(g, H, x) = (g .= 2 .* x; H .= 2 * I(length(x)); sq(x))
    for H_bad in (zeros(2, 2), spzeros(4, 4), zeros(3, 2))
        @test_throws DimensionMismatch optimize(sq, sq_fdf!, ones(3), OptimizationOptions(); H=H_bad)
        @test_throws DimensionMismatch OptimizationWorkspace(ones(3), OptimizationOptions(); H=H_bad)
    end
    # M4: the Hessian is Float64; another sparse element type is a clear error
    err = try optimize(sq, sq_fdf!, ones(3), OptimizationOptions(); H=sparse(Float32(1) * I(3))); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("Float64", sprint(showerror, err))
end
