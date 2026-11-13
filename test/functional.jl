using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Test, Static

function rosenbrock_cost(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_fdf!(g, H, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    g[2] = 200.0 * (x[2] - x[1]^2)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    return rosenbrock_cost(x)
end

@testset "functional.jl - Rosenbrock optimization" begin
    x0_static = @SVector [-1.2, 1.0]
    x0_dense = [-1.2, 1.0]

    # Test second-order iterators with StaticVector
    for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
        opts = OptimizationOptions(iterator=it, maxiters=100, dstep=1e-12)
        x_opt, res = optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts)
        @test isapprox(x_opt, [1.0, 1.0]; atol=1e-4)
        @test res.bestcost < 1e-6
    end

    # Test gradient descent reduces cost
    opts_gd = OptimizationOptions(iterator=:gradientdescent, maxiters=200)
    x_opt_gd, res_gd = optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_gd)
    @test res_gd.bestcost < rosenbrock_cost(x0_static) * 0.05

    # Test each iterator with dynamic Vector
    for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
        opts = OptimizationOptions(iterator=it, maxiters=100)
        x_opt, res = optimize(copy(x0_dense), HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts)
        @test isapprox(x_opt, [1.0, 1.0]; atol=1e-4)
    end

    # Test linear solver options
    for solver in (:auto, :cholesky, :ldl, :qr)
        opts = OptimizationOptions(iterator=:dogleg, linsolver=solver, maxiters=100)
        x_opt, res = optimize(copy(x0_dense), HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts)
        @test isapprox(x_opt, [1.0, 1.0]; atol=1e-4)
    end

    # Test gtol termination
    opts_gtol = OptimizationOptions(gtol=1e-3, maxiters=100)
    x_opt, res = optimize(copy(x0_dense), HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_gtol)
    @test isapprox(x_opt, [1.0, 1.0]; atol=1e-2)
    @test (res.termination & TERMINATION_GRAD_TOL) != 0
    g_final = zeros(2)
    H_final = zeros(2, 2)
    rosenbrock_fdf!(g_final, H_final, x_opt)
    @test maximum(abs, g_final) < 1e-3

    # Test gtol termination at initial point (0 iterations)
    x_at_min = @SVector [1.0, 1.0]
    x_opt0, res0 = optimize(x_at_min, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_gtol)
    @test res0.niterations == 0
    @test (res0.termination & TERMINATION_GRAD_TOL) != 0
    # M18: termination reasons without the raw bits
    @test termination_reasons(res0) == [:grad_tol]
    @test :grad_tol in termination_reasons(res)
    @test occursin("Gradient norm below threshold.", sprint(show, res0))
    bits = [bit for (bit, _, _) in SecondOrderOptimization.TERMINATION_REASONS]
    @test allunique(bits) && all(ispow2, bits) && reduce(|, bits) == (1 << 15) - 1
    _, res_cb = optimize(copy(x0_dense), HessianObjective(rosenbrock_cost, rosenbrock_fdf!), OptimizationOptions(maxiters=50); callback=(c, args...) -> (c, 1))
    @test termination_reasons(res_cb) == [:callback]
    @test !isconverged(res_cb)

    # Test OptimizationResult addition order
    r_sum = res + res0
    @test r_sum.timeinit == res.timeinit + res0.timeinit
    @test r_sum.timecost == res.timecost + res0.timecost

    # Test callbacks
    traj = CostTrajectory()
    opts_cb = OptimizationOptions(maxiters=20)
    x_opt, res = optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_cb; callback=storecostscallback(traj))
    @test length(traj.costs) > 0
    @test length(traj.times_ns) == length(traj.costs)
    @test length(traj.steps) == length(traj.costs)
    # M16: `steps` holds the step vectors
    @test all(length(s) == 2 for s in traj.steps)

    # Test zero-allocation on SVector with OptimizationWorkspace
    opts_alloc = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=30)
    ws = OptimizationWorkspace(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_alloc)
    # Warmup
    optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_alloc; workspace=ws)
    # Allocation test
    # Measured inside a function: at testset top level Julia 1.10 boxes the captured variables
    rosen_allocs(x, obj, o, w) = @allocated optimize(x, obj, o; workspace=w)
    rosen_allocs(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_alloc, ws)
    allocs = rosen_allocs(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_alloc, ws)
    @test allocs == 0

    # Test printoutcallback and OptimizationResult show
    io = IOBuffer()
    show(io, res)
    str = String(take!(io))
    @test occursin("SecondOrderOptimization", str)
    @test occursin("termination", str)

    # Test enum options (OptimizationIterator & OptimizationLinearSolver)
    opts_enum = OptimizationOptions(iterator=SecondOrderOptimization.dogleg, linsolver=SecondOrderOptimization.cholesky_solver, maxiters=50)
    x_enum, _ = optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_enum)
    @test isapprox(x_enum, [1.0, 1.0]; atol=1e-4)

    # Test that optimize does not mutate x0
    opts_test = OptimizationOptions(maxiters=100)
    x0_orig = [-1.2, 1.0]
    x0_copy = copy(x0_orig)
    x_opt_nomut, _ = optimize(x0_copy, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_test)
    @test x0_copy == x0_orig
    @test x_opt_nomut !== x0_copy
    @test isapprox(x_opt_nomut, [1.0, 1.0]; atol=1e-4)

    # Test optimize! mutates mutable vector in-place, and returns the result only
    x_mut = copy(x0_orig)
    res_mut = optimize!(x_mut, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_test)
    @test res_mut isa OptimizationResult
    @test isapprox(x_mut, [1.0, 1.0]; atol=1e-4)

    # optimize! refuses an immutable StaticVector; optimize handles it
    @test_throws ArgumentError optimize!(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_test)
    x_stat_opt, _ = optimize(x0_static, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), opts_test)
    @test isapprox(x_stat_opt, [1.0, 1.0]; atol=1e-4)
end
