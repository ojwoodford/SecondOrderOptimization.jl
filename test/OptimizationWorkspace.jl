using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Test, Static

function quad_cost(x)
    return 0.5 * dot(x, x)
end

function quad_fdf!(g, H, x)
    g .= x
    if H isa SparseMatrixCSC
        for i in 1:size(H, 1)
            H[i, i] = 1.0
        end
    else
        fill!(H, 0.0)
        for i in axes(H, 1)
            H[i, i] = 1.0
        end
    end
    return quad_cost(x)
end

@testset "OptimizationWorkspace.jl" begin
    x0_static = @SVector [2.0, -3.0]
    opts = OptimizationOptions(iterator=:dogleg)

    # 1. Eagerly constructed workspace
    ws = OptimizationWorkspace(quad_cost, quad_fdf!, x0_static, opts)
    @test !isempty(ws)
    x_opt, res = optimize(quad_cost, quad_fdf!, x0_static, opts; workspace=ws)
    @test isapprox(x_opt, [0.0, 0.0]; atol=1e-8)

    # Re-running with same workspace achieves 0 allocations
    allocs = @allocated optimize(quad_cost, quad_fdf!, x0_static, opts; workspace=ws)
    @test allocs == 0

    # 2. Key invalidation when options change (and safe fallback on concretely-typed workspace)
    opts_new = OptimizationOptions(iterator=:levenbergmarquardt)
    @test !SecondOrderOptimization.workspacevalid(ws, x0_static, opts_new, nothing)
    x_opt_new, _ = optimize(quad_cost, quad_fdf!, x0_static, opts_new; workspace=ws)
    @test isapprox(x_opt_new, [0.0, 0.0]; atol=1e-8)
    @test isempty(ws)

    # 3. Dynamic workspace
    ws_dyn = OptimizationWorkspace()
    @test isempty(ws_dyn)
    x_opt, res = optimize(quad_cost, quad_fdf!, x0_static, opts; workspace=ws_dyn)
    @test !isempty(ws_dyn)
    @test isapprox(x_opt, [0.0, 0.0]; atol=1e-8)

    # Emptying workspace
    empty!(ws_dyn)
    @test isempty(ws_dyn)

    # 4. SparseMatrixCSC workspace
    n = 20
    H_sparse = spdiagm(0 => ones(n))
    x0_vec = fill(1.5, n)
    opts_sparse = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=10)
    ws_sp = OptimizationWorkspace(quad_cost, quad_fdf!, x0_vec, opts_sparse; H=H_sparse)
    @test !isempty(ws_sp)

    x_opt_sp, res_sp = optimize(quad_cost, quad_fdf!, copy(x0_vec), opts_sparse; H=H_sparse, workspace=ws_sp)
    @test isapprox(x_opt_sp, zeros(n); atol=1e-8)

    # Warmup and test allocations with sparse workspace
    x_buf = copy(x0_vec)
    optimize(quad_cost, quad_fdf!, x_buf, opts_sparse; H=H_sparse, workspace=ws_sp)
    @test ws_sp.data[1].variables !== ws_sp.data[1].varnext
    x_buf .= x0_vec
    allocs_sp = @allocated optimize(quad_cost, quad_fdf!, x_buf, opts_sparse; H=H_sparse, workspace=ws_sp)
    @test ws_sp.data[1].variables !== ws_sp.data[1].varnext
    @test allocs_sp == 0
end
