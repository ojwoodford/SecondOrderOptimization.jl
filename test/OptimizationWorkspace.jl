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

    # Warmup and test allocations with sparse workspace (in-place optimize! achieves 0 allocations)
    x_buf = copy(x0_vec)
    optimize!(quad_cost, quad_fdf!, x_buf, opts_sparse; H=H_sparse, workspace=ws_sp)
    @test ws_sp.data[1].variables !== ws_sp.data[1].varnext
    x_buf .= x0_vec
    allocs_sp = @allocated optimize!(quad_cost, quad_fdf!, x_buf, opts_sparse; H=H_sparse, workspace=ws_sp)
    @test ws_sp.data[1].variables !== ws_sp.data[1].varnext
    @test allocs_sp == 0

    # B7: block vectors with the same type and total dimension but different block layouts
    # must not share a cached workspace
    BlockT = Union{SVector{2, Float64}, SVector{3, Float64}}
    blk_cost(x) = 0.5 * sum(sum(abs2, xi) for xi in x)
    function blk_fdf!(g, H, x)
        k = 0
        for xi in x, v in xi
            k += 1
            g[k] = v
        end
        fill!(H, 0.0)
        for i in axes(H, 1)
            H[i, i] = 1.0
        end
        return blk_cost(x)
    end
    xa = BlockT[SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
    xb = BlockT[SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
    xc = BlockT[SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0), SVector(6.0, 7.0)]
    xd = BlockT[SVector(1.0, 2.0), SVector(3.0, 4.0, 5.0), SVector(6.0, 7.0)]
    opts_blk = OptimizationOptions(iterator=:levenbergmarquardt)
    ws_blk = OptimizationWorkspace()
    for x0 in (xa, xb, xa, xc, xd)
        x_blk, res_blk = optimize(blk_cost, blk_fdf!, x0, opts_blk; workspace=ws_blk)
        @test length(x_blk) == length(x0)
        @test all(length(x_blk[i]) == length(x0[i]) for i in eachindex(x0))
        @test isapprox(blk_cost(x_blk), 0.0; atol=1e-12)
    end
    @test SecondOrderOptimization.cachekey(xa, opts_blk, nothing) != SecondOrderOptimization.cachekey(xb, opts_blk, nothing)
    @test SecondOrderOptimization.cachekey(xc, opts_blk, nothing) != SecondOrderOptimization.cachekey(xd, opts_blk, nothing)
    @test SecondOrderOptimization.cachekey(xa, opts_blk, nothing) == SecondOrderOptimization.cachekey(copy(xa), opts_blk, nothing)
end
