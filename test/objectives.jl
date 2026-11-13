using SecondOrderOptimization, StaticArrays, LinearAlgebra, Test

# A1: HessianObjective and the optimize / optimize! entry points

# Quadratic with minimum at `target`, written for any variable whose values can be flattened
const QUAD_TARGET = [1.0, -2.0, 3.0, 0.5]
flatvals(x::Number) = [x]
flatvals(x::AbstractVector{<:Number}) = collect(x)
flatvals(x::AbstractVector) = reduce(vcat, collect.(x))
flatvals(x::Tuple) = reduce(vcat, flatvals.(x))
quad_cost(x) = (v = flatvals(x); 0.5 * sum(abs2, v .- QUAD_TARGET[1:length(v)]))
function quad_fdf!(g, H, x)
    v = flatvals(x)
    g .= v .- QUAD_TARGET[1:length(v)]
    H .= 0
    for i in axes(H, 1)
        H[i, i] = 1.0
    end
    return quad_cost(x)
end
const QUAD = HessianObjective(quad_cost, quad_fdf!)

# Allocation-free cost and gradient for a static 2-vector, for the inference and allocation tests
quad2_cost(x) = 0.5 * ((x[1] - 1.0)^2 + (x[2] + 2.0)^2)
function quad2_fdf!(g, H, x)
    g[1] = x[1] - 1.0
    g[2] = x[2] + 2.0
    H[1, 1] = 1.0; H[1, 2] = 0.0; H[2, 1] = 0.0; H[2, 2] = 1.0
    return quad2_cost(x)
end

@testset "objectives.jl" begin
    @testset "HessianObjective" begin
        obj = HessianObjective(quad_cost, quad_fdf!)
        @test obj.f === quad_cost && obj.fdf! === quad_fdf! && obj.H === nothing
        H = zeros(4, 4)
        @test HessianObjective(quad_cost, quad_fdf!; H).H === H
    end

    @testset "optimize! refuses variables it cannot update in place" begin
        for x0 in (1.0, SVector(1.0, 2.0), (SVector(1.0, 2.0), 3.0), view([1.0, 2.0, 3.0], 1:2))
            err = try optimize!(x0, QUAD); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("optimize(", sprint(showerror, err))
        end
    end

    @testset "optimize! updates mutable variables in place and returns the result" begin
        for x0 in ([0.0, 0.0, 0.0, 0.0], MVector(0.0, 0.0), [SVector(0.0, 0.0), SVector(0.0, 0.0)])
            x = copy(x0)
            res = optimize!(x, QUAD, OptimizationOptions(gtol=1e-12))
            @test res isa OptimizationResult
            @test isconverged(res)
            @test isapprox(flatvals(x), QUAD_TARGET[1:length(flatvals(x))]; atol=1e-10)
        end
    end

    @testset "optimize leaves x0 untouched and matches optimize! on a copy" begin
        opts = OptimizationOptions(gtol=1e-12)
        for x0 in ([0.0, 0.0, 0.0, 0.0], MVector(0.0, 0.0), [SVector(0.0, 0.0), SVector(0.0, 0.0)])
            x0_before = deepcopy(x0)
            x, res = optimize(x0, QUAD, opts)
            @test x0 == x0_before
            @test x !== x0
            x_inplace = copy(x0)
            res_inplace = optimize!(x_inplace, QUAD, opts)
            @test x == x_inplace
            @test res.bestcost == res_inplace.bestcost && res.niterations == res_inplace.niterations
        end
        # and works for immutable variables
        for x0 in (0.0, SVector(0.0, 0.0), (SVector(0.0, 0.0), 0.0))
            x, res = optimize(x0, QUAD, opts)
            @test typeof(x) == typeof(x0)
            @test isapprox(flatvals(x), QUAD_TARGET[1:length(flatvals(x))]; atol=1e-10)
        end
    end

    @testset "Inference and zero allocations with a workspace" begin
        obj = HessianObjective(quad2_cost, quad2_fdf!)
        opts = OptimizationOptions()
        # static x0: optimize only
        xs = SVector(0.0, 0.0)
        @inferred optimize(xs, obj, opts)
        ws_s = OptimizationWorkspace(xs, obj, opts)
        @inferred optimize(xs, obj, opts; workspace=ws_s)
        allocs_s(x, o, w) = @allocated optimize(x, o, opts; workspace=w)
        allocs_s(xs, obj, ws_s)
        @test allocs_s(xs, obj, ws_s) == 0
        # dynamic x0: both entry points
        xd = [0.0, 0.0]
        @inferred optimize(xd, obj, opts)
        @inferred optimize!(copy(xd), obj, opts)
        ws_d = OptimizationWorkspace(xd, obj, opts)
        x_buf = copy(xd)
        @inferred optimize!(x_buf, obj, opts; workspace=ws_d)
        allocs_d(x, o, w) = (x .= 0.0; @allocated optimize!(x, o, opts; workspace=w))
        allocs_d(x_buf, obj, ws_d)
        @test allocs_d(x_buf, obj, ws_d) == 0
        @test isapprox(x_buf, [1.0, -2.0]; atol=1e-10)
    end

    @testset "The old (f, fdf!, x) forms are gone" begin
        for fn in (optimize, optimize!)
            @test !hasmethod(fn, Tuple{typeof(quad_cost), typeof(quad_fdf!), Vector{Float64}})
            @test !hasmethod(fn, Tuple{typeof(quad_cost), typeof(quad_fdf!), Vector{Float64}, OptimizationOptions})
            # no method takes the cost and its derivatives as the first two arguments
            @test !hasmethod(fn, Tuple{Function, Function, Any})
            @test !hasmethod(fn, Tuple{Function, Function, Any, OptimizationOptions})
        end
        @test !hasmethod(OptimizationWorkspace, Tuple{typeof(quad_cost), typeof(quad_fdf!), Vector{Float64}})
        # and the H keyword
        @test_throws MethodError optimize([0.0, 0.0], QUAD; H=zeros(2, 2))
        @test_throws MethodError OptimizationWorkspace([0.0, 0.0], OptimizationOptions(); H=zeros(2, 2))
    end
end
