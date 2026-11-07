using SecondOrderOptimization, LinearAlgebra, SparseArrays, StaticArrays, Test
using SecondOrderOptimization: fast_dot, fast_bAb

test_fast_bAb(A, b) = @test fast_bAb(A, b) ≈ dot(b, A, b)

@testset "utils.jl" begin
    # fast_dot
    a = randn(1001)
    b = randn(1001)
    @test fast_dot(a, b) ≈ dot(a, b)
    @test fast_dot(Float64[], Float64[]) == 0.0
    @test fast_dot([1, 2, 3], [0.5, 0.5, 0.5]) == 3.0
    @test fast_dot([1, 2, 3], [0.5, 0.5, 0.5]) isa Float64
    @test fast_dot(view(a, 2:2:1000), view(b, 1:2:999)) ≈ dot(view(a, 2:2:1000), view(b, 1:2:999))
    @test fast_dot(SVector(1.0, 2.0), SVector(3.0, 4.0)) == 11.0

    # fast_bAb
    test_fast_bAb(randn(20, 20), randn(20))
    test_fast_bAb(sprandn(100, 100, 0.05), randn(100))
    test_fast_bAb(randn(SMatrix{10, 10, Float64, 100}), randn(SVector{10, Float64}))
    test_fast_bAb(randn(MMatrix{4, 4, Float64, 16}), randn(MVector{4, Float64}))
end
