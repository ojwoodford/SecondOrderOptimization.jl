using SecondOrderOptimization, LinearAlgebra, SparseArrays, StaticArrays, Test
using SecondOrderOptimization: fast_dot, quadform

# A Hessian storage with only a three-argument `dot` method, to exercise quadform's fallback (test only)
struct DotOnly{M}
    A::M
end
LinearAlgebra.dot(x::AbstractVector, D::DotOnly, y::AbstractVector) = dot(x, D.A, y)

quadform_allocations(A, v) = @allocated quadform(A, v)   # behind a function barrier

# quadform agrees with v' A v, is type-stable, and allocates nothing
function test_quadform(A, v, ref = v' * (A isa DotOnly ? A.A : A) * v)
    @test quadform(A, v) ≈ ref
    @test (@inferred quadform(A, v)) isa Float64
    quadform(A, v)
    @test quadform_allocations(A, v) == 0
end

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

    # quadform: each hand-written method, and the fallback to dot(v, A, v)
    test_quadform(randn(20, 20), randn(20))
    test_quadform(sprandn(100, 100, 0.05), randn(100))
    test_quadform(randn(SMatrix{10, 10, Float64, 100}), randn(SVector{10, Float64}))
    test_quadform(randn(MMatrix{4, 4, Float64, 16}), randn(MVector{4, Float64}))
    test_quadform(DotOnly(Diagonal(randn(7))), randn(7))
    @test which(quadform, (DotOnly{Diagonal{Float64, Vector{Float64}}}, Vector{Float64})) ==
          which(quadform, (Any, Any))   # the fallback
end
