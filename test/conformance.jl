using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Static, Test
const SOO = SecondOrderOptimization

# A3: the linear-system interface, checked on every core system, and the checker itself checked
# on deliberately broken systems

# A system that forwards to a DenseLinearSystem, with one behaviour broken on request
mutable struct BrokenSystem <: AbstractLinearSystem
    inner::SOO.DenseLinearSystem{Float64, Nothing}
    broken::Symbol
end
BrokenSystem(n::Integer, broken) = BrokenSystem(SOO.DenseLinearSystem{Float64}(zeros(n, n), zeros(n), zeros(n)), broken)
SOO.getA(w::BrokenSystem) = w.inner.A
SOO.getb(w::BrokenSystem) = w.inner.b
SOO.getx(w::BrokenSystem) = w.inner.x
function SOO.solve!(w::BrokenSystem, s::StaticSymbol)
    status = SOO.solve!(w.inner, s)
    w.broken === :solve_mutates_A && (w.inner.A[1, 1] += w.inner.lambda)   # leaves damping behind
    return status
end
SOO.dampen!(w::BrokenSystem, λ::Real) = (w.broken === :dampen_keeps_scaling ? (w.inner.lambda = λ) : SOO.dampen!(w.inner, λ); w)
SOO.dampen!(w::BrokenSystem, λ::Real, s) = (SOO.dampen!(w.inner, λ, s); w)
function SOO.copy_structure(w::BrokenSystem)
    inner = SOO.copy_structure(w.inner)
    if w.broken === :copy_aliases_A
        inner = SOO.DenseLinearSystem{Float64}(w.inner.A, inner.b, inner.x, inner.A_scratch, inner.lambda, inner.scaling)
    end
    return BrokenSystem(inner, w.broken)
end
SOO.sharex(w::BrokenSystem, from) = w.broken === :sharex_copies ? BrokenSystem(SOO.copy_structure(w.inner), w.broken) :
                                                                   BrokenSystem(SOO.sharex(w.inner, from.inner), w.broken)
SOO.zero!(w::BrokenSystem) = (w.broken === :zero_keeps_b ? fill!(w.inner.A, 0.0) : SOO.zero!(w.inner); w)
SOO.dampingfloor(w::BrokenSystem) = w.broken === :floor_zero ? 0.0 : SOO.dampingfloor(w.inner)
SOO.max_diag(w::BrokenSystem) = w.broken === :max_diag_skips_nan ? maximum(x -> isnan(x) ? 0.0 : abs(x), diag(w.inner.A)) : SOO.max_diag(w.inner)

checker_message(make_ls; kwargs...) = try check_linear_system(make_ls; kwargs...); nothing catch e; e end

@testset "conformance.jl" begin
    solvers = (:auto, :cholesky, :ldl, :qr)

    @testset "StaticLinearSystem" begin
        for n in (1, 2, 5)
            @test check_linear_system(s -> SOO.makelinearsystem(zeros(SVector{n}), nothing, s); n, linsolvers=solvers) === nothing
            @test check_linear_system(s -> SOO.makelinearsystem(zeros(SVector{n}), zeros(MMatrix{n, n}), s); n, linsolvers=solvers) === nothing
            @test SOO.makelinearsystem(zeros(SVector{n}), nothing, :auto) isa SOO.StaticLinearSystem
        end
    end

    @testset "DenseLinearSystem" begin
        for n in (1, 6)
            @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), zeros(n, n), s); n, linsolvers=solvers) === nothing
            # also without the preallocated :ldl workspace
            @test check_linear_system(() -> SOO.DenseLinearSystem{Float64}(zeros(n, n), zeros(n), zeros(n)); n, linsolvers=solvers) === nothing
            @test SOO.makelinearsystem(zeros(n), zeros(n, n), :auto) isa SOO.DenseLinearSystem
        end
    end

    @testset "SparseLinearSystem" begin
        n = 8
        pattern = spdiagm(-2 => ones(n - 2), -1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1), 2 => ones(n - 2))
        pattern[1, n] = pattern[n, 1] = 1.0
        @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), copy(pattern), s); n, linsolvers=solvers) === nothing
        # a prototype missing diagonal entries (they are inserted) and an Int32 index type
        missing_diag = sparse([1, 2, 3, 1, 4, 5, 6, 7, 8, 8, 7], [2, 1, 1, 3, 4, 5, 6, 7, 8, 7, 8], ones(11), n, n)
        @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), copy(missing_diag), s); n, linsolvers=solvers) === nothing
        @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), SparseMatrixCSC{Float64, Int32}(pattern), s); n, linsolvers=solvers) === nothing
        @test SOO.makelinearsystem(zeros(n), copy(pattern), :auto) isa SOO.SparseLinearSystem
    end

    @testset "Krylov solvers" begin
        using Krylov
        n = 8
        pattern = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
        @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), copy(pattern), s); n, linsolvers=(:cg, :minres, :krylov), rtol=1e-6) === nothing
        @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), zeros(n, n), s); n, linsolvers=(:cg, :minres, :krylov), rtol=1e-6) === nothing
    end

    @testset "Pardiso" begin
        using Pardiso
        if Pardiso.mkl_is_available() || Pardiso.panua_is_available()
            n = 8
            pattern = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
            @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), copy(pattern), s); n, linsolvers=(:pardiso,)) === nothing
            @test check_linear_system(s -> SOO.makelinearsystem(zeros(n), zeros(n, n), s); n, linsolvers=(:pardiso,)) === nothing
        else
            @test_skip "Pardiso library not available"
        end
    end

    @testset "The checker catches broken systems" begin
        n = 4
        @test checker_message(() -> BrokenSystem(n, :none); n, linsolvers=solvers) === nothing
        for (broken, rule) in ((:copy_aliases_A, "copy_structure gives its own A"),
                               (:solve_mutates_A, "solve! leaves A as it was written"),
                               (:dampen_keeps_scaling, "dampen!(ls, λ) solves (A + λI) x = b"),
                               (:sharex_copies, "sharex(trial, ls) aliases x"),
                               (:zero_keeps_b, "zero! zeroes b"),
                               (:max_diag_skips_nan, "max_diag is NaN if a diagonal entry is NaN"),
                               (:floor_zero, "dampingfloor is a relative damping in (0, 1e-6]"))
            err = checker_message(() -> BrokenSystem(n, broken); n, linsolvers=(:cholesky,))
            @test err isa ErrorException
            @test occursin("rule \"$rule\"", err.msg)
        end
        # the wrong dimension
        err = checker_message(() -> BrokenSystem(n + 1, :none); n, linsolvers=(:auto,))
        @test err isa ErrorException && occursin("getb has length n", err.msg)
    end
end
