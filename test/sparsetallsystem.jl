using SecondOrderOptimization, SparseArrays, LinearAlgebra, Static, Random, Test
import DampedLinearSolve
using SecondOrderOptimization: dampen!, zero!, copy_structure, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!
const SOO = SecondOrderOptimization
include("fixtures/termbits.jl")

# TP1: the sparse tall system (a SparseMatrixCSC{Float64, Int} Jacobian, solved by SPQR) against
# the dense tall system on the same J

# Both systems set to J and r as a linearization leaves them (b = Jᵀr)
function tp1_pair(J::SparseMatrixCSC, r)
    ls, ld = DampedLinearSolve.SparseTallLinearSystem(copy(J)), DampedLinearSolve.DenseTallLinearSystem(Matrix(J))
    for s in (ls, ld)
        zero!(s)
        getA(s).r .= r
        DampedLinearSolve.jtr!(getb(s), getA(s).J, getA(s).r)
    end
    return ls, ld
end

tp1_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))
tp1_allocations(ls, lambda) = (tp1_solve!(ls, lambda, nothing); Base.@allocations tp1_solve!(ls, lambda, nothing))
tp1_bytes(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); @allocated f(args...))

# The exponential fit of the JacobianObjective docstring, written without allocating, with J's
# stored entries set through `nonzeros` (the prototype stores both full columns)
const TP1_T = collect(range(0, 1; length = 20))
const TP1_Y = @. 2.0 * exp(-1.5 * TP1_T)
function tp1_cost(x)
    total = 0.0
    for i in eachindex(TP1_T)
        total += (x[1] * exp(x[2] * TP1_T[i]) - TP1_Y[i])^2
    end
    return 0.5 * total
end
function tp1_resjac!(r, J, x)
    m = length(TP1_T)
    for i in 1:m
        e = exp(x[2] * TP1_T[i])
        r[i] = x[1] * e - TP1_Y[i]
        if J isa SparseMatrixCSC
            nonzeros(J)[i] = e
            nonzeros(J)[m + i] = x[1] * TP1_T[i] * e
        else
            J[i, 1] = e
            J[i, 2] = x[1] * TP1_T[i] * e
        end
    end
    return tp1_cost(x)
end
tp1_run(x, obj, opts, ws) = (x .= (1.0, 0.0); optimize!(x, obj, opts; workspace = ws))

@testset "sparsetallsystem.jl" begin
    @testset "Every Jacobian iterator: :$it, callbackchangesstate = $ccs" for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
        opts = OptimizationOptions(; iterator = it, callbackchangesstate = ccs)
        xs, rs = optimize([1.0, 0.0], JacobianObjective(tp1_cost, tp1_resjac!, sparse(ones(20, 2))), opts)
        xd, rd = optimize([1.0, 0.0], JacobianObjective(tp1_cost, tp1_resjac!, ones(20, 2)), opts)
        @test isconverged(rs)
        @test xs ≈ [2.0, -1.5] rtol = 1e-8
        @test rs.niterations == rd.niterations
        @test rs.bestcost ≈ rd.bestcost atol = 1e-20
    end

    @testset "Allocation: only SPQR's working memory, one fixed set per solve" begin
        # a warm run with a workspace allocates exactly SPQR's allocations for each solve
        @testset "run with :$it, callbackchangesstate = $ccs" for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
            opts = OptimizationOptions(; iterator = it, callbackchangesstate = ccs)
            obj = JacobianObjective(tp1_cost, tp1_resjac!, sparse(ones(20, 2)))
            ws = OptimizationWorkspace([1.0, 0.0], obj, opts)
            x = [1.0, 0.0]
            tp1_run(x, obj, opts, ws)
            GC.gc()   # pending finalizers (SuiteSparse objects of earlier tests) would count as allocations
            res = tp1_run(x, obj, opts, ws)
            k = tp1_allocations(ws.data[1].linsystem, 1.0)
            @test isconverged(res) && res.linearsolves > 0
            @test res.initallocations == 0
            @test res.optimizeallocations == res.linearsolves * k
        end
    end

    @testset "Prototypes, patterns and workspaces" begin
        f(x) = 0.5 * sum(abs2, x)
        rj!(r, J, x) = (r .= x; nonzeros(J) .= 1.0; f(x))
        # other sparse types are rejected up front, before f or resjac! is called
        called = Ref(false)
        g(x) = (called[] = true; f(x))
        for P in (SparseMatrixCSC{Float64, Int32}(sparse(1.0I, 2, 2)), SparseMatrixCSC{Float32, Int}(sparse(1.0f0I, 2, 2)), spzeros(0, 2))
            err = try optimize([1.0, 2.0], JacobianObjective(g, rj!, P)); nothing catch e; e end
            @test err isa ArgumentError
            @test !called[]
        end
        @test_throws DimensionMismatch optimize([1.0, 2.0], JacobianObjective(f, rj!, sparse(1.0I, 3, 3)))
        # resjac! writing outside the prototype's pattern throws
        bad!(r, J, x) = (r .= x; J[2, 1] = 1.0; f(x))
        @test_throws ArgumentError optimize([1.0, 2.0], JacobianObjective(f, bad!, sparse(1.0I, 2, 2)))
        # a workspace is reused for the same pattern, and rebuilt for another
        opts = OptimizationOptions()
        P = sparse(1.0I, 2, 2)
        ws = OptimizationWorkspace([1.0, 2.0], JacobianObjective(f, rj!, P), opts)
        sys = ws.data[1].linsystem
        x, res = optimize([1.0, 2.0], JacobianObjective(f, rj!, copy(P)), opts; workspace = ws)
        @test ws.data[1].linsystem === sys && isconverged(res) && norm(x) < 1e-12
        P2 = sparse([1.0 1.0; 0.0 1.0])
        rj2!(r, J, x) = (r .= x; nonzeros(J) .= (1.0, 0.0, 1.0); f(x))
        x, res = optimize([1.0, 2.0], JacobianObjective(f, rj2!, P2), opts; workspace = ws)
        @test ws.data[1].linsystem !== sys && isconverged(res)
        # and a sparse Jacobian's workspace is not reused for a sparse Hessian of the same pattern
        # (an untyped workspace, which can hold either)
        wsany = OptimizationWorkspace()
        optimize([1.0, 2.0], JacobianObjective(f, rj!, sparse(1.0I, 2, 2)), opts; workspace = wsany)
        @test wsany.data[1].linsystem isa DampedLinearSolve.SparseTallLinearSystem
        fdf!(gr, H, x) = (gr .= x; H[1, 1] = H[2, 2] = 1.0; f(x))
        x, res = optimize([1.0, 2.0], HessianObjective(f, fdf!; H = sparse(1.0I, 2, 2)), opts; workspace = wsany)
        @test wsany.data[1].linsystem isa DampedLinearSolve.SparseLinearSystem && isconverged(res)
    end
end
