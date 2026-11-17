using SecondOrderOptimization, SparseArrays, LinearAlgebra, Static, Random, Test
import DampedLinearSolve
using Krylov
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
const SOO = SecondOrderOptimization

# TP2: LSMR on the Jacobian of the dense and sparse tall systems (KrylovExt), against their QR solve

function tp2_set!(ls, r)
    zero!(ls)
    getA(ls).r .= r
    DampedLinearSolve.jtr!(getb(ls), getA(ls).J, getA(ls).r)
    return ls
end
tp2_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
tp2_bytes(ls, lambda) = (tp2_solve!(ls, :lsmr, lambda, nothing); @allocated tp2_solve!(ls, :lsmr, lambda, nothing))

# The exponential fit, without allocating; a sparse J written through `nonzeros` (both columns
# stored)
const TP2_T = collect(range(0, 1; length = 20))
const TP2_Y = @. 2.0 * exp(-1.5 * TP2_T)
function tp2_cost(x)
    total = 0.0
    for i in eachindex(TP2_T)
        total += (x[1] * exp(x[2] * TP2_T[i]) - TP2_Y[i])^2
    end
    return 0.5 * total
end
function tp2_resjac!(r, J, x)
    m = length(TP2_T)
    for i in 1:m
        e = exp(x[2] * TP2_T[i])
        r[i] = x[1] * e - TP2_Y[i]
        if J isa SparseMatrixCSC
            nonzeros(J)[i] = e
            nonzeros(J)[m + i] = x[1] * TP2_T[i] * e
        else
            J[i, 1] = e
            J[i, 2] = x[1] * TP2_T[i] * e
        end
    end
    return tp2_cost(x)
end
tp2_run(x, obj, opts, ws) = (x .= (1.0, 0.0); @allocated optimize!(x, obj, opts; workspace = ws))

@testset "lsmr.jl" begin
    @testset "Every Jacobian iterator with :lsmr: :$it, callbackchangesstate = $ccs" for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
        opts = OptimizationOptions(; iterator = it, linsolver = :lsmr, callbackchangesstate = ccs)
        qopts = OptimizationOptions(; iterator = it, callbackchangesstate = ccs)
        _, rq = optimize([1.0, 0.0], JacobianObjective(tp2_cost, tp2_resjac!, 20), qopts)
        for P in (20, ones(20, 2), sparse(ones(20, 2)))
            obj = JacobianObjective(tp2_cost, tp2_resjac!, P isa Integer ? P : copy(P))
            x, res = optimize([1.0, 0.0], obj, opts)
            @test isconverged(res)
            @test x ≈ [2.0, -1.5] rtol = 1e-6
            @test res.bestcost ≈ rq.bestcost atol = 1e-12
            # no allocation in a warm run with a workspace
            ws = OptimizationWorkspace([1.0, 0.0], obj, opts)
            xw = [1.0, 0.0]
            tp2_run(xw, obj, opts, ws)
            @test tp2_run(xw, obj, opts, ws) == 0
        end
    end

    @testset ":lsmr with a HessianObjective throws up front" begin
        called = Ref(false)
        f(x) = (called[] = true; 0.5 * sum(abs2, x))
        fdf!(g, H, x) = (called[] = true; g .= x; H .= I(2); 0.5 * sum(abs2, x))
        for H in (nothing, zeros(2, 2), sparse(1.0I, 2, 2))
            err = try optimize([1.0, 2.0], HessianObjective(f, fdf!; H), OptimizationOptions(linsolver = :lsmr)); nothing catch e; e end
            @test err isa ArgumentError && occursin("only for a JacobianObjective", err.msg)
            @test_throws ArgumentError OptimizationWorkspace([1.0, 2.0], HessianObjective(f, fdf!; H), OptimizationOptions(linsolver = :lsmr))
        end
        @test !called[]
        # and :cg, :minres stay rejected for a JacobianObjective
        for s in (:cg, :minres)
            @test_throws ArgumentError optimize([1.0, 2.0], JacobianObjective(f, (r, J, x) -> f(x), 2), OptimizationOptions(linsolver = s))
        end
        @test !called[]
    end
end
