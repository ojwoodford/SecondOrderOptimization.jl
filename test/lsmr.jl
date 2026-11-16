using SecondOrderOptimization, SparseArrays, LinearAlgebra, Static, Random, Test
using Krylov
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
const SOO = SecondOrderOptimization

# TP2: LSMR on the Jacobian of the dense and sparse tall systems (KrylovExt), against their QR solve

function tp2_set!(ls, r)
    zero!(ls)
    getA(ls).r .= r
    SOO.jtr!(getb(ls), getA(ls).J, getA(ls).r)
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
    @testset "Steps match QR: $kind ($m, $n)" for kind in (:dense, :sparse), (m, n) in ((40, 6), (6, 6), (40, 1), (4, 6))
        rng = Xoshiro(m + 7n)
        J = sprandn(rng, m, n, 0.5) + sparse(1:min(m, n), 1:min(m, n), 1.0, m, n)
        r = randn(rng, m)
        ll = kind === :dense ? SOO.DenseTallLinearSystem(Matrix(J); linsolver = :lsmr) : SOO.SparseTallLinearSystem(copy(J); linsolver = :lsmr)
        lq = SOO.DenseTallLinearSystem(Matrix(J))
        tp2_set!(ll, r); tp2_set!(lq, r)
        for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, 0.5 .+ rand(rng, n)), (1e-8, nothing))
            a, b = tp2_solve!(ll, :lsmr, lambda, scaling), tp2_solve!(lq, :qr, lambda, scaling)
            @test a == b
            # to LSMR's tolerance (√eps), on these well-conditioned J
            a == Success && @test getx(ll) ≈ getx(lq) rtol = 1e-7
        end
        @test tp2_bytes(ll, 0.3) == 0
        @test tp2_bytes(ll, 0.0) == 0
        # an LSMR system holds no QR factorization, and a QR system no LSMR workspace
        @test_throws ArgumentError tp2_solve!(ll, :qr, 0.3, nothing)
        @test_throws ArgumentError tp2_solve!(lq, :lsmr, 0.3, nothing)
    end

    @testset "Non-finite J or r" begin
        for (J, r) in (([1.0 NaN; 0.0 1.0; 1.0 1.0], ones(3)), ([1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, Inf, 0.0]))
            for ll in (SOO.DenseTallLinearSystem(copy(J); linsolver = :lsmr), SOO.SparseTallLinearSystem(sparse(J); linsolver = :lsmr))
                tp2_set!(ll, r)
                @test tp2_solve!(ll, :lsmr, 1.0, nothing) == NotPositiveDefinite
                @test all(isnan, getx(ll))
            end
        end
    end

    @testset "Meets the linear-system interface: $kind ($m, $n)" for kind in (:dense, :sparse), (m, n) in ((8, 5), (5, 5), (3, 5), (8, 1))
        make = kind === :dense ? (s -> SOO.DenseTallLinearSystem(m, n; linsolver = :lsmr)) : (s -> SOO.SparseTallLinearSystem(sparse(ones(m, n)); linsolver = :lsmr))
        @test isnothing(check_linear_system(make; n, linsolvers = (:lsmr,), rtol = 1e-6))
    end

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
