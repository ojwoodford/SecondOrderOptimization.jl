using SecondOrderOptimization, StaticArrays, LinearAlgebra, Static, Random, Test
import DampedLinearSolve
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
const SOO = SecondOrderOptimization

# TP4: the static tall system for small problems, against the dense tall system

function tp4_set!(ls, r)
    zero!(ls)
    getA(ls).r .= r
    DampedLinearSolve.jtr!(getb(ls), getA(ls).J, getA(ls).r)
    return ls
end
tp4_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))

# The exponential fit (20 residuals, 2 variables: m·n = 40, under the cap), without allocating
const TP4_T = collect(range(0, 1; length = 20))
const TP4_Y = @. 2.0 * exp(-1.5 * TP4_T)
function tp4_cost(x)
    total = 0.0
    for i in eachindex(TP4_T)
        total += (x[1] * exp(x[2] * TP4_T[i]) - TP4_Y[i])^2
    end
    return 0.5 * total
end
function tp4_resjac!(r, J, x)
    for i in eachindex(TP4_T)
        e = exp(x[2] * TP4_T[i])
        r[i] = x[1] * e - TP4_Y[i]
        J[i, 1] = e
        J[i, 2] = x[1] * TP4_T[i] * e
    end
    return tp4_cost(x)
end
tp4_run(x0, obj, opts, ws) = optimize(x0, obj, opts; workspace = ws)
tp4_bytes(x0, obj, opts, ws) = (tp4_run(x0, obj, opts, ws); @allocated tp4_run(x0, obj, opts, ws))

@testset "staticjacobiansystem.jl" begin
    @testset "The static choice is made exactly when the rule says" begin
        x2 = SVector(1.0, 0.0)
        @test DampedLinearSolve.MAX_STATIC_JACOBIAN == 48
        mk(x0, P, s = :auto) = SOO.objective_linearsystem(x0, JacobianObjective(tp4_cost, tp4_resjac!, P), static(s))
        @test mk(x2, 24) isa DampedLinearSolve.StaticJacobianLinearSystem{24, 2}           # m·n = 48
        @test mk(x2, 25) isa DampedLinearSolve.DenseJacobianLinearSystem                   # m·n = 50
        @test mk(x2, ones(24, 2)) isa DampedLinearSolve.StaticJacobianLinearSystem{24, 2}
        @test mk(x2, ones(25, 2)) isa DampedLinearSolve.DenseJacobianLinearSystem
        @test mk(x2, 20, :qr) isa DampedLinearSolve.StaticJacobianLinearSystem
        @test mk([1.0, 0.0], 20) isa DampedLinearSolve.DenseJacobianLinearSystem           # not a static number of variables
        @test mk(1.0, 48) isa DampedLinearSolve.StaticJacobianLinearSystem{48, 1}
        @test mk(1.0, 49) isa DampedLinearSolve.DenseJacobianLinearSystem
        @test mk(SVector{17}(zeros(17)), 2) isa DampedLinearSolve.DenseJacobianLinearSystem  # above MAX_STATIC_SYSTEM
        # the workspace's system follows the same rule
        @test OptimizationWorkspace(x2, JacobianObjective(tp4_cost, tp4_resjac!, ones(20, 2))).data[1].linsystem isa DampedLinearSolve.StaticJacobianLinearSystem
    end

    @testset "Every Jacobian iterator: :$it, callbackchangesstate = $ccs" for it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton), ccs in (false, true)
        opts = OptimizationOptions(; iterator = it, callbackchangesstate = ccs)
        _, rd = optimize([1.0, 0.0], JacobianObjective(tp4_cost, tp4_resjac!, 20), opts)
        for P in (20, zeros(20, 2))
            obj = JacobianObjective(tp4_cost, tp4_resjac!, P)
            x, rs = optimize(SVector(1.0, 0.0), obj, opts)
            @test x isa SVector{2, Float64}
            @test isconverged(rs)
            @test x ≈ [2.0, -1.5] rtol = 1e-8
            @test rs.niterations == rd.niterations
            @test rs.bestcost ≈ rd.bestcost atol = 1e-20
            ws = OptimizationWorkspace(SVector(1.0, 0.0), obj, opts)
            @test tp4_bytes(SVector(1.0, 0.0), obj, opts, ws) == 0
            @inferred optimize(SVector(1.0, 0.0), obj, opts; workspace = ws)
        end
    end

    @testset "A Matrix prototype's entries that resjac! never writes keep the prototype's values" begin
        # resjac! writes only the first column; the second is the prototype's constant column
        f(x) = 0.5 * ((x[1] + x[2] - 1)^2 + (x[1] - x[2])^2 + (2x[1] + x[2])^2)
        rj!(r, J, x) = (r[1] = x[1] + x[2] - 1; r[2] = x[1] - x[2]; r[3] = 2x[1] + x[2]; J[1, 1] = 1.0; J[2, 1] = 1.0; J[3, 1] = 2.0; f(x))
        P = [0.0 1.0; 0.0 -1.0; 0.0 1.0]
        xs, rs = optimize(SVector(0.0, 0.0), JacobianObjective(f, rj!, P))
        xd, rd = optimize([0.0, 0.0], JacobianObjective(f, rj!, copy(P)))
        @test isconverged(rs)
        @test xs ≈ xd rtol = 1e-12
        @test P == [0.0 1.0; 0.0 -1.0; 0.0 1.0]    # the static system works on its own copy
    end
end
