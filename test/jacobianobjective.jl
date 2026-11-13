using SecondOrderOptimization, StaticArrays, SparseArrays, LinearAlgebra, Static, Test
const SOO = SecondOrderOptimization

# J1, J4: JacobianObjective, its validation, and the optimize / optimize! entry points

# A two-residual Rosenbrock, counting the calls to f and resjac!
const JCALLS = Ref(0)
jrosen_cost(x) = (JCALLS[] += 1; 0.5 * ((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2))
function jrosen_resjac!(r, J, x)
    JCALLS[] += 1
    r[1] = 1 - x[1]
    r[2] = 10 * (x[2] - x[1]^2)
    J[1, 1] = -1.0
    J[2, 1] = -20 * x[1]
    J[2, 2] = 10.0
    return 0.5 * (r[1]^2 + r[2]^2)
end

@testset "jacobianobjective.jl" begin
    @testset "JacobianObjective" begin
        obj = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        @test obj.f === jrosen_cost && obj.resjac! === jrosen_resjac! && obj.J === 2
        J = zeros(2, 2)
        @test JacobianObjective(jrosen_cost, jrosen_resjac!, J).J === J
        @test SOO.nresiduals(obj) == 2 && SOO.nresiduals(JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(5, 2))) == 5
    end

    @testset "Unsupported Jacobian prototypes are rejected when the objective is built" begin
        for (J, needle) in ((spzeros(2, 2), "sparse"), (zeros(Float32, 2, 2), "Float64"), (zeros(Int, 2, 2), "Float64"),
                            (zeros(MMatrix{2, 2}), "not supported"), (view(zeros(3, 2), 1:2, :), "not supported"),
                            (Diagonal(ones(2)), "not supported"), (nothing, "not supported"), (2.0, "not supported"),
                            (true, "not supported"), (0, "at least 1"), (-3, "at least 1"), (zeros(0, 2), "no rows"))
            err = try JacobianObjective(jrosen_cost, jrosen_resjac!, J); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin(needle, err.msg)
            @test occursin("Matrix{Float64}", err.msg) && occursin(":levenbergmarquardt", err.msg)
        end
    end

    @testset "Validation at entry: iterator, linsolver and the prototype's columns" begin
        obj = JacobianObjective(jrosen_cost, jrosen_resjac!, 2)
        for it in (:levenbergmarquardt_dtd, :newton, :dogleg, :gradientdescent)
            err = try SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(iterator=it)); nothing catch e; e end
            @test err isa ArgumentError && occursin(":$it", err.msg) && occursin("iterator = :levenbergmarquardt", err.msg)
        end
        for s in (:cholesky, :ldl)
            err = try SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(linsolver=s)); nothing catch e; e end
            @test err isa ArgumentError && occursin(":$s", err.msg) && occursin(":auto or :qr", err.msg)
        end
        for s in (:auto, :qr, auto_solver, qr_solver)
            @test SOO.checkobjective([0.0, 0.0], obj, OptimizationOptions(linsolver=s)) === nothing
        end
        err = try SOO.checkobjective([0.0, 0.0, 0.0], JacobianObjective(jrosen_cost, jrosen_resjac!, zeros(2, 2)), OptimizationOptions()); nothing catch e; e end
        @test err isa DimensionMismatch && occursin("3 columns", err.msg)
        # the integer form has no columns to check, and any m >= 1 is accepted
        @test SOO.checkobjective([0.0, 0.0, 0.0], JacobianObjective(jrosen_cost, jrosen_resjac!, 1), OptimizationOptions()) === nothing
        # no user function was called by any of this
        @test JCALLS[] == 0
    end
end
