using SecondOrderOptimization, BlockLinearSystems, Krylov, Pardiso, StaticArrays, SparseArrays, LinearAlgebra, Random, Test
import DampedLinearSolve
include("fixtures/load.jl"); loadfixtures("tp5.jl")
const SOO = SecondOrderOptimization

# TP5: the supported-combinations table of the JacobianObjective docstring and the Jacobian tutorial. Every
# listed (prototype, linsolver, iterator) runs; every other one throws an ArgumentError before f or resjac!
# is called. The table, as data:
const TP5_ITERATORS = (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton, :gradientdescent)
const TP5_TABLE = Dict(
    :integer => (:auto, :qr, :lsmr),                # also with a static x0 (the static system)
    :matrix => (:auto, :qr, :lsmr),
    :sparse => (:auto, :qr, :lsmr),
    :schur_dense => (:auto, :qr, :cholesky, :ldl, :lsmr),
    :schur_sparse => (:auto, :qr, :cholesky, :ldl, :lsmr),
)
# Every solver symbol any extension accepts in OptimizationOptions, and every iterator
const TP5_ALL_SOLVERS = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :lsmr, :pardiso)
const TP5_ALL_ITERATORS = TP5_ITERATORS   # every iterator (IT2)

@testset "jacobiancombinations.jl" begin
    xref = TP5_V \ TP5_Y
    @testset "$kind$(static ? " (static x0)" : ""), :$s, :$it" for kind in keys(TP5_TABLE), static in (false, true), s in TP5_ALL_SOLVERS, it in TP5_ALL_ITERATORS
        static && kind !== :integer && kind !== :matrix && continue
        x0 = static ? zeros(SVector{4, Float64}) : zeros(4)
        obj = JacobianObjective(tp5_f, tp5_resjac!, tp5_prototype(kind))
        opts = OptimizationOptions(; iterator = it, linsolver = s, maxiters = 100)
        TP5_CALLED[] = false
        if s in TP5_TABLE[kind] && it in TP5_ITERATORS
            x, res = optimize(x0, obj, opts)
            if it === :gradientdescent
                # linear convergence: its accuracy is tested against the HessianObjective run below
                @test res.bestcost < res.startcost
            else
                @test isconverged(res)
                @test x ≈ xref rtol = s === :lsmr ? 1e-6 : 1e-8
            end
        else
            err = try optimize(x0, obj, opts); nothing catch e; e end
            @test err isa ArgumentError
            @test !TP5_CALLED[]
            @test_throws ArgumentError OptimizationWorkspace(x0, obj, opts)
            @test !TP5_CALLED[]
        end
    end
    # the static system is the one a static x0 gets with :auto and :qr; :lsmr has no static form
    @test SOO.objective_linearsystem(zeros(SVector{4}), JacobianObjective(tp5_f, tp5_resjac!, 8), :qr) isa DampedLinearSolve.StaticTallLinearSystem
    @test SOO.objective_linearsystem(zeros(SVector{4}), JacobianObjective(tp5_f, tp5_resjac!, 8), :lsmr) isa DampedLinearSolve.DenseTallLinearSystem
end

# IT2: gradient descent on a JacobianObjective. It uses only the gradient Jᵀr, f and update!, which every
# tall system provides, so it runs with every prototype kind and every solver its row lists, and follows
# the HessianObjective run of the same problem. resjac! and f here write preallocated storage, so the
# allocation count is the optimizer's own.

@testset "IT2: :gradientdescent on a JacobianObjective" begin
# No stall or cost-change stop, so both runs go on to the minimum (they end on the step size); the
# number of iterations differs by rounding, so they are compared on cost (rules.md 31)
strict = (; iterator = :gradientdescent, maxiters = 5000, stalltol = 0.0, reldcost = 0.0, absdcost = 0.0)
# The HessianObjective reference is the same run in every cell: solved once per kind of x0
reference_obj, reference_opts = HessianObjective(gd_f, gd_fdf!), OptimizationOptions(; strict...)
reference = Dict(false => optimize(zeros(4), reference_obj, reference_opts),
    true => optimize(zeros(SVector{4, Float64}), reference_obj, reference_opts))
fmin = gd_f(TP5_V \ TP5_Y)
@testset "$kind$(static ? " (static x0)" : ""), :$s" for kind in keys(TP5_TABLE), static in (false, true), s in TP5_TABLE[kind]
    static && kind !== :integer && kind !== :matrix && continue
    static && s === :lsmr && continue   # :lsmr has no static form; the static x0 runs the dense system
    x0() = static ? zeros(SVector{4, Float64}) : zeros(4)
    proto = tp5_prototype(kind)
    layout = proto isa SchurBlockJacobian ? gd_layout(proto) : nothing
    obj = JacobianObjective(gd_f, GDResJac(layout), proto)
    opts = OptimizationOptions(; strict..., linsolver = s)
    x, res = optimize(x0(), obj, opts)
    xH, resH = reference[static]
    @test isconverged(res) && isconverged(resH)
    @test abs(res.bestcost - resH.bestcost) <= 1e-13 * fmin
    @test abs(res.bestcost - fmin) <= 1e-13 * fmin
    @test x ≈ xH rtol = 1e-6
    # nothing allocates after warm-up
    ws = OptimizationWorkspace(x0(), obj, opts)
    gd_allocs(x, o, w) = @allocated optimize!(x, obj, o; workspace = w)
    static || gd_allocs(x0(), opts, ws)
    if static
        gd_sallocs(o, w) = @allocated optimize(zeros(SVector{4, Float64}), obj, o; workspace = w)
        gd_sallocs(opts, ws)
        GC.gc()
        @test gd_sallocs(opts, ws) == 0
    else
        GC.gc()
        @test gd_allocs(x0(), opts, ws) == 0
    end
end
end
