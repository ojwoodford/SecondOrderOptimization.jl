using SecondOrderOptimization, BlockLinearSystems, Krylov, Pardiso, StaticArrays, SparseArrays, LinearAlgebra, Random, Test
const SOO = SecondOrderOptimization
const CEXT = Base.get_extension(SecondOrderOptimization, :BlockLinearSystemsExt)

# TP5: the supported-combinations table of the JacobianObjective docstring and the Jacobian tutorial. Every
# listed (prototype, linsolver, iterator) runs; every other one throws an ArgumentError before f or resjac!
# is called. The table, as data:
const TP5_ITERATORS = (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
const TP5_TABLE = Dict(
    :integer => (:auto, :qr, :lsmr),                # also with a static x0 (the static system)
    :matrix => (:auto, :qr, :lsmr),
    :sparse => (:auto, :qr, :lsmr),
    :schur_dense => (:auto, :qr, :cholesky, :ldl, :lsmr),
    :schur_sparse => (:auto, :cholesky, :ldl, :lsmr),
)
# Every solver symbol any extension accepts in OptimizationOptions, and every iterator
const TP5_ALL_SOLVERS = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :lsmr, :pardiso)
const TP5_ALL_ITERATORS = (TP5_ITERATORS..., :gradientdescent)

# One small linear least-squares problem, min ½‖V x − y‖², n = 4 (a retained block of 2 and two eliminated
# blocks of 1), m = 8
const TP5_JST = SchurJacobianStructure([2], [1, 1], [[1], [1]], [3, 3]; free_rows = [2], free_retained = [[1]])
function tp5_problem()
    rng = Xoshiro(5)
    V = zeros(8, 4)
    # the pattern of the structure: rows 1–3 touch block 1 and eliminated 1, rows 4–6 block 1 and eliminated
    # 2, rows 7–8 block 1 only; columns 1–2 retained, 3–4 eliminated (the table order)
    V[1:3, [1, 2, 3]] .= randn(rng, 3, 3)
    V[4:6, [1, 2, 4]] .= randn(rng, 3, 3)
    V[7:8, [1, 2]] .= randn(rng, 2, 2)
    return V, randn(rng, 8)
end
const TP5_V, TP5_Y = tp5_problem()

# resjac! for each kind of prototype; `called` records whether f or resjac! ran
const TP5_CALLED = Ref(false)
tp5_f(x) = (TP5_CALLED[] = true; 0.5 * sum(abs2, TP5_V * collect(x) - TP5_Y))
function tp5_resjac!(r, J, x)
    TP5_CALLED[] = true
    res = TP5_V * collect(x) - TP5_Y
    if J isa SchurBlockJacobian
        row0 = 0
        CEXT.foreach_table_block(J) do t, rows, cols, tcols
            CEXT.table_view(J, t)[1:length(rows), tcols] .= TP5_V[rows, cols]
        end
        for t in 1:J.structure.ntables
            m = CEXT.data_rows(J.structure, t)
            CEXT.table_view(J, t)[1:m, end] .= res[row0 .+ (1:m)]
            row0 += m
        end
    elseif J isa SparseMatrixCSC
        for j in axes(J, 2), q in nzrange(J, j)
            nonzeros(J)[q] = TP5_V[rowvals(J)[q], j]
        end
        r .= res
    else
        J .= TP5_V
        r .= res
    end
    return 0.5 * sum(abs2, res)
end
function tp5_prototype(kind)
    kind === :integer && return 8
    kind === :matrix && return zeros(8, 4)
    kind === :sparse && return sparse(TP5_V .!= 0) .* 1.0
    kind === :schur_dense && return SchurBlockJacobian{Float64, Matrix{Float64}}(TP5_JST)
    return SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}(TP5_JST)
end

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
            @test isconverged(res)
            @test x ≈ xref rtol = s === :lsmr ? 1e-6 : 1e-8
        else
            err = try optimize(x0, obj, opts); nothing catch e; e end
            @test err isa ArgumentError
            @test !TP5_CALLED[]
            @test_throws ArgumentError OptimizationWorkspace(x0, obj, opts)
            @test !TP5_CALLED[]
        end
    end
    # the static system is the one a static x0 gets with :auto and :qr; :lsmr has no static form
    @test SOO.makelinearsystem(zeros(SVector{4}), JacobianObjective(tp5_f, tp5_resjac!, 8), :qr) isa SOO.StaticTallLinearSystem
    @test SOO.makelinearsystem(zeros(SVector{4}), JacobianObjective(tp5_f, tp5_resjac!, 8), :lsmr) isa SOO.DenseTallLinearSystem
end
