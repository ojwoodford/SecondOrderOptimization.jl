using SecondOrderOptimization, BlockLinearSystems, Krylov, Pardiso, StaticArrays, SparseArrays, LinearAlgebra, Random, Test
import DampedLinearSolve
const SOO = SecondOrderOptimization
const CEXT = Base.get_extension(SecondOrderOptimization, :BlockLinearSystemsExt)

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
const GD_R = zeros(8)
gd_f(x) = (mul!(GD_R, TP5_V, x); GD_R .-= TP5_Y; 0.5 * sum(abs2, GD_R))
# For a SchurBlockJacobian: where each entry of V and of the residual goes in J.data, found once by
# writing markers through the table accessors
function gd_layout(J::SchurBlockJacobian)
    fill!(J.data, 0.0)
    CEXT.foreach_table_block(J) do t, rows, cols, tcols
        CEXT.table_view(J, t)[1:length(rows), tcols] .= [1000.0 * i + j for i in rows, j in cols]
    end
    row0 = 0
    for t in 1:J.structure.ntables
        m = CEXT.data_rows(J.structure, t)
        CEXT.table_view(J, t)[1:m, end] .= -(row0 .+ (1:m))
        row0 += m
    end
    vidx = [(k, Int(J.data[k]) ÷ 1000, Int(J.data[k]) % 1000) for k in eachindex(J.data) if J.data[k] > 0]
    ridx = [(k, -Int(J.data[k])) for k in eachindex(J.data) if J.data[k] < 0]
    return vidx, ridx
end
function gd_resjac!(r, J, x, layout)
    c = gd_f(x)
    if J isa SchurBlockJacobian
        vidx, ridx = layout
        @inbounds for (k, i, j) in vidx
            J.data[k] = TP5_V[i, j]
        end
        @inbounds for (k, i) in ridx
            J.data[k] = GD_R[i]
        end
    elseif J isa SparseMatrixCSC
        @inbounds for j in axes(J, 2), q in nzrange(J, j)
            nonzeros(J)[q] = TP5_V[rowvals(J)[q], j]
        end
        r .= GD_R
    else
        J .= TP5_V
        r .= GD_R
    end
    return c
end
function gd_fdf!(g, H, x)
    c = gd_f(x)
    mul!(g, TP5_V', GD_R)
    mul!(H, TP5_V', TP5_V)
    return c
end

@testset "IT2: :gradientdescent on a JacobianObjective" begin
@testset "$kind$(static ? " (static x0)" : ""), :$s" for kind in keys(TP5_TABLE), static in (false, true), s in TP5_TABLE[kind]
    static && kind !== :integer && kind !== :matrix && continue
    static && s === :lsmr && continue   # :lsmr has no static form; the static x0 runs the dense system
    x0() = static ? zeros(SVector{4, Float64}) : zeros(4)
    proto = tp5_prototype(kind)
    layout = proto isa SchurBlockJacobian ? gd_layout(proto) : nothing
    obj = JacobianObjective(gd_f, (r, J, x) -> gd_resjac!(r, J, x, layout), proto)
    # No stall or cost-change stop, so both runs go on to the minimum (they end on the step size); the
    # number of iterations differs by rounding, so they are compared on cost (rules.md 31)
    strict = (; iterator = :gradientdescent, maxiters = 5000, stalltol = 0.0, reldcost = 0.0, absdcost = 0.0)
    opts = OptimizationOptions(; strict..., linsolver = s)
    x, res = optimize(x0(), obj, opts)
    xH, resH = optimize(x0(), HessianObjective(gd_f, gd_fdf!), OptimizationOptions(; strict...))
    fmin = gd_f(TP5_V \ TP5_Y)
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
