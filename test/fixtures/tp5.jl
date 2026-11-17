using SecondOrderOptimization, BlockLinearSystems, SparseArrays, LinearAlgebra, Random
const CEXT = Base.get_extension(BlockLinearSystems, :DampedLinearSolveExt)

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
# gd_resjac! with its layout, the function handed to the objective
struct GDResJac{L}
    layout::L
end
(c::GDResJac)(r, J, x) = gd_resjac!(r, J, x, c.layout)
function gd_fdf!(g, H, x)
    c = gd_f(x)
    mul!(g, TP5_V', GD_R)
    mul!(H, TP5_V', TP5_V)
    return c
end
