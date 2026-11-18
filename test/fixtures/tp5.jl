using SecondOrderOptimization, BlockLinearSystems, SparseArrays, LinearAlgebra, Random

# One small linear least-squares problem, min ½‖V x − y‖², n = 4 (a retained block of 2 and two eliminated
# blocks of 1), m = 8. Described for BlockLinearSystems: rows 1–3 touch block 1 and eliminated block 2, rows 4–6
# block 1 and eliminated block 3, rows 7–8 block 1 only; its layout keeps that order.
const TP5_ROWBLOCKS = [[1, 2], [1, 3], [1]]
const TP5_ROWS = [1:3, 4:6, 7:8]
const TP5_COLS = [1:2, 3:3, 4:4]   # the columns of each column block
const TP5_JST = SchurJacobianStructure([2, 1, 1], TP5_ROWBLOCKS, length.(TP5_ROWS); eliminated = [false, true, true])
@assert blockorder(TP5_JST) == 1:3
function tp5_problem()
    rng = Xoshiro(5)
    V = zeros(8, 4)
    # the pattern of the structure: rows 1–3 touch block 1 and eliminated 1, rows 4–6 block 1 and eliminated
    # 2, rows 7–8 block 1 only; columns 1–2 retained, 3–4 eliminated (the layout order)
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
        for (k, rb) in enumerate(TP5_ROWBLOCKS), b in rb
            block(J, k, blockentry(J.structure, k, b), length(TP5_ROWS[k]), length(TP5_COLS[b])) .= TP5_V[TP5_ROWS[k], TP5_COLS[b]]
        end
        foreach(k -> residual_block(J, k) .= res[TP5_ROWS[k]], eachindex(TP5_ROWS))
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
# For a SchurBlockJacobian: for each row block, its rows and, for each block it touches, the handle and the columns,
# found once, so that the fill allocates nothing
gd_layout(J::SchurBlockJacobian) =
    [(TP5_ROWS[k], [(blockentry(J.structure, k, b), TP5_COLS[b]) for b in rb]) for (k, rb) in enumerate(TP5_ROWBLOCKS)]
function gd_resjac!(r, J, x, layout)
    c = gd_f(x)
    if J isa SchurBlockJacobian
        for (k, (rows, entries)) in enumerate(layout)
            for (entry, cols) in entries
                B = block(J, k, entry, length(rows), length(cols))
                @inbounds for (jj, j) in enumerate(cols), (ii, i) in enumerate(rows)
                    B[ii, jj] = TP5_V[i, j]
                end
            end
            R = residual_block(J, k, length(rows))
            @inbounds for (ii, i) in enumerate(rows)
                R[ii] = GD_R[i]
            end
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
