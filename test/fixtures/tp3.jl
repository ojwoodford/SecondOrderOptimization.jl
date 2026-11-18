using SecondOrderOptimization, BlockLinearSystems, SparseArrays, LinearAlgebra, Static, Random
using SecondOrderOptimization: dampen!, zero!, getA, getb, solve!   # internal names

# TP3: QR elimination on the Jacobian path (a SchurBlockJacobian prototype), against the normal path's Schur
# system and a dense least-squares solve of the same problem

const TP3_SOLVERS = Dict(Matrix{Float64} => (:auto, :qr, :cholesky, :ldl, :lsmr),
                         SparseMatrixCSC{Float64, Int} => (:auto, :qr, :cholesky, :ldl, :lsmr))

# Three retained blocks; six eliminated blocks, one above D_STATIC (9) and, with `under`, one with fewer rows
# than dimensions; a free group of rows coupled to every retained block. Described in this problem's own numbering:
# column blocks 1–3 retained, 4–9 eliminated; row block e (1–6) the rows of eliminated block e and its couplings, row
# block 7 the free rows.
const TP3_R_DIMS, TP3_E_DIMS = [6, 6, 6], [3, 3, 3, 9, 2, 1]
const TP3_COUPLINGS = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
function tp3_description(; under = false)
    e_rows = [8, 10, 6, 12, under ? 1 : 4, 3]
    rowblocks = [[[c; 3 + e] for (e, c) in enumerate(TP3_COUPLINGS)]; [[1, 2, 3]]]
    return (blocksizes = [TP3_R_DIMS; TP3_E_DIMS], rowblocks, rowblocksizes = [e_rows; 20],
            eliminated = [falses(3); trues(6)])
end
# `nchunks` and `min_work` set BlockLinearSystems' threading (`min_work = 0`: thread however small the problem)
function tp3_structure(; under = false, nchunks = 1, min_work = 4096)
    d = tp3_description(; under)
    return SchurJacobianStructure(d.blocksizes, d.rowblocks, d.rowblocksizes; eliminated = d.eliminated, nchunks, min_work)
end

# The dense problem's rows are the row blocks' in order, and its columns the layout's. For each row block: its dense
# rows, and for each column block it touches the handle to write through and the dense columns. (The row counts are
# read back through the public views, so that a structure of either variant gives its own.)
function tp3_layout(jst)
    probe = SchurBlockJacobian{Float64, Matrix{Float64}}(jst)
    d0 = tp3_description()
    d = merge(d0, (; rowblocksizes = [length(residual_block(probe, k)) for k in eachindex(d0.rowblocks)]))
    sizes = blocksizes(jst)
    off = computestartindices(sizes)
    lay_of = invperm(blockorder(jst))   # layout block of each column block of the description
    rows = UnitRange{Int}[]
    entries = Vector{Tuple{Int, UnitRange{Int}}}[]
    row0 = 0
    for (k, rb) in enumerate(d.rowblocks)
        m = d.rowblocksizes[k]
        push!(rows, row0 .+ (1:m))
        push!(entries, [(blockentry(jst, k, lay_of[b]), off[lay_of[b]] .+ (0:sizes[lay_of[b]] - 1)) for b in rb])
        row0 += m
    end
    return (; rows, entries, m = row0, n = sum(sizes), d, lay_of, sizes, off)
end

# The problem r(x) = V (x + 0.3 sin x) − y, with V random inside the pattern: its Jacobian keeps the pattern
struct TP3Problem{L <: NamedTuple}
    jst::SchurJacobianStructure
    lay::L
    V::Matrix{Float64}
    y::Vector{Float64}
    u::Vector{Float64}
    rv::Vector{Float64}
end
# V is drawn as before the front door, entry by entry, in the description's own block order (each row in turn; in
# a row, the retained blocks of its row block in ascending order, then its eliminated block), then its columns are
# put in the layout's order
function TP3Problem(jst; seed = 1)
    rng = Xoshiro(seed)
    lay = tp3_layout(jst)
    d = lay.d
    own_off = computestartindices(d.blocksizes)                      # the description's own column offsets
    own_cols(b) = own_off[b] .+ (0:d.blocksizes[b] - 1)
    V0 = zeros(lay.m, lay.n)
    for (k, rb) in enumerate(d.rowblocks)
        blocks = sort(rb; by = b -> (d.eliminated[b], b))
        for i in lay.rows[k], b in blocks, c in own_cols(b)
            V0[i, c] = randn(rng)
        end
    end
    perm = reduce(vcat, [own_cols(b) for b in blockorder(jst)])        # own column of each layout column
    return TP3Problem(jst, lay, V0[:, perm], randn(rng, lay.m), zeros(lay.n), zeros(lay.m))
end
function tp3_residual!(p::TP3Problem, x)
    @inbounds for j in eachindex(p.u)
        p.u[j] = x[j] + 0.3 * sin(x[j])
    end
    @inbounds for i in eachindex(p.rv)
        p.rv[i] = -p.y[i]
    end
    @inbounds for j in eachindex(p.u), i in eachindex(p.rv)
        p.rv[i] += p.V[i, j] * p.u[j]
    end
    return p.rv
end
function tp3_cost(p::TP3Problem, x)
    r = tp3_residual!(p, x)
    total = 0.0
    @inbounds for i in eachindex(r)
        total += r[i]^2
    end
    return 0.5 * total
end
# resjac! for the tables, and for a dense J
function tp3_tables!(p::TP3Problem, J::SchurBlockJacobian, x)
    r = tp3_residual!(p, x)
    lay = p.lay
    for k in eachindex(lay.rows)
        rows = lay.rows[k]
        for (entry, cols) in lay.entries[k]
            B = block(J, k, entry, length(rows), length(cols))
            @inbounds for (jj, c) in enumerate(cols), (ii, i) in enumerate(rows)
                B[ii, jj] = p.V[i, c] * (1 + 0.3 * cos(x[c]))
            end
        end
        R = residual_block(J, k, length(rows))
        @inbounds for (ii, i) in enumerate(rows)
            R[ii] = r[i]
        end
    end
    return tp3_cost(p, x)
end
function tp3_dense!(p::TP3Problem, rr, J::Matrix, x)
    rr .= tp3_residual!(p, x)
    @inbounds for j in axes(J, 2), i in axes(J, 1)
        J[i, j] = p.V[i, j] * (1 + 0.3 * cos(x[j]))
    end
    return tp3_cost(p, x)
end
# The problem functions handed to the objectives, as callable structs
struct TP3Cost{P <: TP3Problem}
    p::P
end
(c::TP3Cost)(x) = tp3_cost(c.p, x)
struct TP3Tables{P <: TP3Problem}
    p::P
end
(c::TP3Tables)(r, J, x) = tp3_tables!(c.p, J, x)
struct TP3Dense{P <: TP3Problem}
    p::P
end
(c::TP3Dense)(r, J, x) = tp3_dense!(c.p, r, J, x)
tp3_objective(p, ::Type{RT}) where RT = JacobianObjective(TP3Cost(p), TP3Tables(p), SchurBlockJacobian{Float64, RT}(p.jst))
tp3_dense_objective(p) = JacobianObjective(TP3Cost(p), TP3Dense(p), p.lay.m)

# A system linearized at x
function tp3_system(p, ::Type{RT}, s, x) where RT
    ls = SecondOrderOptimization.makejacobiansystem(SchurBlockJacobian{Float64, RT}(p.jst), p.lay.n, s)
    zero!(ls)
    SecondOrderOptimization.TallLinearize(TP3Tables(p))(getb(ls), getA(ls), x)
    return ls
end

# The normal path's dense Schur system for the same JᵀJ and Jᵀr
function tp3_schur(p, x)
    Jd = zeros(p.lay.m, p.lay.n); rd = zeros(p.lay.m)
    tp3_dense!(p, rd, Jd, x)
    H = SchurBlockHessian{Float64, Matrix{Float64}}(SchurStructure(p.jst))
    SecondOrderOptimization.conformance_fill!(H, Jd' * Jd)
    ls = SecondOrderOptimization.makehessiansystem(H, static(:cholesky))
    getb(ls) .= Jd' * rd
    return ls, Jd, rd
end

# A warm threaded run allocates nothing, except on Julia 1.10, where each threaded BlockLinearSystems kernel call
# may allocate a few KB per thread (rules.md 21; `threaded_allocs_ok` in blocklinearsystems_ext.jl). A run makes many such calls, so
# on 1.10 the bound is per run: 64 KB per thread, a starting value above the largest measured. Measured on 4 threads,
# nchunks = 4, for :lsmr: 35 160 bytes per thread on 2026-11-15; 48 808 at BlockLinearSystems 328b4f7 (2026-11-18), which
# had reached the earlier bound of 48 KB; 52 336 at 9f7d9cf, where the structure carries its threading (each threaded call
# allocates 96-384 bytes more on 1.10, within rules.md 21's 3 072 per thread). 1.11 and later, and every serial run: zero.
const TP3_110_RUN_BYTES_PER_THREAD = 64 * 1024
tp3_threaded_run_ok(bytes) = bytes == 0 ||
    (VERSION < v"1.11" && Threads.nthreads() > 1 && bytes <= TP3_110_RUN_BYTES_PER_THREAD * Threads.nthreads())

tp3_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
tp3_run(x, obj, opts, ws) = (x .= 0.1; @allocated optimize!(x, obj, opts; workspace = ws))
# Allocations of one warm :qr solve, behind a function barrier (as TP1's tp1_allocations)
# (a literal static(:qr): `static(s)` of a Symbol variable is a dynamic call, which allocates once)
tp6_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))
tp6_allocations(ls, lambda, scaling) = (tp6_solve!(ls, lambda, scaling); Base.@allocations tp6_solve!(ls, lambda, scaling))
# Bytes allocated by a warm run with a workspace, behind a function barrier (Julia 1.10 boxes variables
# captured at testset top level)
function tp3_warm_bytes(p, ::Type{RT}, opts) where RT
    obj = tp3_objective(p, RT)
    ws = OptimizationWorkspace(fill(0.1, p.lay.n), obj, opts)
    x = fill(0.1, p.lay.n)
    tp3_run(x, obj, opts, ws)
    GC.gc()   # @allocated counts every thread: no finalizer from earlier tests runs inside the measurement
    return tp3_run(x, obj, opts, ws)
end

# LP1: the TP3 problem with the retained columns of each block scaled by powers of ten from 1e-2 to 1e2
function tp3_scaled_problem(; seed = 1)
    p = TP3Problem(tp3_structure(); seed)
    for r in 1:nretained(p.jst)
        dr = p.lay.sizes[r]
        for k in 1:dr
            p.V[:, p.lay.off[r] + k - 1] .*= exp10(-2 + 4 * (k - 1) / (dr - 1))
        end
    end
    return p
end
