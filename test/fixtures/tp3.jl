const QREXT = Base.get_extension(BlockLinearSystems, :DampedLinearSolveExt)

# TP3: QR elimination on the Jacobian path (a SchurBlockJacobian prototype), against the normal path's Schur
# system and a dense least-squares solve of the same problem

const TP3_SOLVERS = Dict(Matrix{Float64} => (:auto, :qr, :cholesky, :ldl, :lsmr),
                         SparseMatrixCSC{Float64, Int} => (:auto, :qr, :cholesky, :ldl, :lsmr))

# Three retained blocks; six eliminated blocks, one above D_STATIC (9) and, with `under`, one with fewer rows
# than dimensions; a free group of rows coupled to every retained block
function tp3_structure(; under = false, nchunks = 1)
    r_dims, e_dims = [6, 6, 6], [3, 3, 3, 9, 2, 1]
    couplings = [[1, 2], [1, 2, 3], [2, 3], [1, 3], [1, 2], [3]]
    e_rows = [8, 10, 6, 12, under ? 1 : 4, 3]
    return SchurJacobianStructure(r_dims, e_dims, couplings, e_rows; free_rows = [20], free_retained = [[1, 2, 3]], nchunks)
end

# Where each Jacobian entry and residual of the dense problem lives in the tables: (table index, dense row, dense
# column) for J, (table index, dense row) for r
function tp3_layout(jst)
    st = jst.schur
    jidx, jrow, jcol, ridx, rrow = Int[], Int[], Int[], Int[], Int[]
    row0 = 0
    for t in 1:jst.ntables
        M, C, off = jst.table_nrows[t], jst.table_ncols[t], jst.table_offsets[t]
        m = M - (t <= st.neblocks ? jst.table_top[t] : 0)
        blocks = Tuple{Vector{Int}, Vector{Int}}[]
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            r = jst.entry_retained[u]
            dr = Int(st.r_dims[r])
            push!(blocks, (collect(st.boffsets_r[r] .+ (0:dr-1)), collect(jst.entry_col[u] .+ (0:dr-1))))
        end
        if t <= st.neblocks
            de = Int(st.e_dims[t])
            push!(blocks, (collect(st.nr .+ st.boffsets_e[t] .+ (0:de-1)), collect((C - de):(C - 1))))
        end
        for i in 1:m
            for (cols, tcols) in blocks, (c, tc) in zip(cols, tcols)
                push!(jidx, off + (tc - 1) * M + i - 1); push!(jrow, row0 + i); push!(jcol, c)
            end
            push!(ridx, off + (C - 1) * M + i - 1); push!(rrow, row0 + i)
        end
        row0 += m
    end
    return (; jidx, jrow, jcol, ridx, rrow, m = row0, n = st.nr + sum(st.e_dims))
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
function TP3Problem(jst; seed = 1)
    rng = Xoshiro(seed)
    lay = tp3_layout(jst)
    V = zeros(lay.m, lay.n)
    for k in eachindex(lay.jidx)
        V[lay.jrow[k], lay.jcol[k]] = randn(rng)
    end
    return TP3Problem(jst, lay, V, randn(rng, lay.m), zeros(lay.n), zeros(lay.m))
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
    lay, D = p.lay, J.data
    @inbounds for k in eachindex(lay.jidx)
        c = lay.jcol[k]
        D[lay.jidx[k]] = p.V[lay.jrow[k], c] * (1 + 0.3 * cos(x[c]))
    end
    @inbounds for k in eachindex(lay.ridx)
        D[lay.ridx[k]] = r[lay.rrow[k]]
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
tp3_objective(p, ::Type{RT}) where RT = JacobianObjective(x -> tp3_cost(p, x), (r, J, x) -> tp3_tables!(p, J, x), SchurBlockJacobian{Float64, RT}(p.jst))
tp3_dense_objective(p) = JacobianObjective(x -> tp3_cost(p, x), (r, J, x) -> tp3_dense!(p, r, J, x), p.lay.m)

# A system linearized at x
function tp3_system(p, ::Type{RT}, s, x) where RT
    ls = QREXT.QRSchurLinearSystem(SchurBlockJacobian{Float64, RT}(p.jst), s)
    zero!(ls)
    SOO.TallLinearize((r, J, x) -> tp3_tables!(p, J, x))(getb(ls), getA(ls), x)
    return ls
end

# The normal path's dense Schur system for the same JᵀJ and Jᵀr
function tp3_schur(p, x)
    st = p.jst.schur
    Jd = zeros(p.lay.m, p.lay.n); rd = zeros(p.lay.m)
    tp3_dense!(p, rd, Jd, x)
    clen = st.coffsets[end] + Int(st.r_dims[st.entry_retained[end]]) * Int(st.e_dims[end]) - 1
    H = SchurBlockHessian(st, zeros(st.nr, st.nr), zeros(sum(abs2, st.e_dims)), zeros(clen))
    SOO.conformance_fill!(H, Jd' * Jd)
    ls = SOO.makehessiansystem(H, static(:cholesky))
    getb(ls) .= Jd' * rd
    return ls, Jd, rd
end

# A warm threaded run allocates nothing, except on Julia 1.10, where each threaded BlockLinearSystems kernel call
# may allocate a few KB per thread (rules.md 21; `threaded_allocs_ok` in blocklinearsystems_ext.jl). A run makes many such calls, so
# on 1.10 the bound is per run: 48 KB per thread, a starting value above the largest measured (2026-11-15, 4
# threads, nchunks = 4: 140 640 bytes for :lsmr, 35 160 per thread). 1.11 and later, and every serial run: zero.
const TP3_110_RUN_BYTES_PER_THREAD = 48 * 1024
tp3_threaded_run_ok(bytes) = bytes == 0 ||
    (VERSION < v"1.11" && Threads.nthreads() > 1 && bytes <= TP3_110_RUN_BYTES_PER_THREAD * Threads.nthreads())

tp3_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
tp3_run(x, obj, opts, ws) = (x .= 0.1; @allocated optimize!(x, obj, opts; workspace = ws))
# Allocations of one warm :qr solve, behind a function barrier (as TP1's tp1_allocations)
# (a literal static(:qr): `static(s)` of a Symbol variable is a dynamic call, which allocates once)
tp6_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))
tp6_allocations(ls, lambda, scaling) = (tp6_solve!(ls, lambda, scaling); Base.@allocations tp6_solve!(ls, lambda, scaling))
tp6_fill_bytes(red, ws) = (reduced_jacobian!(red.Jt, red.rt, ws); @allocated reduced_jacobian!(red.Jt, red.rt, ws))
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
    st = p.jst.schur
    for r in 1:st.nrblocks
        dr = Int(st.r_dims[r])
        for k in 1:dr
            p.V[:, st.boffsets_r[r] + k - 1] .*= exp10(-2 + 4 * (k - 1) / (dr - 1))
        end
    end
    return p
end
