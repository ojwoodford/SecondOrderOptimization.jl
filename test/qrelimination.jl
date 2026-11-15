using SecondOrderOptimization, BlockLinearSystems, Krylov, SparseArrays, LinearAlgebra, Static, Random, Test
using Polyester   # BlockLinearSystems' kernels thread only with Polyester loaded
using SecondOrderOptimization: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system   # internal names
using SecondOrderOptimization: getA, getb, getx, solve!
const SOO = SecondOrderOptimization
const QREXT = Base.get_extension(SecondOrderOptimization, :BlockLinearSystemsExt)

# TP3: QR elimination on the Jacobian path (a SchurBlockJacobian prototype), against the normal path's Schur
# system and a dense least-squares solve of the same problem

const TP3_SOLVERS = Dict(Matrix{Float64} => (:auto, :qr, :cholesky, :ldl, :lsmr),
                         SparseMatrixCSC{Float64, Int} => (:auto, :cholesky, :ldl, :lsmr))

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
    H = SchurBlockMatrix(st, zeros(st.nr, st.nr), zeros(sum(abs2, st.e_dims)), zeros(clen))
    SOO.conformance_fill!(H, Jd' * Jd)
    ls = SOO.makelinearsystem(x, H, static(:cholesky))
    getb(ls) .= Jd' * rd
    return ls, Jd, rd
end

# A warm threaded run allocates nothing, except on Julia 1.10, where each threaded BlockLinearSystems kernel call
# may allocate up to 2 048 bytes per thread (rules.md 21, BlockLinearSystems R2). A run makes many such calls, so
# on 1.10 the bound is per run: 48 KB per thread, a starting value above the largest measured (2026-11-15, 4
# threads, nchunks = 4: 140 640 bytes for :lsmr, 35 160 per thread). 1.11 and later, and every serial run: zero.
const TP3_110_RUN_BYTES_PER_THREAD = 48 * 1024
tp3_threaded_run_ok(bytes) = bytes == 0 ||
    (VERSION < v"1.11" && Threads.nthreads() > 1 && bytes <= TP3_110_RUN_BYTES_PER_THREAD * Threads.nthreads())

tp3_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
tp3_run(x, obj, opts, ws) = (x .= 0.1; @allocated optimize!(x, obj, opts; workspace = ws))
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

@testset "qrelimination.jl" begin
    @testset "Steps equal the Schur system's: $RT, under-determined block = $under" for RT in keys(TP3_SOLVERS), under in (false, true)
        p = TP3Problem(tp3_structure(; under))
        x = 0.1 .* randn(Xoshiro(2), p.lay.n)
        lsS, Jd, rd = tp3_schur(p, x)
        sc = 0.5 .+ rand(Xoshiro(3), p.lay.n)
        for s in TP3_SOLVERS[RT]
            ls = tp3_system(p, RT, s, x)
            @test getb(ls) ≈ Jd' * rd rtol = 1e-12
            @test SOO.quadform(getA(ls), x) ≈ norm(Jd * x)^2 rtol = 1e-12
            @test SOO.max_diag(getA(ls)) ≈ maximum(sum(abs2, Jd; dims = 1)) rtol = 1e-12
            for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, sc))
                a = tp3_solve!(ls, s, lambda, scaling)
                if under && lambda == 0
                    # an under-determined block without damping is singular, as on the Schur path
                    @test a == NotPositiveDefinite
                    @test tp3_solve!(lsS, :cholesky, lambda, scaling) == NotPositiveDefinite
                    continue
                end
                @test a == Success
                @test tp3_solve!(lsS, :cholesky, lambda, scaling) == Success
                xref = (Jd' * Jd + lambda * Diagonal(something(scaling, ones(p.lay.n)))) \ (Jd' * rd)
                tol = s === :lsmr ? 1e-6 : 1e-10
                @test getx(ls) ≈ getx(lsS) rtol = tol
                @test getx(ls) ≈ xref rtol = tol
            end
        end
    end

    @testset "Meets the linear-system interface: $RT" for RT in keys(TP3_SOLVERS)
        jst = tp3_structure()
        n = jst.schur.nr + sum(jst.schur.e_dims)
        @test isnothing(check_linear_system(s -> QREXT.QRSchurLinearSystem(SchurBlockJacobian{Float64, RT}(jst), s); n, linsolvers = TP3_SOLVERS[RT], rtol = 1e-6))
    end

    @testset "Every Jacobian iterator: $RT, :$s, :$it, callbackchangesstate = $ccs" for RT in keys(TP3_SOLVERS), s in TP3_SOLVERS[RT], it in SOO.JACOBIAN_ITERATORS, ccs in (false, true)
        p = TP3Problem(tp3_structure())
        opts = OptimizationOptions(; iterator = it, linsolver = s, callbackchangesstate = ccs, maxiters = 200)
        _, rd = optimize(fill(0.1, p.lay.n), tp3_dense_objective(p), OptimizationOptions(; iterator = it, callbackchangesstate = ccs, maxiters = 200))
        obj = tp3_objective(p, RT)
        x, res = optimize(fill(0.1, p.lay.n), obj, opts)
        @test isconverged(res)
        @test res.bestcost ≈ rd.bestcost rtol = 1e-8
        # a warm run with a workspace: no allocation, except CHOLMOD's working memory in the sparse Cholesky
        # (the stated exception)
        if RT <: SparseMatrixCSC && s in (:auto, :cholesky)
            @test tp3_warm_bytes(p, RT, opts) > 0
        else
            @test tp3_warm_bytes(p, RT, opts) == 0
        end
    end

    @testset "Threaded elimination: the same steps; allocation" begin
        @test Base.get_extension(BlockLinearSystems, :PolyesterExt) !== nothing
        old = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
        try
            # One chunk per thread and no threshold, so that on more than one thread the kernels run threaded
            # (rules.md 30)
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = 0
            p = TP3Problem(tp3_structure(; nchunks = Threads.nthreads()))
            x = 0.1 .* randn(Xoshiro(2), p.lay.n)
            lsS, Jd, rd = tp3_schur(p, x)
            for RT in keys(TP3_SOLVERS), s in TP3_SOLVERS[RT]
                ls = tp3_system(p, RT, s, x)
                @test tp3_solve!(ls, s, 0.3, nothing) == Success == tp3_solve!(lsS, :cholesky, 0.3, nothing)
                @test getx(ls) ≈ getx(lsS) rtol = s === :lsmr ? 1e-6 : 1e-10
                RT <: SparseMatrixCSC && s in (:auto, :cholesky) && continue   # CHOLMOD's working memory
                for it in SOO.JACOBIAN_ITERATORS
                    @test tp3_threaded_run_ok(tp3_warm_bytes(p, RT, OptimizationOptions(; iterator = it, linsolver = s)))
                end
            end
        finally
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = old
        end
    end

    @testset "Invalid prototypes and (RType, linsolver) pairs throw up front" begin
        p = TP3Problem(tp3_structure())
        called = Ref(false)
        f(x) = (called[] = true; tp3_cost(p, x))
        rj!(r, J, x) = (called[] = true; tp3_tables!(p, J, x))
        x0 = fill(0.1, p.lay.n)
        for (RT, s) in ((SparseMatrixCSC{Float64, Int}, :qr), (Matrix{Float64}, :cg), (Matrix{Float64}, :minres),
                        (SparseMatrixCSC{Float64, Int}, :cg))
            obj = JacobianObjective(f, rj!, SchurBlockJacobian{Float64, RT}(p.jst))
            err = try optimize(x0, obj, OptimizationOptions(linsolver = s)); nothing catch e; e end
            @test err isa ArgumentError && occursin("SchurBlockJacobian", err.msg) && occursin(":$s", err.msg)
            @test_throws ArgumentError OptimizationWorkspace(x0, obj, OptimizationOptions(linsolver = s))
        end
        @test !called[]
        @test_throws ArgumentError JacobianObjective(f, rj!, SchurBlockJacobian{Float32, Matrix{Float32}}(p.jst))
        @test_throws DimensionMismatch optimize(zeros(3), JacobianObjective(f, rj!, SchurBlockJacobian{Float64, Matrix{Float64}}(p.jst)))
        @test !called[]
        # iterator = :gradientdescent stays rejected
        @test_throws ArgumentError optimize(x0, JacobianObjective(f, rj!, SchurBlockJacobian{Float64, Matrix{Float64}}(p.jst)), OptimizationOptions(iterator = :gradientdescent))
        # ...and its message, shared by every Jacobian-path error, names the solvers each SchurBlockJacobian takes
        msg = sprint(showerror, try optimize(x0, JacobianObjective(f, rj!, SchurBlockJacobian{Float64, Matrix{Float64}}(p.jst)), OptimizationOptions(iterator = :gradientdescent)); nothing catch e; e end)
        @test occursin("SchurBlockJacobian{Float64, Matrix{Float64}}, linsolver = :auto, :qr, :cholesky or :ldl", msg)
        @test occursin("SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, linsolver = :auto, :cholesky or :ldl", msg)
        @test !called[]
    end

    @testset "Workspaces: reused for the same structure, rebuilt for another" begin
        p = TP3Problem(tp3_structure())
        opts = OptimizationOptions()
        ws = OptimizationWorkspace(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts)
        data = ws.data[1]        # (the loop may swap the main and trial systems, so compare the data)
        x, res = optimize(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts; workspace = ws)
        @test ws.data[1] === data && isconverged(res)
        p2 = TP3Problem(tp3_structure(; under = true))
        wsany = OptimizationWorkspace()
        optimize(fill(0.1, p.lay.n), tp3_objective(p, Matrix{Float64}), opts; workspace = wsany)
        data1 = wsany.data[1]
        optimize(fill(0.1, p2.lay.n), tp3_objective(p2, Matrix{Float64}), opts; workspace = wsany)
        @test wsany.data[1] !== data1
    end
end
