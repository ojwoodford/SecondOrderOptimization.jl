using Test, Random, LinearAlgebra, SparseArrays, StaticArrays, Static
import DampedLinearSolve
using SecondOrderOptimization, BlockLinearSystems, Polyester   # Polyester: BlockLinearSystems' threaded kernels
using SecondOrderOptimization: check_linear_system   # internal names

# The Schur-complement systems of a SchurBlockHessian (BlockLinearSystems' DampedLinearSolveExt) under the
# iterators and the workspace. Their solve-level tests are BlockLinearSystems'.
# Every Schur solve is compared against a dense Cholesky solve of the full damped system.

const SOOExt = Base.get_extension(BlockLinearSystems, :DampedLinearSolveExt)

# rules.md 21: a warm kernel call allocates nothing, except that on Julia 1.10 a threaded call may allocate up to
# 3 072 bytes per thread. BlockLinearSystems R2 set 2 048 (2026-11-15); from its R10 the mixed-dims reduction takes
# 8 304 bytes on 4 threads, 2 076 per thread, so this is raised by half (decided 2026-11-16)
threaded_allocs_ok(bytes) = bytes == 0 || (VERSION < v"1.11" && Threads.nthreads() > 1 && bytes <= 3072 * Threads.nthreads())

"""
    SchurTestProblem

A random, strictly diagonally dominant (hence positive definite) bipartite system with retained
variables first and eliminated variables second, in full dense form plus everything needed to build
`SchurBlockHessian` objects from it.
"""
struct SchurTestProblem
    H::Matrix{Float64}
    b::Vector{Float64}
    r_dims::Vector{Int}
    e_dims::Vector{Int}
    couplings::Vector{Vector{Int}}
    rr_pairs::Vector{Tuple{Int, Int}}   # retained-retained block couplings (r > r), beyond the diagonal
end

function SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs=Tuple{Int, Int}[])
    nr, ne = sum(r_dims), sum(e_dims)
    n = nr + ne
    roff = cumsum([1; r_dims[1:end-1]])
    eoff = nr .+ cumsum([1; e_dims[1:end-1]])
    H = zeros(n, n)
    function fillblock!(r0, nrow, c0, ncol)
        H[r0:r0+nrow-1, c0:c0+ncol-1] .= 2 .* rand(rng, nrow, ncol) .- 1
    end
    for (r, d) in enumerate(r_dims)
        fillblock!(roff[r], d, roff[r], d)
    end
    for (e, d) in enumerate(e_dims)
        fillblock!(eoff[e], d, eoff[e], d)
    end
    for (e, coupled) in enumerate(couplings), r in coupled
        fillblock!(roff[r], r_dims[r], eoff[e], e_dims[e])
    end
    for (r1, r2) in rr_pairs
        fillblock!(roff[r1], r_dims[r1], roff[r2], r_dims[r2])
    end
    # Symmetrize from the upper triangle (block diagonals are symmetrized element-wise)
    H = Matrix(Symmetric(H, :U))
    # Make strictly diagonally dominant, but keep the conditioning moderate
    for i in 1:n
        H[i, i] = sum(abs, view(H, i, :)) - abs(H[i, i]) + 0.5 + rand(rng)
    end
    return SchurTestProblem(H, randn(rng, n), r_dims, e_dims, couplings, rr_pairs)
end

nr(p::SchurTestProblem) = sum(p.r_dims)

function schur_structure(p::SchurTestProblem; nchunks=1)
    return SchurStructure(p.r_dims, p.e_dims, p.couplings; nchunks)
end

function fill_ce!(p::SchurTestProblem, structure::SchurStructure, E_data, C_data)
    nr_ = nr(p)
    for e in eachindex(p.e_dims)
        de = p.e_dims[e]
        eo = structure.eoffsets[e]
        eoff = nr_ + structure.boffsets_e[e]
        E_data[eo:eo+de^2-1] .= vec(p.H[eoff:eoff+de-1, eoff:eoff+de-1])
        for c in structure.eliminated_ptr[e]:structure.eliminated_ptr[e+1]-1
            r = structure.entry_retained[c]
            dr = p.r_dims[r]
            ro = structure.boffsets_r[r]
            co = structure.coffsets[c]
            # Stored as de x dr column-major: C_data[co + (r-1)*de + i-1] = H[c_r, p_i]
            C_data[co:co+dr*de-1] .= vec(p.H[ro:ro+dr-1, eoff:eoff+de-1]')
        end
    end
    return nothing
end

function dense_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    n_r = nr(p)
    R = p.H[1:n_r, 1:n_r]
    E_data = zeros(sum(abs2, p.e_dims))
    C_data = zeros(structure.coffsets[end] + Int(structure.r_dims[structure.entry_retained[end]]) * Int(structure.e_dims[end]) - 1)
    fill_ce!(p, structure, E_data, C_data)
    return SchurBlockHessian(structure, R, E_data, C_data)
end

function sparse_schur_matrix(p::SchurTestProblem; nchunks=1)
    structure = schur_structure(p; nchunks)
    pattern = build_schur_pattern(structure)
    coords = Tuple{Int, Int}[]
    cb = pattern.block_colptr
    for r2 in 1:structure.nrblocks, k in cb[r2]:cb[r2+1]-1
        r1 = pattern.block_rowval[k]       # r1 <= r2 in the upper-triangle pattern
        push!(coords, (r2, r1))            # store lower-triangle block (row = r2, col = r1)
    end
    append!(coords, p.rr_pairs)
    unique!(coords)
    R = SparseBlockLinearSystem{Float64}(coords, p.r_dims)
    roff = structure.boffsets_r
    for (r1, r2) in coords
        block(R, r1, r2) .= p.H[roff[r1]:roff[r1]+p.r_dims[r1]-1, roff[r2]:roff[r2]+p.r_dims[r2]-1]
    end
    E_data = zeros(sum(abs2, p.e_dims))
    C_data = zeros(structure.coffsets[end] + Int(structure.r_dims[structure.entry_retained[end]]) * Int(structure.e_dims[end]) - 1)
    fill_ce!(p, structure, E_data, C_data)
    return SchurBlockHessian(structure, R, E_data, C_data)
end

function make_schur_ls(kind::Symbol, p::SchurTestProblem, scaling=ones(length(p.b)); nchunks=1)
    n = length(p.b)
    if kind === :dense
        return SOOExt.DenseSchurLinearSystem(dense_schur_matrix(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    else
        return SOOExt.SparseSchurLinearSystem(sparse_schur_matrix(p; nchunks), copy(p.b), zeros(n), copy(scaling))
    end
end

"Reference solution: dense Cholesky of the full system with `lambda * diag(damping)` added to the diagonal."
function reference_solve(p::SchurTestProblem, lambda, damping)
    Hd = p.H + lambda * Diagonal(damping)
    return cholesky(Symmetric(Hd)) \ p.b
end

function standard_problem(seed=1; rr_pairs=Tuple{Int, Int}[])
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    # Retained blocks 1-2 and 3-4 are coupled to eliminated blocks together; 2-3 are linked through a few eliminated blocks
    couplings = [
        [1, 2], [1, 2], [1, 2, 3], [2, 3], [3, 4], [3, 4], [1, 4],
        [1, 2], [2, 3], [3, 4], [1, 2, 3, 4], [4],
        [1], [2], [3], [4], [1, 2], [3, 4], [2, 3], [1, 4],
    ]
    return SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs)
end

# Eliminated blocks of every size class: tiny closed forms (1-3), the generic static tier (4..D_STATIC)
# and the dynamic tier above D_STATIC
function mixed_dims_problem(seed=11)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 3, 9, 6]
    e_dims = [12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    coupled = 1:4
    couplings = [sort(unique([rand(rng, coupled) for _ in 1:3])) for _ in e_dims]
    return SchurTestProblem(rng, r_dims, e_dims, couplings)
end


# Copy the numerical content of `src` (a SchurBlockHessian) into `dst`
function copy_content!(dst::SchurBlockHessian, src::SchurBlockHessian)
    dst.R isa Matrix ? copyto!(dst.R, src.R) : copyto!(dst.R.data, src.R.data)
    copyto!(dst.E_data, src.E_data)
    copyto!(dst.C_data, src.C_data)
    return dst
end


# Two groups of retained blocks ({1, 2} and {3, 4}) that share no eliminated block with each other, coupled only by
# direct retained-retained blocks (odometry / loop-closure style), which have no eliminated variable.
function coupled_groups_problem(seed=3)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 2, 2, 1, 1]
    couplings = [[1, 2], [1, 2], [3, 4], [3, 4], [1], [4], [2, 1], [3, 4]]
    return SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs=[(3, 1), (4, 2), (2, 1)])
end





# Measured through a function barrier: at testset scope Julia 1.10 boxes the Float64 result
schur_quadform_allocations(H, v) = @allocated SecondOrderOptimization.quadform(H, v)


@testset "BlockLinearSystemsExt: the type of H selects elimination; linsolver names the reduced solver (A2)" begin
    p = standard_problem()
    Hd, Hs = dense_schur_matrix(p), sparse_schur_matrix(p)
    mk(H, s) = SecondOrderOptimization.makehessiansystem(H, s)
    # The enum spellings of a solver reach the systems as the options hold them, as a StaticSymbol
    opt(s) = OptimizationOptions(linsolver = s).linsolver
    # The storage of R picks the dense or sparse system, whatever the reduced-system solver
    for s in (:auto, :cholesky, :ldl, static(:auto), static(:ldl), opt(SecondOrderOptimization.auto_solver),
              opt(SecondOrderOptimization.cholesky_solver), opt(SecondOrderOptimization.ldl_solver))
        @test mk(Hd, s) isa SOOExt.DenseSchurLinearSystem
        @test mk(Hs, s) isa SOOExt.SparseSchurLinearSystem
    end
    # Solvers not implemented for the reduced system are rejected when it is built (not treated as :auto)
    sq(x) = sum(abs2, x)
    sq_fdf!(g, H, x) = sq(x)
    for H in (Hd, Hs)
        for s in (:qr, :pardiso, :cg, :minres, static(:qr), opt(SecondOrderOptimization.qr_solver))
            err = try mk(H, s); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("reduced system", sprint(showerror, err))
        end
        opts_qr = OptimizationOptions(linsolver=:qr)
        @test_throws ArgumentError optimize(zeros(length(p.b)), HessianObjective(sq, sq_fdf!; H), opts_qr)
        @test_throws ArgumentError OptimizationWorkspace(zeros(length(p.b)), HessianObjective(sq, sq_fdf!; H), opts_qr)
    end
    # ...and by solve! on a system built directly
    for kind in (:dense, :sparse)
        @test_throws ArgumentError SecondOrderOptimization.solve!(make_schur_ls(kind, p), :qr)
    end
    # The old symbols are unknown even with the extension loaded; other Hessians need no check
    for s in (:schur, :dense_schur, :sparse_schur)
        @test !SecondOrderOptimization.validatelinsolver(s)
        @test_throws ArgumentError OptimizationOptions(linsolver=s)
    end
    @test_throws MethodError OptimizationOptions(reduction=:eliminate)
    for H_other in (nothing, zeros(3, 3), sparse(1.0I, 3, 3))
        x, res = optimize(ones(3), HessianObjective(sq, (g, H, x) -> (g .= 2 .* x; H[diagind(H)] .= 2; sq(x)); H = H_other))
        @test isconverged(res) && norm(x) < 1e-8
    end

    # solve! accepts Symbols and static symbols (the enum spellings as the options hold them) without ambiguity
    n = length(p.b)
    x_ref = reference_solve(p, 0.3, ones(n))
    for kind in (:dense, :sparse), s in (:auto, :cholesky, :ldl, static(:auto), static(:ldl),
                                        opt(SecondOrderOptimization.auto_solver), opt(SecondOrderOptimization.cholesky_solver), opt(SecondOrderOptimization.ldl_solver))
        ls = make_schur_ls(kind, p)
        SecondOrderOptimization.dampen!(ls, 0.3)
        @test SecondOrderOptimization.solve!(ls, s) == SecondOrderOptimization.Success
        @test norm(ls.x - x_ref) <= 1e-10 * norm(x_ref)
    end
end


function to_float32(H::SchurBlockHessian)
    R = if H.R isa Matrix
        Float32.(H.R)
    else
        S = H.R
        SparseBlockLinearSystem{Float32}(Float32.(S.data), S.indices, S.indicestransposed, S.rowblocksizes, S.columnblocksizes, S.m, S.n)
    end
    return SchurBlockHessian(H.structure, R, Float32.(H.E_data), Float32.(H.C_data))
end

@testset "A SchurBlockHessian must be Float64; Float32 is rejected up front" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse), x0 in (zeros(Float32, n), zeros(n))
        H = to_float32(kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p))
        calls = Ref(0)
        obj = HessianObjective(x -> (calls[] += 1; 0.0), (g, Hs, x) -> (calls[] += 1; 0.0); H)
        for s in (:auto, :ldl)
            @test_throws ArgumentError optimize(x0, obj, OptimizationOptions(; linsolver = s))
            @test_throws ArgumentError OptimizationWorkspace(x0, obj, OptimizationOptions(; linsolver = s))
        end
        @test calls[] == 0
        err = try optimize(x0, obj, OptimizationOptions()) catch e; e end
        @test occursin("SchurBlockHessian{Float32", sprint(showerror, err))
    end
    # The sparse system's factor type is concrete
    ls = SOOExt.SparseSchurLinearSystem(sparse_schur_matrix(p), copy(p.b), zeros(n))
    @test isconcretetype(typeof(ls.S_factor))
    @test isconcretetype(fieldtype(typeof(ls), :S_factor))

    # Singularity tolerance scales with the precision: a block that is singular to Float32 precision is rejected
    H2 = SMatrix{2, 2, Float32, 4}(1, 1, 1, 1 + 1f-6)
    @test !BlockLinearSystems.inv_sym_block(H2, 0f0, SVector{2, Float32}(1, 1)).success
    H2d = SMatrix{2, 2, Float64, 4}(1, 1, 1, 1 + 1e-6)
    @test BlockLinearSystems.inv_sym_block(H2d, 0.0, SVector{2, Float64}(1, 1)).success
end





# A gauge-deficient problem: every coupling constrains only the *difference* x_retained[1:3] - x_eliminated, so
# translating all retained and eliminated blocks together changes nothing and H is singular along that direction.
# Dimensions 4:6 of the retained blocks only have a tiny prior, which makes H ill conditioned.
function gauge_problem(seed=21; prior=1e-8)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 6, 6, 6]
    e_dims = [3, 3, 3, 3, 3, 3]
    couplings = [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [1, 4], [1, 2, 3, 4]]
    nr_, np_ = sum(r_dims), sum(e_dims)
    H = zeros(nr_ + np_, nr_ + np_)
    for (e, coupled) in enumerate(couplings), r in coupled
        w = 0.5 + rand(rng)
        ci = 6 * (r - 1) .+ (1:3)
        li = nr_ + 3 * (e - 1) .+ (1:3)
        H[ci, ci] .+= w * I(3); H[li, li] .+= w * I(3)
        H[ci, li] .-= w * I(3); H[li, ci] .-= w * I(3)
    end
    for r in 1:4
        ci = 6 * (r - 1) .+ (4:6)
        H[ci, ci] .+= prior * I(3)
    end
    return SchurTestProblem(H, randn(rng, nr_ + np_), r_dims, e_dims, couplings, Tuple{Int, Int}[])
end





# Add d to the diagonal of the full matrix a SchurBlockHessian represents
function add_schur_diag!(H::SchurBlockHessian, d)
    st = H.structure
    for r in 1:st.nrblocks, i in 1:Int(st.r_dims[r])
        k = st.boffsets_r[r] + i - 1
        if H.R isa Matrix
            H.R[k, k] += d[k]
        else
            block(H.R, r, r)[i, i] += d[k]
        end
    end
    for e in 1:st.neblocks
        de = Int(st.e_dims[e])
        for i in 1:de
            H.E_data[st.eoffsets[e] + (i - 1) * (de + 1)] += d[st.nr + st.boffsets_e[e] + i - 1]
        end
    end
    return H
end

# A quadratic plus a quartic term, so the Hessian H + diag(3x^2) changes every iteration. `kind` is
# :dense or :sparse (a SchurBlockHessian with that storage of R) or :full (a plain dense Hessian).
function quartic_run(p, kind, linsolver, iterator)
    cost(x) = 0.5 * dot(x, p.H * x) - dot(p.b, x) + 0.25 * sum(xi -> xi^4, x)
    if kind === :full
        Hproto = zeros(length(p.b), length(p.b))
        fdf! = (g, H, x) -> (g .= p.H * x .- p.b .+ x .^ 3; H .= p.H; H[diagind(H)] .+= 3 .* x .^ 2; cost(x))
    else
        Href = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
        Hproto = kind === :dense ? dense_schur_matrix(p) : sparse_schur_matrix(p)
        fdf! = (g, H, x) -> (g .= p.H * x .- p.b .+ x .^ 3; copy_content!(H, Href); add_schur_diag!(H, 3 .* x .^ 2); cost(x))
    end
    traj = CostTrajectory()
    x, res = optimize(fill(3.0, length(p.b)), HessianObjective(cost, fdf!; H = Hproto),
                      OptimizationOptions(; iterator, linsolver, gtol=1e-10); callback=storecostscallback(traj))
    return (; x, res, costs = copy(traj.costs), steps = deepcopy(traj.steps))
end

@testset "Full optimization with a SchurBlockHessian: :auto, :cholesky and :ldl (A2)" begin
    p = standard_problem()
    for kind in (:dense, :sparse), it in (:levenbergmarquardt, :levenbergmarquardt_dtd, :newton, :dogleg)
        @testset "$kind, $it" begin
            ra = quartic_run(p, kind, :auto, it)
            rc = quartic_run(p, kind, :cholesky, it)
            rl = quartic_run(p, kind, :ldl, it)
            rf = quartic_run(p, :full, :cholesky, it)
            @test isconverged(ra.res) && isconverged(rl.res)
            @test ra.res.niterations > 3
            # :auto and :cholesky are the same solver for the reduced system: identical iterates
            @test ra.x == rc.x && ra.costs == rc.costs && ra.steps == rc.steps
            # the same optimum as :ldl and as the same problem with a plain dense Hessian
            @test norm(rl.x - ra.x) <= 1e-8 * norm(ra.x)
            @test norm(rf.x - ra.x) <= 1e-8 * norm(ra.x)
            @test ra.res.niterations == rf.res.niterations
        end
    end
end


# AS1: a workspace for a SchurBlockHessian. The quartic of `quartic_run`, evaluated without allocating
function as1_objective(p, Href)
    d = similar(p.b)
    cost(x) = 0.5 * dot(x, p.H, x) - dot(p.b, x) + 0.25 * sum(xi -> xi^4, x)
    function fdf!(g, H, x)
        mul!(g, p.H, x)
        g .+= x .^ 3 .- p.b
        copy_content!(H, Href)
        d .= 3 .* x .^ 2
        add_schur_diag!(H, d)
        return cost(x)
    end
    return cost, fdf!
end
as1_allocs(x, obj, o, w) = (x .= 3.0; @allocated optimize!(x, obj, o; workspace = w))
as1_valid_allocs(w, x, o, H) = @allocated SecondOrderOptimization.workspacevalid(w, x, o, H)

@testset "OptimizationWorkspace with a SchurBlockHessian (AS1)" begin
    p = standard_problem()
    n = length(p.b)
    opts = OptimizationOptions(; iterator = :levenbergmarquardt, gtol = 1e-10)
    for kind in (:dense, :sparse)
        schur(q) = kind === :dense ? dense_schur_matrix(q) : sparse_schur_matrix(q)
        H = schur(p)
        cost, fdf! = as1_objective(p, schur(p))
        obj = HessianObjective(cost, fdf!; H)
        x_ref, res_ref = optimize(fill(3.0, n), obj, opts)
        @test isconverged(res_ref)
        for typed in (true, false)
            @testset "$kind, $(typed ? "typed" : "untyped") workspace" begin
                ws = typed ? OptimizationWorkspace(fill(3.0, n), obj, opts) : OptimizationWorkspace()
                # The first solve with a workspace gives the result of a solve without one
                x1 = fill(3.0, n)
                res1 = optimize!(x1, obj, opts; workspace = ws)
                @test x1 == x_ref && res1.niterations == res_ref.niterations
                @test !isempty(ws)
                built = ws.data
                # A second solve with the same H reuses it. With a typed workspace, matching the key allocates
                # nothing, and the whole dense run allocates nothing, as for a dense Hessian (the sparse run's
                # CHOLMOD allocates, rules.md 21; an untyped workspace dispatches dynamically)
                x2 = fill(3.0, n)
                optimize!(x2, obj, opts; workspace = ws)
                @test ws.data === built && x2 == x_ref
                if typed
                    as1_valid_allocs(ws, x2, opts, H)
                    @test as1_valid_allocs(ws, x2, opts, H) == 0
                end
                if typed && kind === :dense
                    as1_allocs(x2, obj, opts, ws)
                    @test as1_allocs(x2, obj, opts, ws) == 0
                end
                # An H with another structure object rebuilds it, and still gives the right result
                H_other = schur(p)
                @test H_other.structure !== H.structure
                obj_other = HessianObjective(cost, fdf!; H = H_other)
                @test !SecondOrderOptimization.workspacevalid(ws, x2, opts, H_other)
                x3 = fill(3.0, n)
                optimize!(x3, obj_other, opts; workspace = ws)
                @test ws.data !== built
                @test x3 == x_ref
            end
        end
        # A workspace for a Schur H is not accepted for a dense H of the same size, nor the reverse
        Hd = zeros(n, n)
        ws_s = OptimizationWorkspace(fill(3.0, n), obj, opts)
        ws_d = OptimizationWorkspace(fill(3.0, n), HessianObjective(cost, (g, Hm, x) -> 0.0; H = Hd), opts)
        @test !SecondOrderOptimization.workspacevalid(ws_s, fill(3.0, n), opts, Hd)
        @test !SecondOrderOptimization.workspacevalid(ws_d, fill(3.0, n), opts, H)
        @test SecondOrderOptimization.workspacevalid(ws_s, fill(3.0, n), opts, H)
        # An x0 of the wrong size is a DimensionMismatch up front, before the objective is evaluated
        calls = Ref(0)
        obj_count = HessianObjective(cost, (g, Hm, x) -> (calls[] += 1; 0.0); H)
        @test_throws DimensionMismatch optimize!(fill(3.0, n + 1), obj_count, opts)
        @test_throws DimensionMismatch OptimizationWorkspace(fill(3.0, n - 1), obj_count, opts)
        @test calls[] == 0
        err = try optimize!(fill(3.0, n + 1), obj_count, opts) catch e; e end
        @test occursin("$(n + 1) degrees of freedom", sprint(showerror, err))
    end
end

@testset "updatescaling! guards a non-finite diagonal entry, as the other systems do (AS2)" begin
    p = standard_problem()
    n = length(p.b)
    for kind in (:dense, :sparse), bad in (NaN, Inf, -Inf), k in (1, nr(p) + 1)   # one retained, one eliminated entry
        @testset "$kind, $bad at $(k == 1 ? "retained" : "eliminated") entry" begin
            ls = make_schur_ls(kind, p)
            d = zeros(n)
            d[k] = bad
            add_schur_diag!(ls.H, d)
            M = SecondOrderOptimization.conformance_matrix(ls.H)
            @test !isfinite(M[k, k])
            ref = DampedLinearSolve.DenseLinearSystem{Float64}(M, copy(p.b), zeros(n))
            for mk in (() -> SecondOrderOptimization.MarquardtData(0.0, zeros(n)),
                       () -> SecondOrderOptimization.MarquardtDataTrial(0.0, zeros(n), nothing, NaN, false))
                lmd, lmd_ref = mk(), mk()
                SecondOrderOptimization.updatescaling!(lmd, ls)
                SecondOrderOptimization.updatescaling!(lmd_ref, ref)
                @test all(isfinite, lmd.scaling)
                @test lmd.scaling == lmd_ref.scaling
                @test lmd.lambda == lmd_ref.lambda
            end
        end
    end
end

# AS3: the dense Schur system's Bunch-Kaufman solve, for :ldl and as :auto's fallback. The reference is
# the solve as it was written before: Cholesky (for :auto), then LinearAlgebra's bunchkaufman
function as3_reference(ls, solver)
    SOOExt.build_reduced!(ls) || return (SecondOrderOptimization.NotPositiveDefinite, nothing)
    S, g = copy(ls.S), copy(ls.g_S)
    if solver === :auto
        F = cholesky(Symmetric(copy(S), :U); check = false)
        issuccess(F) && return (SecondOrderOptimization.Success, F \ g)
    end
    bk = bunchkaufman(Symmetric(S, :U); check = false)
    issuccess(bk) || return (SecondOrderOptimization.NotPositiveDefinite, nothing)
    xr = bk \ g
    return (DampedLinearSolve.is_indefinite(bk) ? SecondOrderOptimization.Indefinite : SecondOrderOptimization.Success, xr)
end
as3_solve_allocs(ls, s) = @allocated SecondOrderOptimization.solve!(ls, s)
as3_allocs(x, obj, o, w) = (x .= 0.1; @allocated optimize!(x, obj, o; workspace = w))

@testset "The dense Schur system's Bunch-Kaufman solve allocates nothing (AS3)" begin
    p = standard_problem()
    n = length(p.b)
    q = standard_problem()
    q.H[1, 1] = -50.0                        # S indefinite, so Cholesky fails (B5)
    zeroS(ls) = (ls.H.R .= 0; ls.H.C_data .= 0; ls)   # S = 0: Bunch-Kaufman finds an exactly zero pivot
    for (name, mk, expected) in (("positive definite", () -> make_schur_ls(:dense, p), Dict(:auto => :Success, :ldl => :Success)),
                                 ("indefinite", () -> make_schur_ls(:dense, q), Dict(:auto => :Indefinite, :ldl => :Indefinite)),
                                 ("singular", () -> zeroS(make_schur_ls(:dense, p)), Dict(:auto => :NotPositiveDefinite, :ldl => :NotPositiveDefinite)))
        for s in (:auto, :ldl)
            @testset "$name, $s" begin
                ls = mk()
                SecondOrderOptimization.dampen!(ls, 0.0)
                st_ref, xr_ref = as3_reference(ls, s)
                @test Symbol(st_ref) === expected[s]
                st = SecondOrderOptimization.solve!(ls, static(s))
                @test st == st_ref
                isnothing(xr_ref) || @test norm(ls.x_r - xr_ref) <= 1e-12 * norm(xr_ref)
                # Warm: a solve allocates nothing (the system is serial here: one chunk)
                as3_solve_allocs(ls, static(s))
                @test as3_solve_allocs(ls, static(s)) == 0
            end
        end
    end
    # A whole Newton or dogleg run on an indefinite Schur problem allocates nothing (typed workspace)
    for it in (:newton, :dogleg), s in (:auto, :ldl)
        @testset "$it, $s run" begin
            cost, fdf! = as1_objective(q, dense_schur_matrix(q))
            obj = HessianObjective(cost, fdf!; H = dense_schur_matrix(q))
            opts = OptimizationOptions(; iterator = it, linsolver = s, maxiters = 30)
            ws = OptimizationWorkspace(fill(0.1, n), obj, opts)
            x = fill(0.1, n)
            as3_allocs(x, obj, opts, ws)
            @test as3_allocs(x, obj, opts, ws) == 0
        end
    end
end

