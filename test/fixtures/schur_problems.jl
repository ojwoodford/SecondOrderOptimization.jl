using Random, LinearAlgebra, SparseArrays, SecondOrderOptimization, BlockLinearSystems

# The test problems are built through BlockLinearSystems' public API only: each is described (block sizes, row blocks,
# eliminated blocks), laid out by `SchurStructure`, allocated by `SchurBlockHessian{T, RType}(structure)` and filled
# with `addblock!`; the systems are DampedLinearSolve's, from `makehessiansystem`.

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

# The problem as a description, in its own block numbering (retained blocks, then eliminated ones): one row block for
# each eliminated block alone, one for each coupling and one for each retained-retained pair
function schur_description(p::SchurTestProblem)
    nrb = length(p.r_dims)
    rowblocks = Vector{Int}[]
    for (e, coupled) in enumerate(p.couplings)
        push!(rowblocks, [nrb + e])
        foreach(r -> push!(rowblocks, [r, nrb + e]), coupled)
    end
    foreach(((r1, r2),) -> push!(rowblocks, [r1, r2]), p.rr_pairs)
    return (blocksizes = [p.r_dims; p.e_dims], rowblocks, eliminated = [falses(nrb); trues(length(p.e_dims))])
end

function schur_structure(p::SchurTestProblem; nchunks=1)
    d = schur_description(p)
    return SchurStructure(d.blocksizes, d.rowblocks; eliminated=d.eliminated, nchunks)
end

# The same problem with its blocks in the layout's order, which BlockLinearSystems chooses: H and b permuted, the
# couplings and the retained pairs renumbered. Applied until the layout keeps the order (at most twice: a second pass
# reorders the eliminated blocks by the retained blocks' new numbers).
function in_layout(p::SchurTestProblem)
    while true
        order = blockorder(schur_structure(p))
        order == eachindex(order) && return p
        nrb = length(p.r_dims)
        sizes = [p.r_dims; p.e_dims]
        off = computestartindices(sizes)
        perm = reduce(vcat, [off[b] .+ (0:sizes[b]-1) for b in order])
        pos = invperm(order)
        couplings = [sort([pos[r] for r in p.couplings[order[q] - nrb]]) for q in nrb+1:length(order)]
        rr_pairs = [(max(pos[a], pos[b]), min(pos[a], pos[b])) for (a, b) in p.rr_pairs]
        p = SchurTestProblem(p.H[perm, perm], p.b[perm], sizes[order[1:nrb]], sizes[order[nrb+1:end]], couplings, rr_pairs)
    end
end

# The scalar range of each layout block of a system
function block_ranges(H::SchurBlockHessian)
    sizes = blocksizes(H.structure)
    off = computestartindices(sizes)
    return [off[b]:off[b] + sizes[b] - 1 for b in eachindex(sizes)]
end

# Add p.H to a system laid out by schur_structure(p) (p in layout order): every block it has storage for
function fill_schur!(H::SchurBlockHessian, p::SchurTestProblem)
    rg = block_ranges(H)
    nrb = nretained(H.structure)
    foreach(b -> addblock!(H, b, b, p.H, rg[b], rg[b]), eachindex(rg))
    for (e, coupled) in enumerate(p.couplings), r in coupled
        addblock!(H, r, nrb + e, p.H, rg[r], rg[nrb + e])
    end
    foreach(((r1, r2),) -> addblock!(H, r1, r2, p.H, rg[r1], rg[r2]), p.rr_pairs)
    return H
end

dense_schur_matrix(p::SchurTestProblem; nchunks=1) =
    fill_schur!(SchurBlockHessian{Float64, Matrix{Float64}}(schur_structure(p; nchunks)), p)
sparse_schur_matrix(p::SchurTestProblem; nchunks=1) =
    fill_schur!(SchurBlockHessian{Float64, SparseBlockLinearSystem{Float64}}(schur_structure(p; nchunks)), p)

# The system of p through DampedLinearSolve's contract, built for `linsolver`, with b = p.b
function make_schur_ls(kind::Symbol, p::SchurTestProblem; linsolver=:auto, nchunks=1)
    H = kind === :dense ? dense_schur_matrix(p; nchunks) : sparse_schur_matrix(p; nchunks)
    ls = SecondOrderOptimization.makehessiansystem(H, linsolver)
    SecondOrderOptimization.getb(ls) .= p.b
    return ls
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
    return in_layout(SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs))
end

# Eliminated blocks of every size class: tiny closed forms (1-3), the generic static tier (4..D_STATIC)
# and the dynamic tier above D_STATIC
function mixed_dims_problem(seed=11)
    rng = Random.Xoshiro(seed)
    r_dims = [6, 3, 9, 6]
    e_dims = [12, 12, 9, 9, 8, 8, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    coupled = 1:4
    couplings = [sort(unique([rand(rng, coupled) for _ in 1:3])) for _ in e_dims]
    return in_layout(SchurTestProblem(rng, r_dims, e_dims, couplings))
end


# Copy the numerical content of `src` (a SchurBlockHessian) into `dst`, of the same structure: whole parts
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
    return in_layout(SchurTestProblem(rng, r_dims, e_dims, couplings; rr_pairs=[(3, 1), (4, 2), (2, 1)]))
end





# Measured through a function barrier: at testset scope Julia 1.10 boxes the Float64 result
schur_quadform_allocations(H, v) = @allocated SecondOrderOptimization.quadform(H, v)
