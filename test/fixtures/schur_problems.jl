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
