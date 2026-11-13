module BlockLinearSystemsExt

# Schur-complement linear systems for bipartite problems, built on the
# block data structures and reduction kernels of BlockLinearSystems.jl. Loaded when both packages are.

using SecondOrderOptimization
import SecondOrderOptimization: OptimizationLinearSolver
import SecondOrderOptimization: AbstractLinearSystem, solve!, dampen!, sharex, getA, getb, getx, SolveStatus, Success, NotPositiveDefinite, Indefinite, makelinearsystem, max_diag, updatescaling!, AnyMarquardtData, is_indefinite
import SecondOrderOptimization.LDLFactorizations: ldl_analyze, ldl_factorize!, factorized
import BlockLinearSystems
import BlockLinearSystems: SchurBlockMatrix, SchurReductionWorkspace, SparseBlockLinearSystem, reset_schur_dense!, reset_schur_sparse!, reduce_schur_dense!, reduce_schur_sparse!, backsubstitute!, build_schur_pattern, build_r_to_s_map, retained_block_pairs, trial_workspace
using LinearAlgebra, SparseArrays, Static

"""
    DenseSchurLinearSystem{T} <: AbstractLinearSystem

Linear system solving bipartite normal equations via dense Schur reduction.
"""
mutable struct DenseSchurLinearSystem{T} <: AbstractLinearSystem
    H::SchurBlockMatrix{T, Matrix{T}}
    b::Vector{T}
    x::Vector{T}
    b_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{T}
    S::Matrix{T}
    lambda::T
    scaling::Vector{T}
    scaling_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function DenseSchurLinearSystem(
    H::SchurBlockMatrix{T, Matrix{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
    nchunks::Int=0
) where T
    nr = H.structure.nr
    tot_dim = length(b)
    @assert tot_dim == nr + sum(Int, H.structure.e_dims; init=0) "Dimensions mismatch"

    b_r = view(b, 1:nr)
    b_e = view(b, nr+1:tot_dim)
    x_r = view(x, 1:nr)
    x_e = view(x, nr+1:tot_dim)
    scaling_r = view(scaling, 1:nr)
    scaling_e = view(scaling, nr+1:tot_dim)

    g_S = zeros(T, nr)
    S = zeros(T, nr, nr)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K; is_dense=true)

    return DenseSchurLinearSystem{T}(
        H, b, x, b_r, b_e, x_r, x_e, g_S, S, zero(T), scaling, scaling_r, scaling_e, ws
    )
end

"""
    SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem

Linear system solving bipartite normal equations via sparse Schur reduction, CHOLMOD sparse Cholesky, and LDL factorizations.
"""
mutable struct SparseSchurLinearSystem{T, Ti, TL, TF, TC} <: AbstractLinearSystem
    H::SchurBlockMatrix{T, SparseBlockLinearSystem{T}}
    b::Vector{T}
    x::Vector{T}
    b_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    b_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    x_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    g_S::Vector{T}
    S::SparseMatrixCSC{T, Ti}
    S_factor::TF                      # CHOLMOD.Factor (its type parameters differ between Julia versions)
    S_factor_ldl::TL
    S_chol::TC                        # CHOLMOD.Sparse copy of S, refreshed in place before each factorization
    r_to_s_map::Vector{Int}
    diag_inds::Vector{Int}
    block_rowval::Vector{Ti}
    block_colptr::Vector{Ti}
    block_row_offsets::Vector{Int}
    col_heights::Vector{Int}
    lambda::T
    scaling::Vector{T}
    scaling_r::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    scaling_e::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    workspace::SchurReductionWorkspace{T}
end

function SparseSchurLinearSystem(
    H::SchurBlockMatrix{T, SparseBlockLinearSystem{T}},
    b::Vector{T}, x::Vector{T},
    scaling::Vector{T}=ones(T, length(b));
    nchunks::Int=0,
    pattern=nothing
) where T
    nr = H.structure.nr
    tot_dim = length(b)
    @assert tot_dim == nr + sum(Int, H.structure.e_dims; init=0) "Dimensions mismatch"

    b_r = view(b, 1:nr)
    b_e = view(b, nr+1:tot_dim)
    x_r = view(x, 1:nr)
    x_e = view(x, nr+1:tot_dim)
    scaling_r = view(scaling, 1:nr)
    scaling_e = view(scaling, nr+1:tot_dim)

    # 1. Build the reduced-system pattern and sparsity structures. Retained-retained couplings stored in R
    # that do not go through an eliminated block need slots in S too.
    # A precomputed `pattern` (the result of an earlier build_schur_pattern call with the same
    # extra blocks, typically carried by `H` itself) can be passed in to avoid rebuilding it. It is
    # used as structure only: it may be shared with other systems, so S gets its own copy of
    # `S_sparse`, whose values this system writes.
    if pattern === nothing
        pattern = build_schur_pattern(H.structure, T; extra_blocks=retained_block_pairs(H.R))
        S = pattern.S_sparse
    else
        S = copy(pattern.S_sparse)
    end
    Ti = eltype(pattern.block_rowval)

    # 2. Map R to S
    r_to_s_map = build_r_to_s_map(H.R, pattern, H.structure.boffsets_r)

    # 3. Pre-analyze CHOLMOD symbolic factor and LDLFactorizations symbolic factor
    # A caller can add an analysed factor to the pattern (`merge(pattern, (S_factor = fac,))`);
    # `hasproperty` on a NamedTuple is resolved at compile time.
    S_chol = SparseArrays.CHOLMOD.Sparse(Symmetric(S, :U))
    S_factor = hasproperty(pattern, :S_factor) ? pattern.S_factor : SparseArrays.CHOLMOD.analyze(S_chol)
    S_factor_ldl = ldl_analyze(Symmetric(S, :U))

    g_S = zeros(T, nr)

    K = nchunks > 0 ? nchunks : (isempty(H.structure.chunks) ? 1 : length(H.structure.chunks[1]))
    ws = SchurReductionWorkspace{T}(H.structure, K; is_dense=false, nnz_S=nnz(S))

    return SparseSchurLinearSystem{T, Ti, typeof(S_factor_ldl), typeof(S_factor), typeof(S_chol)}(
        H, b, x, b_r, b_e, x_r, x_e, g_S, S, S_factor, S_factor_ldl, S_chol,
        r_to_s_map, pattern.diag_inds,
        pattern.block_rowval, pattern.block_colptr, pattern.block_row_offsets, pattern.col_heights,
        zero(T), scaling, scaling_r, scaling_e, ws
    )
end

# -----------------------------------------------------------------------------
# Interface Contracts
# -----------------------------------------------------------------------------
SecondOrderOptimization.getA(ls::DenseSchurLinearSystem) = ls.H
SecondOrderOptimization.getb(ls::DenseSchurLinearSystem) = ls.b
SecondOrderOptimization.getx(ls::DenseSchurLinearSystem) = ls.x

SecondOrderOptimization.getA(ls::SparseSchurLinearSystem) = ls.H
SecondOrderOptimization.getb(ls::SparseSchurLinearSystem) = ls.b
SecondOrderOptimization.getx(ls::SparseSchurLinearSystem) = ls.x

SecondOrderOptimization.fast_bAb(H::SchurBlockMatrix, dx::AbstractVector) = BlockLinearSystems.fast_bAb(H, dx)

function SecondOrderOptimization.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real) where T
    ls.lambda = convert(T, lambda)
    return ls
end

function SecondOrderOptimization.dampen!(ls::Union{DenseSchurLinearSystem{T}, SparseSchurLinearSystem{T}}, lambda::Real, scaling) where T
    ls.lambda = convert(T, lambda)
    if scaling !== nothing
        copyto!(ls.scaling, scaling)
    end
    return ls
end

function SecondOrderOptimization.max_diag(H::SchurBlockMatrix{T}) where T
    m = zero(T)
    if H.R isa Matrix
        for i in 1:H.structure.nr
            m = max(m, abs(H.R[i, i]))
        end
    elseif H.R isa SparseBlockLinearSystem
        for b in 1:H.structure.nrblocks
            ptr = H.R.indices[b, b]
            if ptr > 0
                db = Int(H.structure.r_dims[b])
                for k in 1:db
                    m = max(m, abs(H.R.data[ptr + (k - 1) * (db + 1)]))
                end
            end
        end
    end
    for e in 1:H.structure.neblocks
        eo = H.structure.eoffsets[e]
        de = Int(H.structure.e_dims[e])
        for k in 1:de
            m = max(m, abs(H.E_data[eo + (k - 1) * (de + 1)]))
        end
    end
    return m
end

SecondOrderOptimization.max_diag(ls::Union{DenseSchurLinearSystem, SparseSchurLinearSystem}) = SecondOrderOptimization.max_diag(ls.H)

function SecondOrderOptimization.updatescaling!(lmd::AnyMarquardtData, ls::DenseSchurLinearSystem{T}) where T
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    R = ls.H.R
    E_data = ls.H.E_data
    struct_info = ls.H.structure
    nr = struct_info.nr
    m = zero(T)

    for i in 1:nr
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(R[i, i]))
        m = max(m, d)
    end
    for e in 1:struct_info.neblocks
        eo = struct_info.eoffsets[e]
        e_off = struct_info.boffsets_e[e]
        de = Int(struct_info.e_dims[e])
        for i in 1:de
            c_ii = E_data[eo + (i - 1)*(de + 1)]
            @inbounds d = lmd.scaling[nr + e_off + i - 1] = max(lmd.scaling[nr + e_off + i - 1], abs(c_ii))
            m = max(m, d)
        end
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

function SecondOrderOptimization.updatescaling!(lmd::AnyMarquardtData, ls::SparseSchurLinearSystem{T}) where T
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    R_sys = ls.H.R
    E_data = ls.H.E_data
    struct_info = ls.H.structure
    nr = struct_info.nr
    m = zero(T)

    for b in 1:struct_info.nrblocks
        ptr = R_sys.indices[b, b]
        db = Int(struct_info.r_dims[b])
        bo = struct_info.boffsets_r[b]
        if ptr > 0
            for i in 1:db
                b_ii = R_sys.data[ptr + (i - 1)*(db + 1)]
                @inbounds d = lmd.scaling[bo + i - 1] = max(lmd.scaling[bo + i - 1], abs(b_ii))
                m = max(m, d)
            end
        end
    end
    for e in 1:struct_info.neblocks
        eo = struct_info.eoffsets[e]
        e_off = struct_info.boffsets_e[e]
        de = Int(struct_info.e_dims[e])
        for i in 1:de
            c_ii = E_data[eo + (i - 1)*(de + 1)]
            @inbounds d = lmd.scaling[nr + e_off + i - 1] = max(lmd.scaling[nr + e_off + i - 1], abs(c_ii))
            m = max(m, d)
        end
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end

function SecondOrderOptimization.zero!(ls::DenseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    return ls
end

function SecondOrderOptimization.zero!(ls::SparseSchurLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    BlockLinearSystems.zero!(ls.H)
    return ls
end

# Build the reduced system S and g_S from scratch. Called at the start of each solve, and again if an
# in-place factorization has overwritten S and a fallback factorization is needed.
function build_reduced!(ls::DenseSchurLinearSystem)
    reset_schur_dense!(ls.S, ls.H.R, ls.scaling_r, ls.lambda, ls.workspace)
    return reduce_schur_dense!(
        ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda,
        ls.H.structure, ls.H.E_data, ls.H.C_data, ls.workspace
    )
end

function solve_bunchkaufman!(ls::DenseSchurLinearSystem)
    bk = bunchkaufman!(Symmetric(ls.S, :U); check=false)
    issuccess(bk) || return NotPositiveDefinite
    ldiv!(ls.x_r, bk, ls.g_S)
    is_indefinite(bk) && return Indefinite
    return Success
end

function SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, solver_opt::StaticSymbol=static(:auto))
    build_reduced!(ls) || return NotPositiveDefinite

    if solver_opt isa StaticSymbol{:ldl}
        status = solve_bunchkaufman!(ls)
    else
        # Cholesky first, whether or not there is damping. It works in place on S (which is rebuilt on
        # every solve anyway); only if it fails do we rebuild S and fall back to Bunch-Kaufman.
        cholfac = cholesky!(Symmetric(ls.S, :U); check=false)
        if issuccess(cholfac)
            ldiv!(ls.x_r, cholfac, ls.g_S)
            status = Success
        else
            build_reduced!(ls) || return NotPositiveDefinite
            status = solve_bunchkaufman!(ls)
        end
    end
    status == Success || return status

    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace.E_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))
SecondOrderOptimization.solve!(ls::DenseSchurLinearSystem, s::OptimizationLinearSolver) = SecondOrderOptimization.solve!(ls, static(SecondOrderOptimization.linsolver_symbol(s)))

# Copy the values of S into the persistent CHOLMOD.Sparse in place. Its pattern (upper triangle of S, same
# order as S.nzval) never changes, so this avoids rebuilding a CHOLMOD.Sparse (pattern and values) per solve.
function refresh_cholmod!(chol::SparseArrays.CHOLMOD.Sparse, S::SparseMatrixCSC)
    cs = unsafe_load(pointer(chol))
    cs.nzmax == length(S.nzval) || error("CHOLMOD.Sparse pattern does not match S")
    GC.@preserve chol begin
        copyto!(unsafe_wrap(Array, Ptr{eltype(S)}(cs.x), cs.nzmax), S.nzval)
    end
    return chol
end

# `ldiv!(x, ::CHOLMOD.Factor, b)` only exists from Julia 1.12; earlier versions need the allocating `\\`.
@static if VERSION >= v"1.12"
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = ldiv!(x, F, b)
else
    cholmod_ldiv!(x, F::SparseArrays.CHOLMOD.Factor, b) = copyto!(x, F \ b)
end

function SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, solver_opt::StaticSymbol=static(:auto))
    reset_schur_sparse!(
        ls.S.nzval, ls.H.R.data, ls.r_to_s_map, ls.diag_inds, ls.scaling_r, ls.lambda, ls.workspace
    )
    success = reduce_schur_sparse!(
        ls.S, ls.g_S, ls.b_r, ls.b_e, ls.scaling_e, ls.lambda,
        ls.H.structure, ls.H.E_data, ls.H.C_data,
        ls.block_colptr, ls.block_rowval, ls.block_row_offsets, ls.col_heights,
        ls.workspace
    )
    if !success
        return NotPositiveDefinite
    end

    if solver_opt isa StaticSymbol{:ldl}
        F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
        if !factorized(F)
            return NotPositiveDefinite
        end
        ldiv!(ls.x_r, F, ls.g_S)
        if is_indefinite(F)
            return Indefinite
        end
    else
        refresh_cholmod!(ls.S_chol, ls.S)
        LinearAlgebra.cholesky!(ls.S_factor, ls.S_chol; check=false)
        if !LinearAlgebra.issuccess(ls.S_factor)
            F = ldl_factorize!(Symmetric(ls.S, :U), ls.S_factor_ldl)
            if !factorized(F)
                return NotPositiveDefinite
            end
            ldiv!(ls.x_r, F, ls.g_S)
            if is_indefinite(F)
                return Indefinite
            end
        else
            cholmod_ldiv!(ls.x_r, ls.S_factor, ls.g_S)
        end
    end

    backsubstitute!(ls.x_e, ls.x_r, ls.H.structure, ls.H.C_data, ls.workspace.E_inv_data, ls.workspace.z_data)
    return Success
end
SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, s::Symbol) = SecondOrderOptimization.solve!(ls, static(s))
SecondOrderOptimization.solve!(ls::SparseSchurLinearSystem, s::OptimizationLinearSolver) = SecondOrderOptimization.solve!(ls, static(SecondOrderOptimization.linsolver_symbol(s)))

# `quadratic_model_change` is intentionally not specialised: SecondOrderOptimization's generic method
# (0.5 * fast_bAb(getA(ls), dx) + fast_dot(getb(ls), dx)) is the undamped model change, and works with
# the `fast_bAb` for `SchurBlockMatrix` defined above. (An earlier override added the damping term with
# the wrong sign, and used a damping value the iterators have already reset to zero when a step is scored.)

function SecondOrderOptimization.copy_structure(ls::DenseSchurLinearSystem{T}) where T
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared
    nr = ls.H.structure.nr
    b = copy(ls.b)
    x = ls.x
    scaling = copy(ls.scaling)
    ws_trial = trial_workspace(ls.workspace)
    return DenseSchurLinearSystem{T}(
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nr), view(b, nr+1:length(b)),
        view(x, 1:nr), view(x, nr+1:length(x)),
        similar(ls.g_S), similar(ls.S), ls.lambda, scaling,
        view(scaling, 1:nr), view(scaling, nr+1:length(scaling)),
        ws_trial
    )
end


function SecondOrderOptimization.copy_structure(ls::SparseSchurLinearSystem{T, Ti, TL, TF, TC}) where {T, Ti, TL, TF, TC}
    # New storage for the Hessian and gradient (the original is kept for retries); x is shared
    nr = ls.H.structure.nr
    b = copy(ls.b)
    x = ls.x
    scaling = copy(ls.scaling)
    ws_trial = trial_workspace(ls.workspace)
    return SparseSchurLinearSystem{T, Ti, TL, TF, TC}(
        BlockLinearSystems.copy_structure(ls.H), b, x,
        view(b, 1:nr), view(b, nr+1:length(b)),
        view(x, 1:nr), view(x, nr+1:length(x)),
        similar(ls.g_S), copy(ls.S), ls.S_factor, ls.S_factor_ldl, copy(ls.S_chol),
        ls.r_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, scaling,
        view(scaling, 1:nr), view(scaling, nr+1:length(scaling)),
        ws_trial
    )
end

BlockLinearSystems.copy_structure(ls::DenseSchurLinearSystem) = SecondOrderOptimization.copy_structure(ls)
BlockLinearSystems.copy_structure(ls::SparseSchurLinearSystem) = SecondOrderOptimization.copy_structure(ls)

function SecondOrderOptimization.sharex(ls::DenseSchurLinearSystem{T}, from) where T
    ls.x === from.x && return ls
    nr = ls.H.structure.nr
    x = from.x
    return DenseSchurLinearSystem{T}(
        ls.H, ls.b, x,
        ls.b_r, ls.b_e,
        view(x, 1:nr), view(x, nr+1:length(x)),
        ls.g_S, ls.S, ls.lambda, ls.scaling,
        ls.scaling_r, ls.scaling_e,
        ls.workspace
    )
end

function SecondOrderOptimization.sharex(ls::SparseSchurLinearSystem{T, Ti, TL, TF, TC}, from) where {T, Ti, TL, TF, TC}
    ls.x === from.x && return ls
    nr = ls.H.structure.nr
    x = from.x
    return SparseSchurLinearSystem{T, Ti, TL, TF, TC}(
        ls.H, ls.b, x,
        ls.b_r, ls.b_e,
        view(x, 1:nr), view(x, nr+1:length(x)),
        ls.g_S, ls.S, ls.S_factor, ls.S_factor_ldl, ls.S_chol,
        ls.r_to_s_map, ls.diag_inds,
        ls.block_rowval, ls.block_colptr, ls.block_row_offsets, ls.col_heights,
        ls.lambda, ls.scaling,
        ls.scaling_r, ls.scaling_e,
        ls.workspace
    )
end

# The Schur solvers, valid only with a SchurBlockMatrix Hessian
SecondOrderOptimization.validatelinsolver(::StaticSymbol{:schur}) = true
SecondOrderOptimization.validatelinsolver(::StaticSymbol{:dense_schur}) = true
SecondOrderOptimization.validatelinsolver(::StaticSymbol{:sparse_schur}) = true
SecondOrderOptimization.checkhessianforsolver(::SchurBlockMatrix, linsolver) = nothing

function SecondOrderOptimization.makelinearsystem(x0, H::SchurBlockMatrix{T}, linsolver) where T
    nr = H.structure.nr
    ne = sum(Int, H.structure.e_dims; init=0)
    tot = nr + ne
    b = zeros(T, tot)
    x = zeros(T, tot)
    sym = SecondOrderOptimization.linsolver_symbol(linsolver)
    sparse_B = H.R isa SparseBlockLinearSystem
    if sym === :sparse_schur && !sparse_B
        throw(ArgumentError("linsolver = :sparse_schur requires a SchurBlockMatrix whose R is a SparseBlockLinearSystem, got $(typeof(H.R))"))
    elseif sym === :dense_schur && sparse_B
        throw(ArgumentError("linsolver = :dense_schur requires a SchurBlockMatrix whose R is a dense Matrix, got $(typeof(H.R))"))
    end
    # :schur (and any other symbol) picks the system from the storage of R
    # The sparse system reuses a Schur pattern carried by H (BlockLinearSystems versions that
    # predate the field have none); the dense system has no use for one.
    pattern = hasfield(typeof(H), :pattern) ? H.pattern : nothing
    return sparse_B ? SparseSchurLinearSystem(H, b, x; pattern) : DenseSchurLinearSystem(H, b, x)
end

end # module BlockLinearSystemsExt
