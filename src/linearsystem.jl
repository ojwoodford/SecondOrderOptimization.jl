using StaticArrays, Static, SparseArrays, LinearAlgebra

@enum SolveStatus Success NotPositiveDefinite Indefinite
@enum OptimizationLinearSolver auto_solver cholesky_solver ldl_solver qr_solver

linsolver_symbol(s::Symbol) = s
linsolver_symbol(s::StaticSymbol) = dynamic(s)
function linsolver_symbol(s::OptimizationLinearSolver)
    s == auto_solver && return :auto
    s == cholesky_solver && return :cholesky
    s == ldl_solver && return :ldl
    return :qr
end

abstract type AbstractLinearSystem end
function solve! end
function dampen! end

# Static linear system
mutable struct StaticLinearSystem{N, N2, T} <: AbstractLinearSystem
    const A::MMatrix{N, N, T, N2}
    const b::MVector{N, T}
    const x::MVector{N, T}
    lambda::T
    scaling::Union{Nothing, SVector{N, T}}
end
function StaticLinearSystem{N, N2, T}(A::MMatrix{N, N, T, N2}, b::MVector{N, T}, x::MVector{N, T}) where {N, N2, T}
    return StaticLinearSystem{N, N2, T}(A, b, x, zero(T), nothing)
end
function StaticLinearSystem{N, T}() where {N, T}
    N2 = N * N
    return StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, T}(undef), MVector{N, T}(undef), zero(T), nothing)
end
StaticLinearSystem{N}() where N = StaticLinearSystem{N, Float64}()

# Dense linear system. `solver_ws` holds preallocated factorization or Krylov workspaces
# (`nothing` when the solver needs none), built by `makedensews`.
mutable struct DenseLinearSystem{T, W} <: AbstractLinearSystem
    const A::Matrix{T}
    const b::Vector{T}
    const x::Vector{T}
    const A_scratch::Matrix{T}
    const solver_ws::W
    lambda::T
    scaling::Union{Nothing, Vector{T}}
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{T}, x::Vector{T}, A_scratch::Matrix{T}, solver_ws::W, lambda, scaling) where {T, W}
    return DenseLinearSystem{T, W}(A, b, x, A_scratch, solver_ws, lambda, scaling)
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{T}, x::Vector{T}, A_scratch::Matrix{T}, lambda, scaling) where T
    return DenseLinearSystem{T, Nothing}(A, b, x, A_scratch, nothing, lambda, scaling)
end
function DenseLinearSystem{T}(n::Integer; solver_ws=nothing) where T
    return DenseLinearSystem{T}(Matrix{T}(undef, n, n), Vector{T}(undef, n), Vector{T}(undef, n), Matrix{T}(undef, n, n), solver_ws, zero(T), nothing)
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{T}, x::Vector{T}, A_scratch::Matrix{T}=Matrix{T}(undef, size(A)); solver_ws=nothing) where T
    return DenseLinearSystem{T}(A, b, x, A_scratch, solver_ws, zero(T), nothing)
end
DenseLinearSystem(n::Integer; kwargs...) = DenseLinearSystem{Float64}(n; kwargs...)
DenseLinearSystem(A::Matrix{T}, b::Vector{T}, x::Vector{T}; kwargs...) where T = DenseLinearSystem{T}(A, b, x; kwargs...)

# Sparse linear system
mutable struct SparseLinearSystem{T, Ti, SWS} <: AbstractLinearSystem
    const A::SparseMatrixCSC{T, Ti}
    const b::Vector{T}
    const x::Vector{T}
    const diag_inds::Vector{Ti}
    const solver_ws::SWS
    const diag_cache::Vector{T}
    lambda::T
    scaling::Union{Nothing, Vector{T}}
end
function SparseLinearSystem(A::SparseMatrixCSC{T, Ti}, b::Vector{T}, x::Vector{T}, diag_inds::Vector{Ti}, solver_ws::SWS, diag_cache::Vector{T}=copy(A.nzval[diag_inds])) where {T, Ti, SWS}
    return SparseLinearSystem{T, Ti, SWS}(A, b, x, diag_inds, solver_ws, diag_cache, zero(T), nothing)
end

function ensure_diag_and_inds(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    @assert A.m == A.n "Hessian matrix must be square"
    n = A.n
    diag_inds = Vector{Ti}(undef, n)
    missing_diag = false
    for col in 1:n
        found = false
        for idx in nzrange(A, col)
            if A.rowval[idx] == col
                diag_inds[col] = idx
                found = true
                break
            end
        end
        if !found
            missing_diag = true
            break
        end
    end

    if !missing_diag
        return A, diag_inds
    end

    # Insert structural zeros on the diagonal. The result is a new matrix with a different
    # `nzval` layout from `A`, so `fdf!` is then handed this matrix, not the caller's prototype.
    I = Ti[]
    J = Ti[]
    V = T[]
    sizehint!(I, nnz(A) + n)
    sizehint!(J, nnz(A) + n)
    sizehint!(V, nnz(A) + n)
    for col in 1:n
        has_diag = false
        for idx in nzrange(A, col)
            r = A.rowval[idx]
            if r == col
                has_diag = true
            end
            push!(I, r)
            push!(J, col)
            push!(V, A.nzval[idx])
        end
        if !has_diag
            push!(I, col)
            push!(J, col)
            push!(V, zero(T))
        end
    end
    new_A = sparse(I, J, V, n, n)::SparseMatrixCSC{T, Ti}
    new_diag_inds = Vector{Ti}(undef, n)
    for col in 1:n
        for idx in nzrange(new_A, col)
            if new_A.rowval[idx] == col
                new_diag_inds[col] = idx
                break
            end
        end
    end
    return new_A, new_diag_inds
end

makesparsews(A, ::Union{StaticSymbol{:auto}, StaticSymbol{:ldl}}) = LDLFactorizations.ldl_analyze(A)
makesparsews(A, ::StaticSymbol{:cholesky}) = SparseArrays.cholesky(Symmetric(A, :U); check=false)
makesparsews(A, ::StaticSymbol{:qr}) = nothing
makesparsews(A, ::StaticSymbol{S}) where S = nothing
makesparsews(A, s::Symbol) = makesparsews(A, static(s))
makesparsews(A, s::OptimizationLinearSolver) = makesparsews(A, static(linsolver_symbol(s)))

function makelinearsystem(x0, H::SparseMatrixCSC{T}, linsolver) where T
    T === Float64 || throw(ArgumentError("The Hessian must be Float64; got a sparse prototype with element type $T. Pass a SparseMatrixCSC{Float64}."))
    A, diag_inds = ensure_diag_and_inds(H)
    sym = static(linsolver_symbol(linsolver))
    ws = makesparsews(A, sym)
    diag_cache = copy(A.nzval[diag_inds])
    return SparseLinearSystem(A, zeros(Float64, size(A, 1)), zeros(Float64, size(A, 1)), diag_inds, ws, diag_cache)
end

function makelinearsystem(x0, H::MMatrix{N, N, Float64, N2}, linsolver) where {N, N2}
    return StaticLinearSystem{N, N2, Float64}(H, MVector{N, Float64}(undef), MVector{N, Float64}(undef))
end

function makelinearsystem(x0, H::StaticMatrix{N, N}, linsolver) where N
    return StaticLinearSystem{N, Float64}()
end

# Preallocated workspace for a dense solve, or `nothing`. Extensions add methods for their solvers.
makedensews(n, s) = makedensews(n, static(linsolver_symbol(s)))
makedensews(n, ::StaticSymbol{:ldl}) = DenseLDLWorkspace(n)
makedensews(n, ::StaticSymbol) = nothing

function makelinearsystem(x0, H::Matrix{Float64}, linsolver)
    n = size(H, 1)
    return DenseLinearSystem{Float64}(H, zeros(Float64, n), zeros(Float64, n); solver_ws=makedensews(n, linsolver))
end

function makelinearsystem(x0, H::AbstractMatrix, linsolver)
    n = size(H, 1)
    return DenseLinearSystem{Float64}(Matrix{Float64}(H), zeros(Float64, n), zeros(Float64, n); solver_ws=makedensews(n, linsolver))
end

function makelinearsystem(x0::Number, ::Nothing, linsolver)
    return StaticLinearSystem{1, Float64}()
end

function makelinearsystem(x0::StaticVector{N}, ::Nothing, linsolver) where N
    return StaticLinearSystem{N, Float64}()
end

function makelinearsystem(x0, ::Nothing, linsolver)
    nv = nvars(x0)
    if dynamic(is_static(nv)) && nv <= 16
        return StaticLinearSystem{dynamic(nv), Float64}()
    end
    return DenseLinearSystem{Float64}(dynamic(nv); solver_ws=makedensews(dynamic(nv), linsolver))
end

getA(ls::AbstractLinearSystem) = ls.A
getb(ls::AbstractLinearSystem) = ls.b
getx(ls::AbstractLinearSystem) = ls.x

getgrad(ls::AbstractLinearSystem) = getb(ls)
gethessian(ls::AbstractLinearSystem) = getA(ls)
gethessgrad(ls::AbstractLinearSystem) = gethessian(ls), getgrad(ls)

function zero!(ls::AbstractLinearSystem)
    fill!(getb(ls), zero(eltype(getb(ls))))
    fill!(getA(ls), zero(eltype(getA(ls))))
    return ls
end
function zero!(ls::SparseLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    fill!(ls.A.nzval, zero(eltype(ls.A.nzval)))
    return ls
end

refresh_diag_cache!(ls::AbstractLinearSystem) = nothing
function refresh_diag_cache!(ls::SparseLinearSystem)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        ls.diag_cache[i] = ls.A.nzval[idx]
    end
    return nothing
end

# Unified damping interface: dampen! records, solve! applies
dampen!(ls::AbstractLinearSystem, lambda::Real) = (ls.lambda = convert(eltype(getb(ls)), lambda); ls.scaling = nothing; ls)
dampen!(ls::AbstractLinearSystem, lambda::Real, scaling) = (ls.lambda = convert(eltype(getb(ls)), lambda); ls.scaling = scaling; ls)

# Copy structure for trial relinearization (LevMarDataTrial)
copy_structure(ls::StaticLinearSystem{N, N2, T}) where {N, N2, T} = StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, T}(undef), ls.x, ls.lambda, ls.scaling)
copy_structure(ls::DenseLinearSystem{T, W}) where {T, W} = DenseLinearSystem{T, W}(Matrix{T}(undef, size(ls.A)), Vector{T}(undef, length(ls.b)), ls.x, Matrix{T}(undef, size(ls.A)), ls.solver_ws, ls.lambda, ls.scaling)
copy_structure(ls::SparseLinearSystem{T, Ti, SWS}) where {T, Ti, SWS} = SparseLinearSystem{T, Ti, SWS}(
    SparseMatrixCSC(ls.A.m, ls.A.n, ls.A.colptr, ls.A.rowval, Vector{T}(undef, length(ls.A.nzval))),
    Vector{T}(undef, length(ls.b)),
    ls.x,
    ls.diag_inds,
    ls.solver_ws,
    copy(ls.diag_cache),
    ls.lambda,
    ls.scaling
)

sharex(ls::StaticLinearSystem{N, N2, T}, from) where {N, N2, T} = ls.x === from.x ? ls : StaticLinearSystem{N, N2, T}(ls.A, ls.b, from.x, ls.lambda, ls.scaling)
sharex(ls::DenseLinearSystem{T, W}, from) where {T, W} = ls.x === from.x ? ls : DenseLinearSystem{T, W}(ls.A, ls.b, from.x, ls.A_scratch, ls.solver_ws, ls.lambda, ls.scaling)
sharex(ls::SparseLinearSystem{T, Ti, SWS}, from) where {T, Ti, SWS} = ls.x === from.x ? ls : SparseLinearSystem{T, Ti, SWS}(ls.A, ls.b, from.x, ls.diag_inds, ls.solver_ws, ls.diag_cache, ls.lambda, ls.scaling)

# Diagonal operations for damping
function uniformscaling!(A::AbstractMatrix, k)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += k
    end
end

function uniformscaling!(ls, k)
    uniformscaling!(getA(ls), k)
end

# The sparse system damps A in place, so these first save the undamped diagonal, from which every
# sparse solve restores it afterwards. Saving here (not in `refresh_diag_cache!`) means a solve
# leaves A as it found it even if A was written without a refresh.
function uniformscaling!(ls::SparseLinearSystem, k)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        ls.diag_cache[i] = ls.A.nzval[idx]
        ls.A.nzval[idx] += k
    end
end

function max_diag(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    m = zero(eltype(A))
    @inbounds for i in axes(A, 1)
        m = max(m, abs(A[i, i]))
    end
    return m
end
function max_diag(ls::SparseLinearSystem)
    m = zero(eltype(ls.A.nzval))
    @inbounds for idx in ls.diag_inds
        m = max(m, abs(ls.A.nzval[idx]))
    end
    return m
end
max_diag(ls) = max_diag(getA(ls))

function damp_marquardt!(A::AbstractMatrix, scaling, dlambda)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += dlambda * scaling[i]
    end
end
function damp_marquardt!(ls::SparseLinearSystem, scaling, dlambda)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        ls.diag_cache[i] = ls.A.nzval[idx]
        ls.A.nzval[idx] += dlambda * scaling[i]
    end
end
damp_marquardt!(ls, scaling, dlambda) = damp_marquardt!(getA(ls), scaling, dlambda)

# Change in the undamped quadratic model, 0.5 dx'H dx + g'dx, for a step dx (negative for a
# descent step)
function quadratic_model_change(ls::AbstractLinearSystem, dx)
    return 0.5 * fast_bAb(getA(ls), dx) + fast_dot(getb(ls), dx)
end
