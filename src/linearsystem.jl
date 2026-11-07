using StaticArrays, Static, SparseArrays, LinearAlgebra

abstract type AbstractLinearSystem end

# Static linear system
struct StaticLinearSystem{N, N2, T} <: AbstractLinearSystem
    A::MMatrix{N, N, T, N2}
    b::MVector{N, T}
    x::MVector{N, T}
end
function StaticLinearSystem{N, T}() where {N, T}
    N2 = N * N
    return StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, T}(undef), MVector{N, T}(undef))
end
StaticLinearSystem{N}() where N = StaticLinearSystem{N, Float64}()

# Dense linear system
struct DenseLinearSystem{T} <: AbstractLinearSystem
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    A_scratch::Matrix{T}
end
function DenseLinearSystem{T}(n::Integer) where T
    return DenseLinearSystem{T}(Matrix{T}(undef, n, n), Vector{T}(undef, n), Vector{T}(undef, n), Matrix{T}(undef, n, n))
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where T
    return DenseLinearSystem{T}(A, b, x, Matrix{T}(undef, size(A)))
end
DenseLinearSystem(n::Integer) = DenseLinearSystem{Float64}(n)
DenseLinearSystem(A::Matrix{T}, b::Vector{T}, x::Vector{T}) where T = DenseLinearSystem{T}(A, b, x)

# Sparse linear system
struct SparseLinearSystem{T, Ti, SWS} <: AbstractLinearSystem
    A::SparseMatrixCSC{T, Ti}
    b::Vector{T}
    x::Vector{T}
    diag_inds::Vector{Ti}
    solver_ws::SWS
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

    # Insert structural zeros on the diagonal
    I = Int[]
    J = Int[]
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
    new_A = sparse(I, J, V, n, n)
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

function makelinearsystem(x0, H::SparseMatrixCSC, linsolver)
    A, diag_inds = ensure_diag_and_inds(H)
    ldlfac = LDLFactorizations.ldl_analyze(A)
    return SparseLinearSystem(A, zeros(Float64, size(A, 1)), zeros(Float64, size(A, 1)), diag_inds, ldlfac)
end

function makelinearsystem(x0, H::MMatrix{N, N, Float64, N2}, linsolver) where {N, N2}
    return StaticLinearSystem{N, N2, Float64}(H, MVector{N, Float64}(undef), MVector{N, Float64}(undef))
end

function makelinearsystem(x0, H::StaticMatrix{N, N}, linsolver) where N
    return StaticLinearSystem{N, Float64}()
end

function makelinearsystem(x0, H::Matrix{Float64}, linsolver)
    n = size(H, 1)
    return DenseLinearSystem{Float64}(H, zeros(Float64, n), zeros(Float64, n))
end

function makelinearsystem(x0, H::AbstractMatrix, linsolver)
    n = size(H, 1)
    return DenseLinearSystem{Float64}(Matrix{Float64}(H), zeros(Float64, n), zeros(Float64, n))
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
    return DenseLinearSystem{Float64}(dynamic(nv))
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

# Copy structure for trial relinearization (LevMarDataTrial)
copy_structure(ls::StaticLinearSystem{N, N2, T}) where {N, N2, T} = StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, T}(undef), ls.x)
copy_structure(ls::DenseLinearSystem{T}) where T = DenseLinearSystem{T}(Matrix{T}(undef, size(ls.A)), Vector{T}(undef, length(ls.b)), ls.x, Matrix{T}(undef, size(ls.A)))
copy_structure(ls::SparseLinearSystem{T, Ti, SWS}) where {T, Ti, SWS} = SparseLinearSystem{T, Ti, SWS}(
    SparseMatrixCSC(ls.A.m, ls.A.n, ls.A.colptr, ls.A.rowval, Vector{T}(undef, length(ls.A.nzval))),
    Vector{T}(undef, length(ls.b)),
    ls.x,
    ls.diag_inds,
    ls.solver_ws
)

sharex(ls::StaticLinearSystem{N, N2, T}, from) where {N, N2, T} = ls.x === from.x ? ls : StaticLinearSystem{N, N2, T}(ls.A, ls.b, from.x)
sharex(ls::DenseLinearSystem{T}, from) where T = ls.x === from.x ? ls : DenseLinearSystem{T}(ls.A, ls.b, from.x, ls.A_scratch)
sharex(ls::SparseLinearSystem{T, Ti, SWS}, from) where {T, Ti, SWS} = ls.x === from.x ? ls : SparseLinearSystem{T, Ti, SWS}(ls.A, ls.b, from.x, ls.diag_inds, ls.solver_ws)

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

function uniformscaling!(ls::SparseLinearSystem, k)
    @inbounds for idx in ls.diag_inds
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
        ls.A.nzval[idx] += dlambda * scaling[i]
    end
end
damp_marquardt!(ls, scaling, dlambda) = damp_marquardt!(getA(ls), scaling, dlambda)
