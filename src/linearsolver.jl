using LDLFactorizations, LinearAlgebra, StaticArrays, SparseArrays, Static

validatelinsolver(::StaticSymbol{:auto}) = true
validatelinsolver(::StaticSymbol{:cholesky}) = true
validatelinsolver(::StaticSymbol{:ldl}) = true
validatelinsolver(::StaticSymbol{:qr}) = true
validatelinsolver(::StaticSymbol{S}) where S = false
validatelinsolver(s::Symbol) = validatelinsolver(static(s))
validatelinsolver(s::OptimizationLinearSolver) = true

function checklinsolver(s::Symbol)
    validatelinsolver(s) && return true
    if s === :pardiso
        throw(ArgumentError("Linear solver :pardiso requires Pardiso.jl. Please run `using Pardiso` first."))
    elseif s in (:cg, :minres, :lsmr, :krylov)
        throw(ArgumentError("Linear solver :$s requires Krylov.jl. Please run `using Krylov` first."))
    end
    throw(ArgumentError("Unknown linear solver: $s"))
end
checklinsolver(s::OptimizationLinearSolver) = true
checklinsolver(s::StaticSymbol) = checklinsolver(dynamic(s))

function getlinsolver(s::Symbol)
    s === :auto && return auto_solver
    s === :cholesky && return cholesky_solver
    s === :ldl && return ldl_solver
    s === :qr && return qr_solver
    checklinsolver(s)
    return s
end
getlinsolver(s::OptimizationLinearSolver) = s
getlinsolver(s::StaticSymbol) = getlinsolver(dynamic(s))

function is_indefinite(bk::BunchKaufman)
    n = length(bk.ipiv)
    k = 1
    @inbounds while k <= n
        if bk.ipiv[k] < 0
            return true
        elseif bk.D[k, k] < 0
            return true
        else
            k += 1
        end
    end
    return false
end

function is_indefinite(F::LDLFactorizations.LDLFactorization)
    @inbounds for d in F.d
        if d < 0
            return true
        end
    end
    return false
end

@inline function has_positive_pivots(C::Cholesky)
    @inbounds for i in 1:size(C.factors, 1)
        if !(C.factors[i, i] > 0)
            return false
        end
    end
    return true
end

function try_cholesky!(x::StaticVector, A::StaticMatrix, b::StaticVector)
    A_ = SMatrix(A)
    b_ = SVector(b)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        x .= cholfac \ b_
        return Success
    end
    return NotPositiveDefinite
end

function try_cholesky!(x, A, b)
    cholfac = cholesky(Symmetric(A); check=false)
    if issuccess(cholfac)
        ldiv!(x, cholfac, b)
        return Success
    end
    return NotPositiveDefinite
end

# Linear solve dispatching on solver symbol: :auto, :cholesky, :ldl, :qr
solve!(ls) = solve!(ls, static(:auto))
solve!(ls, solver::OptimizationLinearSolver) = solve!(ls, static(linsolver_symbol(solver)))

# StaticLinearSystem
function solve!(ls::StaticLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}, StaticSymbol{:ldl}})
    A_ = SMatrix(ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            A_ = A_ + ls.lambda * I
        else
            A_ = A_ + Diagonal(ls.lambda .* ls.scaling)
        end
    end
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        ls.x .= cholfac \ SVector(ls.b)
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::StaticLinearSystem, ::StaticSymbol{:qr})
    A_ = SMatrix(ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            A_ = A_ + ls.lambda * I
        else
            A_ = A_ + Diagonal(ls.lambda .* ls.scaling)
        end
    end
    ls.x .= A_ \ SVector(ls.b)
    return Success
end

# DenseLinearSystem
function solve!(ls::DenseLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    cholfac = cholesky!(Symmetric(ls.A_scratch); check=false)
    if issuccess(cholfac)
        ldiv!(ls.x, cholfac, ls.b)
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:ldl})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    bk = bunchkaufman!(Symmetric(ls.A_scratch); check=false)
    if issuccess(bk)
        ldiv!(ls.x, bk, ls.b)
        if is_indefinite(bk)
            return Indefinite
        end
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:qr})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    qrfac = qr!(ls.A_scratch, ColumnNorm())
    ldiv!(ls.x, qrfac, ls.b)
    return Success
end

# SparseLinearSystem
function solve!(ls::SparseLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:ldl}})
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    F = ldl_factorize!(ls.A, ls.solver_ws)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    if LDLFactorizations.factorized(F)
        ldiv!(ls.x, F, ls.b)
        if is_indefinite(F)
            return Indefinite
        end
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cholesky})
    if isnothing(ls.scaling)
        shift_val = Float64(ls.lambda)
        cholfac = cholesky!(ls.solver_ws, Symmetric(ls.A, :U); shift=shift_val, check=false)
    else
        if ls.lambda > 0
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
        cholfac = cholesky!(ls.solver_ws, Symmetric(ls.A, :U); check=false)
        if ls.lambda > 0
            @inbounds for (i, idx) in enumerate(ls.diag_inds)
                ls.A.nzval[idx] = ls.diag_cache[i]
            end
        end
    end
    if issuccess(cholfac)
        ls.x .= cholfac \ ls.b
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:qr})
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    ls.x .= qr(ls.A) \ ls.b
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return Success
end

# Method dispatch for Symbol and OptimizationLinearSolver
solve!(ls::AbstractLinearSystem, s::Symbol) = solve!(ls, static(s))
solve!(ls::AbstractLinearSystem, s::OptimizationLinearSolver) = solve!(ls, static(linsolver_symbol(s)))

# Generic matrix fallback for direct solve!(x, A, b)
solve!(x, A, b) = try_cholesky!(x, A, b)
solve!(x, A::SparseMatrixCSC, b) = ldiv!(x, ldl(A), b)
solve!(x::StaticVector, A::StaticMatrix, b::StaticVector) = try_cholesky!(x, A, b)

# Symmetric matrix inversion
invsym(A::AbstractMatrix) = inv(bunchkaufman(A))
invsym(A::StaticMatrix) = Size(A)[1] <= 14 ? inv(A) : inv(bunchkaufman(A))
