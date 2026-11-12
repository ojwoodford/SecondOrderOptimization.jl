using LDLFactorizations, LinearAlgebra, StaticArrays, SparseArrays, Static

validatelinsolver(::StaticSymbol{:auto}) = true
validatelinsolver(::StaticSymbol{:cholesky}) = true
validatelinsolver(::StaticSymbol{:ldl}) = true
validatelinsolver(::StaticSymbol{:qr}) = true
validatelinsolver(::StaticSymbol{S}) where S = false
validatelinsolver(s::Symbol) = validatelinsolver(static(s))
validatelinsolver(s::OptimizationLinearSolver) = true

# Implemented by the BlockLinearSystems extension, for a `SchurBlockMatrix` Hessian only
const SCHUR_SOLVERS = (:schur, :dense_schur, :sparse_schur)

# Checked before the linear system is built, so a Schur solver with another Hessian type fails
# with a clear message rather than a `MethodError` in the first `solve!`. The BlockLinearSystems
# extension adds the method that accepts a `SchurBlockMatrix`.
function checkhessianforsolver(H, linsolver)
    sym = linsolver_symbol(linsolver)
    if sym in SCHUR_SOLVERS
        throw(ArgumentError("Linear solver :$sym requires H to be a BlockLinearSystems.SchurBlockMatrix, got $(typeof(H))."))
    end
    return nothing
end

checkhessiansize(x0, H) = nothing
function checkhessiansize(x0, H::AbstractMatrix)
    n = dynamic(nvars(x0))
    size(H) == (n, n) || throw(DimensionMismatch("Hessian prototype H has size $(size(H)), but the variables have $n degrees of freedom, so H must be $n×$n."))
    return nothing
end

# Checks on the Hessian prototype, run before the linear system is built
checkhessian(x0, H, linsolver) = (checkhessiansize(x0, H); checkhessianforsolver(H, linsolver))

function checklinsolver(s::Symbol)
    validatelinsolver(s) && return true
    if s === :pardiso
        throw(ArgumentError("Linear solver :pardiso requires Pardiso.jl. Please run `using Pardiso` first."))
    elseif s in (:cg, :minres, :krylov)
        throw(ArgumentError("Linear solver :$s requires Krylov.jl. Please run `using Krylov` first."))
    elseif s in SCHUR_SOLVERS
        throw(ArgumentError("Linear solver :$s requires BlockLinearSystems.jl. Please run `using BlockLinearSystems` first."))
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

# Indefiniteness tests on a factorization's pivots. A pivot counts as negative when it is below
# `-rtol` times the largest pivot magnitude, so the test is independent of the matrix's scale.

# Eigenvalues of the symmetric 2x2 block [a b; b c], smallest first
@inline function eig2x2(a, b, c)
    m = 0.5 * (a + c)
    r = hypot(0.5 * (a - c), b)
    return m - r, m + r
end

# Indefiniteness of the block-diagonal factor of a (non-rook) Bunch-Kaufman factorization, read
# straight from the packed factor `LD` and pivots `ipiv` as LAPACK's `sytrf` leaves them: a
# negative `ipiv[k]` starts a 2x2 block at (k, k+1), whose off-diagonal entry is stored in the
# triangle `uplo`. Reading the packed factor avoids materialising `bk.D`.
function is_indefinite_packed(LD::AbstractMatrix, ipiv::AbstractVector, uplo::Char, rtol::Float64=1e-6)
    n = length(ipiv)
    # The largest pivot magnitude, and the smallest pivot (eigenvalue, for 2x2 blocks)
    maxabs = 0.0
    minpivot = Inf
    k = 1
    @inbounds while k <= n
        if ipiv[k] < 0 && k < n
            off = uplo == 'U' ? LD[k, k+1] : LD[k+1, k]
            lo, hi = eig2x2(LD[k, k], off, LD[k+1, k+1])
            maxabs = max(maxabs, abs(lo), abs(hi))
            minpivot = min(minpivot, lo)
            k += 2
        else
            d = LD[k, k]
            maxabs = max(maxabs, abs(d))
            minpivot = min(minpivot, d)
            k += 1
        end
    end
    return minpivot < -rtol * maxabs
end

function is_indefinite(bk::BunchKaufman, rtol::Float64=1e-6)
    bk.rook || return is_indefinite_packed(bk.LD, bk.ipiv, bk.uplo, rtol)
    # Rook pivoting stores its pivots differently, so go through `bk.D`, materialised once
    # (it rebuilds the block-diagonal matrix on every access)
    D = bk.D
    n = length(bk.ipiv)
    maxabs = 0.0
    minpivot = Inf
    k = 1
    @inbounds while k <= n
        if bk.ipiv[k] < 0
            lo, hi = eig2x2(D[k, k], D[k, k+1], D[k+1, k+1])
            maxabs = max(maxabs, abs(lo), abs(hi))
            minpivot = min(minpivot, lo)
            k += 2
        else
            d = D[k, k]
            maxabs = max(maxabs, abs(d))
            minpivot = min(minpivot, d)
            k += 1
        end
    end
    return minpivot < -rtol * maxabs
end

function is_indefinite(F::LDLFactorizations.LDLFactorization, rtol::Float64=1e-6)
    maxabs = 0.0
    minpivot = Inf
    @inbounds for d in F.d
        maxabs = max(maxabs, abs(d))
        minpivot = min(minpivot, d)
    end
    return minpivot < -rtol * maxabs
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
@inline function damped_static(ls::StaticLinearSystem)
    A_ = SMatrix(ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            A_ = A_ + ls.lambda * I
        else
            A_ = A_ + Diagonal(ls.lambda .* ls.scaling)
        end
    end
    return A_
end

function solve!(ls::StaticLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}})
    A_ = damped_static(ls)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        ls.x .= cholfac \ SVector(ls.b)
        return Success
    end
    return NotPositiveDefinite
end

# Static `:ldl`: Cholesky when positive definite, otherwise LU (StaticArrays has no
# Bunch-Kaufman, and its `eigen` allocates above 3x3). Once Cholesky has failed, a nonsingular
# symmetric matrix has a negative eigenvalue, so the solve is reported as `Indefinite`, as the
# dense and sparse `:ldl` do.
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:ldl})
    A_ = damped_static(ls)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        ls.x .= cholfac \ SVector(ls.b)
        return Success
    end
    all(isfinite, A_) || return NotPositiveDefinite
    lufac = lu(A_; check=false)
    issuccess(lufac) || return NotPositiveDefinite
    x_ = lufac \ SVector(ls.b)
    all(isfinite, x_) || return NotPositiveDefinite
    ls.x .= x_
    return Indefinite
end

# Static `:qr` uses LU (`\`): StaticArrays has no pivoted QR
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:qr})
    A_ = damped_static(ls)
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

const BlasInt = LinearAlgebra.BlasInt

# Preallocated pivots and LAPACK work array for dense Bunch-Kaufman (`dsytrf`), so the
# factorization doesn't allocate in the loop
struct DenseLDLWorkspace
    ipiv::Vector{BlasInt}
    work::Vector{Float64}
end
function DenseLDLWorkspace(n::Integer)
    ipiv = Vector{BlasInt}(undef, n)
    work = Vector{Float64}(undef, 1)
    if n > 0
        # Workspace size query (lwork = -1): only the dimensions are read
        A = Matrix{Float64}(undef, n, n)
        info = Ref{BlasInt}()
        ccall((LinearAlgebra.BLAS.@blasfunc(dsytrf_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}, Clong),
              'U', n, A, max(1, n), ipiv, work, BlasInt(-1), info, 1)
        resize!(work, max(1, Int(work[1])))
    end
    return DenseLDLWorkspace(ipiv, work)
end

# Bunch-Kaufman factorization of the upper triangle of `A` in place; returns LAPACK's `info`
# (positive when the block-diagonal factor is exactly singular)
function sytrf_ws!(A::Matrix{Float64}, ws::DenseLDLWorkspace)
    n = LinearAlgebra.checksquare(A)
    length(ws.ipiv) == n || throw(DimensionMismatch("DenseLDLWorkspace is for size $(length(ws.ipiv)), got $n"))
    n == 0 && return BlasInt(0)
    info = Ref{BlasInt}()
    ccall((LinearAlgebra.BLAS.@blasfunc(dsytrf_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}, Clong),
          'U', n, A, max(1, stride(A, 2)), ws.ipiv, ws.work, BlasInt(length(ws.work)), info, 1)
    LinearAlgebra.LAPACK.chkargsok(info[])
    return info[]
end

function solve!(ls::DenseLinearSystem{Float64, DenseLDLWorkspace}, ::StaticSymbol{:ldl})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    sytrf_ws!(ls.A_scratch, ls.solver_ws) == 0 || return NotPositiveDefinite
    copyto!(ls.x, ls.b)
    LinearAlgebra.LAPACK.sytrs!('U', ls.A_scratch, ls.solver_ws.ipiv, ls.x)
    return is_indefinite_packed(ls.A_scratch, ls.solver_ws.ipiv, 'U') ? Indefinite : Success
end

# Without a preallocated workspace (a system not built by `makelinearsystem`)
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
        @static if VERSION >= v"1.12.0-"
            ldiv!(ls.x, cholfac, ls.b)   # in place; this method first appears in Julia 1.12
        else
            ls.x .= cholfac \ ls.b
        end
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
