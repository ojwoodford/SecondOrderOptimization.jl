# The residual-and-Jacobian linear system: A = JᵀJ held implicitly, b = Jᵀr. Solved by a
# Householder QR of [J r] once per linearization, then, per damping value, a QR of the small
# stacked matrix [R q; √λ D^½ 0]. JᵀJ is never formed.

# Householder QR of the strided matrix `A` in place (LAPACK `dgeqrf`), with preallocated `tau`
# and work array, so that it does not allocate
function geqrf_ws!(A::StridedMatrix{Float64}, tau::Vector{Float64}, work::Vector{Float64})
    m, n = size(A)
    length(tau) >= min(m, n) || throw(DimensionMismatch("tau has length $(length(tau)), needs $(min(m, n))"))
    (m == 0 || n == 0) && return A
    info = Ref{BlasInt}()
    ccall((LinearAlgebra.BLAS.@blasfunc(dgeqrf_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}),
          m, n, A, max(1, stride(A, 2)), tau, work, BlasInt(length(work)), info)
    LinearAlgebra.LAPACK.chkargsok(info[])
    return A
end

# The optimal LAPACK work-array length for `dgeqrf` on an m×n matrix. A workspace query
# (lwork = -1) reads only the dimensions and lda, and touches neither the matrix nor tau, so
# one-element stand-ins are passed for both, with the true m, n and lda.
function geqrf_worksize(m::Integer, n::Integer)
    (m == 0 || n == 0) && return 1
    A = Ref(0.0)
    tau = Ref(0.0)
    work = Ref(0.0)
    info = Ref{BlasInt}()
    ccall((LinearAlgebra.BLAS.@blasfunc(dgeqrf_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{Float64}, Ref{BlasInt},
           Ref{Float64}, Ref{Float64}, Ref{BlasInt}, Ref{BlasInt}),
          m, n, A, max(1, m), tau, work, BlasInt(-1), info)
    LinearAlgebra.LAPACK.chkargsok(info[])
    return max(1, Int(work[]))
end

# The factorization scratch: [J r] factored in place, the stacked matrix of the damped solve,
# and their LAPACK arrays. A system and its `copy_structure` copy share one: only one of them
# has a live factorization at a time (the trial system is not solved until it becomes the main
# one). `owner` is the token of the system whose factorization the scratch holds (0: none).
mutable struct TallQRScratch
    const Jr::Matrix{Float64}     # m × (n+1): [J r], factored in place; R and q = (Qᵀr)[1:k] on top
    const tau::Vector{Float64}
    const S::Matrix{Float64}      # 2n × (n+1): [R q; √λ D^½ 0], factored in place per damping
    const tau2::Vector{Float64}
    const work::Vector{Float64}
    owner::Int
end
function TallQRScratch(m::Integer, n::Integer)
    lwork = max(geqrf_worksize(m, n + 1), geqrf_worksize(2n, n + 1))
    return TallQRScratch(Matrix{Float64}(undef, m, n + 1), Vector{Float64}(undef, min(m, n + 1)),
                         Matrix{Float64}(undef, 2n, n + 1), Vector{Float64}(undef, min(2n, n + 1)),
                         Vector{Float64}(undef, lwork), 0)
end

# The tall systems, dense and sparse: they share the Marquardt scaling (from the squared column
# norms of J, `colnorms2!`) and the damping floor
abstract type AbstractTallLinearSystem <: AbstractLinearSystem end

const TALL_TOKENS = Threads.Atomic{Int}(0)

# The state of one system's linearization: whether its factorization and the squared column norms
# of J (the diagonal of JᵀJ) are current. `stale` and `normsstale` are both set, through
# `markstale!`, by `zero!` and by the linearization wrapper after `resjac!` has written J and r;
# the QR clears `stale`, and computing the column norms clears `normsstale`. The factorization
# itself is in the (possibly shared) scratch, and is this system's only while the scratch's owner
# is this system's token: a solve refactors otherwise.
mutable struct TallQRWorkspace
    const scratch::TallQRScratch
    const token::Int
    const colnorm2::Vector{Float64}   # n: squared column norms of J, the diagonal of JᵀJ
    stale::Bool
    normsstale::Bool
    nonfinite::Bool               # J or r held a NaN or Inf at the last factorization
    nfactor::Int                  # number of QR factorizations of [J r], for tests
    nnorms::Int                   # number of column-norm computations, for tests
end
TallQRWorkspace(scratch::TallQRScratch, n::Integer) =
    TallQRWorkspace(scratch, Threads.atomic_add!(TALL_TOKENS, 1) + 1, Vector{Float64}(undef, n), true, true, false, 0, 0)
TallQRWorkspace(m::Integer, n::Integer) = TallQRWorkspace(TallQRScratch(m, n), n)
# Whether the scratch holds this system's current factorization
factorized(ws::TallQRWorkspace) = !ws.stale && ws.scratch.owner == ws.token

"""
    TallHessian

The Hessian storage of a [`DenseTallLinearSystem`](@ref): the Jacobian `J` (`m × n`) and the
residual `r` (length `m`), standing for `A = JᵀJ`, which is never formed.
`quadform(A, v)` is `‖J v‖²`, and the diagonal of `A` is the squared column norms of `J`.
"""
struct TallHessian
    J::Matrix{Float64}
    r::Vector{Float64}
    Jv::Vector{Float64}           # scratch for quadform
    ws::TallQRWorkspace
end
TallHessian(J::Matrix{Float64}) = TallHessian(J, zeros(size(J, 1)), zeros(size(J, 1)), TallQRWorkspace(size(J)...))

Base.size(A::TallHessian) = (size(A.J, 2), size(A.J, 2))
Base.size(A::TallHessian, d::Integer) = size(A)[d]
markstale!(A::TallHessian) = (A.ws.stale = true; A.ws.normsstale = true; A)
# The storage the linearization writes, J and r, and the gradient Jᵀr it then forms
getJ(A::TallHessian) = A.J
getr(A::TallHessian) = A.r
gradient!(g::AbstractVector, A::TallHessian) = jtr!(g, A.J, A.r)

# y = J v, one scaled column at a time: a plain loop, not BLAS gemv, so that per-cost work stays
# single-threaded (rules.md 23). The first column is written, not added, so y needs no zeroing.
function jmul!(y::AbstractVector, J::Matrix{Float64}, v::AbstractVector)
    m, n = size(J)
    if n == 0
        fill!(y, 0.0)
        return y
    end
    @inbounds begin
        v1 = v[1]
        @fastmath @simd for i in 1:m
            y[i] = J[i, 1] * v1
        end
        for j in 2:n
            vj = v[j]
            @fastmath @simd for i in 1:m
                y[i] += J[i, j] * vj
            end
        end
    end
    return y
end

# ‖J v‖², with J v formed into the scratch `Jv`
function quadform(A::TallHessian, v::AbstractVector)
    size(A.J, 2) == 0 && return 0.0
    Jv = jmul!(A.Jv, A.J, v)
    total = 0.0
    @inbounds @simd for i in eachindex(Jv)
        total += Jv[i] * Jv[i]
    end
    return total
end

# g = Jᵀ r as one dot product per column of J: a plain loop, not BLAS gemv (rules.md 23)
function jtr!(g::AbstractVector, J::Matrix{Float64}, r::Vector{Float64})
    m, n = size(J)
    @inbounds for j in 1:n
        total = 0.0
        @fastmath @simd for i in 1:m
            total += J[i, j] * r[i]
        end
        g[j] = total
    end
    return g
end

# Squared norm of column j of J, summed in row order (as `conformance_matrix` sums), so the two agree exactly
@inline function sqcolnorm(J::Matrix{Float64}, j::Integer)
    total = 0.0
    @inbounds for i in axes(J, 1)
        total += J[i, j] * J[i, j]
    end
    return total
end

# The squared column norms of J, the diagonal of JᵀJ: computed once per linearization and shared
# by `max_diag` and the Marquardt scaling
function colnorms2!(A::TallHessian)
    ws = A.ws
    if ws.normsstale
        @inbounds for j in axes(A.J, 2)
            ws.colnorm2[j] = sqcolnorm(A.J, j)
        end
        ws.normsstale = false
        ws.nnorms += 1
    end
    return ws.colnorm2
end

# The largest squared column norm; NaN if any is NaN (`max` propagates NaN)
function max_diag(A::TallHessian)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

"""
    DenseTallLinearSystem(J::Matrix{Float64})
    DenseTallLinearSystem(m::Integer, n::Integer)

The linear system of a [`JacobianObjective`](@ref): the normal equations `(JᵀJ + λD) x = Jᵀr`,
with `J` dense (`m × n`, writing into the given matrix, or its own) and `JᵀJ` never formed.
It meets the [`AbstractLinearSystem`](@ref) interface with `getA(ls)` a [`TallHessian`](@ref)
holding `J` and `r`, and `getb(ls)` the gradient `Jᵀr`.

`solve!(ls, :auto)` (or `:qr`) factors `[J r]` by Householder QR once per linearization
(`O(mn²)`), and then, for each damping value, factors the `2n × n` stacked matrix
`[R; √λ D^½]` (`O(n³)`, independent of `m`), so retries with a larger `λ` are cheap.
Memory: `J`, one `m × (n+1)` scratch copy of `[J r]` for the factorization, and `O(n²)` more.
`copy_structure` (the trial system, with `callbackchangesstate = false`) adds its own `J` and
shares the scratch: about three times the storage of `J` in all. An
[`OptimizationWorkspace`](@ref) with a `Matrix` prototype owns its own `J`, so the caller's
prototype is one more. An undamped
solve on a numerically rank-deficient `J` (smallest `|R[i,i]|` at or below
`max(m, n)·eps` times the largest, which includes `m < n`) returns `NotPositiveDefinite`, as
Cholesky does on singular normal equations; a `NaN` or `Inf` in `J` or `r` also returns
`NotPositiveDefinite`, with `x` set to `NaN`. Nothing is allocated after construction.

The factorization and the squared column norms of `J` (the diagonal of `JᵀJ`, used by `max_diag`
and by the Marquardt scaling of `:levenbergmarquardt_dtd`) are computed once per linearization.
`zero!(ls)` does not zero `J` or `r`: `resjac!` sets them, so `zero!` only invalidates the
factorization and the column norms (and zeroes `b`, which the linearization then overwrites).
Code that writes `getA(ls).J` or `.r` directly must call `zero!` (before or after writing), or
`SecondOrderOptimization.markstale!(getA(ls))`; the linearization wrapper does this itself.

`DenseTallLinearSystem(J)` uses `J` as its storage with the values it holds, as does
[`optimize`](@ref) with a `Matrix` prototype; with an integer `m`, `J` starts as zeros.
`copy_structure` copies `J`, so a trial system starts with the same entries.
"""
mutable struct DenseTallLinearSystem{W} <: AbstractTallLinearSystem
    const A::TallHessian
    const b::Vector{Float64}      # the gradient Jᵀr
    const x::Vector{Float64}
    const solver_ws::W            # an iterative solver's workspace (`maketallws`), or nothing for QR
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
function DenseTallLinearSystem(J::Matrix{Float64}; linsolver = :auto)
    m, n = size(J)
    # An iterative solver needs no QR scratch, only its own workspace
    scratch = needsqr(linsolver) ? TallQRScratch(m, n) : TallQRScratch(0, n)
    A = TallHessian(J, zeros(m), zeros(m), TallQRWorkspace(scratch, n))
    return DenseTallLinearSystem(A, zeros(n), zeros(n), maketallws(m, n, linsolver), 0.0, nothing)
end
DenseTallLinearSystem(m::Integer, n::Integer; linsolver = :auto) = DenseTallLinearSystem(zeros(Float64, m, n); linsolver)
isfinitesystem(ls::DenseTallLinearSystem) = all(isfinite, ls.A.J) && all(isfinite, ls.A.r)

# The solvers of the tall systems: QR (`:auto`, `:qr`) in this package, and those an extension
# adds through `maketallws` (LSMR, in KrylovExt), which builds the solver's workspace
needsqr(linsolver) = linsolver_symbol(linsolver) in (:auto, :qr)
maketallws(m, n, linsolver) = maketallws(m, n, static(linsolver_symbol(linsolver)))
maketallws(m, n, ::StaticSymbol) = nothing

# LSMR's stopping tolerances, for every system it solves (KrylovExt on J, BlockLinearSystemsKrylovExt on the
# reduced Jacobian of QR elimination). Starting values, not measured (rules.md 26; improvements.md item 5):
# Krylov's own defaults, stated here so that a change in Krylov does not change them. itmax is Krylov's
# default, m + n iterations.
const LSMR_ETOL = sqrt(eps(Float64))     # stopping on the normal-equation residual
const LSMR_AXTOL = sqrt(eps(Float64))
const LSMR_BTOL = sqrt(eps(Float64))
const LSMR_CONLIM = 1 / sqrt(eps(Float64))

# The reduced solver of QR elimination (BlockLinearSystemsExt): its storage, built once, and its solve. Declared
# here so that an extension triggered by two packages can add a method without naming the other extension's types.
function make_reduced end
function solve_reduced! end

# The tall system of `n` variables for the Jacobian prototype `J`. A Matrix prototype is the storage
# resjac! writes into, used with the values it holds: entries resjac! never writes keep the caller's
# values. With an integer m the matrix starts as zeros. The prototype's type selects the system (a
# sparse one in sparsetallsystem.jl).
# A small problem with a static number of variables (`n` a `StaticInt`) gets the static system
# (statictallsystem.jl). The choice depends on m, known only at run time, so it is made here, once,
# when the system is built.
function tall_linearsystem(m::Integer, n, linsolver)
    usesstatictall(m, n, linsolver) && return StaticTallLinearSystem(zeros(MMatrix{Int(m), dynamic(n), Float64}))
    return DenseTallLinearSystem(m, dynamic(n); linsolver)
end
function tall_linearsystem(J::Matrix{Float64}, n, linsolver)
    usesstatictall(size(J, 1), n, linsolver) && return StaticTallLinearSystem(J)
    return DenseTallLinearSystem(J; linsolver)
end

# resjac! sets every entry of r and every entry of J that can be non-zero, so a fresh
# linearization needs no zeroing: only the cached factorization and column norms are invalidated.
# b is zeroed (n entries) so that zero! leaves the system as the interface describes; the
# linearization wrapper overwrites it.
function zero!(ls::DenseTallLinearSystem)
    fill!(ls.b, 0.0)
    markstale!(ls.A)
    return ls
end

# The trial system starts from the same J (its unwritten entries must match the main system's),
# and shares the factorization scratch: its own J, r, b, stale flags and column norms, so it costs
# one more J, not two
function copy_structure(ls::DenseTallLinearSystem)
    A = ls.A
    m, n = size(A.J)
    At = TallHessian(copy(A.J), zeros(m), zeros(m), TallQRWorkspace(A.ws.scratch, n))
    return DenseTallLinearSystem(At, zeros(length(ls.b)), ls.x, ls.solver_ws, ls.lambda, ls.scaling)
end
sharex(ls::DenseTallLinearSystem, from) = ls.x === from.x ? ls : DenseTallLinearSystem(ls.A, ls.b, from.x, ls.solver_ws, ls.lambda, ls.scaling)

# QR of [J r]; afterwards Jr[1:k, 1:n] holds R (upper trapezoidal) and Jr[1:k, n+1] holds
# q = (Qᵀr)[1:k], with k = min(m, n)
function factor_tall!(A::TallHessian)
    ws = A.ws
    J, r = A.J, A.r
    m, n = size(J)
    sc = ws.scratch
    ws.stale = false
    sc.owner = ws.token
    ws.nonfinite = !(all(isfinite, J) && all(isfinite, r))
    ws.nonfinite && return nothing
    Jr = sc.Jr
    copyto!(view(Jr, :, 1:n), J)
    copyto!(view(Jr, :, n + 1), r)
    geqrf_ws!(Jr, sc.tau, sc.work)
    ws.nfactor += 1
    return nothing
end

# Numerical full rank of the n×n upper triangle stored in the top of the factored `R` (of
# size(R, 1) rows): the smallest |R[i,i]| must exceed max(rows, n)·eps times the largest. A
# threshold of n·eps alone misses an exactly duplicated column of a tall J, whose R[i,i] is left
# at the rounding error of the Householder reflections, which grows with the number of rows.
function fullrank_upper(R::Matrix{Float64}, n::Integer)
    tol = max(size(R, 1), n) * eps(Float64)
    maxabs = 0.0
    minabs = Inf
    @inbounds for i in 1:n
        d = abs(R[i, i])
        maxabs = max(maxabs, d)
        minabs = min(minabs, d)
    end
    return minabs > tol * maxabs
end

# x = U \ x for the upper triangle U = R[1:n, 1:n], column-oriented
function backsubstitute!(x::Vector{Float64}, R::Matrix{Float64}, n::Integer)
    @inbounds for j in n:-1:1
        x[j] /= R[j, j]
        xj = x[j]
        for i in 1:j-1
            x[i] -= R[i, j] * xj
        end
    end
    return x
end

function solve!(ls::DenseTallLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}})
    A = ls.A
    ws = A.ws
    m, n = size(A.J)
    size(ws.scratch.Jr, 1) == m || solvermismatch(ls, :qr)
    factorized(ws) || factor_tall!(A)
    if ws.nonfinite
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    sc = ws.scratch
    k = min(m, n)
    Jr = sc.Jr
    lambda = ls.lambda
    if !(lambda > 0)
        # Undamped: R x = q, if R is square and numerically of full rank
        (m >= n && fullrank_upper(Jr, n)) || return NotPositiveDefinite
        @inbounds for i in 1:n
            ls.x[i] = Jr[i, n + 1]
        end
        backsubstitute!(ls.x, Jr, n)
        return Success
    end
    # Damped: QR of [R q; √λ D^½ 0], then back-substitution
    S = sc.S
    scaling = ls.scaling
    @inbounds begin
        for j in 1:n+1, i in 1:2n
            S[i, j] = 0.0
        end
        for j in 1:n+1, i in 1:min(j, k)
            S[i, j] = Jr[i, j]
        end
        for i in 1:n
            S[n + i, i] = sqrt(lambda * (isnothing(scaling) ? 1.0 : scaling[i]))
        end
    end
    geqrf_ws!(S, sc.tau2, sc.work)
    fullrank_upper(S, n) || return NotPositiveDefinite
    @inbounds for i in 1:n
        ls.x[i] = S[i, n + 1]
    end
    backsubstitute!(ls.x, S, n)
    return Success
end

function solve!(ls::DenseTallLinearSystem, ::StaticSymbol{S}) where S
    throw(ArgumentError("linsolver = :$S is not supported by a DenseTallLinearSystem; use :auto or :qr."))
end

# QR of J works with cond(J), not cond(JᵀJ) = cond(J)², so a damping down to about eps² times the
# diagonal of JᵀJ still changes the solve
dampingfloor(ls::AbstractTallLinearSystem) = eps(Float64)^2

# The diagonal of JᵀJ, the squared column norms of J (computed once per linearization)
getdiag!(d::AbstractVector, ls::AbstractTallLinearSystem) = copyto!(d, colnorms2!(ls.A))

# The Jacobian prototypes the tall systems are built from: their checks, size and solvers
const JACOBIAN_SUPPORT = "JacobianObjective supports every iterator, with a Matrix{Float64} or SparseMatrixCSC{Float64, Int} Jacobian prototype or an integer number of residuals m, and linsolver = :auto or :qr;" *
    " or, with BlockLinearSystems.jl loaded, a SchurBlockJacobian{Float64, Matrix{Float64}} or SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, linsolver = :auto, :qr, :cholesky or :ldl." *
    " Every prototype also takes linsolver = :lsmr, with Krylov.jl loaded"

# The Jacobian prototype: a Matrix{Float64} with at least one row, or an integer m >= 1
function checkjacobian(J)
    if J isa Integer && !(J isa Bool)
        J >= 1 || throw(ArgumentError("The number of residuals m must be at least 1; got $J. $JACOBIAN_SUPPORT."))
    elseif J isa Matrix{Float64}
        size(J, 1) >= 1 || throw(ArgumentError("The Jacobian prototype has no rows (size $(size(J))): it needs one row per residual. $JACOBIAN_SUPPORT."))
    elseif J isa SparseMatrixCSC{Float64, Int}
        size(J, 1) >= 1 || throw(ArgumentError("The Jacobian prototype has no rows (size $(size(J))): it needs one row per residual. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractSparseMatrix
        throw(ArgumentError("A sparse Jacobian prototype of type $(typeof(J)) is not supported. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractMatrix && eltype(J) !== Float64
        throw(ArgumentError("The Jacobian must be Float64; got a prototype with element type $(eltype(J)). $JACOBIAN_SUPPORT."))
    else
        throw(ArgumentError("A Jacobian prototype of type $(typeof(J)) is not supported. $JACOBIAN_SUPPORT."))
    end
    return nothing
end

# The number of residuals a Jacobian prototype stands for
nresiduals_of(J) = J isa Integer ? Int(J) : size(J, 1)

# The prototype's column count (nothing for an integer m, whose J is sized from x0), and the solvers it
# accepts. An extension adds methods for its prototype types.
jacobian_ncols(J::AbstractMatrix) = size(J, 2)
jacobian_ncols(J) = nothing
function check_jacobian_linsolver(J, s::Symbol)
    s in (:auto, :qr, :lsmr) || throw(ArgumentError("linsolver = :$s is not supported with a JacobianObjective. $JACOBIAN_SUPPORT."))
    return nothing
end
