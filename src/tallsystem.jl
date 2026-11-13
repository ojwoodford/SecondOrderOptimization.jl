using LinearAlgebra

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

# The optimal LAPACK work-array length for `dgeqrf` on an m×n matrix
function geqrf_worksize(m::Integer, n::Integer)
    (m == 0 || n == 0) && return 1
    A = Matrix{Float64}(undef, m, n)
    tau = Vector{Float64}(undef, min(m, n))
    work = Vector{Float64}(undef, 1)
    info = Ref{BlasInt}()
    ccall((LinearAlgebra.BLAS.@blasfunc(dgeqrf_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}),
          m, n, A, max(1, m), tau, work, BlasInt(-1), info)
    return max(1, Int(work[1]))
end

# The state of one linearization: the factorization and the squared column norms of J (the
# diagonal of JᵀJ). `stale` and `normsstale` are both set whenever J or r is written (by `zero!`,
# by the linearization wrapper, or by `conformance_fill!`, through `markstale!`); the QR clears
# `stale`, and computing the column norms clears `normsstale`.
mutable struct TallQRWorkspace
    const Jr::Matrix{Float64}     # m × (n+1): [J r], factored in place; R and q = (Qᵀr)[1:k] on top
    const tau::Vector{Float64}
    const S::Matrix{Float64}      # 2n × (n+1): [R q; √λ D^½ 0], factored in place per damping
    const tau2::Vector{Float64}
    const work::Vector{Float64}
    const colnorm2::Vector{Float64}   # n: squared column norms of J, the diagonal of JᵀJ
    stale::Bool
    normsstale::Bool
    nonfinite::Bool               # J or r held a NaN or Inf at the last factorization
    nfactor::Int                  # number of QR factorizations of [J r], for tests
    nnorms::Int                   # number of column-norm computations, for tests
end
function TallQRWorkspace(m::Integer, n::Integer)
    lwork = max(geqrf_worksize(m, n + 1), geqrf_worksize(2n, n + 1))
    return TallQRWorkspace(Matrix{Float64}(undef, m, n + 1), Vector{Float64}(undef, min(m, n + 1)),
                           Matrix{Float64}(undef, 2n, n + 1), Vector{Float64}(undef, min(2n, n + 1)),
                           Vector{Float64}(undef, lwork), Vector{Float64}(undef, n), true, true, false, 0, 0)
end

"""
    TallHessian

The Hessian storage of a [`DenseTallLinearSystem`](@ref): the Jacobian `J` (`m × n`) and the
residual `r` (length `m`), standing for `A = JᵀJ`, which is never formed.
`fast_bAb(A, v)` is `‖J v‖²`, and the diagonal of `A` is the squared column norms of `J`.
"""
struct TallHessian
    J::Matrix{Float64}
    r::Vector{Float64}
    Jv::Vector{Float64}           # scratch for fast_bAb
    ws::TallQRWorkspace
end
TallHessian(J::Matrix{Float64}) = TallHessian(J, zeros(size(J, 1)), zeros(size(J, 1)), TallQRWorkspace(size(J)...))

Base.size(A::TallHessian) = (size(A.J, 2), size(A.J, 2))
Base.size(A::TallHessian, d::Integer) = size(A)[d]
markstale!(A::TallHessian) = (A.ws.stale = true; A.ws.normsstale = true; A)

function fast_bAb(A::TallHessian, v::AbstractVector)
    mul!(A.Jv, A.J, v)
    total = 0.0
    @inbounds @simd for i in eachindex(A.Jv)
        total += A.Jv[i] * A.Jv[i]
    end
    return total
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
`[R; √λ D^½]` (`O(n³)`, independent of `m`), so retries with a larger `λ` are cheap. An undamped
solve on a numerically rank-deficient `J` (smallest `|R[i,i]|` at or below
`max(m, n)·eps` times the largest, which includes `m < n`) returns `NotPositiveDefinite`, as
Cholesky does on singular normal equations; a `NaN` or `Inf` in `J` or `r` also returns
`NotPositiveDefinite`, with `x` set to `NaN`. Nothing is allocated after construction.

The factorization and the squared column norms of `J` (the diagonal of `JᵀJ`, used by `max_diag`
and by the Marquardt scaling of `:levenbergmarquardt_dtd`) are computed once per linearization,
and redone after `zero!`, after the optimizer writes `J` and `r`, and after `conformance_fill!`;
code that writes `getA(ls).J` or `.r` directly must call `zero!` first, or
`SecondOrderOptimization.markstale!(getA(ls))`.
"""
mutable struct DenseTallLinearSystem <: AbstractLinearSystem
    const A::TallHessian
    const b::Vector{Float64}      # the gradient Jᵀr
    const x::Vector{Float64}
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
DenseTallLinearSystem(J::Matrix{Float64}) = DenseTallLinearSystem(TallHessian(J), zeros(size(J, 2)), zeros(size(J, 2)), 0.0, nothing)
DenseTallLinearSystem(m::Integer, n::Integer) = DenseTallLinearSystem(zeros(Float64, m, n))

function makelinearsystem(x0, obj::JacobianObjective, linsolver)
    obj.J isa Integer && return DenseTallLinearSystem(nresiduals(obj), dynamic(nvars(x0)))
    return DenseTallLinearSystem(obj.J)
end

function zero!(ls::DenseTallLinearSystem)
    fill!(ls.A.J, 0.0)
    fill!(ls.A.r, 0.0)
    fill!(ls.b, 0.0)
    markstale!(ls.A)
    return ls
end

copy_structure(ls::DenseTallLinearSystem) = DenseTallLinearSystem(TallHessian(similar(ls.A.J)), similar(ls.b), ls.x, ls.lambda, ls.scaling)
sharex(ls::DenseTallLinearSystem, from) = ls.x === from.x ? ls : DenseTallLinearSystem(ls.A, ls.b, from.x, ls.lambda, ls.scaling)

# How the iterators and the outer loop drive the tall system unchanged: they call
# `fdf!(getb(ls), getA(ls), x)`, and this wrapper turns that into `resjac!(r, J, x)` and g = Jᵀr
struct TallLinearize{RJ}
    resjac!::RJ
end
function (t::TallLinearize)(g::AbstractVector, A::TallHessian, x)
    cost = t.resjac!(A.r, A.J, x)
    mul!(g, transpose(A.J), A.r)
    markstale!(A)
    return cost
end

# QR of [J r]; afterwards Jr[1:k, 1:n] holds R (upper trapezoidal) and Jr[1:k, n+1] holds
# q = (Qᵀr)[1:k], with k = min(m, n)
function factor_tall!(A::TallHessian)
    ws = A.ws
    J, r = A.J, A.r
    m, n = size(J)
    ws.stale = false
    ws.nonfinite = !(all(isfinite, J) && all(isfinite, r))
    ws.nonfinite && return nothing
    Jr = ws.Jr
    copyto!(view(Jr, :, 1:n), J)
    copyto!(view(Jr, :, n + 1), r)
    geqrf_ws!(Jr, ws.tau, ws.work)
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
    ws.stale && factor_tall!(A)
    if ws.nonfinite
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    k = min(m, n)
    Jr = ws.Jr
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
    S = ws.S
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
    geqrf_ws!(S, ws.tau2, ws.work)
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

# Marquardt scaling (`:levenbergmarquardt_dtd`) from the diagonal of JᵀJ, the squared column norms
# of J. The rule is the generic method's (in iterators.jl), which reads `A[i, i]`: a running
# maximum of the diagonal, a non-finite entry counting as 1, then a floor of 1e-8 times the
# largest scale (and at least 1e-12). The scale is in the units of diag(JᵀJ), as `solve!` expects:
# it puts √(λ sᵢ) on the diagonal of the stacked matrix, adding λ sᵢ to JᵀJ.
function updatescaling!(lmd::AnyMarquardtData, ls::DenseTallLinearSystem)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(lmd.scaling))
    d2 = colnorms2!(ls.A)
    for i in eachindex(d2)
        val = abs(d2[i])
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], isfinite(val) ? val : 1.0)
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, max(m * 1e-8, 1e-12))
end
