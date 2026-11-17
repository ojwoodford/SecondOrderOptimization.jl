# The static residual-and-Jacobian linear system, for small problems (TP4): J an MMatrix{M, N}, r an MVector{M},
# solved as the dense tall system is (a QR of J once per linearization, then per damping value a QR of the
# stacked [R; √λ D^½]), with StaticArrays' QR. Chosen at construction, without a user option, when the
# number of variables is static (a `StaticInt`, as the optimizer passes it when `usesstaticsystem(x0)`
# holds) and M·N is at most MAX_STATIC_TALL.

"""
    MAX_STATIC_TALL = 48

The largest `m·n` (residuals times variables) for which a tall system with a static number of variables
(`n` a `StaticInt`, as for a [`JacobianObjective`](@ref) with `usesstaticsystem(x0)`) and a dense Jacobian
(an integer `m` or a `Matrix` prototype) is a static tall system. A starting value (rules.md 26): measured 2026-11-15 against the dense tall system on
Julia 1.10 and 1.13, the static system was faster, per linearization and per retry, at every measured size with
`m·n ≤ 48` (`n` from 1 to 6), and slower somewhere at every larger `m·n` measured (`n = 8, m = 8` already
retries 1.26 times slower on 1.10).
"""
const MAX_STATIC_TALL = 48

# Whether n variables and an m-residual dense Jacobian get the static tall system: n static, a QR solver,
# and m·n small
usesstatictall(m::Integer, n, linsolver) = n isa StaticInt && needsqr(linsolver) && m * dynamic(n) <= MAX_STATIC_TALL

"""
    StaticTallHessian

The Hessian storage of a [`StaticTallLinearSystem`](@ref): `J` (an `MMatrix{M, N}`) and `r` (an
`MVector{M}`), standing for `A = JᵀJ`, which is never formed.
"""
struct StaticTallHessian{M, N, MN}
    J::MMatrix{M, N, Float64, MN}
    r::MVector{M, Float64}
    colnorm2::MVector{N, Float64}     # squared column norms of J, the diagonal of JᵀJ
    stale::Base.RefValue{Bool}         # the factorization (R, q) is out of date
    normsstale::Base.RefValue{Bool}
end
StaticTallHessian(J::MMatrix{M, N}) where {M, N} = StaticTallHessian(J, zeros(MVector{M, Float64}), zeros(MVector{N, Float64}), Ref(true), Ref(true))

Base.size(A::StaticTallHessian{M, N}) where {M, N} = (N, N)
Base.size(A::StaticTallHessian, d::Integer) = size(A)[d]
markstale!(A::StaticTallHessian) = (A.stale[] = true; A.normsstale[] = true; A)

function jmul!(y::AbstractVector, J::MMatrix, v::AbstractVector)
    m, n = size(J)
    @inbounds for i in 1:m
        total = 0.0
        for j in 1:n
            total += J[i, j] * v[j]
        end
        y[i] = total
    end
    return y
end
function jtr!(g::AbstractVector, J::MMatrix, r::AbstractVector)
    m, n = size(J)
    @inbounds for j in 1:n
        total = 0.0
        for i in 1:m
            total += J[i, j] * r[i]
        end
        g[j] = total
    end
    return g
end

function quadform(A::StaticTallHessian{M, N}, v::AbstractVector) where {M, N}
    J = A.J
    total = 0.0
    @inbounds for i in 1:M
        s = 0.0
        for j in 1:N
            s += J[i, j] * v[j]
        end
        total += s * s
    end
    return total
end

function colnorms2!(A::StaticTallHessian{M, N}) where {M, N}
    if A.normsstale[]
        @inbounds for j in 1:N
            total = 0.0
            for i in 1:M
                total += A.J[i, j] * A.J[i, j]
            end
            A.colnorm2[j] = total
        end
        A.normsstale[] = false
    end
    return A.colnorm2
end
function max_diag(A::StaticTallHessian)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

"""
    StaticTallLinearSystem{M, N, MN, K, KN}

The tall system of [`DenseTallLinearSystem`](@ref), with `J` an `MMatrix{M, N}` and the QR factorizations done
by StaticArrays, for problems with `M·N ≤ MAX_STATIC_TALL`. Same solves, statuses and damping floor. `R` and
`q` (`K = min(M, N)` rows) are kept per linearization.
"""
mutable struct StaticTallLinearSystem{M, N, MN, K, KN} <: AbstractTallLinearSystem
    const A::StaticTallHessian{M, N, MN}
    const b::MVector{N, Float64}
    const x::MVector{N, Float64}
    const R::MMatrix{K, N, Float64, KN}
    const q::MVector{K, Float64}
    nonfinite::Bool
    lambda::Float64
    scaling::Union{Nothing, MVector{N, Float64}, Vector{Float64}}
end
function StaticTallLinearSystem(J::MMatrix{M, N}) where {M, N}
    K = min(M, N)
    return StaticTallLinearSystem{M, N, M * N, K, K * N}(StaticTallHessian(J), zeros(MVector{N, Float64}), zeros(MVector{N, Float64}),
                                                        zeros(MMatrix{K, N, Float64, K * N}), zeros(MVector{K, Float64}), false, 0.0, nothing)
end
isfinitesystem(ls::StaticTallLinearSystem) = all(isfinite, ls.A.J) && all(isfinite, ls.A.r)
StaticTallLinearSystem(J::Matrix{Float64}) = StaticTallLinearSystem(MMatrix{size(J, 1), size(J, 2), Float64}(J))

zero!(ls::StaticTallLinearSystem) = (fill!(ls.b, 0.0); markstale!(ls.A); ls)
initjacobian!(ls::StaticTallLinearSystem, ::Integer) = (fill!(ls.A.J, 0.0); markstale!(ls.A); ls)
initjacobian!(ls::StaticTallLinearSystem, P::Matrix{Float64}) = (copyto!(ls.A.J, P); markstale!(ls.A); ls)

function copy_structure(ls::StaticTallLinearSystem{M, N, MN, K, KN}) where {M, N, MN, K, KN}
    return StaticTallLinearSystem{M, N, MN, K, KN}(StaticTallHessian(copy(ls.A.J)), zeros(MVector{N, Float64}), ls.x,
                                                  zeros(MMatrix{K, N, Float64, KN}), zeros(MVector{K, Float64}), false, ls.lambda, ls.scaling)
end
sharex(ls::StaticTallLinearSystem{M, N, MN, K, KN}, from) where {M, N, MN, K, KN} = ls.x === from.x ? ls :
    StaticTallLinearSystem{M, N, MN, K, KN}(ls.A, ls.b, from.x, ls.R, ls.q, ls.nonfinite, ls.lambda, ls.scaling)

function (t::TallLinearize)(g::AbstractVector, A::StaticTallHessian, x)
    cost = t.resjac!(A.r, A.J, x)
    jtr!(g, A.J, A.r)
    markstale!(A)
    return cost
end

# R and q = (Qᵀr)[1:K] from the thin QR of J
function factor_static!(ls::StaticTallLinearSystem{M, N, MN, K}) where {M, N, MN, K}
    A = ls.A
    A.stale[] = false
    ls.nonfinite = !(all(isfinite, A.J) && all(isfinite, A.r))
    ls.nonfinite && return nothing
    F = qr(SMatrix(A.J))
    ls.R .= F.R
    ls.q .= F.Q' * SVector(A.r)
    return nothing
end

# Numerical full rank of the N×N upper triangle of R, as `fullrank_upper` decides for the dense system
@inline function static_fullrank(R::StaticMatrix, rows::Integer, n::Integer)
    tol = max(rows, n) * eps(Float64)
    maxabs, minabs = 0.0, Inf
    @inbounds for i in 1:n
        d = abs(R[i, i])
        maxabs = max(maxabs, d)
        minabs = min(minabs, d)
    end
    return minabs > tol * maxabs
end

function solve!(ls::StaticTallLinearSystem{M, N, MN, K}, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}) where {M, N, MN, K}
    ls.A.stale[] && factor_static!(ls)
    if ls.nonfinite
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    R, q = SMatrix(ls.R), SVector(ls.q)
    lambda = ls.lambda
    if !(lambda > 0)
        (M >= N && static_fullrank(R, M, N)) || return NotPositiveDefinite
        ls.x .= UpperTriangular(SMatrix{N, N}(R[SOneTo(N), :])) \ SVector{N}(q[SOneTo(N)])
        return Success
    end
    scaling = ls.scaling
    d = isnothing(scaling) ? ones(SVector{N, Float64}) : SVector{N, Float64}(scaling)
    S = vcat(R, SMatrix{N, N, Float64}(Diagonal(sqrt.(lambda .* d))))
    F = qr(S)
    static_fullrank(F.R, K + N, N) || return NotPositiveDefinite
    qq = F.Q' * vcat(q, zeros(SVector{N, Float64}))
    ls.x .= UpperTriangular(F.R) \ qq
    return Success
end
function solve!(ls::StaticTallLinearSystem, ::StaticSymbol{S}) where S
    throw(ArgumentError("linsolver = :$S is not supported by a StaticTallLinearSystem; use :auto or :qr."))
end

conformance_matrix(A::StaticTallHessian) = conformance_gram(Matrix(A.J))
conformance_fill!(A::StaticTallHessian, M::AbstractMatrix) = (A.J .= conformance_factor(M, size(A.J)...); A)
conformance_traits(A::StaticTallHessian) = (exact = false, maxrank = min(size(A.J)...), indefinite = false, accumulates = false)
