module KrylovExt

using SecondOrderOptimization
using Krylov
using SparseArrays
using LinearAlgebra
using Static

import SecondOrderOptimization: validatelinsolver, makesparsews, makedensews, solve!
import SecondOrderOptimization: AbstractLinearSystem, SparseLinearSystem, DenseLinearSystem
import SecondOrderOptimization: SolveStatus, Success, NotPositiveDefinite, Indefinite
import SecondOrderOptimization: uniformscaling!, damp_marquardt!, builtfor, solvermismatch
import SecondOrderOptimization: DenseTallLinearSystem, SparseTallLinearSystem, maketallws, jmul!, jtr!
import SecondOrderOptimization: LSMR_ETOL, LSMR_AXTOL, LSMR_BTOL, LSMR_CONLIM

# Validation of Krylov solver symbols
validatelinsolver(::StaticSymbol{:cg}) = true
validatelinsolver(::StaticSymbol{:minres}) = true
validatelinsolver(::StaticSymbol{:lsmr}) = true   # for a JacobianObjective only; a HessianObjective rejects it

# A Krylov workspace plus storage for a Jacobi (diagonal) preconditioner
struct JacobiWorkspace{W}
    ws::W
    invdiag::Vector{Float64}
end
JacobiWorkspace(ws, n::Integer) = JacobiWorkspace(ws, Vector{Float64}(undef, n))

# Fill the inverse of the (damped) diagonal of `ls.A`. Returns false, meaning "don't
# precondition", when a diagonal entry is not positive and finite: the Jacobi preconditioner
# must be positive definite, and an indefinite matrix should still be reported as such.
function jacobi!(invdiag::Vector{Float64}, ls::SparseLinearSystem)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        d = ls.A.nzval[idx]
        (d > 0 && isfinite(d)) || return false
        invdiag[i] = inv(d)
    end
    return true
end

# Preallocated workspace creation for SparseLinearSystem
makesparsews(A, ::StaticSymbol{:cg}) = JacobiWorkspace(Krylov.CgWorkspace(A, zeros(eltype(A), size(A, 1))), size(A, 1))
makesparsews(A, ::StaticSymbol{:minres}) = JacobiWorkspace(Krylov.MinresWorkspace(A, zeros(eltype(A), size(A, 1))), size(A, 1))

# Preallocated workspace creation for DenseLinearSystem
makedensews(n, ::StaticSymbol{:cg}) = Krylov.CgWorkspace(n, n, Vector{Float64})
makedensews(n, ::StaticSymbol{:minres}) = Krylov.MinresWorkspace(n, n, Vector{Float64})

builtfor(::JacobiWorkspace{<:Krylov.CgWorkspace}) = ":cg"
builtfor(::JacobiWorkspace{<:Krylov.MinresWorkspace}) = ":minres"

# The sparse workspace is built once, by `makesparsews`, and `solver_ws` is a `const` field,
# so it cannot be replaced when it doesn't match the requested solver.
function check_krylov_ws(ls, ::Type{W}, asked::Symbol) where W
    ls.solver_ws isa W && return ls.solver_ws
    solvermismatch(ls, asked)
end

# With `linesearch=true`, nonpositive curvature sets both `indefinite` and `solved`, so check it first
function cg_status(stats)
    stats.indefinite && return Indefinite
    return stats.solved ? Success : NotPositiveDefinite
end

# SparseLinearSystem execution
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cg})
    jws = check_krylov_ws(ls, JacobiWorkspace{<:Krylov.CgWorkspace}, :cg)
    ws = jws.ws
    # Krylov's CG throws, rather than reporting, on a non-finite operator
    SecondOrderOptimization.isfinitesystem(ls) || return NotPositiveDefinite
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    # `linesearch=true` makes CG stop and flag `stats.indefinite` on nonpositive curvature;
    # without it, CG keeps iterating on an indefinite matrix. Jacobi-preconditioned when the
    # diagonal is positive (`M` is applied as a product, so it holds the inverse diagonal).
    if jacobi!(jws.invdiag, ls)
        Krylov.cg!(ws, ls.A, ls.b; M=Diagonal(jws.invdiag), linesearch=true)
    else
        Krylov.cg!(ws, ls.A, ls.b; linesearch=true)
    end
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return cg_status(ws.stats)
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:minres})
    jws = check_krylov_ws(ls, JacobiWorkspace{<:Krylov.MinresWorkspace}, :minres)
    ws = jws.ws
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    if jacobi!(jws.invdiag, ls)
        Krylov.minres!(ws, ls.A, ls.b; M=Diagonal(jws.invdiag))
    else
        Krylov.minres!(ws, ls.A, ls.b)
    end
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return ws.stats.solved ? Success : NotPositiveDefinite
end

# DenseLinearSystem execution
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:cg})
    SecondOrderOptimization.isfinitesystem(ls) || return NotPositiveDefinite   # Krylov's CG would throw
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if ws isa Krylov.CgWorkspace
        Krylov.cg!(ws, ls.A_scratch, ls.b; linesearch=true)
        copyto!(ls.x, ws.x)
        return cg_status(ws.stats)
    end
    # Not built by `makelinearsystem` for :cg, so no preallocated workspace
    (x, stats) = Krylov.cg(ls.A_scratch, ls.b; linesearch=true)
    copyto!(ls.x, x)
    return cg_status(stats)
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:minres})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if ws isa Krylov.MinresWorkspace
        Krylov.minres!(ws, ls.A_scratch, ls.b)
        copyto!(ls.x, ws.x)
        return ws.stats.solved ? Success : NotPositiveDefinite
    end
    (x, stats) = Krylov.minres(ls.A_scratch, ls.b)
    copyto!(ls.x, x)
    return stats.solved ? Success : NotPositiveDefinite
end

# ------------------------------------------------------------------------------------------------
# LSMR on the Jacobian of a tall system (TP2)
#
# The damped step solves min ‖J x − r‖² + λ Σᵢ sᵢ xᵢ², with λ the optimizer's damping and s the
# Marquardt scaling (all ones without it). That is LSMR's regularized problem
# min ‖b − A x‖² + λₗ² ‖x‖²_F with A = J, b = r, λₗ = √λ and F = Diagonal(s), passed as LSMR's
# preconditioner N = F⁻¹. Nothing is augmented and JᵀJ is never formed. J's products are this
# package's plain loops (`jmul!`, `jtr!`), not BLAS (rules.md 23).

# The tolerances (LSMR_ETOL and the rest, in tallsystem.jl) are starting values, not measured.

struct TallLsmrWorkspace{W}
    ws::W
    invscale::Vector{Float64}     # N = F⁻¹, the inverse of the Marquardt scaling
end
maketallws(m, n, ::StaticSymbol{:lsmr}) = TallLsmrWorkspace(Krylov.LsmrWorkspace(m, n, Vector{Float64}), Vector{Float64}(undef, n))
builtfor(::TallLsmrWorkspace) = ":lsmr"

# J as an operator, and its adjoint, for Krylov
struct TallOperator{M}
    J::M
end
struct TallOperatorT{M}
    J::M
end
Base.size(A::TallOperator) = size(A.J)
Base.size(A::TallOperator, d::Integer) = size(A.J, d)
Base.size(A::TallOperatorT) = reverse(size(A.J))
Base.size(A::TallOperatorT, d::Integer) = size(A)[d]
Base.eltype(::Union{TallOperator, TallOperatorT}) = Float64
Base.adjoint(A::TallOperator) = TallOperatorT(A.J)
LinearAlgebra.mul!(y::AbstractVector, A::TallOperator, v::AbstractVector) = jmul!(y, A.J, v)
LinearAlgebra.mul!(y::AbstractVector, A::TallOperatorT, w::AbstractVector) = jtr!(y, A.J, w)

const LsmrTallSystem = Union{DenseTallLinearSystem{<:TallLsmrWorkspace}, SparseTallLinearSystem{<:TallLsmrWorkspace}}

function solve!(ls::LsmrTallSystem, ::StaticSymbol{:lsmr})
    A = ls.A
    J, r = A.J, A.r
    m, n = size(J)
    lw = ls.solver_ws
    vals = J isa SparseMatrixCSC ? nonzeros(J) : J
    if !(all(isfinite, vals) && all(isfinite, r))
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    lambda = max(ls.lambda, 0.0)
    # Undamped with fewer residuals than variables, J is rank deficient: report it as the QR solve
    # does. (LSMR cannot detect numerical rank deficiency with m ≥ n; it returns a least-squares
    # solution there.)
    lambda == 0 && m < n && return NotPositiveDefinite
    op = TallOperator(J)
    if lambda > 0 && !isnothing(ls.scaling)
        @inbounds for i in 1:n
            lw.invscale[i] = inv(ls.scaling[i])
        end
        Krylov.lsmr!(lw.ws, op, r; λ = sqrt(lambda), N = Diagonal(lw.invscale),
                     etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    else
        Krylov.lsmr!(lw.ws, op, r; λ = sqrt(lambda),
                     etol = LSMR_ETOL, axtol = LSMR_AXTOL, btol = LSMR_BTOL, conlim = LSMR_CONLIM)
    end
    x = lw.ws.x
    (lw.ws.stats.solved && all(isfinite, x)) || return NotPositiveDefinite
    copyto!(ls.x, x)
    return Success
end
solve!(ls::DenseTallLinearSystem, ::StaticSymbol{:lsmr}) = solvermismatch(ls, :lsmr)
solve!(ls::SparseTallLinearSystem, ::StaticSymbol{:lsmr}) = solvermismatch(ls, :lsmr)

end # module KrylovExt
