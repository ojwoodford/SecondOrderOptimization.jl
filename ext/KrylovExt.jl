module KrylovExt

using SecondOrderOptimization
using Krylov
using SparseArrays
using LinearAlgebra
using Static

import SecondOrderOptimization: validatelinsolver, makesparsews, makedensews, solve!
import SecondOrderOptimization: AbstractLinearSystem, SparseLinearSystem, DenseLinearSystem
import SecondOrderOptimization: SolveStatus, Success, NotPositiveDefinite, Indefinite
import SecondOrderOptimization: uniformscaling!, damp_marquardt!

# Validation of Krylov solver symbols
validatelinsolver(::StaticSymbol{:cg}) = true
validatelinsolver(::StaticSymbol{:minres}) = true
validatelinsolver(::StaticSymbol{:lsmr}) = true
validatelinsolver(::StaticSymbol{:krylov}) = true

# Preallocated workspace creation for SparseLinearSystem
makesparsews(A, ::StaticSymbol{:cg}) = Krylov.CgWorkspace(A, zeros(eltype(A), size(A, 1)))
makesparsews(A, ::StaticSymbol{:minres}) = Krylov.MinresWorkspace(A, zeros(eltype(A), size(A, 1)))
makesparsews(A, ::StaticSymbol{:lsmr}) = Krylov.LsmrWorkspace(A, zeros(eltype(A), size(A, 1)))
makesparsews(A, ::StaticSymbol{:krylov}) = makesparsews(A, static(:cg))

# Preallocated workspace creation for DenseLinearSystem
makedensews(n, ::StaticSymbol{:cg}) = Krylov.CgWorkspace(n, n, Vector{Float64})
makedensews(n, ::StaticSymbol{:minres}) = Krylov.MinresWorkspace(n, n, Vector{Float64})
makedensews(n, ::StaticSymbol{:lsmr}) = Krylov.LsmrWorkspace(n, n, Vector{Float64})
makedensews(n, ::StaticSymbol{:krylov}) = makedensews(n, static(:cg))

# The sparse workspace is built once, by `makesparsews`, and `solver_ws` is a `const` field,
# so it cannot be replaced when it doesn't match the requested solver.
function check_krylov_ws(ws, ::Type{W}, s::Symbol) where W
    ws isa W && return ws
    throw(ArgumentError("This linear system was not built for linsolver=:$s; construct it with `makelinearsystem(x0, H, :$s)`."))
end

# With `linesearch=true`, nonpositive curvature sets both `indefinite` and `solved`, so check it first
function cg_status(stats)
    stats.indefinite && return Indefinite
    return stats.solved ? Success : NotPositiveDefinite
end

# SparseLinearSystem execution
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cg})
    ws = check_krylov_ws(ls.solver_ws, Krylov.CgWorkspace, :cg)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    # `linesearch=true` makes CG stop and flag `stats.indefinite` on nonpositive curvature;
    # without it, CG keeps iterating on an indefinite matrix.
    Krylov.cg!(ws, ls.A, ls.b; linesearch=true)
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return cg_status(ws.stats)
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:minres})
    ws = check_krylov_ws(ls.solver_ws, Krylov.MinresWorkspace, :minres)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    Krylov.minres!(ws, ls.A, ls.b)
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return ws.stats.solved ? Success : NotPositiveDefinite
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:lsmr})
    ws = check_krylov_ws(ls.solver_ws, Krylov.LsmrWorkspace, :lsmr)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    Krylov.lsmr!(ws, ls.A, ls.b)
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return ws.stats.solved ? Success : NotPositiveDefinite
end

solve!(ls::SparseLinearSystem, ::StaticSymbol{:krylov}) = solve!(ls, static(:cg))

# DenseLinearSystem execution
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:cg})
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

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:lsmr})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if ws isa Krylov.LsmrWorkspace
        Krylov.lsmr!(ws, ls.A_scratch, ls.b)
        copyto!(ls.x, ws.x)
        return ws.stats.solved ? Success : NotPositiveDefinite
    end
    (x, stats) = Krylov.lsmr(ls.A_scratch, ls.b)
    copyto!(ls.x, x)
    return stats.solved ? Success : NotPositiveDefinite
end

solve!(ls::DenseLinearSystem, ::StaticSymbol{:krylov}) = solve!(ls, static(:cg))

end # module KrylovExt
