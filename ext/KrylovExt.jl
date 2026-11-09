module KrylovExt

using SecondOrderOptimization
using Krylov
using SparseArrays
using LinearAlgebra
using Static

import SecondOrderOptimization: validatelinsolver, makesparsews, solve!
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

# SparseLinearSystem execution
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cg})
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if !(ws isa Krylov.CgWorkspace)
        ws = Krylov.CgWorkspace(ls.A, ls.b)
        ls.solver_ws = ws
    end
    Krylov.cg!(ws, ls.A, ls.b)
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    if ws.stats.solved
        return Success
    elseif occursin("negative curvature", ws.stats.status)
        return Indefinite
    else
        return NotPositiveDefinite
    end
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:minres})
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if !(ws isa Krylov.MinresWorkspace)
        ws = Krylov.MinresWorkspace(ls.A, ls.b)
        ls.solver_ws = ws
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
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    if !(ws isa Krylov.LsmrWorkspace)
        ws = Krylov.LsmrWorkspace(ls.A, ls.b)
        ls.solver_ws = ws
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
    (x, stats) = Krylov.cg(ls.A_scratch, ls.b)
    copyto!(ls.x, x)
    if stats.solved
        return Success
    elseif occursin("negative curvature", stats.status)
        return Indefinite
    else
        return NotPositiveDefinite
    end
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
    (x, stats) = Krylov.lsmr(ls.A_scratch, ls.b)
    copyto!(ls.x, x)
    return stats.solved ? Success : NotPositiveDefinite
end

solve!(ls::DenseLinearSystem, ::StaticSymbol{:krylov}) = solve!(ls, static(:cg))

end # module KrylovExt
