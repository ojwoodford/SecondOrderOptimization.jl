module PardisoExt

using SecondOrderOptimization
using Pardiso
using SparseArrays
using Static

import SecondOrderOptimization: validatelinsolver, makesparsews, solve!
import SecondOrderOptimization: AbstractLinearSystem, SparseLinearSystem, DenseLinearSystem
import SecondOrderOptimization: SolveStatus, Success, NotPositiveDefinite, Indefinite
import SecondOrderOptimization: uniformscaling!, damp_marquardt!

# Validation of Pardiso solver symbol
validatelinsolver(::StaticSymbol{:pardiso}) = true

function get_or_create_pardiso(ws)
    if ws isa Pardiso.AbstractPardisoSolver
        return ws
    end
    if Pardiso.mkl_is_available()
        ps = Pardiso.MKLPardisoSolver()
    elseif Pardiso.panua_is_available()
        ps = Pardiso.PardisoSolver()
    else
        error("Pardiso.jl is loaded, but neither Intel MKL Pardiso nor Panua Pardiso native shared library is available.")
    end
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    Pardiso.pardisoinit(ps)
    return ps
end

function makesparsews(A, ::StaticSymbol{:pardiso})
    if Pardiso.mkl_is_available()
        ps = Pardiso.MKLPardisoSolver()
    elseif Pardiso.panua_is_available()
        ps = Pardiso.PardisoSolver()
    else
        return nothing
    end
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    Pardiso.pardisoinit(ps)
    return ps
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:pardiso})
    ps = get_or_create_pardiso(ls.solver_ws)
    ls.solver_ws = ps
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    status = Success
    try
        Pardiso.solve!(ps, ls.x, ls.A, ls.b)
    catch e
        if isa(e, Pardiso.PardisoPosDefException)
            status = NotPositiveDefinite
        elseif isa(e, Pardiso.PardisoException)
            status = NotPositiveDefinite
        else
            rethrow(e)
        end
    finally
        if ls.lambda > 0
            @inbounds for (i, idx) in enumerate(ls.diag_inds)
                ls.A.nzval[idx] = ls.diag_cache[i]
            end
        end
    end
    return status
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:pardiso})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ps = get_or_create_pardiso(nothing)
    A_sp = sparse(ls.A_scratch)
    try
        Pardiso.solve!(ps, ls.x, A_sp, ls.b)
        return Success
    catch e
        if isa(e, Union{Pardiso.PardisoPosDefException, Pardiso.PardisoException})
            return NotPositiveDefinite
        else
            rethrow(e)
        end
    end
end

end # module PardisoExt
