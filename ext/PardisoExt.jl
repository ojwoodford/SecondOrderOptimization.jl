module PardisoExt

using SecondOrderOptimization
using Pardiso
using SparseArrays
using Static

import SecondOrderOptimization: validatelinsolver, makesparsews, solve!
import SecondOrderOptimization: AbstractLinearSystem, SparseLinearSystem, DenseLinearSystem
import SecondOrderOptimization: SolveStatus, Success, NotPositiveDefinite, Indefinite
import SecondOrderOptimization: uniformscaling!, damp_marquardt!, ensure_diag_and_inds

# Validation of Pardiso solver symbol
validatelinsolver(::StaticSymbol{:pardiso}) = true

pardiso_is_available() = Pardiso.mkl_is_available() || Pardiso.panua_is_available()

function pardiso_unavailable_error()
    error("Pardiso.jl is loaded, but neither Intel MKL Pardiso nor Panua Pardiso native shared library is available.")
end

# Pardiso reads a symmetric matrix as the upper triangle of a CSR matrix, which is the lower
# triangle of the same matrix in CSC. `L` holds that triangle, in Pardiso's index type, and
# `L.nzval[k] == A.nzval[map[k]]`, so the values are refreshed without reallocating.
function lower_triangle_with_map(A::SparseMatrixCSC{T}, ::Type{Ti}) where {T, Ti}
    n = size(A, 2)
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Ti[]
    map = Int[]
    colptr[1] = 1
    for col in 1:n
        for idx in nzrange(A, col)
            r = A.rowval[idx]
            if r >= col
                push!(rowval, r)
                push!(map, idx)
            end
        end
        colptr[col+1] = length(rowval) + 1
    end
    L = SparseMatrixCSC{T, Ti}(size(A, 1), n, colptr, rowval, A.nzval[map])
    return L, map
end

# Workspace for the phased Pardiso API: the symbolic analysis runs once, here, and each solve
# runs only the numerical factorization and the solve. It is shared by a system and its
# `copy_structure` copies, which have the same pattern.
struct PardisoWorkspace{PS <: Pardiso.AbstractPardisoSolver, T, Ti}
    ps::PS
    L::SparseMatrixCSC{T, Ti}
    map::Vector{Int}
end

function PardisoWorkspace(A::SparseMatrixCSC{Float64})
    ps = Pardiso.mkl_is_available() ? Pardiso.MKLPardisoSolver() : Pardiso.PardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    Pardiso.pardisoinit(ps)   # also registers a finalizer that releases Pardiso's memory
    Pardiso.fix_iparm!(ps, :N)
    L, map = lower_triangle_with_map(A, eltype(ps.iparm))
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, L, zeros(Float64, size(A, 1)))
    return PardisoWorkspace(ps, L, map)
end

function makesparsews(A, ::StaticSymbol{:pardiso})
    pardiso_is_available() || return nothing
    return PardisoWorkspace(A)
end

# The sparse workspace is built once, by `makesparsews`, and `solver_ws` is a `const` field,
# so it cannot be replaced here.
function check_pardiso_ws(ws)
    ws isa PardisoWorkspace && return ws
    pardiso_is_available() || pardiso_unavailable_error()
    throw(ArgumentError("This linear system was not built for linsolver=:pardiso; construct it with `makelinearsystem(x0, H, :pardiso)`."))
end

# Factorize and solve with the current (already damped) values of `A`, whose pattern must be
# the one `ws` was analysed with
function pardiso_solve!(ws::PardisoWorkspace, x, A::SparseMatrixCSC, b)
    @inbounds for (k, idx) in enumerate(ws.map)
        ws.L.nzval[k] = A.nzval[idx]
    end
    try
        Pardiso.set_phase!(ws.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ws.ps, ws.L, b)
    catch e
        # A zero or negative pivot in the positive-definite factorization
        isa(e, Union{Pardiso.PardisoPosDefException, Pardiso.PardisoException}) || rethrow()
        return NotPositiveDefinite
    end
    Pardiso.set_phase!(ws.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ws.ps, x, ws.L, b)
    return Success
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:pardiso})
    ws = check_pardiso_ws(ls.solver_ws)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    try
        return pardiso_solve!(ws, ls.x, ls.A, ls.b)
    finally
        if ls.lambda > 0
            @inbounds for (i, idx) in enumerate(ls.diag_inds)
                ls.A.nzval[idx] = ls.diag_cache[i]
            end
        end
    end
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:pardiso})
    pardiso_is_available() || pardiso_unavailable_error()
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    # Pardiso needs every diagonal entry stored, and `sparse` drops zeros
    A_sp = first(ensure_diag_and_inds(sparse(ls.A_scratch)))
    ws = PardisoWorkspace(A_sp)
    try
        return pardiso_solve!(ws, ls.x, A_sp, ls.b)
    finally
        Pardiso.set_phase!(ws.ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ws.ps)
    end
end

end # module PardisoExt
