using LDLFactorizations, LinearAlgebra, StaticArrays, SparseArrays, Static

@enum OptimizationLinearSolver auto_solver cholesky_solver ldl_solver qr_solver

linsolver_symbol(s::Symbol) = s
function linsolver_symbol(s::OptimizationLinearSolver)
    s == auto_solver && return :auto
    s == cholesky_solver && return :cholesky
    s == ldl_solver && return :ldl
    return :qr
end

function try_cholesky!(x::StaticVector, A::StaticMatrix, b::StaticVector)
    A_ = SMatrix(A)
    b_ = SVector(b)
    cholfac = StaticArrays._cholesky(Size(A_), A_, false)
    if issuccess(cholfac)
        return x .= cholfac \ b_
    end
    return x .= A_ \ b_
end

function try_cholesky!(x, A, b)
    cholfac = cholesky(A; check=false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    return ldiv!(x, qr(A), b)
end

# Linear solve dispatching on solver symbol: :auto, :cholesky, :ldl, :qr
solve!(ls) = solve!(ls, static(:auto))
solve!(ls, solver::OptimizationLinearSolver) = solve!(ls, static(linsolver_symbol(solver)))

# StaticLinearSystem
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:auto})
    try_cholesky!(ls.x, ls.A, ls.b)
    return ls.x
end
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:cholesky})
    try_cholesky!(ls.x, ls.A, ls.b)
    return ls.x
end
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:ldl})
    A_ = SMatrix(ls.A)
    b_ = SVector(ls.b)
    ls.x .= bunchkaufman(A_) \ b_
    return ls.x
end
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:qr})
    A_ = SMatrix(ls.A)
    b_ = SVector(ls.b)
    ls.x .= A_ \ b_
    return ls.x
end

# DenseLinearSystem
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:auto})
    try_cholesky!(ls.x, ls.A, ls.b)
    return ls.x
end
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:cholesky})
    try_cholesky!(ls.x, ls.A, ls.b)
    return ls.x
end
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:ldl})
    bk = bunchkaufman(Symmetric(ls.A); check=false)
    if issuccess(bk)
        ldiv!(ls.x, bk, ls.b)
    else
        ldiv!(ls.x, qr(ls.A), ls.b)
    end
    return ls.x
end
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:qr})
    ldiv!(ls.x, qr(ls.A), ls.b)
    return ls.x
end

# SparseLinearSystem
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:auto})
    ldiv!(ls.x, ldl_factorize!(ls.A, ls.solver_ws), ls.b)
    return ls.x
end
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:ldl})
    ldiv!(ls.x, ldl_factorize!(ls.A, ls.solver_ws), ls.b)
    return ls.x
end
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cholesky})
    cholfac = SparseArrays.cholesky(ls.A; check=false)
    if issuccess(cholfac)
        ls.x .= cholfac \ ls.b
    else
        ls.x .= qr(ls.A) \ ls.b
    end
    return ls.x
end
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:qr})
    ls.x .= qr(ls.A) \ ls.b
    return ls.x
end

# Generic matrix fallback for direct solve!(x, A, b)
solve!(x, A, b) = try_cholesky!(x, A, b)
solve!(x, A::SparseMatrixCSC, b) = ldiv!(x, ldl(A), b)
solve!(x::StaticVector, A::StaticMatrix, b::StaticVector) = try_cholesky!(x, A, b)

# Symmetric matrix inversion
invsym(A::AbstractMatrix) = inv(bunchkaufman(A))
invsym(A::StaticMatrix) = Size(A)[1] <= 14 ? inv(A) : inv(bunchkaufman(A))
