# The sparse residual-and-Jacobian linear system: A = JᵀJ held implicitly, with J a
# SparseMatrixCSC{Float64, Int}, b = Jᵀr. Each damped solve is the least-squares solution of
# [J; √λ D^½] x = [r; 0] by SuiteSparse's SPQR. The column ordering and symbolic analysis of the
# augmented pattern are computed once, when the system is built; each solve writes the values into
# the augmented matrix in place and runs only SPQR's numeric factorization, Qᵀb and the triangular
# solve. JᵀJ is never formed.
#
# SPQR allocates its working memory inside each of those three calls (Julia's SparseArrays points
# SuiteSparse's malloc at Julia's allocator, so it shows in `@allocated`), and frees it before
# returning: the exception to rules.md 21 decided 2026-11-15. Everything on this side of the calls
# is preallocated. The calls go through SparseArrays' own bindings (`LibSuiteSparse`), which are
# not part of SparseArrays' public API: there is no public route to SPQR's symbolic/numeric split.

const LSS = SparseArrays.LibSuiteSparse

# The SPQR state shared by a system and its `copy_structure` copy (only one of them has a live
# factorization at a time). The augmented matrix [J; √λ D^½] is held in CHOLMOD's zero-based
# layout in Julia arrays, which the `cholmod_sparse` header points into; column j holds J's stored
# entries of column j, then the damping entry in row m + j. The right-hand side [r; 0] likewise.
mutable struct SPQRWorkspace
    const common::Base.RefValue{LSS.cholmod_common}
    const p::Vector{Int}                 # n + 1 column pointers, zero based
    const i::Vector{Int}                 # row indices, zero based
    const x::Vector{Float64}             # values: J's, then √(λ sⱼ) last in each column
    const A::LSS.cholmod_sparse_struct   # header over p, i, x
    const rhs::Vector{Float64}           # m + n
    const B::LSS.cholmod_dense_struct    # header over rhs
    const out::Base.RefValue{Ptr{LSS.cholmod_dense}}   # for freeing SPQR's results
    QR::Ptr{LSS.SuiteSparseQR_C_factorization}
end

function SPQRWorkspace(J::SparseMatrixCSC{Float64, Int})
    m, n = size(J)
    # Julia's SparseArrays sets up SuiteSparse (its malloc, among others) on first use of a common
    SparseArrays.CHOLMOD.getcommon(Int)
    common = Ref(LSS.cholmod_common())
    LSS.cholmod_l_start(common) == 1 || error("cholmod_l_start failed")
    common[].print = 0
    nz = nnz(J) + n
    p = Vector{Int}(undef, n + 1)
    i = Vector{Int}(undef, nz)
    x = zeros(Float64, nz)
    k = 0
    @inbounds for j in 1:n
        p[j] = k
        for q in nzrange(J, j)
            k += 1
            i[k] = J.rowval[q] - 1
        end
        k += 1
        i[k] = m + j - 1
    end
    p[n + 1] = k
    A = LSS.cholmod_sparse_struct()
    A.nrow = m + n; A.ncol = n; A.nzmax = nz
    A.p = pointer(p); A.i = pointer(i); A.nz = C_NULL; A.x = pointer(x); A.z = C_NULL
    A.stype = 0; A.itype = LSS.CHOLMOD_LONG; A.xtype = LSS.CHOLMOD_REAL; A.dtype = LSS.CHOLMOD_DOUBLE
    A.sorted = 1; A.packed = 1
    rhs = zeros(Float64, m + n)
    B = LSS.cholmod_dense_struct()
    B.nrow = m + n; B.ncol = 1; B.nzmax = m + n; B.d = m + n
    B.x = pointer(rhs); B.z = C_NULL; B.xtype = LSS.CHOLMOD_REAL; B.dtype = LSS.CHOLMOD_DOUBLE
    ws = SPQRWorkspace(common, p, i, x, A, rhs, B, Ref(Ptr{LSS.cholmod_dense}(C_NULL)), C_NULL)
    GC.@preserve ws begin
        ws.QR = LSS.SuiteSparseQR_C_symbolic(LSS.SPQR_ORDERING_DEFAULT, 1, spqr_A(ws), common)
    end
    ws.QR == C_NULL && error("SPQR's symbolic analysis failed (status $(common[].status))")
    finalizer(free_spqr!, ws)
    return ws
end

spqr_A(ws::SPQRWorkspace) = Ptr{LSS.cholmod_sparse}(pointer_from_objref(ws.A))
spqr_B(ws::SPQRWorkspace) = Ptr{LSS.cholmod_dense}(pointer_from_objref(ws.B))

function free_spqr!(ws::SPQRWorkspace)
    if ws.QR != C_NULL
        LSS.SuiteSparseQR_C_free(Ref(ws.QR), ws.common)
        ws.QR = C_NULL
    end
    LSS.cholmod_l_finish(ws.common)
    return nothing
end

# The x array of a cholmod_dense that SPQR returned, read without building a Julia copy of the header
const DENSE_X_OFFSET = fieldoffset(LSS.cholmod_dense_struct, findfirst(==(:x), fieldnames(LSS.cholmod_dense_struct)))
dense_x(D::Ptr{LSS.cholmod_dense}) = unsafe_load(Ptr{Ptr{Float64}}(D + DENSE_X_OFFSET))

function free_dense!(ws::SPQRWorkspace, D::Ptr{LSS.cholmod_dense})
    ws.out[] = D
    LSS.cholmod_l_free_dense(ws.out, ws.common)
    return nothing
end

# The per-system state: squared column norms of J and whether they are current
mutable struct SparseTallWorkspace
    const spqr::Union{Nothing, SPQRWorkspace}   # nothing for an iterative solver
    const colnorm2::Vector{Float64}
    const nnzJ::Int               # stored entries of J, which resjac! must not change
    normsstale::Bool
    nfactor::Int                  # number of SPQR numeric factorizations, for tests
end
SparseTallWorkspace(spqr, J::SparseMatrixCSC) = SparseTallWorkspace(spqr, Vector{Float64}(undef, size(J, 2)), nnz(J), true, 0)

"""
    SparseTallHessian

The Hessian storage of a [`SparseTallLinearSystem`](@ref): the sparse Jacobian `J` (`m × n`) and
the residual `r` (length `m`), standing for `A = JᵀJ`, which is never formed.
"""
struct SparseTallHessian
    J::SparseMatrixCSC{Float64, Int}
    r::Vector{Float64}
    Jv::Vector{Float64}           # scratch for quadform
    ws::SparseTallWorkspace
end

Base.size(A::SparseTallHessian) = (size(A.J, 2), size(A.J, 2))
Base.size(A::SparseTallHessian, d::Integer) = size(A)[d]
markstale!(A::SparseTallHessian) = (A.ws.normsstale = true; A)

# The linearization changed J's stored entries if resjac! wrote outside the pattern (which
# inserts entries): the augmented pattern SPQR analysed would no longer be J's
function checkpattern(A::SparseTallHessian)
    nnz(A.J) == A.ws.nnzJ || throw(ArgumentError("resjac! changed the sparsity pattern of J (it had $(A.ws.nnzJ) stored entries, now $(nnz(A.J))): write only entries the prototype stores."))
    return nothing
end

# y = J v, accumulated column by column over the stored entries: plain loops (rules.md 23)
function jmul!(y::AbstractVector, J::SparseMatrixCSC{Float64, Int}, v::AbstractVector)
    fill!(y, 0.0)   # each entry of J v sums over a row's stored entries, so it starts from zero
    rv, nzv = rowvals(J), nonzeros(J)
    @inbounds for j in axes(J, 2)
        vj = v[j]
        for q in nzrange(J, j)
            y[rv[q]] += nzv[q] * vj
        end
    end
    return y
end

# ‖J v‖²
function quadform(A::SparseTallHessian, v::AbstractVector)
    Jv = jmul!(A.Jv, A.J, v)
    total = 0.0
    @inbounds @simd for i in eachindex(Jv)
        total += Jv[i] * Jv[i]
    end
    return total
end

# g = Jᵀ r, one sparse dot product per column (rules.md 23)
function jtr!(g::AbstractVector, J::SparseMatrixCSC{Float64, Int}, r::Vector{Float64})
    rv, nzv = rowvals(J), nonzeros(J)
    @inbounds for j in axes(J, 2)
        total = 0.0
        for q in nzrange(J, j)
            total += nzv[q] * r[rv[q]]
        end
        g[j] = total
    end
    return g
end

function colnorms2!(A::SparseTallHessian)
    ws = A.ws
    if ws.normsstale
        nzv = nonzeros(A.J)
        @inbounds for j in axes(A.J, 2)
            total = 0.0
            for q in nzrange(A.J, j)
                total += nzv[q] * nzv[q]
            end
            ws.colnorm2[j] = total
        end
        ws.normsstale = false
    end
    return ws.colnorm2
end

function max_diag(A::SparseTallHessian)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

"""
    SparseTallLinearSystem(J::SparseMatrixCSC{Float64, Int})

The linear system of a [`JacobianObjective`](@ref) with a sparse Jacobian prototype: the normal
equations `(JᵀJ + λD) x = Jᵀr`, with `JᵀJ` never formed. `getA(ls)` is a
[`SparseTallHessian`](@ref) holding `J` (the given matrix, used as storage) and `r`; `getb(ls)`
is the gradient `Jᵀr`.

`solve!(ls, :auto)` (or `:qr`) solves `min ‖[J; √λ D^½] x − [r; 0]‖` by SuiteSparse's SPQR. The
column ordering and symbolic analysis of the augmented pattern are done once, at construction;
each solve runs SPQR's numeric factorization, so a retry with another `λ` costs a full numeric
factorization (unlike the dense system, which refactors only an `n × n` block). SPQR allocates
its working memory inside each solve and frees it again (the stated exception to "nothing
allocates in the loop"); nothing else allocates after construction.

As for [`DenseTallLinearSystem`](@ref): an undamped solve on a numerically rank-deficient `J`
(SPQR's rank estimate below `n` at a column-norm tolerance of `max(m + n, n)·eps` times the
largest column norm of the augmented matrix, which includes `m < n`) returns
`NotPositiveDefinite`, as does a damped one whose augmented matrix is rank deficient at that
tolerance, and a `NaN` or `Inf` in `J` or `r` returns `NotPositiveDefinite` with `x` set to
`NaN`. `zero!` does not zero `J` or `r`. `resjac!` must write only entries the prototype stores:
writing any other entry inserts it, which changes the pattern, and the linearization throws an
`ArgumentError`.
"""
mutable struct SparseTallLinearSystem{W} <: AbstractTallLinearSystem
    const A::SparseTallHessian
    const b::Vector{Float64}
    const x::Vector{Float64}
    const solver_ws::W            # an iterative solver's workspace (`maketallws`), or nothing for SPQR
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
function SparseTallLinearSystem(J::SparseMatrixCSC{Float64, Int}; linsolver = :auto)
    m, n = size(J)
    spqr = needsqr(linsolver) ? SPQRWorkspace(J) : nothing   # an iterative solver needs no analysis
    A = SparseTallHessian(J, zeros(m), zeros(m), SparseTallWorkspace(spqr, J))
    return SparseTallLinearSystem(A, zeros(n), zeros(n), maketallws(m, n, linsolver), 0.0, nothing)
end

tall_linearsystem(x0, J::SparseMatrixCSC{Float64, Int}, linsolver) = SparseTallLinearSystem(J; linsolver)
workspace_tall_linearsystem(x0, J::SparseMatrixCSC{Float64, Int}, linsolver) = SparseTallLinearSystem(copy(J); linsolver)

zero!(ls::SparseTallLinearSystem) = (fill!(ls.b, 0.0); markstale!(ls.A); ls)

# A reused workspace's J has the prototype's pattern (the workspace key checks it), so only the
# values are set
function initjacobian!(ls::SparseTallLinearSystem, P::SparseMatrixCSC{Float64, Int})
    ls.A.J === P || copyto!(nonzeros(ls.A.J), nonzeros(P))
    markstale!(ls.A)
    return ls
end

# The trial system: its own J (same pattern and entries), r, b and column norms; the SPQR state
# is shared
function copy_structure(ls::SparseTallLinearSystem)
    A = ls.A
    m, n = size(A.J)
    J = copy(A.J)
    At = SparseTallHessian(J, zeros(m), zeros(m), SparseTallWorkspace(A.ws.spqr, J))
    return SparseTallLinearSystem(At, zeros(length(ls.b)), ls.x, ls.solver_ws, ls.lambda, ls.scaling)
end
sharex(ls::SparseTallLinearSystem, from) = ls.x === from.x ? ls : SparseTallLinearSystem(ls.A, ls.b, from.x, ls.solver_ws, ls.lambda, ls.scaling)

function (t::TallLinearize)(g::AbstractVector, A::SparseTallHessian, x)
    cost = t.resjac!(A.r, A.J, x)
    checkpattern(A)
    jtr!(g, A.J, A.r)
    markstale!(A)
    return cost
end

function solve!(ls::SparseTallLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}})
    A = ls.A
    J, r = A.J, A.r
    m, n = size(J)
    if !(all(isfinite, nonzeros(J)) && all(isfinite, r))
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    sp = A.ws.spqr
    isnothing(sp) && solvermismatch(ls, :qr)
    lambda = max(ls.lambda, 0.0)
    scaling = ls.scaling
    nzv = nonzeros(J)
    # The augmented values, and the largest squared column norm of the augmented matrix
    maxnorm2 = 0.0
    k = 0
    @inbounds for j in 1:n
        total = 0.0
        for q in nzrange(J, j)
            k += 1
            sp.x[k] = nzv[q]
            total += nzv[q] * nzv[q]
        end
        d2 = lambda * (isnothing(scaling) ? 1.0 : scaling[j])
        k += 1
        sp.x[k] = sqrt(d2)
        maxnorm2 = max(maxnorm2, total + d2)
    end
    @inbounds for i in 1:m
        sp.rhs[i] = r[i]
    end
    @inbounds for i in m+1:m+n
        sp.rhs[i] = 0.0
    end
    tol = max(m + n, n) * eps(Float64) * sqrt(maxnorm2)
    GC.@preserve sp begin
        ok = LSS.SuiteSparseQR_C_numeric(tol, spqr_A(sp), sp.QR, sp.common)
        ok == 1 || error("SPQR's numeric factorization failed (status $(sp.common[].status))")
        A.ws.nfactor += 1
        # SPQR_istat[4] (zero based) is the rank estimate at `tol`
        sp.common[].SPQR_istat[5] < n && return NotPositiveDefinite
        Y = LSS.SuiteSparseQR_C_qmult(LSS.SPQR_QTX, sp.QR, spqr_B(sp), sp.common)
        Y == C_NULL && error("SPQR's Qᵀb failed (status $(sp.common[].status))")
        X = LSS.SuiteSparseQR_C_solve(LSS.SPQR_RETX_EQUALS_B, sp.QR, Y, sp.common)
        free_dense!(sp, Y)
        X == C_NULL && error("SPQR's triangular solve failed (status $(sp.common[].status))")
        xs = dense_x(X)
        @inbounds for i in 1:n
            ls.x[i] = unsafe_load(xs, i)
        end
        free_dense!(sp, X)
    end
    return Success
end

function solve!(ls::SparseTallLinearSystem, ::StaticSymbol{S}) where S
    throw(ArgumentError("linsolver = :$S is not supported by a SparseTallLinearSystem; use :auto or :qr."))
end
