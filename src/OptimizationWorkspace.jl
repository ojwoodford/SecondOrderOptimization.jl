mutable struct OptimizationWorkspace{K, D}
    key::Union{Nothing, K}
    data::Union{Nothing, D}
end
OptimizationWorkspace() = OptimizationWorkspace{Any, Any}(nothing, nothing)

Base.empty!(ws::OptimizationWorkspace) = (ws.key = nothing; ws.data = nothing; ws)
Base.isempty(ws::OptimizationWorkspace) = isnothing(ws.data)

function storeworkspace!(ws::OptimizationWorkspace{K, D}, key, data) where {K, D}
    if key isa K && data isa D
        ws.key = key
        ws.data = data
    else
        @warn "OptimizationWorkspace{$K, $D} can't hold the data for this problem, so it was emptied and every call with it pays the full setup cost. Use `OptimizationWorkspace()` for a workspace that adapts to any problem." maxlog=1
        empty!(ws)
    end
    return ws
end

h_structure_key(::Nothing) = nothing
h_structure_key(H::AbstractMatrix) = (typeof(H), size(H))

# The structure of a sparse Hessian. It keeps the arrays it was built from so that a later call
# passing the same `H` (or one sharing its `colptr` and `rowval`) matches without rehashing the
# O(nnz) pattern; any other matrix is compared by hashes.
struct SparseStructureKey{M <: SparseMatrixCSC, Ti}
    size::Tuple{Int, Int}
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    hcolptr::UInt
    hrowval::UInt
    nnz::Int
end
h_structure_key(H::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    SparseStructureKey{typeof(H), Ti}(size(H), H.colptr, H.rowval, hash(H.colptr), hash(H.rowval), length(H.nzval))

function Base.:(==)(a::SparseStructureKey, b::SparseStructureKey)
    return typeof(a) === typeof(b) && a.size == b.size && a.nnz == b.nnz && a.hcolptr == b.hcolptr && a.hrowval == b.hrowval
end
Base.hash(k::SparseStructureKey, h::UInt) = hash((typeof(k), k.size, k.nnz, k.hcolptr, k.hrowval), h)

hkeyvalid(key, H) = key == h_structure_key(H)
function hkeyvalid(key::SparseStructureKey{M}, H::M) where M <: SparseMatrixCSC
    (key.size == size(H) && key.nnz == length(H.nzval)) || return false
    # Same arrays: the pattern is the one that was hashed. Inserting or removing stored entries
    # in place changes the entry count, which was checked above.
    key.colptr === H.colptr && key.rowval === H.rowval && return true
    return key.hcolptr == hash(H.colptr) && key.hrowval == hash(H.rowval)
end

# A `Vector` of variable blocks of one type and total dimension can still differ in the number
# and sizes of its blocks (e.g. two 3-vectors versus three 2-vectors), so key on those too.
blocklayout_key(x0) = nothing
blocklayout_key(x0::Vector{<:Number}) = nothing
function blocklayout_key(x0::Vector)
    h = hash(length(x0))
    for xi in x0
        h = hash(dynamic(nvars(xi)), h)
    end
    return (length(x0), h)
end

function cachekey(x0, options::OptimizationOptions, H)
    return (typeof(x0), dynamic(nvars(x0)), blocklayout_key(x0), typeof(options), h_structure_key(H))
end

workspacevalid(ws::OptimizationWorkspace, x0, options::OptimizationOptions, H) = keyvalid(ws.key, x0, options, H)
keyvalid(key, x0, options::OptimizationOptions, H) = false
function keyvalid(key::Tuple, x0, options::OptimizationOptions, H)
    key[1] === typeof(x0) || return false
    key[2] == dynamic(nvars(x0)) || return false
    key[3] == blocklayout_key(x0) || return false
    key[4] === typeof(options) || return false
    return hkeyvalid(key[5], H)
end

function OptimizationWorkspace(x0, obj::HessianObjective, options::OptimizationOptions=OptimizationOptions(); startstats::Stats=Stats())
    H = obj.H
    checkhessian(x0, H)
    ls = makelinearsystem(x0, H, options.linsolver)
    data = OptimizationInternal(ls, x0, startstats)
    iteratedata = construct(options.iterator, ls, options.callbackchangesstate)
    key = cachekey(x0, options, H)
    return OptimizationWorkspace{typeof(key), Tuple{typeof(data), typeof(iteratedata)}}(key, (data, iteratedata))
end
