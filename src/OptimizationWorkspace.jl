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
        empty!(ws)
    end
    return ws
end

h_structure_key(::Nothing) = nothing
h_structure_key(H::AbstractMatrix) = (typeof(H), size(H))
h_structure_key(H::SparseMatrixCSC) = (typeof(H), size(H), hash(H.colptr), hash(H.rowval), length(H.nzval))

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
    return key == cachekey(x0, options, H)
end

function OptimizationWorkspace(x0, options::OptimizationOptions=OptimizationOptions(); H=nothing, startstats::Stats=Stats())
    ls = makelinearsystem(x0, H, options.linsolver)
    data = OptimizationInternal(ls, x0, startstats)
    iteratedata = construct(options.iterator, ls, options.callbackchangesstate)
    key = cachekey(x0, options, H)
    return OptimizationWorkspace{typeof(key), Tuple{typeof(data), typeof(iteratedata)}}(key, (data, iteratedata))
end

# Backward compatibility overload
OptimizationWorkspace(f, fdf!, x0, options::OptimizationOptions=OptimizationOptions(); H=nothing, startstats::Stats=Stats()) =
    OptimizationWorkspace(x0, options; H=H, startstats=startstats)
