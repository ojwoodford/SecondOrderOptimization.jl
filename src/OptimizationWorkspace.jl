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
h_structure_key(H::SparseMatrixCSC) = (typeof(H), size(H), objectid(H.colptr), objectid(H.rowval), length(H.nzval))

function cachekey(x0, options::OptimizationOptions, H)
    return (typeof(x0), dynamic(nvars(x0)), typeof(options), h_structure_key(H))
end

workspacevalid(ws::OptimizationWorkspace, x0, options::OptimizationOptions, H) = keyvalid(ws.key, x0, options, H)
keyvalid(key, x0, options::OptimizationOptions, H) = false
function keyvalid(key::Tuple, x0, options::OptimizationOptions, H)
    return key == cachekey(x0, options, H)
end

function OptimizationWorkspace(x0, options::OptimizationOptions=OptimizationOptions(); H=nothing, retract! = update!, startstats::Stats=Stats())
    return OptimizationWorkspace(nothing, nothing, x0, options; H=H, retract! = retract!, startstats=startstats)
end

function OptimizationWorkspace(f, fdf!, x0, options::OptimizationOptions=OptimizationOptions(); H=nothing, retract! = update!, startstats::Stats=Stats())
    ls = makelinearsystem(x0, H, options.linsolver)
    data = OptimizationInternal(ls, x0, startstats)
    iteratedata = construct(options.iterator, ls, options.callbackchangesstate)
    key = cachekey(x0, options, H)
    return OptimizationWorkspace{typeof(key), Tuple{typeof(data), typeof(iteratedata)}}(key, (data, iteratedata))
end
