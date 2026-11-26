# The shared trial linearization (IteratorInterface): an iterator that expects its trial point to be accepted
# linearizes there, into a spare system, while it evaluates the cost; the next `linearize!` swaps that system in
# instead of linearizing again. Built according to `callbackchangesstate`: a callback that may change the state may
# also change the point or the cost after the trial, so then the component holds no spare and always evaluates `f`.
mutable struct TrialLinearization{L}
    trial::L          # the spare linear system, or nothing
    cost::Float64     # the cost at the trial point, when ready
    ready::Bool       # whether `trial` holds the linearization at the trial point the loop will move to
end
TrialLinearization(ls, ::True) = TrialLinearization{Nothing}(nothing, NaN, false)
TrialLinearization(ls, ::False) = TrialLinearization(copy_structure(ls), NaN, false)

reset!(tl::TrialLinearization) = (tl.ready = false; tl.cost = NaN; nothing)
discardtrial!(tl::TrialLinearization) = (tl.ready = false; nothing)

@inline trialcost!(tl::TrialLinearization{Nothing}, data::OptimizationInternal, f, fdf!, likely::Bool) = trialcost!(data, f)
@inline function trialcost!(tl::TrialLinearization, data::OptimizationInternal, f, fdf!, likely::Bool)
    likely || return trialcost!(data, f)
    tl.trial = sharex(tl.trial, data.linsystem)
    zero!(tl.trial)
    data.gradients += @elapsed_ns_if collectsstats(data) cost = fdf!(getb(tl.trial), getA(tl.trial), data.varnext)
    tl.ready = true
    tl.cost = cost
    return cost
end

# The iterator's trial linearization, if it holds one
triallinearization(itdata) = nothing

# The default `linearize!`: swap in a ready trial linearization, otherwise linearize at the current point
linearize!(itdata, data, fdf!) = linearizefrom!(triallinearization(itdata), data, fdf!)
linearizefrom!(::Union{Nothing, TrialLinearization{Nothing}}, data, fdf!) = computelinearization!(data, fdf!, data.variables)
function linearizefrom!(tl::TrialLinearization, data, fdf!)
    tl.ready || return computelinearization!(data, fdf!, data.variables)
    tl.ready = false
    trial = sharex(tl.trial, data.linsystem)
    tl.trial = data.linsystem
    data.linsystem = trial
    return tl.cost
end

# The spare system, for whatever prepares a run's systems (`initlinearization!`), or nothing
trialsystem(itdata) = trialsystem(triallinearization(itdata))
trialsystem(::Nothing) = nothing
trialsystem(tl::TrialLinearization) = tl.trial
