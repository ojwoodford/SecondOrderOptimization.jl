# The functions an iterator calls on the loop's state `data` (an `OptimizationInternal`), so that an iterator reads no
# field of it and calls no function of the linear-system contract (`SecondOrderOptimization.IteratorInterface`). Each
# is a thin method over the linear system's function; the timing of the iterator's work lives here (in `solvestep!`,
# `trialcost!` and `linearizetrial!`) and nowhere else.

@inline function solvestep!(data::OptimizationInternal, options::OptimizationOptions)
    status = Success
    data.solves += @elapsed_ns_if collectsstats(data) status = solve!(data.linsystem, options.linsolver)
    status == Success || return false
    negate!(getx(data.linsystem))   # a successful solve's x is finite (DampedLinearSolve's contract)
    return true
end
@inline function solvestep!(data::OptimizationInternal, options::OptimizationOptions, λ)
    dampen!(data.linsystem, λ)
    return solvestep!(data, options)
end
@inline function solvestep!(data::OptimizationInternal, options::OptimizationOptions, λ, scaling)
    dampen!(data.linsystem, λ, scaling)
    return solvestep!(data, options)
end
@inline undamp!(data::OptimizationInternal) = (dampen!(data.linsystem, 0.0); nothing)
@inline systemisfinite(data::OptimizationInternal) = isfinitesystem(data.linsystem)
@inline systemeltype(data::OptimizationInternal) = systemeltype(data.linsystem)
@inline dampingfloor(data::OptimizationInternal) = dampingfloor(data.linsystem)
@inline getgradient(data::OptimizationInternal) = getb(data.linsystem)
@inline getstep(data::OptimizationInternal) = getx(data.linsystem)
@inline similarstep(ls::AbstractLinearSystem) = similar(getx(ls))
@inline predictedchange(data::OptimizationInternal, s) = quadratic_model_change(data.linsystem, s)
@inline curvature(data::OptimizationInternal, v) = quadform(getA(data.linsystem), v)
@inline maxdiagonal(data::OptimizationInternal) = max_diag(data.linsystem)
@inline getdiagonal!(out, data::OptimizationInternal) = getdiag!(out, data.linsystem)
@inline getbestcost(data::OptimizationInternal) = data.bestcost
# The assignment, because an immutable variable is rebuilt by `update!`
@inline settrial!(data::OptimizationInternal, s = getstep(data)) = (data.varnext = update!(data.varnext, data.variables, s); nothing)
@inline function trialcost!(data::OptimizationInternal, f)
    data.costs += @elapsed_ns_if collectsstats(data) cost = f(data.varnext)
    return cost
end
function computelinearization!(data, fdf!, x)
    zero!(data.linsystem)
    cost = 0.0
    data.gradients += @elapsed_ns_if collectsstats(data) cost = fdf!(getb(data.linsystem), getA(data.linsystem), x)
    return cost
end
@inline function linearizetrial!(data::OptimizationInternal, fdf!)
    zero!(data.linsystem)
    data.gradients += @elapsed_ns_if collectsstats(data) cost = fdf!(getb(data.linsystem), getA(data.linsystem), data.varnext)
    return cost
end
