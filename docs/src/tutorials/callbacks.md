# Callbacks & Step Monitoring

`SecondOrderOptimization.jl` provides fine-grained control over the optimization process through callback functions.

## Defining a Callback

A callback has the 4-argument signature:
```julia
function my_callback(cost, x_trial, data::OptimizationInternal, iteratedata)
    # Return a tuple (cost, terminate_flags):
    # terminate_flags == 0 -> continue optimization normally
    # terminate_flags != 0 -> stop optimization (stored in result.termination >> 16)
    return (cost, 0)
end
```

> [!NOTE]
> If your callback modifies the optimization state or returns a modified `cost`, pass `callbackchangesstate = true` in [`OptimizationOptions`](@ref).

## Built-In Callbacks

- [`nullcallback`](@ref): Default no-op callback (zero overhead).
- [`printoutcallback`](@ref): Prints iteration number, cost, cost reduction, step norm, and trust-region radius.
- [`storecostscallback`](@ref): Appends per-iteration costs to a `Vector{Float64}` or records costs, timestamps, and step vectors in a [`CostTrajectory`](@ref).

## Example: Recording a `CostTrajectory` and Custom Termination

```julia
using SecondOrderOptimization
using StaticArrays

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function fdf!(g, H, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    g[2] = 200.0 * (x[2] - x[1]^2)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    return f(x)
end

x0 = @SVector [-1.2, 1.0]
options = OptimizationOptions(iterator = :dogleg)

# 1. Using CostTrajectory
traj = CostTrajectory()
x_opt, result = optimize(f, fdf!, x0, options; callback = storecostscallback(traj))
println("Recorded ", length(traj.costs), " iterations.")

# 2. Custom early-stopping callback
function stop_below_1e4(cost, x_trial, data, iteratedata)
    terminate = cost < 1e-4 ? 1 : 0
    return (cost, terminate)
end
x_early, res_early = optimize(f, fdf!, x0, options; callback = stop_below_1e4)
println("User termination flag: ", res_early.termination >> 16)
```
