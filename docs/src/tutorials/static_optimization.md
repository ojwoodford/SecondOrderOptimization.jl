# Zero-Allocation Static Optimization

A primary design goal of `SecondOrderOptimization.jl` is zero dynamic memory allocation when optimizing fixed-size or repeated problems. This makes it ideal for real-time robotics, embedded vision, high-frequency filtering, and inner loops of simulation pipelines.

## Static Rosenbrock Example

By using `StaticArrays.SVector` for variables and preallocating an [`OptimizationWorkspace`](@ref), the entire optimization loop runs without a single allocation on the heap.

```julia
using SecondOrderOptimization
using StaticArrays

function rosenbrock_cost(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_fdf!(g, H, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    g[2] = 200.0 * (x[2] - x[1]^2)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
    return rosenbrock_cost(x)
end

x0 = @SVector [-1.2, 1.0]
options = OptimizationOptions(iterator = :levenbergmarquardt)
ws = OptimizationWorkspace(x0, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options)

# Warmup run
x_opt, result = optimize(x0, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options; workspace = ws)

# Verify zero heap allocations
allocs = @allocated optimize(x0, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options; workspace = ws)
println("Heap allocations: ", allocs) # 0 bytes!
```

## Reusing Preallocated Workspaces Across Trials

When calling [`optimize`](@ref) thousands of times in a loop with varying initial states `x_init`, reuse the same [`OptimizationWorkspace`](@ref):

```julia
ws = OptimizationWorkspace(x0, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options)

for trial in 1:1000
    x_init = @SVector [-1.2 + 0.001 * trial, 1.0]
    x_opt, res = optimize(x_init, HessianObjective(rosenbrock_cost, rosenbrock_fdf!), options; workspace = ws)
end
```
