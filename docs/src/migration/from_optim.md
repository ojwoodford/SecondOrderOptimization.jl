# Migrating from Optim.jl

[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is a general-purpose optimization library in the Julia ecosystem. Here is a guide comparing concepts and migrating second-order optimization code to `SecondOrderOptimization.jl`.

## Conceptual Differences

| Feature | Optim.jl | SecondOrderOptimization.jl |
| :--- | :--- | :--- |
| **Primary Focus** | General first & second order, line search | High-performance second-order trust-region & LM |
| **Combined Evaluation** | Separate `f`, `g!`, `h!` or `only_fgh!` | Combined `cost = fdf!(g, H, x)` single pass |
| **Static Allocations** | Allocations on state/results | Zero heap allocations on `StaticArrays` with [`OptimizationWorkspace`](@ref) |
| **Manifolds** | `Manifold` wrapper | Native `nvars`/`update` or `retract!(to, from, step)` |
| **Iteration Algorithms** | `Newton()`, `NewtonTrustRegion()` | `:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`, `:newton`, `:gradientdescent` |

## Code Translation Example

### In Optim.jl:
```julia
using Optim

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    G[2] = 200.0 * (x[2] - x[1]^2)
end
function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

res = optimize(f, g!, h!, [-1.2, 1.0], NewtonTrustRegion())
```

### In SecondOrderOptimization.jl:
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
x_opt, res = optimize(f, fdf!, x0, options)
```
Notice how `fdf!` allows sharing intermediate terms between cost, gradient, and Hessian computation in a single pass.
