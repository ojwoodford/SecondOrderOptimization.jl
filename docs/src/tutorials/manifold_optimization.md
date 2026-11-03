# Manifold Optimization

`SecondOrderOptimization.jl` natively supports optimization over non-Euclidean manifolds, such as $SO(3)$, $SE(3)$, unit spheres $\mathcal{S}^n$, or custom domain-constrained parameter structs.

The solver computes steps in local tangent coordinates ($\Delta x \in T_x \mathcal{M}$) and applies them via either:
1. Extending [`SecondOrderOptimization.nvars`](@ref) and [`SecondOrderOptimization.update`](@ref) for a custom variable type, or
2. Passing a 3-argument retraction `retract!(to, from, step) -> to` to [`optimize`](@ref).

## Method 1: Custom Variable Struct (`nvars` & `update`)

For immutable manifold types, define `SecondOrderOptimization.nvars` and `SecondOrderOptimization.update`:

```julia
using SecondOrderOptimization
using LinearAlgebra
using Static, StaticArrays

struct UnitVector3
    v::SVector{3, Float64}
    UnitVector3(v::SVector{3, Float64}) = new(normalize(v))
end

SecondOrderOptimization.nvars(::UnitVector3) = static(3)
function SecondOrderOptimization.update(u::UnitVector3, step, start=1)
    s = @inbounds SVector(step[start], step[start + 1], step[start + 2])
    return UnitVector3(u.v + s)
end

target = SVector(0.0, 0.0, 1.0)
cost(u::UnitVector3) = 0.5 * sum(abs2, u.v - target)
function fdf!(g, H, u::UnitVector3)
    g .= u.v .- target
    H .= @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    return cost(u)
end

x0 = UnitVector3(SVector(1.0, 0.0, 0.1))
x_opt, res = optimize(cost, fdf!, x0, OptimizationOptions(iterator = :levenbergmarquardt))
println("Optimized unit vector: ", x_opt.v)
println("Norm: ", norm(x_opt.v))
```

## Method 2: Custom `retract!(to, from, step)` Keyword

Alternatively, pass a custom `retract!(to, from, step)` function directly to [`optimize`](@ref):

```julia
using SecondOrderOptimization
using LinearAlgebra
using StaticArrays

target = SVector(0.0, 0.0, 1.0)
sphere_cost(x) = 0.5 * sum(abs2, x - target)

function sphere_fdf!(g, H, x)
    g .= x .- target
    H .= @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    return sphere_cost(x)
end

# 3-argument retraction: (to, from, step) -> updated state
sphere_retract!(to, from, step) = normalize(from + step)

x0 = SVector(1.0, 0.0, 0.1)
x_opt, res = optimize(sphere_cost, sphere_fdf!, x0; retract! = sphere_retract!)
println("Optimized state on sphere: ", x_opt)
println("Norm: ", norm(x_opt))
```
