# Manifold Optimization and Custom Variable Types

`SecondOrderOptimization.jl` optimizes over any variable type that meets the contract of
[OptimizationVariableInterface.jl](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/):
numbers, static and dynamic vectors, collections of these, and your own types, such as points on
$SO(3)$, $SE(3)$ or the unit sphere $\mathcal{S}^n$, or bounded parameters.

The optimizer computes each step as a flat vector in the tangent space of the variables
($\Delta x \in T_x \mathcal{M}$), and asks the variables to apply it. It never looks inside them.

## The contract

A variable type `T` provides:

```julia
nvars(x::T)                        # number of degrees of freedom: a StaticInt, or an Int
update(x::T, step, start = 1)      # the variable moved along step[start:start+nvars(x)-1]; x is unchanged
update!(to::T, from::T, step)      # the same, written into `to` (only for in-place, mutable types)
```

An immutable type needs only `nvars` and `update`; the default `update!` calls `update`.
Return a `StaticInt` from `nvars` (`static(3)`) when the size is fixed by the type, so that
the optimizer can use static arrays.

These functions belong to OptimizationVariableInterface, not to this package. To extend them,
add it to your environment and load it:

```julia
using OptimizationVariableInterface
```

## Example: a unit vector

```julia
using SecondOrderOptimization, OptimizationVariableInterface
using LinearAlgebra, Static, StaticArrays

struct UnitVector3
    v::SVector{3, Float64}
    UnitVector3(v::SVector{3, Float64}) = new(normalize(v))
end

OptimizationVariableInterface.nvars(::UnitVector3) = static(3)
function OptimizationVariableInterface.update(u::UnitVector3, step, start = 1)
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
x_opt, res = optimize(x0, HessianObjective(cost, fdf!), OptimizationOptions(iterator = :levenbergmarquardt))
println("Optimized unit vector: ", x_opt.v)
println("Norm: ", norm(x_opt.v))
```

The gradient and Hessian are taken with respect to the step passed to `update`, so in the
tangent space, not in the coordinates the type stores.

## Further reading

OptimizationVariableInterface documents the contract in full:

- [Defining a variable type](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/custom_variables/):
  the guide, with collections of variables, in-place types and the built-in bounded types.
- [API reference](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/api/):
  the docstrings of `nvars`, `update` and `update!`.
