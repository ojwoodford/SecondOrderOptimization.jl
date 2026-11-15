# Migrating from JuliaSmoothOptimizers (NLPModels / JSOSolvers)

The [JuliaSmoothOptimizers (JSO)](https://github.com/JuliaSmoothOptimizers) ecosystem provides packages such as `NLPModels.jl` and `JSOSolvers.jl` (`trunk`, `tron`, `lbfgs`, etc.).

## Key Differences

1. **Problem Abstraction**:
   - `NLPModels.jl` defines an `AbstractNLPModel` representing objective, gradient, and Hessian evaluations.
   - `SecondOrderOptimization.jl` takes a [`HessianObjective`](@ref) built from two plain Julia callables, `f(x)` and `fdf!(g, H, x)`.
2. **Allocation & Overhead**:
   - JSO models operate on dynamically allocated `Vector{Float64}` arrays.
   - `SecondOrderOptimization.jl` supports stack-allocated `StaticArrays` (`SVector`, `MMatrix`) as well as preallocated dense and sparse workspaces with zero heap allocations inside the solve loop.

## Native Zero-Allocation Translation

1. Convert `obj(nlp, x)` to `f(x)`.
2. Combine `grad!(nlp, x, g)` and `hess` into `fdf!(g, H, x)` (note that `g` and `H` are zeroed before each `fdf!` call).
3. Wrap small fixed-size parameters in `SVector` and reuse an [`OptimizationWorkspace`](@ref).
4. Wrap them as `obj = HessianObjective(f, fdf!)` and call `x, result = optimize(x0, obj, OptimizationOptions(iterator = :dogleg))` (or `:levenbergmarquardt`); see [`optimize`](@ref).
