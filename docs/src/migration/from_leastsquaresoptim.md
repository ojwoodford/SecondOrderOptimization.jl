# Migrating from LeastSquaresOptim.jl

[LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl) focuses on nonlinear least squares problems $\min_x \frac{1}{2} \|r(x)\|^2$.

## Key Comparisons

- **General vs Least Squares**: `LeastSquaresOptim.jl` formulates problems as a residual vector $r(x)$ and Jacobian matrix $J(x)$. `SecondOrderOptimization.jl` is a general second-order solver: it optimizes arbitrary functions $f(x)$ with gradient $g(x)$ and Hessian/Gauss-Newton approximation $H(x)$. (To model structured nonlinear least squares graphs with automatic differentiation and robust kernels, use `NLLSModels.jl`, which is built on `SecondOrderOptimization.jl`.)
- **Allocation Profiles**: `SecondOrderOptimization.jl` achieves zero heap allocations when using `StaticArrays` and [`OptimizationWorkspace`](@ref).

## Translation Example

### In LeastSquaresOptim.jl:
```julia
using LeastSquaresOptim

function rosenbrock_residuals!(r, x)
    r[1] = 1.0 - x[1]
    r[2] = 10.0 * (x[2] - x[1]^2)
end

res = optimize!(LeastSquaresProblem(x = [-1.2, 1.0], f! = rosenbrock_residuals!, output_length = 2), Dogleg())
```

### In SecondOrderOptimization.jl:
```julia
using SecondOrderOptimization
using StaticArrays

rosenbrock_cost(x) = 0.5 * ((1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2)

function rosenbrock_fdf!(g, H, x)
    r1 = 1.0 - x[1]
    r2 = 10.0 * (x[2] - x[1]^2)
    # g = J' * r
    g[1] = -r1 - 20.0 * x[1] * r2
    g[2] = 10.0 * r2
    # H = J' * J (Gauss-Newton approximation)
    H[1, 1] = 1.0 + 400.0 * x[1]^2
    H[1, 2] = -200.0 * x[1]
    H[2, 1] = -200.0 * x[1]
    H[2, 2] = 100.0
    return 0.5 * (r1^2 + r2^2)
end

x0 = @SVector [-1.2, 1.0]
x_opt, res = optimize(rosenbrock_cost, rosenbrock_fdf!, x0, OptimizationOptions(iterator = :dogleg))
```
