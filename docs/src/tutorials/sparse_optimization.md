# Sparse Large-Scale Optimization

For large-scale problems (e.g. chain systems, pose graphs, bundle adjustment, spline fitting), forming or factoring dense Hessians is prohibitive. `SecondOrderOptimization.jl` natively supports sparse Hessians by passing a `SparseMatrixCSC` prototype via the `H` keyword argument of [`optimize`](@ref).

## Passing a Sparse Hessian Prototype `H`

When `H::SparseMatrixCSC` is provided:
- `SecondOrderOptimization.jl` automatically ensures all diagonal entries are structurally present (for Levenberg-Marquardt damping) and pre-computes `diag_inds` for $O(n)$ diagonal updates.
- Symbolic factorization is performed once during workspace initialization via `LDLFactorizations.ldl_analyze`, and numerical factorization (`LDLFactorizations.ldl_factorize!`) runs in-place without heap allocations inside the loop.

```julia
using SecondOrderOptimization
using SparseArrays
using LinearAlgebra

# Tridiagonal nonlinear objective:
# f(x) = 0.5 * x_1^2 + sum_{i=1}^{n-1} 0.5 * (x_{i+1} - x_i - 1.0)^2
n = 100
function chain_cost(x)
    c = 0.5 * x[1]^2
    @inbounds for i in 1:length(x)-1
        c += 0.5 * (x[i+1] - x[i] - 1.0)^2
    end
    return c
end

function chain_fdf!(g, H::SparseMatrixCSC, x)
    g[1] += x[1]
    H[1, 1] += 1.0
    @inbounds for i in 1:length(x)-1
        r = x[i+1] - x[i] - 1.0
        g[i] -= r
        g[i+1] += r
        H[i, i] += 1.0
        H[i, i+1] -= 1.0
        H[i+1, i] -= 1.0
        H[i+1, i+1] += 1.0
    end
    return chain_cost(x)
end

# Build tridiagonal sparsity pattern prototype
H_proto = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
x0 = zeros(n)
opts = OptimizationOptions(iterator = :levenbergmarquardt, linsolver = :ldl)

# Preallocate workspace for zero-allocation repeated solves
ws = OptimizationWorkspace(chain_cost, chain_fdf!, x0, opts; H = H_proto)
x_opt, res = optimize(chain_cost, chain_fdf!, copy(x0), opts; H = H_proto, workspace = ws)
```

## Choosing the Sparse Linear Solver

You can select the sparse factorization backend via `options = OptimizationOptions(linsolver = ...)`:
- `:auto` or `:ldl`: In-place sparse $LDL^T$ factorization via `LDLFactorizations.jl` (zero allocation during solve).
- `:cholesky`: Sparse Cholesky via SuiteSparse CHOLMOD with QR fallback.
- `:qr`: Sparse QR via SuiteSparse SPQR.
