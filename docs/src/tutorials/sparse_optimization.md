# Sparse Large-Scale Optimization

For large-scale problems (e.g. chain systems, pose graphs, bundle adjustment, spline fitting), forming or factoring dense Hessians is prohibitive. `SecondOrderOptimization.jl` natively supports sparse Hessians: give a `SparseMatrixCSC` prototype as the `H` keyword of [`HessianObjective`](@ref), and pass the objective to [`optimize`](@ref).

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
ws = OptimizationWorkspace(x0, HessianObjective(chain_cost, chain_fdf!; H = H_proto), opts)
x_opt, res = optimize(copy(x0), HessianObjective(chain_cost, chain_fdf!; H = H_proto), opts; workspace = ws)
```

## Choosing the Sparse Linear Solver

You can select the sparse linear solver backend via the `linsolver` keyword argument of [`OptimizationOptions`](@ref):

| Solver Symbol | Backend | Solver Class | Recommended Use Case | Setup |
| :--- | :--- | :--- | :--- | :--- |
| `:auto` / `:ldl` | `LDLFactorizations.jl` | Direct $LDL^T$ | Small to medium sparse systems ($< 10^5$ DoF); zero allocation during solve | Built-in |
| `:cholesky` | `SparseArrays.CHOLMOD` | Direct Supernodal Cholesky | Sparse positive-definite systems; leverages native CHOLMOD $\lambda$ shift | Built-in |
| `:qr` | `SuiteSparse.SPQR` | Direct Column-Pivoted QR | Rank-deficient sparse systems | Built-in |
| `:pardiso` | `Pardiso.jl` | Direct Multi-Threaded Factorization | Large sparse systems ($10^4\text{--}10^6$ DoF) leveraging multi-core CPUs (MKL / Panua) | `using Pardiso` |
| `:cg` | `Krylov.jl` | Iterative Conjugate Gradient | Ultra large-scale SPD systems ($10^5\text{--}10^7$ DoF) or when direct fill-in exhausts RAM | `using Krylov` |
| `:minres` | `Krylov.jl` | Iterative MINRES | Large symmetric indefinite systems, saddle-point problems, and Newton steps | `using Krylov` |
| `:krylov` | `Krylov.jl` | Iterative (defaults to `:cg`) | General iterative Krylov subspace solve | `using Krylov` |

---

## Extension Solvers: When and How to Use Them

`SecondOrderOptimization.jl` integrates with `Pardiso.jl` and `Krylov.jl` through Julia 1.10+ package extensions. These optional backends are activated automatically whenever their respective packages are loaded in your Julia session.

### 1. `Pardiso.jl` (`linsolver = :pardiso`)

#### When to Use
- **Multi-threaded CPU Factorization**: Standard `LDLFactorizations.jl` is single-threaded. When optimizing medium-to-large sparse problems on modern workstations or servers with 8–64 CPU cores, Intel MKL PARDISO parallelizes sparse numerical factorization across threads.
- **Exact Direct Solves**: When you need exact machine-precision step computations without tuning iterative convergence tolerances.

#### How to Use
Simply load `Pardiso` alongside `SecondOrderOptimization`:

```julia
using SecondOrderOptimization
using SparseArrays
using Pardiso

# Large-scale objective with sparse prototype H_proto
options = OptimizationOptions(
    iterator = :newton,
    linsolver = :pardiso,
    gtol = 1e-6
)

x_opt, result = optimize(x0, HessianObjective(chain_cost, chain_fdf!; H = H_proto), options)
println("Final cost: ", result.bestcost)
```

> **Note**: `PardisoExt` automatically detects whether Intel MKL (`MKLPardisoSolver`) or Panua Pardiso (`PardisoSolver`) is available in your environment, configures the solver for symmetric positive-definite systems (`REAL_SYM_POSDEF`), and maps breakdown/singularity into [`NotPositiveDefinite`](@ref).

---

### 2. `Krylov.jl` (`linsolver = :cg`, `:minres`, `:krylov`)

#### When to Use
- **Memory-Constrained & Ultra Large-Scale Optimization**: For problems with hundreds of thousands or millions of variables ($10^5\text{--}10^7$ DoF), direct factorizations ($LDL^T$, Cholesky) suffer from catastrophic sparse fill-in (where $L$ contains orders of magnitude more non-zeros than $A$), causing out-of-memory errors. Krylov subspace methods require only $O(N)$ storage.
- **Iterative Solver Selection**:
  - **`:cg` (Conjugate Gradient)**: Use for symmetric positive-definite systems. Ideal in combination with **Levenberg-Marquardt** (`:levenbergmarquardt`), where damping $\lambda I$ guarantees positive definiteness.
  - **`:minres` (Minimal Residual)**: Use for symmetric indefinite systems, such as saddle-point formulations or unconstrained Newton steps with negative curvature.

#### How to Use
Simply load `Krylov` in your session:

```julia
using SecondOrderOptimization
using SparseArrays
using Krylov

# Configure Levenberg-Marquardt with Conjugate Gradient linear solver
options = OptimizationOptions(
    iterator = :levenbergmarquardt,
    linsolver = :cg,
    maxiters = 50,
    gtol = 1e-5
)

# Zero-allocation workspace preallocates Krylov basis vectors in `solver_ws`
ws = OptimizationWorkspace(x0, HessianObjective(chain_cost, chain_fdf!; H = H_proto), options)
x_opt, result = optimize(copy(x0), HessianObjective(chain_cost, chain_fdf!; H = H_proto), options; workspace = ws)
```

#### Zero-Allocation Iterative Solves
When `OptimizationWorkspace` is constructed with a Krylov solver, `SecondOrderOptimization.jl` preallocates the appropriate Krylov workspace (`Krylov.CgWorkspace` or `MinresWorkspace`) inside `solver_ws`. During the optimization loop, `cg!` and `minres!` execute in-place with **zero heap allocations** per iteration.

If the linear solver detects negative curvature ($x^T A x < 0$), it reports [`Indefinite`](@ref), allowing Newton and Dogleg iterators to apply regularizing spectral shifts or fall back to trust-region boundaries.
