"""
    HessianObjective(f, fdf!; H = nothing)

An objective given by its cost and by its gradient and Hessian, for [`optimize`](@ref) and
[`optimize!`](@ref).

- `f(x) -> cost` returns the scalar cost at `x`.
- `fdf!(g, H, x) -> cost` writes the gradient into `g` and the Hessian into `H` (both arrive
  zeroed), and returns the cost at `x`.
- `H` is the Hessian prototype: `nothing` (a static or dense system sized from `x0`), a `Matrix`,
  an `MMatrix`, a `SparseMatrixCSC`, or, with BlockLinearSystems.jl loaded, a `SchurBlockMatrix`.
  The type of `H` selects the linear system, including elimination for a `SchurBlockMatrix`; the
  `linsolver` option then names the solver for that system.

# The Hessian prototype
- **Fill both triangles.** `fdf!` must write the full symmetric Hessian. Some solvers read only
  the upper triangle (dense Cholesky and Bunch-Kaufman, sparse `:cholesky`), but sparse
  `:auto`/`:ldl` need both, and `:qr` and the predicted cost reduction read every stored entry,
  so an upper-triangle-only `H` gives a wrong step or a wrong step-quality estimate.
- **Size and element type.** `H` must be `n×n`, where `n == nvars(x0)`; a mismatch is a
  `DimensionMismatch`. The Hessian is `Float64`: a sparse prototype must be
  `SparseMatrixCSC{Float64}`, and a dense one of another type is copied into a `Matrix{Float64}`.
- **Missing diagonal entries.** A sparse prototype should store every diagonal entry (explicit
  zeros are fine). If one is missing, a new matrix with explicit zeros on the diagonal is built
  and handed to `fdf!` in place of the prototype; its `nzval` layout differs from the
  prototype's, so an `fdf!` that writes by precomputed `nzval` positions of its own prototype
  would write to the wrong entries. Writing through `H[i, j]` works either way.
- **Which matrix holds the final Hessian.** `fdf!` writes into the linear system's own matrix.
  That is the prototype itself for a `Matrix{Float64}`, `MMatrix` or a sparse matrix with every
  diagonal entry stored, and a copy otherwise. With a reused `workspace`, the matrix is the one
  from the call that built the workspace: an objective whose `H` is a different object of the
  same type and size (or, for a sparse `H`, the same pattern) reuses the cached system, and its
  `H` is left untouched.

# Example
```julia
f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function fdf!(g, H, x)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    return f(x)
end
obj = HessianObjective(f, fdf!)
```
"""
struct HessianObjective{F, FDF, HM}
    f::F        # f(x) -> cost
    fdf!::FDF   # fdf!(g, H, x) -> cost
    H::HM       # Hessian prototype, or nothing
end
HessianObjective(f, fdf!; H = nothing) = HessianObjective(f, fdf!, H)
