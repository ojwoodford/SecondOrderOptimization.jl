"""
    HessianObjective(f, fdf!; H = nothing)

An objective given by its cost and by its gradient and Hessian, for [`optimize`](@ref) and
[`optimize!`](@ref).

- `f(x) -> cost` returns the scalar cost at `x`.
- `fdf!(g, H, x) -> cost` writes the gradient into `g` and the Hessian into `H` (both arrive
  zeroed), and returns the cost at `x`.
- `H` is the Hessian prototype: `nothing` (a static or dense system sized from `x0`), a `Matrix`,
  an `MMatrix`, a `SparseMatrixCSC`, or another package's prototype for which that package implements
  DampedLinearSolve's linear-system contract (a `SchurBlockHessian`, for Schur-complement elimination).
  The type of `H` selects the linear system, including elimination for a `SchurBlockHessian`; the
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

"""
    JacobianObjective(f, resjac!, J)

A least-squares objective given by its cost and by its residual and Jacobian, for
[`optimize`](@ref) and [`optimize!`](@ref). The linear systems are solved by a QR factorization
of the Jacobian, never by forming `JᵀJ`, which is more accurate when `J` is ill conditioned: a
dense Householder QR for a dense `J`, and SuiteSparse's sparse QR (SPQR) for a sparse one.

- `f(x) -> cost` returns the true scalar cost at `x`. Trial points are scored with `f`: the
  optimizer never derives the cost from the residual, so the cost need not be `½‖r‖²` (with robust
  weighting it is not).
- `resjac!(r, J, x) -> cost` writes the residual into `r` (length `m`) and the Jacobian into `J`
  (`m × nvars(x)`, the derivative of `r` along the update `dx` of `update(x, dx)`), and returns
  the cost at `x`, equal to `f(x)`.
- **`resjac!` sets `r` and `J`; it does not add to them.** Unlike `g` and `H` for `fdf!` in
  [`HessianObjective`](@ref), they are not zeroed before each call: the optimizer never writes
  to `r` or `J`. So `resjac!` must set every entry of `r` on every call, and every entry of `J`
  that it writes at all must be set on every call. Entries of `J` that `resjac!` never writes keep
  the values the caller put in the prototype: a caller whose `resjac!` writes only part of `J`
  supplies a prototype with the rest already set (normally `zeros(m, n)`, for structural zeros).
  With an integer `m` the optimizer allocates `J` itself, as zeros. An entry written at some `x`
  and skipped at another keeps its old value, which is a bug in `resjac!`.
- `J` is required: a `Matrix{Float64}` or `SparseMatrixCSC{Float64, Int}` prototype
  (`m × nvars(x0)`), or an integer `m`, the number of residuals, for a dense system with its own
  matrix. A sparse prototype's stored entries are its pattern: `resjac!` must write only stored
  entries (through `nonzeros(J)`, or `J[i, j]` for a stored `(i, j)`). Writing any other entry
  inserts it, which changes the pattern, and the run throws an `ArgumentError`.
- `J` may also be another package's prototype for which that package implements DampedLinearSolve's
  linear-system contract, such as a `SchurBlockJacobian{Float64, RType}`: the variables are
  partitioned into retained and eliminated blocks, and each step eliminates the eliminated blocks by
  QR, solves the reduced problem in the retained variables and back-substitutes. `resjac!` writes
  the Jacobian blocks and the residuals into `J` by row block (through that package's `blockentry`,
  `block(J, k, entry, nrows, ncols)` and `residual_block(J, k)`); its `r` argument is then an empty
  vector. `RType`, `Matrix{Float64}` or
  `SparseMatrixCSC{Float64, Int}`, is the storage of the reduced system. A prototype is the matrix `resjac!` writes
  into, with its values (a trial system, for `callbackchangesstate = false`, works on a copy
  made at the start of the run), except with an [`OptimizationWorkspace`](@ref): a workspace owns
  its Jacobian storage, set from the prototype handed in at the start of each run, and the
  prototype itself is then never written.

The quadratic model at `x` is `cost + (Jᵀr)ᵀdx + ½‖J dx‖²`: the gradient `Jᵀr` is computed by the
optimizer, and there is no residual-only callable. The model's gradient matches the true cost's
gradient only when the cost is `½‖r‖²` near `x`; for a reweighted cost, scale `r` and `J` so
that this holds.

# Supported combinations
| Jacobian prototype `J` | Linear system | `linsolver` |
|:---|:---|:---|
| integer `m ≥ 1`, or `Matrix{Float64}` (`m × n`) | dense QR of `J`; static QR when `x0` has a static size and `m·n ≤ 48` | `:auto`, `:qr`; `:lsmr` (with Krylov.jl) |
| `SparseMatrixCSC{Float64, Int}` | sparse QR of `J` (SPQR) | `:auto`, `:qr`; `:lsmr` (with Krylov.jl) |
| `SchurBlockJacobian{Float64, Matrix{Float64}}` (another package's) | QR elimination, dense reduced system | `:auto`, `:qr` (QR of the reduced Jacobian); `:cholesky`, `:ldl` (factor its normal equations); `:lsmr` (with Krylov.jl) |
| `SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}` (another package's) | QR elimination, sparse reduced system | `:auto`, `:qr` (sparse QR of the reduced Jacobian, SPQR); `:cholesky` (sparse Cholesky of the reduced normal equations, falling back to LDLᵀ), `:ldl`; `:lsmr` (with Krylov.jl) |

Every row takes every iterator: `:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`, `:newton`
(Gauss-Newton) and `:gradientdescent`, which uses only the gradient `Jᵀr` and never solves the linear
system. Everything else is rejected with an `ArgumentError` before `f` or `resjac!` is called: any
other prototype type, element type or index type, and any other `linsolver`.

The model Hessian is `JᵀJ`, so `iterator = :newton` is Gauss-Newton: each step solves
`JᵀJ dx = -Jᵀr` by QR of `J`, with Newton's diagonal shift if `J` is numerically rank deficient.
`:levenbergmarquardt_dtd` scales the damping by the running maximum of the squared column norms
of `J` (the diagonal of `JᵀJ`).

With a sparse `J`, each solve (each damping value tried) is a numeric SPQR factorization of
`[J; √λ D^½]`, whose ordering and symbolic analysis are done once, when the system is built. SPQR
allocates its working memory inside each factorization and frees it again; nothing else allocates
in the loop. The same holds for `:auto` and `:qr` with a `SchurBlockJacobian{Float64,
SparseMatrixCSC{Float64, Int}}`: the reduced Jacobian `J̃` has a fixed pattern, filled after each
elimination, and `[J̃; √λ D_r^½]` is factored by SPQR in the same way.

`linsolver = :lsmr` (with `using Krylov`) solves each damped step iteratively by LSMR on `J`,
dense or sparse, with the damping as LSMR's regularization and the Marquardt scaling as its
preconditioner `N`: nothing is factorized or augmented, `JᵀJ` is never formed, and nothing
allocates. Its tolerances are Krylov's defaults (`etol`, `axtol`, `btol` = √eps, `conlim` = 1/√eps,
at most `m + n` iterations), starting values that have not been measured. LSMR does not detect a
numerically rank-deficient `J`, except `m < n` undamped, which is reported as the QR solve reports
it.

For a small problem with a static number of variables (`x0` a `Number`, a `StaticVector` or a tuple of
them, with at most `MAX_STATIC_SYSTEM` variables) and a dense `J` with `m·n ≤ 48`, the optimizer uses a
static system: `J` is an `MMatrix` and `r` an `MVector`, and the QR solves are StaticArrays'. So `J`
may be any `AbstractMatrix` the optimizer allocates: write it by `J[i, j]`, not by assuming a `Matrix`.
With a `Matrix` prototype the static system works on a copy, and the prototype itself is not
written. The cap is a starting value, measured to be where the static system is faster.

A prototype whose column count is not `nvars(x0)` is a `DimensionMismatch`.

# Example
```julia
t = collect(range(0, 1; length = 20))
y = @. 2.0 * exp(-1.5 * t)
f(x) = 0.5 * sum(abs2, @. x[1] * exp(x[2] * t) - y)
function resjac!(r, J, x)
    @. r = x[1] * exp(x[2] * t) - y
    @. J[:, 1] = exp(x[2] * t)
    @. J[:, 2] = x[1] * t * exp(x[2] * t)
    return 0.5 * sum(abs2, r)
end
x, result = optimize([1.0, 0.0], JacobianObjective(f, resjac!, length(t)))
```
"""
struct JacobianObjective{F, RJ, JM}
    f::F            # f(x) -> cost
    resjac!::RJ     # resjac!(r, J, x) -> cost
    J::JM           # Matrix{Float64} prototype, or an Int: the number of residuals m

    function JacobianObjective(f::F, resjac!::RJ, J::JM) where {F, RJ, JM}
        checkjacobian(J)
        return new{F, RJ, JM}(f, resjac!, J)
    end
end

nresiduals(obj::JacobianObjective) = jacobian_nrows(obj.J)

# Everything checked before anything is allocated or any user function called
function checkobjective(x0, obj::JacobianObjective, options::OptimizationOptions)
    checkjacobian(obj.J)
    n = dynamic(nvars(x0))
    nc = jacobian_ncols(obj.J)
    if !isnothing(nc) && nc != n
        throw(DimensionMismatch("Jacobian prototype J has $nc columns, but the variables have $n degrees of freedom, so J must have $n columns."))
    end
    check_jacobian_linsolver(obj.J, linsolver_symbol(options.linsolver))
    return nothing
end
