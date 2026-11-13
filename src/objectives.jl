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

"""
    JacobianObjective(f, resjac!, J)

A least-squares objective given by its cost and by its residual and Jacobian, for
[`optimize`](@ref) and [`optimize!`](@ref). The linear systems are solved by a QR factorization
of the Jacobian, never by forming `JᵀJ`, which is more accurate when `J` is ill conditioned.

- `f(x) -> cost` returns the true scalar cost at `x`. Trial points are scored with `f`: the
  optimizer never derives the cost from the residual, so the cost need not be `½‖r‖²` (with robust
  weighting it is not).
- `resjac!(r, J, x) -> cost` writes the residual into `r` (length `m`) and the Jacobian into `J`
  (`m × nvars(x)`, the derivative of `r` along the update `dx` of `update(x, dx)`), and returns
  the cost at `x`, equal to `f(x)`. `r` and `J` arrive zeroed, as `g` and `H` do for `fdf!` in
  [`HessianObjective`](@ref), so `resjac!` may accumulate into them.
- `J` is required: a `Matrix{Float64}` prototype (`m × nvars(x0)`), which the linear system then
  writes into, or an integer `m`, the number of residuals, for a system with its own matrix.

The quadratic model at `x` is `cost + (Jᵀr)ᵀdx + ½‖J dx‖²`: the gradient `Jᵀr` is computed by the
optimizer, and there is no residual-only callable. The model's gradient matches the true cost's
gradient only when the cost is `½‖r‖²` near `x`; for a reweighted cost, scale `r` and `J` so
that this holds.

# Supported so far
| Accepted | Rejected (`ArgumentError`, before `f` or `resjac!` is called) |
|:---|:---|
| `J` a `Matrix{Float64}` prototype, or an integer `m ≥ 1` | a sparse or any other prototype; another element type |
| `iterator = :levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg` or `:newton` | `:gradientdescent`, which uses only the gradient `Jᵀr` and so gains nothing from the Jacobian form |
| `linsolver = :auto` or `:qr` (the same QR solve) | every other linear solver |

The model Hessian is `JᵀJ`, so `iterator = :newton` is Gauss-Newton: each step solves
`JᵀJ dx = -Jᵀr` by QR of `J`, with Newton's diagonal shift if `J` is numerically rank deficient.
`:levenbergmarquardt_dtd` scales the damping by the running maximum of the squared column norms
of `J` (the diagonal of `JᵀJ`).

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

# The iterators accepted with a JacobianObjective
const JACOBIAN_ITERATORS = (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)

const JACOBIAN_SUPPORT = "JacobianObjective supports a Matrix{Float64} Jacobian prototype or an integer number of residuals m, with iterator = " *
    join((":$s" for s in JACOBIAN_ITERATORS), ", ", " or ") * " and linsolver = :auto or :qr"

# The Jacobian prototype: a Matrix{Float64} with at least one row, or an integer m >= 1
function checkjacobian(J)
    if J isa Integer && !(J isa Bool)
        J >= 1 || throw(ArgumentError("The number of residuals m must be at least 1; got $J. $JACOBIAN_SUPPORT."))
    elseif J isa Matrix{Float64}
        size(J, 1) >= 1 || throw(ArgumentError("The Jacobian prototype has no rows (size $(size(J))): it needs one row per residual. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractSparseMatrix
        throw(ArgumentError("A sparse Jacobian prototype ($(typeof(J))) is not supported yet. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractMatrix && eltype(J) !== Float64
        throw(ArgumentError("The Jacobian must be Float64; got a prototype with element type $(eltype(J)). $JACOBIAN_SUPPORT."))
    else
        throw(ArgumentError("A Jacobian prototype of type $(typeof(J)) is not supported. $JACOBIAN_SUPPORT."))
    end
    return nothing
end

nresiduals(obj::JacobianObjective) = obj.J isa Integer ? Int(obj.J) : size(obj.J, 1)

# Everything checked before anything is allocated or any user function called
function checkobjective(x0, obj::JacobianObjective, options::OptimizationOptions)
    checkjacobian(obj.J)
    n = dynamic(nvars(x0))
    if obj.J isa AbstractMatrix && size(obj.J, 2) != n
        throw(DimensionMismatch("Jacobian prototype J has size $(size(obj.J)), but the variables have $n degrees of freedom, so J must have $n columns."))
    end
    iter = dynamic(options.iterator)
    iter === :gradientdescent && throw(ArgumentError("iterator = :gradientdescent is not supported with a JacobianObjective: gradient descent uses only the gradient Jᵀr, so it gains nothing from the residual-and-Jacobian form; use a HessianObjective. $JACOBIAN_SUPPORT."))
    iter in JACOBIAN_ITERATORS || throw(ArgumentError("iterator = :$iter is not supported with a JacobianObjective. $JACOBIAN_SUPPORT."))
    s = dynamic(options.linsolver)
    s in (:auto, :qr) || throw(ArgumentError("linsolver = :$s is not supported with a JacobianObjective. $JACOBIAN_SUPPORT."))
    return nothing
end
