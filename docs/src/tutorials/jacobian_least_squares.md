# Least Squares from a Jacobian

Many objectives are sums of squares, $f(x) = \tfrac12 \lVert r(x) \rVert^2$. You can give such a
problem to [`optimize`](@ref) in two ways:

- as a [`HessianObjective`](@ref), whose `fdf!` forms the gradient $J^\top r$ and the
  Gauss-Newton Hessian $J^\top J$ itself; or
- as a [`JacobianObjective`](@ref), whose `resjac!` writes only the residual $r$ and the
  Jacobian $J$. The optimizer then solves each step by a QR factorization of $J$ and never forms
  $J^\top J$.

Both give the same iterations, with any iterator but `:gradientdescent`. They differ in how
accurately each step is solved, and in what that accuracy costs. With a `JacobianObjective` the
model Hessian is $J^\top J$, so `iterator = :newton` is Gauss-Newton.

## The same problem, both ways

Fit $y \approx a\, e^{b t}$ to some data:

```julia
using SecondOrderOptimization, LinearAlgebra

t = collect(range(0, 2; length = 25))
y = @. 3.0 * exp(-1.3 * t) + 0.05 * sin(7 * t)

residual(x) = @. x[1] * exp(x[2] * t) - y
cost(x) = 0.5 * sum(abs2, residual(x))

# As a HessianObjective: form g = Jᵀr and H = JᵀJ
function fdf!(g, H, x)
    r = residual(x)
    J = hcat(exp.(x[2] .* t), x[1] .* t .* exp.(x[2] .* t))
    g .= J' * r
    H .= J' * J
    return 0.5 * sum(abs2, r)
end
x1, res1 = optimize([1.0, 0.0], HessianObjective(cost, fdf!; H = zeros(2, 2)))

# As a JacobianObjective: write r and J, and give the number of residuals
function resjac!(r, J, x)
    @. r = x[1] * exp(x[2] * t) - y
    @. J[:, 1] = exp(x[2] * t)
    @. J[:, 2] = x[1] * t * exp(x[2] * t)
    return 0.5 * sum(abs2, r)
end
x2, res2 = optimize([1.0, 0.0], JacobianObjective(cost, resjac!, length(t)))
```

On a well-conditioned problem like this one the two runs agree: the same damping and the same
cost at every iteration, to rounding.

The contract of `resjac!(r, J, x)`:
- it writes the residual into `r` (length `m`) and the Jacobian into `J` (`m × nvars(x)`, taken
  with respect to the step of `update`, so in the tangent space for a manifold variable; see
  [Manifold Optimization and Custom Variable Types](manifold_optimization.md)),
  and returns the cost at `x`;
- it sets `r` and `J`, and does not add to them. They are not zeroed before each call (unlike
  `g` and `H` for `fdf!`), and the optimizer never writes them. So every entry of `r`, and every
  entry of `J` that `resjac!` writes at all, must be set on every call. Entries it never writes
  keep the prototype's values (zeros for an integer `m`), so a `resjac!` that writes only the
  non-zero pattern of `J` is handed a prototype of zeros;
- the cost it returns must equal `f(x)`. Trial points are scored with `f`, never with
  $\tfrac12 \lVert r \rVert^2$, so `f` can be any cost whose gradient is $J^\top r$. For a robust
  cost, for example, `resjac!` writes the reweighted residual and Jacobian.

`J` is a `Matrix{Float64}` prototype, which the optimizer then writes into, or the integer `m`.

## When the Jacobian form is worth it

Forming $J^\top J$ squares the condition number: $\kappa(J^\top J) = \kappa(J)^2$. When
$\kappa(J)$ approaches $10^8$, the normal equations have no correct digits left in their weakest
directions. Cholesky may still succeed, but on a matrix that rounding has already changed. A QR
factorization of $J$ works with $\kappa(J)$ itself.

The test suite checks this on a degree-13 polynomial fit with $\kappa(J) \approx 4 \times 10^9$.
With `:levenbergmarquardt` the Jacobian form reaches the least-squares minimum to rounding in
about 20 iterations, and with `:dogleg` or `:newton` (Gauss-Newton) in under 10. The normal
equations with `linsolver = :cholesky` are still above the minimum after hundreds of iterations:
their steps are only as good as the rounded $J^\top J$.

`:levenbergmarquardt_dtd` reaches the minimum too in the Jacobian form, in about 20 iterations.
Here the smallest squared singular value is about $10^{-19}$ of the diagonal of $J^\top J$, and
the damping of Levenberg-Marquardt can fall to what the linear solver resolves (see
`dampingfloor` in the internal linear-system interface): about $\epsilon^2$ times the
diagonal for QR of $J$, but only about $\epsilon$ for Cholesky of $J^\top J$, so the normal
equations stay above the minimum. Ill-conditioned Jacobians come from
polynomial and spline fits in a monomial basis, from parameters of very different scales, and
from nearly redundant parameterizations.

When $J$ is well conditioned, the two forms give the same answer and the normal equations are
cheaper.

## What it costs

For $J$ of size $m \times n$:

| | Normal equations (`HessianObjective`) | QR of a dense Jacobian (`JacobianObjective`) |
|:---|:---|:---|
| Once per linearization | form $J^\top J$ and $J^\top r$: $O(mn^2)$, in your `fdf!` | Householder QR of $[J\ r]$: $O(mn^2)$, about twice the flops of forming $J^\top J$ |
| Per damping value tried | Cholesky of $J^\top J + \lambda D$: $O(n^3)$ | QR of the stacked $2n \times n$ matrix $[R;\ \sqrt{\lambda} D^{1/2}]$: $O(n^3)$, independent of $m$ |
| Storage | $n \times n$ (twice that with trial reuse) | $J$, plus one $m \times (n + 1)$ scratch copy of $[J\ r]$ for the factorization, plus $2n \times (n + 1)$: about twice the storage of $J$. With trial reuse (`callbackchangesstate = false`, the default) a second $J$, which shares the scratch: about three times |

With a sparse `J` (a `SparseMatrixCSC{Float64, Int}` prototype) the system is solved by
SuiteSparse's sparse QR (SPQR) of $[J;\ \sqrt{\lambda} D^{1/2}]$. Its column ordering and symbolic
analysis are computed once, when the system is built; every damping value tried is a numeric
factorization of the whole augmented matrix, so a retry costs as much as the first solve. SPQR
allocates its working memory inside each factorization (and frees it again); nothing else
allocates in the loop. `resjac!` must write only the entries the prototype stores, for example
through `nonzeros(J)`.

For a dense `J`, the per-damping cost does not depend on $m$, so the retries with a larger $\lambda$ after a
rejected step stay cheap however many residuals there are. An undamped solve on a numerically
rank-deficient $J$ (including $m < n$) is reported as not positive definite, just as Cholesky
reports singular normal equations: Levenberg-Marquardt raises $\lambda$, and Newton and dogleg
shift the diagonal.

## What is supported

| Jacobian prototype `J` | Linear system | `linsolver` |
|:---|:---|:---|
| integer `m ≥ 1`, or `Matrix{Float64}` (`m × n`) | dense QR of `J`; static QR when `x0` has a static size and `m·n ≤ 48` | `:auto`, `:qr`; `:lsmr` (with Krylov.jl) |
| `SparseMatrixCSC{Float64, Int}` | sparse QR of `J` (SPQR) | `:auto`, `:qr`; `:lsmr` (with Krylov.jl) |
| `SchurBlockJacobian{Float64, Matrix{Float64}}` (with BlockLinearSystems.jl) | QR elimination, dense reduced system | `:auto`, `:qr` (QR of the reduced Jacobian); `:cholesky`, `:ldl` (factor its normal equations); `:lsmr` (with Krylov.jl) |
| `SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}` (with BlockLinearSystems.jl) | QR elimination, sparse reduced system | `:auto`, `:cholesky` (sparse Cholesky of the reduced normal equations, falling back to LDLᵀ); `:ldl`; `:lsmr` (with Krylov.jl) |

Every row takes `iterator = :levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg` or `:newton`
(Gauss-Newton). Everything else is rejected with an `ArgumentError` before `f` or `resjac!` is
called: any other prototype type, element type or index type; `iterator = :gradientdescent`, which
uses only the gradient $J^\top r$ and gains nothing from the Jacobian form; and any other `linsolver`.

With `linsolver = :lsmr` each damped step is solved iteratively by LSMR on $J$ (dense or sparse):
the damping enters as LSMR's regularization parameter and the Marquardt scaling as its
preconditioner, so nothing is augmented, factorized or formed, and nothing allocates. Its stopping
tolerances are Krylov's defaults, starting values that have not been measured. It suits a large
$J$ whose QR factorization is too expensive; it does not detect a numerically rank-deficient $J$.
A `HessianObjective` rejects `:lsmr`: on the normal equations it would square their condition
number again.

## Small problems

With a static number of variables (`x0` a `Number`, a `StaticVector`, or a tuple of them) and a dense
$J$ with $m n \le 48$, the optimizer uses a static system: $J$ is an `MMatrix`, $r$ an `MVector`, and
the QR solves are StaticArrays'. Measured, this is faster than the dense system for those sizes (by
up to 30× for one variable) and slower above them. Nothing changes for the caller, except that
`resjac!` must write `J` by index, as any `AbstractMatrix`, and with a `Matrix` prototype the
static system works on its own copy of it.

## Eliminating variables: QR elimination

When the variables split into retained ones and many small, mutually independent eliminated
blocks (each residual involving at most one eliminated block, as for points in bundle
adjustment), a `SchurBlockJacobian{Float64, RType}` prototype from BlockLinearSystems.jl
eliminates them by QR, the Jacobian counterpart of the Schur complement
(BlockLinearSystems' QR-elimination tutorial explains the layout). Each step factors every
eliminated block's rows with its damping, solves the reduced problem in the retained variables
and back-substitutes; it never forms $J^\top J$ unless the chosen reduced solver does. `RType` is
the storage of the reduced system, and `linsolver` its solver: the last two rows of the table
above.

`resjac!` writes the Jacobian blocks and the residuals into the tables of `J` through
BlockLinearSystems' assembly views; its first argument is then an empty vector. An eliminated
block with fewer residuals than dimensions is singular without damping, and is reported as the
Schur path reports it.

Everything else works as for a `HessianObjective`: every variable type, including immutable and
manifold variables through [`optimize`](@ref); both variants of each iterator
(`callbackchangesstate`); callbacks and [`CostTrajectory`](@ref); the termination criteria, with
`gtol` applied to $\lVert J^\top r \rVert_\infty$; and an [`OptimizationWorkspace`](@ref), with
which a repeated `optimize!` allocates nothing. Time spent in `resjac!` is counted under gradients
in the [`OptimizationResult`](@ref).
