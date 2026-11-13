# Least Squares from a Jacobian

Many objectives are sums of squares, $f(x) = \tfrac12 \lVert r(x) \rVert^2$. You can give such a
problem to [`optimize`](@ref) in two ways:

- as a [`HessianObjective`](@ref), whose `fdf!` forms the gradient $J^\top r$ and the
  Gauss-Newton Hessian $J^\top J$ itself; or
- as a [`JacobianObjective`](@ref), whose `resjac!` writes only the residual $r$ and the
  Jacobian $J$. The optimizer then solves each step by a QR factorization of $J$ and never forms
  $J^\top J$.

Both give the same Levenberg-Marquardt iteration. They differ in how accurately each step is
solved, and in what that accuracy costs.

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
  with respect to the step of [`update`](@ref), so in the tangent space for a manifold variable),
  and returns the cost at `x`;
- `r` and `J` arrive zeroed, so `resjac!` may accumulate into them;
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
The Jacobian form reaches the least-squares minimum to rounding in about 20 iterations. The
normal equations with `linsolver = :cholesky` are still above the minimum after 500 iterations:
their steps are only as good as the rounded $J^\top J$. Ill-conditioned Jacobians come from
polynomial and spline fits in a monomial basis, from parameters of very different scales, and
from nearly redundant parameterizations.

When $J$ is well conditioned, the two forms give the same answer and the normal equations are
cheaper.

## What it costs

For $J$ of size $m \times n$:

| | Normal equations (`HessianObjective`) | QR of the Jacobian (`JacobianObjective`) |
|:---|:---|:---|
| Once per linearization | form $J^\top J$ and $J^\top r$: $O(mn^2)$, in your `fdf!` | Householder QR of $[J\ r]$: $O(mn^2)$, about twice the flops of forming $J^\top J$ |
| Per damping value tried | Cholesky of $J^\top J + \lambda D$: $O(n^3)$ | QR of the stacked $2n \times n$ matrix $[R;\ \sqrt{\lambda} D^{1/2}]$: $O(n^3)$, independent of $m$ |
| Storage | $n \times n$ | $J$, plus a copy of $[J\ r]$ for the factorization (about twice the storage of $J$), plus $2n \times (n + 1)$ |

The per-damping cost does not depend on $m$, so the retries with a larger $\lambda$ after a
rejected step stay cheap however many residuals there are. An undamped solve on a numerically
rank-deficient $J$ (including $m < n$) is reported as `NotPositiveDefinite`, just as Cholesky
reports singular normal equations, and Levenberg-Marquardt raises $\lambda$.

## What this first version supports

| Accepted | Rejected for now, with an `ArgumentError` before `f` or `resjac!` is called |
|:---|:---|
| `J` a `Matrix{Float64}` prototype, or an integer `m ≥ 1` | a sparse or any other prototype, or another element type |
| `iterator = :levenbergmarquardt` | `:levenbergmarquardt_dtd`, `:newton`, `:dogleg`, `:gradientdescent` |
| `linsolver = :auto` or `:qr` (both the QR solve above) | every other linear solver |

Everything else works as for a `HessianObjective`: every variable type, including immutable and
manifold variables through [`optimize`](@ref); both Levenberg-Marquardt variants
(`callbackchangesstate`); callbacks and [`CostTrajectory`](@ref); the termination criteria, with
`gtol` applied to $\lVert J^\top r \rVert_\infty$; and an [`OptimizationWorkspace`](@ref), with
which a repeated `optimize!` allocates nothing. Time spent in `resjac!` is counted under gradients
in the [`OptimizationResult`](@ref).
