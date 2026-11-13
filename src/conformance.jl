using LinearAlgebra, SparseArrays

# Reading and writing the Hessian storage of a linear system as a dense matrix, for
# `check_linear_system`. A new kind of Hessian storage needs a method of each.

"""
    conformance_matrix(A) -> Matrix{Float64}

The matrix that the Hessian storage `A` (as returned by `getA`) represents, as a dense `Matrix`.
Used by [`check_linear_system`](@ref); a new kind of Hessian storage needs a method.
"""
conformance_matrix(A::AbstractMatrix) = Matrix{Float64}(A)

"""
    conformance_fill!(A, M) -> A

Write the entries of the dense matrix `M` into the Hessian storage `A`, wherever `A` can hold a
value (every entry of a dense matrix, the stored entries of a sparse one); other entries of `M`
are ignored. Used by [`check_linear_system`](@ref); a new kind of Hessian storage needs a method.
"""
conformance_fill!(A::AbstractMatrix, M::AbstractMatrix) = copyto!(A, M)
function conformance_fill!(A::SparseMatrixCSC, M::AbstractMatrix)
    @inbounds for col in 1:size(A, 2), k in nzrange(A, col)
        A.nzval[k] = M[A.rowval[k], col]
    end
    return A
end

"""
    conformance_fillb!(ls, b) -> ls

Write the right-hand side `b` into the linear system `ls`, after its Hessian storage has been
filled with [`conformance_fill!`](@ref). The default sets `getb(ls) .= b`; a system whose `b` is
derived from other storage (as `b = Jᵀr` for a [`DenseTallLinearSystem`](@ref)) needs a method
that also fills that storage. Used by [`check_linear_system`](@ref).
"""
conformance_fillb!(ls, b::AbstractVector) = (getb(ls) .= b; ls)

"""
    conformance_traits(A) -> (; exact, maxrank, indefinite)

What [`check_linear_system`](@ref) can expect of the Hessian storage `A`:
- `exact`: whether [`conformance_fill!`](@ref) followed by [`conformance_matrix`](@ref) gives the
  matrix back exactly; otherwise to a relative `1e-10` (as for `JᵀJ` built from a factor).
- `maxrank`: the largest rank of matrix the storage can hold; below `n`, the checker uses a dense
  positive semidefinite test matrix of that rank, a right-hand side in its range, and a positive
  base damping where it would otherwise solve undamped, and checks that an undamped solve does not
  return `Success`.
- `indefinite`: whether the storage can hold an indefinite matrix; if not, the indefinite-`A`
  checks are skipped.

The default, `(exact = true, maxrank = typemax(Int), indefinite = true)`, fits every square storage.
"""
conformance_traits(A) = (exact = true, maxrank = typemax(Int), indefinite = true)

# The tall storage: A = JᵀJ. Filling factors M (its largest min(m, n) eigenpairs) into J through
# a fixed orthonormal basis, so J is dense; a non-finite diagonal entry of M is written into the
# first row of J, so the corresponding diagonal of JᵀJ is non-finite too.
function conformance_matrix(A::TallHessian)
    J = A.J
    n = size(J, 2)
    M = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        total = 0.0
        for k in axes(J, 1)
            total += J[k, i] * J[k, j]
        end
        M[i, j] = total
    end
    return M
end
function conformance_fill!(A::TallHessian, M::AbstractMatrix)
    m, n = size(A.J)
    k = min(m, n)
    Mf = [isfinite(M[i, j]) ? Float64(M[i, j]) : 0.0 for i in 1:n, j in 1:n]
    E = eigen(Symmetric((Mf + Mf') / 2))
    idx = n-k+1:n
    W = Matrix(qr([sin(0.9 * i + 1.1 * j + 0.2 * i * j) for i in 1:m, j in 1:k]).Q)[:, 1:k]
    A.J .= W * Diagonal(sqrt.(max.(E.values[idx], 0.0))) * E.vectors[:, idx]'
    for i in 1:n
        isfinite(M[i, i]) || (A.J[1, i] = M[i, i])
    end
    markstale!(A)
    return A
end
function conformance_fillb!(ls::DenseTallLinearSystem, b::AbstractVector)
    J = ls.A.J
    ls.b .= b
    if all(isfinite, J)
        ls.A.r .= transpose(J) \ b    # Jᵀr = b: minimum norm for m ≥ n, b in the range of Jᵀ otherwise
    else
        fill!(ls.A.r, NaN)
    end
    markstale!(ls.A)
    return ls
end
conformance_traits(A::TallHessian) = (exact = false, maxrank = min(size(A.J)...), indefinite = false)

# Deterministic test data, so the checker needs no random number generator
conformance_symmetric(n) = [sin(1.7 * (i + j) + 0.3 * i * j) for i in 1:n, j in 1:n]
conformance_vector(n, phase) = [cos(1.3 * i + phase) for i in 1:n]

function conformance_error(rule, linsolver, detail="")
    error("check_linear_system: rule \"$rule\" violated" * (isnothing(linsolver) ? "" : " (linsolver = :$linsolver)") *
          (isempty(detail) ? "" : ": $detail"))
end

relerr(x, y) = norm(x - y) / max(norm(y), floatmin(Float64))

# A fresh system, filled with the matrix `Q` and right-hand side `b`
function conformance_system(make_ls, s, Q, b)
    ls = applicable(make_ls, s) ? make_ls(s) : make_ls()
    conformance_fill!(getA(ls), Q)
    conformance_fillb!(ls, b)
    return ls
end

# Solve with the current damping and check the status and that A and b are untouched. `Qw` is
# the matrix the storage held after it was filled.
function conformance_solve!(ls, s, Qw, b, what)
    status = solve!(ls, s)
    status == Success || conformance_error("solve! returns Success for a positive-definite system", s, "$what returned $status")
    conformance_matrix(getA(ls)) == Qw || conformance_error("solve! leaves A as it was written", s, what)
    getb(ls) == b || conformance_error("solve! leaves b as it was written", s, what)
    return copy(getx(ls))
end

"""
    check_linear_system(make_ls; n, linsolvers, rtol = 1e-8)

Check that the linear systems built by `make_ls` meet the [`AbstractLinearSystem`](@ref)
interface, for each solver in `linsolvers` (symbols such as `:auto`, `:cholesky`, `:ldl`).
Throws an `ErrorException` naming the first rule that is broken; returns `nothing` otherwise.

`make_ls(linsolver)` (or `make_ls()`, if that is the only method) must return a fresh system of
dimension `n` whose Hessian storage can hold a dense symmetric matrix wherever its pattern allows.
The test matrices are built inside that pattern and made diagonally dominant, so they are
positive definite. Reading and writing the storage uses [`conformance_matrix`](@ref),
[`conformance_fill!`](@ref) and [`conformance_fillb!`](@ref), which have methods for `Matrix`,
`MMatrix`, `SparseMatrixCSC`, the [`TallHessian`](@ref) of a `DenseTallLinearSystem` and (with
BlockLinearSystems loaded) `SchurBlockMatrix`; [`conformance_traits`](@ref) says what the storage
can represent. For the tall storage, `JᵀJ` is compared with the test matrix to a relative `1e-10`
(then exactly with itself), the indefinite-`A` checks are skipped, and when `m < n` the test
matrix has rank `m`, the undamped solve must not return `Success`, and the solves that are
otherwise undamped use a damping of `0.25`. Solutions are compared with `rtol`; use a looser
value for iterative solvers.

The rules, for each solver:
- `getA`, `getb`, `getx` have dimension `n`.
- `solve!` returns `Success` for a positive-definite system and solves `A * x = b` into `getx`
  (the iterators negate it to get the step), leaving `A` and `b` exactly as they were written:
  undamped, damped, and also when the solve fails.
- `dampen!(ls, λ)` solves `(A + λI) x = b`; `dampen!(ls, λ, s)` solves `(A + λ Diagonal(s)) x = b`;
  each replaces the damping set before, so returning to an earlier damping gives the identical `x`.
- `copy_structure(ls)` has its own `A` and `b`: filling and solving it changes neither the
  original's `A` and `b` nor the original's next solution.
- `sharex(trial, ls)` returns a system whose `getx` is `getx(ls)`.
- `fast_bAb(getA(ls), v) ≈ v' A v` and `quadratic_model_change(ls, v) ≈ v' A v / 2 + b' v`.
- `max_diag(ls)` is the largest absolute diagonal entry of `A`, and `NaN` if a diagonal entry is.
- `zero!(ls)` zeroes `A` and `b`, and the system can be refilled and solved afterwards.
- For an indefinite `A`, `:cholesky` does not return `Success` and `:ldl` returns `Indefinite`
  (checked only for the solvers in `linsolvers`).

# Example
```julia
check_linear_system(s -> makelinearsystem(zeros(5), zeros(5, 5), s); n = 5, linsolvers = (:auto, :cholesky, :ldl, :qr))
check_linear_system(s -> SecondOrderOptimization.DenseTallLinearSystem(8, 5); n = 5, linsolvers = (:auto, :qr))
```
"""
function check_linear_system(make_ls; n::Integer, linsolvers, rtol::Real = 1e-8)
    for s in linsolvers
        check_linear_system_solver(make_ls, n, s, rtol)
    end
    return nothing
end

function check_linear_system_solver(make_ls, n, s, rtol)
    # Test data inside the system's pattern: fill a symmetric matrix, read back what was kept, and
    # make it diagonally dominant (so symmetric positive definite)
    ls = applicable(make_ls, s) ? make_ls(s) : make_ls()
    length(getb(ls)) == n || conformance_error("getb has length n", s, "got $(length(getb(ls))), n = $n")
    length(getx(ls)) == n || conformance_error("getx has length n", s, "got $(length(getx(ls))), n = $n")
    traits = conformance_traits(getA(ls))
    conformance_fill!(getA(ls), conformance_symmetric(n))
    P = conformance_matrix(getA(ls))
    size(P) == (n, n) || conformance_error("getA is n×n", s, "got $(size(P)), n = $n")
    deficient = traits.maxrank < n
    if deficient
        # Storage of limited rank (a tall system with fewer rows than columns): a dense positive
        # semidefinite matrix of that rank, and right-hand sides in its range
        C = [cos(0.7 * i + 1.9 * j + 0.1 * i * j) + (i == j ? 3.0 : 0.0) for i in 1:traits.maxrank, j in 1:n]
        Q = C' * C
        Q = (Q + Q') / 2
        b = Q * conformance_vector(n, 0.0)
    else
        P = (P + P') / 2
        Q = P + Diagonal(vec(sum(abs, P; dims=2)) .+ 1)
        b = conformance_vector(n, 0.0)
    end
    sc = [1.0 + i / n for i in 1:n]
    λ1, λ2 = 0.37, 2.9
    λ0 = deficient ? 0.25 : 0.0     # the base damping: undamped unless A cannot be definite
    base = deficient ? "solve with base damping" : "undamped solve"

    ls = conformance_system(make_ls, s, Q, b)
    Qw = conformance_matrix(getA(ls))   # what the storage holds: Q itself when the storage is exact
    if traits.exact
        Qw == Q || conformance_error("conformance_fill! and conformance_matrix round-trip", s,
            "the test matrix lies inside the pattern, so writing and reading it back must give it unchanged")
    else
        relerr(Qw, Q) <= 1e-10 || conformance_error("conformance_fill! and conformance_matrix round-trip", s,
            "the test matrix lies inside the pattern, so writing and reading it back must give it to a relative 1e-10; got $(relerr(Qw, Q))")
    end

    # A singular A is not solved as definite
    if deficient
        dampen!(ls, 0.0)
        solve!(ls, s) == Success && conformance_error("an undamped solve of a singular A does not return Success", s,
            "the storage holds rank at most $(traits.maxrank) < n = $n")
        conformance_matrix(getA(ls)) == Qw || conformance_error("solve! leaves A as it was written", s, "after an undamped solve of a singular A")
        getb(ls) == b || conformance_error("solve! leaves b as it was written", s, "after an undamped solve of a singular A")
    end

    # Undamped, then each kind of damping; then back to an earlier damping (no drift)
    dampen!(ls, λ0)
    x0 = conformance_solve!(ls, s, Qw, b, base)
    relerr(x0, (Qw + λ0 * I) \ b) <= rtol || conformance_error("solve! solves A * x = b", s, "relative error $(relerr(x0, (Qw + λ0 * I) \ b))")
    dampen!(ls, λ2, sc)
    x2 = conformance_solve!(ls, s, Qw, b, "solve with dampen!(ls, λ, s)")
    relerr(x2, (Qw + λ2 * Diagonal(sc)) \ b) <= rtol ||
        conformance_error("dampen!(ls, λ, s) solves (A + λ Diagonal(s)) x = b", s, "relative error $(relerr(x2, (Qw + λ2 * Diagonal(sc)) \ b))")
    dampen!(ls, λ1)
    x1 = conformance_solve!(ls, s, Qw, b, "solve with dampen!(ls, λ)")
    relerr(x1, (Qw + λ1 * I) \ b) <= rtol ||
        conformance_error("dampen!(ls, λ) solves (A + λI) x = b", s, "relative error $(relerr(x1, (Qw + λ1 * I) \ b)) (after dampen!(ls, λ, s): the scaling must not persist)")
    dampen!(ls, λ2, sc)
    conformance_solve!(ls, s, Qw, b, "second solve with dampen!(ls, λ, s)") == x2 ||
        conformance_error("dampen! replaces the damping set before (no drift)", s, "returning to dampen!(ls, λ, s) gave a different x")
    dampen!(ls, λ1)
    conformance_solve!(ls, s, Qw, b, "second solve with dampen!(ls, λ)") == x1 ||
        conformance_error("dampen! replaces the damping set before (no drift)", s, "returning to dampen!(ls, λ) gave a different x")
    dampen!(ls, λ0)
    conformance_solve!(ls, s, Qw, b, "second $base") == x0 ||
        conformance_error("dampen! replaces the damping set before (no drift)", s, "returning to no damping gave a different x")

    # copy_structure: its own A and b
    trial = copy_structure(ls)
    getA(trial) !== getA(ls) || conformance_error("copy_structure gives its own A", s, "getA(trial) === getA(ls)")
    getb(trial) !== getb(ls) || conformance_error("copy_structure gives its own b", s, "getb(trial) === getb(ls)")
    Q2 = deficient ? 2.0 * Q : Q + Diagonal(fill(1.5, n))
    b2 = deficient ? Q2 * conformance_vector(n, 0.7) : conformance_vector(n, 0.7)
    conformance_fill!(getA(trial), Q2)
    conformance_fillb!(trial, b2)
    Q2w = conformance_matrix(getA(trial))
    conformance_matrix(getA(ls)) == Qw || conformance_error("copy_structure gives its own A", s, "filling the copy changed the original's A")
    getb(ls) == b || conformance_error("copy_structure gives its own b", s, "filling the copy changed the original's b")
    dampen!(trial, λ0)
    solve!(trial, s) == Success || conformance_error("solve! returns Success for a positive-definite system", s, "on the copy_structure copy")
    relerr(getx(trial), (Q2w + λ0 * I) \ b2) <= rtol || conformance_error("solve! solves A * x = b", s, "on the copy_structure copy")
    conformance_matrix(getA(ls)) == Qw || conformance_error("copy_structure gives its own A", s, "solving the copy changed the original's A")
    conformance_solve!(ls, s, Qw, b, "solve after solving the copy") == x0 ||
        conformance_error("copy_structure shares nothing a solve corrupts", s, "solving the copy changed the original's next solution")

    # sharex, on a separate system (copy_structure may already share x with the original)
    other = applicable(make_ls, s) ? make_ls(s) : make_ls()
    getx(other) !== getx(ls) || conformance_error("make_ls returns a fresh system", s, "two calls returned systems sharing x")
    shared = sharex(other, ls)
    getx(shared) === getx(ls) || conformance_error("sharex(trial, ls) aliases x", s, "getx(sharex(trial, ls)) !== getx(ls)")

    # Quadratic forms, with vectors of the types the iterators pass (the gradient's and the step's)
    v = copy(getb(ls)); v .= conformance_vector(n, 2.1)
    isapprox(fast_bAb(getA(ls), v), dot(v, Qw * v); rtol=1e-10) || conformance_error("fast_bAb(getA(ls), v) == v' A v", s)
    dx = copy(getx(ls)); dx .= conformance_vector(n, 2.1)
    isapprox(quadratic_model_change(ls, dx), 0.5 * dot(dx, Qw * dx) + dot(b, dx); rtol=1e-10) ||
        conformance_error("quadratic_model_change(ls, dx) == dx' A dx / 2 + b' dx", s)

    # max_diag, including a NaN on the diagonal
    max_diag(ls) == maximum(abs, diag(Qw)) || conformance_error("max_diag is the largest absolute diagonal entry", s,
        "got $(max_diag(ls)), expected $(maximum(abs, diag(Qw)))")
    Qnan = copy(Q); Qnan[n, n] = NaN
    lsnan = conformance_system(make_ls, s, Qnan, b)
    isnan(max_diag(lsnan)) || conformance_error("max_diag is NaN if a diagonal entry is NaN", s)

    # zero!, then refill and solve
    zero!(ls)
    all(iszero, conformance_matrix(getA(ls))) || conformance_error("zero! zeroes A", s)
    all(iszero, getb(ls)) || conformance_error("zero! zeroes b", s)
    conformance_fill!(getA(ls), Q)
    conformance_fillb!(ls, b)
    dampen!(ls, λ0)
    relerr(conformance_solve!(ls, s, Qw, b, "solve after zero! and refilling"), (Qw + λ0 * I) \ b) <= rtol ||
        conformance_error("solve! solves A * x = b", s, "after zero! and refilling")

    # Indefinite A: a negative first diagonal entry larger than its row, so nonsingular. Skipped
    # for storage that cannot hold an indefinite matrix (JᵀJ).
    if s in (:cholesky, :ldl) && traits.indefinite
        Qi = copy(Q); Qi[1, 1] = -Q[1, 1]
        lsi = conformance_system(make_ls, s, Qi, b)
        status = solve!(lsi, s)
        s === :cholesky && status == Success && conformance_error("an indefinite A is not solved as positive definite by :cholesky", s)
        s === :ldl && status != Indefinite && conformance_error("an indefinite A gives Indefinite from :ldl", s, "got $status")
        conformance_matrix(getA(lsi)) == Qi || conformance_error("solve! leaves A as it was written", s, "after a solve of an indefinite A")
    end
    return nothing
end
