using LinearAlgebra, Random

# Least-squares test problems as one concrete type: the data in fields, the problem chosen by `kind`. Every
# problem of a testset then gives the same residual and Jacobian types, and so one Jacobian-objective type and
# one Hessian-objective type, where a pair of anonymous functions per problem gives a type per problem.
#   :rosenbrock  Rosenbrock as two residuals, m = 2
#   :expfit      an exponential fit with a nonzero residual, x[1] exp(x[2] t) − y, m = 25
#   :sin30       M (x + sin(x)/2) − y, n = 30, m = 45
#   :tanh30      M tanh(x) − y, n = 30, m = 45
struct LSQProblem
    kind::Symbol
    t::Vector{Float64}
    y::Vector{Float64}
    M::Matrix{Float64}
end

# The residual and Jacobian functions of a problem, each allocating its result
struct LSQResidual
    p::LSQProblem
end
struct LSQJacobian
    p::LSQProblem
end
function (f::LSQResidual)(x)
    p = f.p
    p.kind === :rosenbrock && return [1 - x[1], 10 * (x[2] - x[1]^2)]
    p.kind === :expfit && return @.(x[1] * exp(x[2] * p.t) - p.y)
    p.kind === :sin30 && return p.M * (x .+ 0.5 .* sin.(x)) .- p.y
    return p.M * tanh.(x) .- p.y
end
function (f::LSQJacobian)(x)
    p = f.p
    p.kind === :rosenbrock && return [-1.0 0.0; -20x[1] 10.0]
    p.kind === :expfit && return hcat(exp.(x[2] .* p.t), x[1] .* p.t .* exp.(x[2] .* p.t))
    p.kind === :sin30 && return p.M * Diagonal(1 .+ 0.5 .* cos.(x))
    return p.M * Diagonal(sech.(x) .^ 2)
end

# The three well-conditioned problems of a testset, as (name, residual, Jacobian, m, x0); `n30` is :sin30 or
# :tanh30
function lsq_problems(n30::Symbol)
    t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
    rng = Xoshiro(11); M = randn(rng, 45, 30) ./ sqrt(45) + I(45)[:, 1:30]; yy = randn(rng, 45)
    none = zeros(0, 0)
    rosen = LSQProblem(:rosenbrock, Float64[], Float64[], none)
    expfit = LSQProblem(:expfit, t, y, none)
    big = LSQProblem(n30, Float64[], yy, M)
    return (
        ("Rosenbrock", LSQResidual(rosen), LSQJacobian(rosen), 2, [-1.2, 1.0]),
        ("exponential fit", LSQResidual(expfit), LSQJacobian(expfit), 25, [1.0, 0.0]),
        ("n = 30", LSQResidual(big), LSQJacobian(big), 45, zeros(30)),
    )
end
