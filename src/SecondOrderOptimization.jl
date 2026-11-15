"""
    SecondOrderOptimization

Second-order (Newton-type) unconstrained optimization of smooth functions, from a gradient and
Hessian ([`HessianObjective`](@ref)) or from residuals and their Jacobian
([`JacobianObjective`](@ref)). Run it with [`optimize`](@ref) or [`optimize!`](@ref), configured
by [`OptimizationOptions`](@ref).
"""
module SecondOrderOptimization

using FastNorm
using Printf: @printf, @sprintf
using OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars, syncvars!, ismutablevar

# The public API: every exported name is documented in the API reference, and nothing else is
# exported. The test suite pins this list (test/publicapi.jl).
export optimize, optimize!
export HessianObjective, JacobianObjective
export OptimizationOptions, OptimizationResult, OptimizationWorkspace, isconverged
export CostTrajectory
export nullcallback, printoutcallback, storecostscallback
export TERMINATION_COST_INF, TERMINATION_COST_NAN, TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST
export TERMINATION_STEP_INF, TERMINATION_STEP_NAN, TERMINATION_STEP_SIZE, TERMINATION_MAX_FAILS
export TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME
export TERMINATION_START_NOT_FINITE, TERMINATION_GRAD_TOL, TERMINATION_GRAD_NOT_FINITE
export TERMINATION_LINSOLVER_FAILED, TERMINATION_STALLED

# Utilities
include("utils.jl")
include("stats.jl")

# Linear systems and solvers
include("linearsystem.jl")
include("linearsolver.jl")

# Core options, results, and internal data structures
include("structs.jl")
include("objectives.jl")
include("OptimizationWorkspace.jl")

# Callbacks
include("callbacks.jl")

# Iterators
include("iterators.jl")

# The residual-and-Jacobian linear systems, dense and sparse (after the iterators, for their
# Marquardt scaling method)
include("tallsystem.jl")
include("sparsetallsystem.jl")

# Optimization driver and loop
include("optimize.jl")

# Conformance check for linear systems
include("conformance.jl")

# API docstrings
include("docstrings.jl")

end # module SecondOrderOptimization
