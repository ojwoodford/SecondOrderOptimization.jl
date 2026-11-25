"""
    SecondOrderOptimization

Second-order (Newton-type) unconstrained optimization of smooth functions, from a gradient and
Hessian ([`HessianObjective`](@ref)) or from residuals and their Jacobian
([`JacobianObjective`](@ref)). Run it with [`optimize`](@ref) or [`optimize!`](@ref), configured
by [`OptimizationOptions`](@ref).
"""
module SecondOrderOptimization

using FastNorm
using LinearAlgebra, SparseArrays, StaticArrays, Static
using DampedLinearSolve
# Two functions of the linear-system contract (DampedLinearSolve's exports) gain a method for the loop's state, for the
# iterator interface
import DampedLinearSolve: systemeltype, dampingfloor
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
export TERMINATION_START_NOT_FINITE, TERMINATION_GRAD_INF, TERMINATION_GRAD_NAN, TERMINATION_GRAD_TOL
export TERMINATION_LINSOLVER_FAILED, TERMINATION_STALLED
export IteratorInterface

# Utilities
include("utils.jl")
include("stats.jl")

# The rule for a static system, decided from the variables (the systems are DampedLinearSolve's)
include("staticrule.jl")

# Core options, results, and internal data structures
include("structs.jl")
include("objectives.jl")
include("OptimizationWorkspace.jl")

# Callbacks
include("callbacks.jl")

# The functions an iterator calls, and the iterators
include("iteratorinterface.jl")
include("triallinearization.jl")
include("iterators.jl")

# Optimization driver and loop
include("optimize.jl")

# API docstrings
include("docstrings.jl")

# The iterator interface, public through this submodule: the functions are the parent's (and two DampedLinearSolve's),
# so that a method a user adds through it extends the same function
"""
    SecondOrderOptimization.IteratorInterface

The interface for writing an iterator: the methods an iterator defines (`iteratorname`, `construct`, `reset!`,
`iterate!`, and optionally `restrictedstep`, `triallinearization` and `linearize!`), and the functions it calls on the loop's state
`data`. See the manual's page on the iterator interface.
"""
module IteratorInterface
import ..SecondOrderOptimization: iteratorname, construct, reset!, iterate!, restrictedstep, triallinearization, linearize!,
    solvestep!, undamp!, systemisfinite, systemeltype, dampingfloor, getgradient, getstep, similarstep,
    predictedchange, curvature, maxdiagonal, getdiagonal!, getbestcost, settrial!, trialcost!, linearizetrial!,
    TrialLinearization, discardtrial!, LINSOLVER_FAILED
export iteratorname, construct, reset!, iterate!, restrictedstep, triallinearization, linearize!
export solvestep!, undamp!, systemisfinite, systemeltype, dampingfloor, getgradient, getstep, similarstep
export predictedchange, curvature, maxdiagonal, getdiagonal!, getbestcost, settrial!, trialcost!, linearizetrial!
export TrialLinearization, discardtrial!, LINSOLVER_FAILED
end

end # module SecondOrderOptimization
