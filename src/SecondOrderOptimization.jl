module SecondOrderOptimization

using FastNorm
using Printf: @printf, @sprintf
using OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars, syncvars!

# Exported public API
export optimize, optimize!
export OptimizationOptions, OptimizationResult, OptimizationWorkspace, isconverged
export CostTrajectory
export nullcallback, printoutcallback, storecostscallback
export AbstractLinearSystem, solve!, dampen!, validatelinsolver, zero!, makelinearsystem, copy_structure, sharex, fast_bAb, fast_dot, quadratic_model_reduction
export SolveStatus, Success, NotPositiveDefinite, Indefinite
export OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver
export OptimizationIterator, newton, levenbergmarquardt, levenbergmarquardt_dtd, dogleg, gradientdescent, getiterator
export NewtonData, NewtonDataRelinearize, DoglegData
export LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData
export OptimizationInternal
export TERMINATION_COST_INF, TERMINATION_COST_NAN, TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST
export TERMINATION_STEP_INF, TERMINATION_STEP_NAN, TERMINATION_STEP_SIZE, TERMINATION_MAX_FAILS
export TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME, TERMINATION_COST_DISCREPANCY
export TERMINATION_START_NOT_FINITE, TERMINATION_GRAD_TOL, TERMINATION_GRAD_NOT_FINITE
export TERMINATION_LINSOLVER_FAILED, LINSOLVER_FAILED

# Utilities
include("utils.jl")
include("stats.jl")

# Linear systems and solvers
include("linearsystem.jl")
include("linearsolver.jl")

# Core options, results, and internal data structures
include("structs.jl")
include("OptimizationWorkspace.jl")

# Callbacks
include("callbacks.jl")

# Iterators
include("iterators.jl")

# Optimization driver and loop
include("optimize.jl")

# API docstrings
include("docstrings.jl")

end # module SecondOrderOptimization
