module SecondOrderOptimization

using FastLinAlg
using MicroInfo
using OptimizationVariableInterface
using LoopVectorization
import OptimizationVariableInterface: nvars, update, update!, copyvars, syncvars!, isbitsvartype

# Exported public API
export optimize
export OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal
export OptimizationIterator, newton, levenbergmarquardt, dogleg, gradientdescent, levenbergmarquardt_dtd, getiterator
export NewtonData, NewtonDataRelinearize, DoglegData, LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData
export OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver
export CostTrajectory
export nullcallback, printoutcallback, storecostscallback
export nvars, update, update!, zero!, solve!

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
