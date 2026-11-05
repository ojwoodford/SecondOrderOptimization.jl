module SecondOrderOptimization

using FastNorm
using Printf: @printf, @sprintf
using OptimizationVariableInterface
using LoopVectorization
import OptimizationVariableInterface: nvars, update, update!, copyvars, syncvars!

# Exported public API
export optimize, optimize!
export OptimizationOptions, OptimizationResult, OptimizationWorkspace
export CostTrajectory
export nullcallback, printoutcallback, storecostscallback

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
