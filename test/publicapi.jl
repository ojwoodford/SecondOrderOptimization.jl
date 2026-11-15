using SecondOrderOptimization, Test

# The public API is exactly what is exported, and every exported name is documented. Adding or
# dropping an export is a change to this list.
const PUBLIC_API = sort([
    :CostTrajectory,
    :HessianObjective,
    :JacobianObjective,
    :OptimizationOptions,
    :OptimizationResult,
    :OptimizationWorkspace,
    :SecondOrderOptimization,
    :TERMINATION_ABS_DCOST,
    :TERMINATION_COST_INF,
    :TERMINATION_COST_NAN,
    :TERMINATION_GRAD_NOT_FINITE,
    :TERMINATION_GRAD_TOL,
    :TERMINATION_LINSOLVER_FAILED,
    :TERMINATION_MAX_FAILS,
    :TERMINATION_MAX_ITERS,
    :TERMINATION_MAX_TIME,
    :TERMINATION_REL_DCOST,
    :TERMINATION_STALLED,
    :TERMINATION_START_NOT_FINITE,
    :TERMINATION_STEP_INF,
    :TERMINATION_STEP_NAN,
    :TERMINATION_STEP_SIZE,
    :isconverged,
    :nullcallback,
    :optimize,
    :optimize!,
    :printoutcallback,
    :storecostscallback,
])

@testset "Public API" begin
    @test issorted(PUBLIC_API)
    @test sort(names(SecondOrderOptimization)) == PUBLIC_API
    # (Base.Docs.hasdoc is not in Julia 1.10, so look the binding up in the module's docstrings)
    docs = Base.Docs.meta(SecondOrderOptimization)
    @testset "$name has a docstring" for name in PUBLIC_API
        @test haskey(docs, Base.Docs.Binding(SecondOrderOptimization, name))
    end
    # Internal names stay defined, under the module
    @testset "$name is defined but internal" for name in (
            :OptimizationInternal, :NewtonData, :NewtonDataRelinearize, :DoglegData,
            :LevenbergData, :LevenbergDataTrial, :MarquardtData, :MarquardtDataTrial, :GradientDescentData,
            :OptimizationIterator, :newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg,
            :gradientdescent, :getiterator,
            :OptimizationLinearSolver, :auto_solver, :cholesky_solver, :ldl_solver, :qr_solver,
            :LINSOLVER_FAILED, :fast_dot,
            :AbstractLinearSystem, :solve!, :dampen!, :zero!, :makelinearsystem, :copy_structure,
            :sharex, :quadform, :quadratic_model_change, :validatelinsolver,
            :SolveStatus, :Success, :NotPositiveDefinite, :Indefinite,
            :check_linear_system, :usesstaticsystem, :MAX_STATIC_SYSTEM)
        @test isdefined(SecondOrderOptimization, name)
        @test !(name in names(SecondOrderOptimization))
    end
end

# The API reference documents this package's exports and nothing else (P3): the variable
# contract (nvars, update, update!) belongs to OptimizationVariableInterface, and is explained in
# the manifold tutorial instead.
@testset "The API page lists exactly the exports (P3)" begin
    api = read(joinpath(@__DIR__, "..", "docs", "src", "api.md"), String)
    documented = Symbol[]
    for block in eachmatch(r"```@docs\n(.*?)```"s, api)
        append!(documented, Symbol.(filter(!isempty, strip.(split(block[1], '\n')))))
    end
    @test allunique(documented)
    @test sort(documented) == filter(!=(:SecondOrderOptimization), PUBLIC_API)
    tutorial = read(joinpath(@__DIR__, "..", "docs", "src", "tutorials", "manifold_optimization.md"), String)
    for sig in ("nvars(x::T)", "update(x::T, step, start = 1)", "update!(to::T, from::T, step)",
                "using OptimizationVariableInterface",
                "https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/custom_variables/",
                "https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/api/")
        @test occursin(sig, tutorial)
    end
end

@testset "OptimizationResult.iterator is the Symbol the options selected (P1)" begin
    obj = HessianObjective(x -> x[1]^2, (g, H, x) -> (g[1] = 2x[1]; H[1, 1] = 2.0; x[1]^2))
    # The names `show` printed before P1, when the field held the internal enum
    shown = (newton = "Newton", levenbergmarquardt = "Levenberg-Marquardt",
             levenbergmarquardt_dtd = "Levenberg-Marquardt (D'D scaling)", dogleg = "Dogleg",
             gradientdescent = "gradient descent")
    @test fieldtype(OptimizationResult, :iterator) === Symbol
    for (s, name) in pairs(shown)
        _, r = optimize([1.0], obj, OptimizationOptions(iterator = s))
        @test r.iterator === s
        @test (r + r).iterator === s
        @test startswith(sprint(show, r), "SecondOrderOptimization optimization using $name took ")
    end
end
