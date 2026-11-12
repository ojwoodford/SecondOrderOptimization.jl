using Documenter
using SecondOrderOptimization
using OptimizationVariableInterface

DocMeta.setdocmeta!(SecondOrderOptimization, :DocTestSetup, :(using SecondOrderOptimization, OptimizationVariableInterface); recursive=true)

makedocs(
    sitename = "SecondOrderOptimization.jl",
    modules = [SecondOrderOptimization, OptimizationVariableInterface],
    checkdocs = :none,
    remotes = nothing,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = false,
        sidebar_sitename = true,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples & Tutorials" => [
            "Zero-Allocation Static Optimization" => "tutorials/static_optimization.md",
            "Manifold Optimization" => "tutorials/manifold_optimization.md",
            "Sparse Large-Scale Optimization" => "tutorials/sparse_optimization.md",
            "Callbacks & Step Monitoring" => "tutorials/callbacks.md",
        ],
        "Migration Guides" => [
            "From Optim.jl" => "migration/from_optim.md",
            "From LeastSquaresOptim.jl" => "migration/from_leastsquaresoptim.md",
            "From JSOSolvers / NLPModels" => "migration/from_jso.md",
        ],
    ]
)

# Publishes to the gh-pages branch when run on GitHub Actions (see .github/workflows/Documentation.yml);
# does nothing when run locally.
deploydocs(
    repo = "github.com/ojwoodford/SecondOrderOptimization.jl.git",
    devbranch = "main",
    push_preview = false,
)
