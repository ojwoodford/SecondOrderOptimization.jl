using SecondOrderOptimization, Static, LinearAlgebra, Test
include("fixtures/load.jl"); loadfixtures("gradientdescent.jl")   # :gradientdescent, the tutorial's user-defined iterator
const II = SecondOrderOptimization.IteratorInterface

# check_iterator (src/iteratorconformance.jl): the built-in iterators and the tutorial's conform; and for each
# requirement it checks, an iterator broken in that one way makes it throw that requirement's message

@testset "check_iterator: the built-in iterators and the tutorial's conform" begin
    for it in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :gradientdescent)
        @test II.check_iterator(it) === true
    end
    @test_throws ArgumentError II.check_iterator(:not_an_iterator)
end

# Gradient descent (the tutorial's), broken in one way for each K: the requirement it breaks, with a second letter
# where a requirement has a second way to fail
mutable struct BrokenIterator{K}
    stepsize::Float64
    calls::Int           # the calls to iterate! since construct or reset!
    g::Vector{Float64}   # storage of its own, for calling fdf! behind the loop's back (K = :8b)
    H::Matrix{Float64}
end
const BROKEN = (Symbol("1a"), Symbol("1b"), Symbol("2"), Symbol("3"), Symbol("5"), Symbol("6"), Symbol("7a"), Symbol("7b"), Symbol("8a"), Symbol("8b"))
for K in BROKEN
    s = QuoteNode(Symbol(:broken, K))
    @eval II.iteratorname(::StaticSymbol{$s}) = $("broken $K")
    @eval II.construct(::StaticSymbol{$s}, ls, ::StaticBool) =
        (n = length(II.similarstep(ls)); BrokenIterator{$(QuoteNode(K))}(1.0, 0, zeros(n), zeros(n, n)))
end
II.reset!(d::BrokenIterator) = (d.stepsize = 1.0; d.calls = 0; nothing)
II.reset!(d::BrokenIterator{Symbol("7a")}) = (d.stepsize = 1.0; nothing)   # 7: forgets the count, so a second run differs
II.restrictedstep(d::BrokenIterator{Symbol("3")}) = 0                 # 3: not a Bool

# The hooks, each a no-op but for its K
before!(d, data, f, fdf!) = nothing
before!(d::BrokenIterator{Symbol("7a")}, data, f, fdf!) = (d.calls += 1; d.calls > 20 && (d.stepsize *= 0.3); nothing)
before!(d::BrokenIterator{Symbol("5")}, data, f, fdf!) = II.systemisfinite(data) || error("a non-finite Hessian")   # 5: throws
before!(d::BrokenIterator{Symbol("7b")}, data, f, fdf!) = (d.g = copy(II.getgradient(data)); nothing)              # 7: allocates
before!(d::BrokenIterator{Symbol("8a")}, data, f, fdf!) = (f(II.getstep(data)); nothing)                           # 8: an uncounted f
before!(d::BrokenIterator{Symbol("8b")}, data, f, fdf!) = (fdf!(d.g, d.H, II.getstep(data)); nothing)              # 8: an uncounted fdf!
after(d, data, cost) = cost
after(d::BrokenIterator{Symbol("1a")}, data, cost) = Float32(cost)                                                  # 1: not a Float64
after(d::BrokenIterator{Symbol("1b")}, data, cost) = cost + 1                                                       # 1: not f at the trial point
after(d::BrokenIterator{Symbol("2")}, data, cost) = (II.getstep(data) .*= 2; cost)                                 # 2: the step is not the trial's
after(d::BrokenIterator{Symbol("6")}, data, cost) = isnan(cost) ? error("a NaN cost") : cost                        # 6: throws

function II.iterate!(d::BrokenIterator, data, f::F, fdf!::FDF, options) where {F, FDF}
    before!(d, data, f, fdf!)
    gradient = II.getgradient(data)
    step = II.getstep(data)
    step .= .-gradient .* d.stepsize
    II.settrial!(data)
    cost = after(d, data, II.trialcost!(data, f))
    inner = 0
    while !(cost <= II.getbestcost(data)) && inner < 100 && d.stepsize >= floatmin(Float64)
        inner += 1
        d.stepsize *= 0.5
        step .= .-gradient .* d.stepsize
        II.settrial!(data)
        cost = after(d, data, II.trialcost!(data, f))
    end
    d.stepsize *= 2.0
    return cost
end

@testset "check_iterator: an iterator broken in one way breaks that requirement" begin
    for (K, k, where_) in ((Symbol("1a"), 1, "at iteration 1: returned a Float32"),
                           (Symbol("1b"), 1, "at iteration 1: returned"),
                           (Symbol("2"), 2, "at iteration 1"),
                           (Symbol("3"), 3, "at iteration 1: it is a Int64"),
                           (Symbol("5"), 5, ": the run threw"),
                           (Symbol("6"), 6, ": the run threw"),
                           (Symbol("7a"), 7, "on the second run: its costs differ"),
                           (Symbol("7b"), 7, "on the second run: it allocated"),
                           (Symbol("8a"), 8, "at the end of the run: "),
                           (Symbol("8b"), 8, "gradient computations counted"))
        @testset "requirement $k ($K)" begin
            err = try II.check_iterator(Symbol(:broken, K)); nothing catch e; e end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("check_iterator(:broken$K): requirement $k (", msg)
            @test occursin(where_, msg)
            # it names the problem
            @test occursin(r" is violated on (the |Rosenbrock)", msg)
        end
    end
end
