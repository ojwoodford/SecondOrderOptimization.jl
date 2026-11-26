# Writing Your Own Iterator

An iterator is the algorithm inside each iteration of the optimization loop: it decides the step. The package has
four built in (`:newton`, `:levenbergmarquardt`, `:levenbergmarquardt_dtd` and `:dogleg`), and you can add your own
through [`SecondOrderOptimization.IteratorInterface`](@ref SecondOrderOptimization.IteratorInterface), without
touching the package. This tutorial writes a backtracking gradient descent, registered as `:gradientdescent`.
Gradient descent is not a second-order method, and it was removed from the package for that reason; it makes a short
example because it uses only the gradient and never solves the linear system.

The whole iterator:

```julia
using SecondOrderOptimization.IteratorInterface, Static, LinearAlgebra

# The iterator's state between iterations: the step size, a multiple of the negative gradient
mutable struct GradientDescentData
    stepsize::Float64
end

# Registers the name: OptimizationOptions(iterator = :gradientdescent) now accepts it
IteratorInterface.iteratorname(::StaticSymbol{:gradientdescent}) = "gradient descent"

# One method, whatever the callback may change: this iterator holds no trial linearization
IteratorInterface.construct(::StaticSymbol{:gradientdescent}, ls, ::StaticBool) = GradientDescentData(1.0)

# A reused workspace repeats a first run
IteratorInterface.reset!(gdd::GradientDescentData) = (gdd.stepsize = 1.0; nothing)

# `f` and `fdf!` are only passed on, so type parameters make Julia specialise on them (otherwise each call allocates)
function IteratorInterface.iterate!(gdd::GradientDescentData, data, f::F, fdf!::FDF, options)::Float64 where {F, FDF}
    gradient = getgradient(data)   # finite: the loop stops on a non-finite gradient before iterating
    step = getstep(data)
    step .= .-gradient .* gdd.stepsize
    settrial!(data)
    cost = trialcost!(data, f)
    inner_iters = 0
    while !(cost <= getbestcost(data))
        inner_iters += 1
        if inner_iters > 100 || !(gdd.stepsize >= floatmin(Float64)) || !isfinite(gdd.stepsize)
            return cost
        end
        if isfinite(cost)
            # Minimiser of the quadratic through the cost and slope at 0 and the cost at the trial
            # step, as a fraction of the step, safeguarded to [0.1, 0.5]. The slope term is not
            # positive and the trial cost is above the best, so `costdiff` is negative, never zero.
            slope = dot(step, gradient)
            costdiff = getbestcost(data) + slope - cost
            gdd.stepsize *= clamp(0.5 * slope / costdiff, 0.1, 0.5)
        else
            gdd.stepsize *= 0.5
        end
        step .= .-gradient .* gdd.stepsize
        settrial!(data)
        cost = trialcost!(data, f)
    end
    gdd.stepsize *= 2.0
    return cost
end
```

## Step by step

**The data.** `GradientDescentData` is the iterator's state between iterations: here only the step size, a multiple
of the negative gradient. A second-order iterator would hold more, such as its damping, its trust radius, or work
vectors made by `similarstep(ls)` in `construct`.

**The name.** Defining `iteratorname` for `StaticSymbol{:gradientdescent}` registers the symbol: from then on
`OptimizationOptions(iterator = :gradientdescent)` accepts it, and `show(result)` prints the name.

**`construct`** builds the data for a run on the linear system `ls`. Its third argument says whether the callback may
change the state. An iterator that holds a `TrialLinearization` (to reuse a linearization at an accepted trial point)
passes it on, `TrialLinearization(ls, callbackchangesstate)`; gradient descent holds none, so it takes any
`StaticBool` with one method.

**`reset!`** restores the state that `construct` returned. A run with a reused `OptimizationWorkspace` calls it before
it starts, so that it repeats a first run.

**`iterate!`** is one iteration. It reads the gradient (`getgradient(data)`), writes its step into the step storage
(`getstep(data)`), where the loop will read it, sets the trial point from that step (`settrial!(data)`), and evaluates
the cost there (`trialcost!(data, f)`, which counts and times the evaluation). The loop moves to the point `iterate!`
returns, even if its cost is higher, so an iterator that wants only descent steps backtracks inside `iterate!`: here
until the cost is no higher than the best so far (`getbestcost(data)`), shrinking the step by the minimiser of a
quadratic fitted along it, or halving it when the trial cost is not finite. After 100 failed trials it returns the
last one, and the loop's own tests then stop the run. It returns the trial cost, a `Float64`.

The iterator defines neither `restrictedstep` (its steps are never held back by a safeguard: the default `false`
stands) nor `linearize!` (the default linearizes at the point the loop moved to). Nor does it print a value of its own
in `printoutcallback`: a user-defined iterator prints the four standard columns.

## Running it

With the iterator defined, it is used like any other:

```julia
using SecondOrderOptimization

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function fdf!(g, H, x)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    H[1, 1] = 2 - 400 * x[2] + 1200 * x[1]^2
    H[1, 2] = H[2, 1] = -400 * x[1]
    H[2, 2] = 200
    return f(x)
end

x, result = optimize([-1.2, 1.0], HessianObjective(f, fdf!), OptimizationOptions(iterator = :gradientdescent, maxiters = 1000))
```

Gradient descent is slow on Rosenbrock's curved valley, as a first-order method is: after its 1000 iterations it is
still short of the minimum at `(1, 1)` (at about `(0.91, 0.82)`), which `iterator = :levenbergmarquardt` reaches in
about 20.

## Checking it

`check_iterator` runs an iterator on small problems of its own and checks the contract of
[The Iterator Interface](../iterator_interface.md): that the cost returned is the cost at the trial point, that the
trial point is the step from the current point, that a second run on a workspace repeats the first and allocates
nothing, that the evaluation counts are right, and that hard problems end without an exception. It returns `true`, or
throws an error naming the first requirement broken:

```julia
using SecondOrderOptimization.IteratorInterface
check_iterator(:gradientdescent)   # true
```
