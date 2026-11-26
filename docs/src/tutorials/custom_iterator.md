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
