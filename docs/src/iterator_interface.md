```@meta
CurrentModule = SecondOrderOptimization
```

# The Iterator Interface

An **iterator** is the algorithm inside each iteration of the optimization loop. The loop linearizes the objective
at the current point, hands the linear system to the iterator, which chooses a step and a trial point, and then
decides from the trial cost whether the run has converged. The four built-in iterators (`:newton`,
`:levenbergmarquardt`, `:levenbergmarquardt_dtd` and `:dogleg`) are written against the interface described here,
and so can yours: [Writing Your Own Iterator](tutorials/custom_iterator.md) works through an example.

The interface is the submodule `SecondOrderOptimization.IteratorInterface`:

```julia
using SecondOrderOptimization.IteratorInterface
```

Its functions belong to `SecondOrderOptimization` (two, `systemeltype` and `dampingfloor`, to DampedLinearSolve),
so a method you add, written `IteratorInterface.iterate!(...) = ...`, extends the function the loop calls.

```@docs; canonical=false
SecondOrderOptimization.IteratorInterface
```

## What an iterator defines

An iterator is named by a symbol, `:name`, which selects it in [`OptimizationOptions`](@ref)`(iterator = :name)`. It
defines four methods, and may define three more:

- [`iteratorname`](@ref) for `StaticSymbol{:name}` (required): its display name. Defining this method registers the
  symbol.
- [`construct`](@ref) for `StaticSymbol{:name}` (required): builds the iterator's own data, its state between
  iterations, from the linear system. It may call [`similarstep`](@ref) for work vectors and
  [`TrialLinearization`](@ref), and nothing else of the system.
- [`reset!`](@ref) (required): restores that data to the state `construct` returned, so that a run with a reused
  [`OptimizationWorkspace`](@ref) repeats a first run.
- [`iterate!`](@ref) (required): one iteration.
- [`restrictedstep`](@ref) (optional, default `false`): whether the step last returned was held back by the
  iterator's safeguard.
- [`triallinearization`](@ref) (optional, default `nothing`): the iterator's trial linearization, if it holds one.
- [`linearize!`](@ref) (optional): the linearization at the point the loop moved to. The default uses the trial
  linearization when it is ready, and otherwise evaluates `fdf!`; override it only to do something else.

```@docs
iteratorname
construct
reset!
iterate!
restrictedstep
triallinearization
linearize!
```

## What `iterate!` returns

`iterate!(itdata, data, f, fdf!, options)` returns a `Float64`, one of:

- **the cost at the trial point**, which it has set with [`settrial!`](@ref), leaving the step it took in
  [`getstep`](@ref)`(data)`;
- [`LINSOLVER_FAILED`](@ref), when no linear solve succeeded: the run stops with `TERMINATION_LINSOLVER_FAILED`;
- `NaN`, when no finite step could be formed: the run stops with `TERMINATION_COST_NAN`.

`data` is the loop's state. Its type and fields are internal: an iterator reads it, and changes it, only through the
functions below. Of `options`, the run's [`OptimizationOptions`](@ref), an iterator reads the fields `dstep` and
`reldcost` (its smallest useful step, and the relative change in cost below which it may stop retrying), and hands
`options` itself to [`solvestep!`](@ref). Annotate `f` and `fdf!` with type parameters (`f::F, fdf!::FDF ... where
{F, FDF}`): an iterator only passes them on, and without the parameters Julia does not specialise the method on them,
so that each evaluation is a dynamic call that allocates.

## What the loop does around `iterate!`

- **It moves to the returned point, even if its cost is higher**, and keeps the best point itself, returning it at the
  end. An iterator that rejects a step (a damping or a trust region that a trial failed) retries inside `iterate!`,
  and returns the trial it settles on, accepted or not.
- **It undamps the system** after `iterate!` returns a cost, before the callback, so that an iterator need not, and
  the callback sees the undamped system. An iterator that needs the undamped model within `iterate!` (to predict the
  change in cost of a damped step, say) calls [`undamp!`](@ref) itself.
- **It reads [`restrictedstep`](@ref)** after an accepted step: a restricted step is small because of the
  iterator's safeguard, not because the run has converged, so the loop does not then test `reldcost` or `dstep`
  (it still tests `absdcost`).
- **It linearizes** at the new point by [`linearize!`](@ref), unless the run has stopped.
- **It counts and times** each linear solve, cost and linearization through the functions that make them
  ([`solvestep!`](@ref), [`trialcost!`](@ref), [`linearizetrial!`](@ref)); an iterator calls `f` and `fdf!` only
  through those, so that the result's counts are right.

## The functions an iterator calls

The linear system, the step and the costs:

```@docs
solvestep!
undamp!
getstep
getgradient
predictedchange
curvature
maxdiagonal
getdiagonal!
systemisfinite
systemeltype(::SecondOrderOptimization.OptimizationInternal)
dampingfloor(::SecondOrderOptimization.OptimizationInternal)
similarstep
```

The trial point and its cost:

```@docs
getbestcost
settrial!
trialcost!
linearizetrial!
LINSOLVER_FAILED
```

## Saving an evaluation: the trial linearization

An iterator that expects its trial point to be accepted can linearize there while it computes the cost: the
linearization (the gradient and the Hessian, or the residual and the Jacobian) gives the cost too, so the iteration
evaluates `fdf!` once instead of `f` and then `fdf!` at the next point. A [`TrialLinearization`](@ref) in the
iterator's data holds a spare system for this. The iterator calls [`trialcost!`](@ref)`(tl, data, f, fdf!, likely)`
for the cost of a trial, with `likely` its guess that the trial will be accepted (Levenberg–Marquardt and dogleg guess
from the predicted reduction of their first trial); [`discardtrial!`](@ref) when it rejects that trial; and returns
`tl` from [`triallinearization`](@ref). The default [`linearize!`](@ref) then swaps the spare system in. Built with
`callbackchangesstate = true` the component holds no spare and always evaluates `f`, since the callback may change
the point or the cost after the trial.

```@docs
TrialLinearization
discardtrial!
```

An iterator that always moves to its trial point (Newton) has no need for the spare: it linearizes at the trial point
into the current system by [`linearizetrial!`](@ref), and returns that cost from its own [`linearize!`](@ref).

## Checking an iterator

[`check_iterator`](@ref) runs an iterator on problems of its own and checks the contract above, throwing an error
that names the first requirement broken. Call it in your iterator's tests.

```@docs
check_iterator
```
