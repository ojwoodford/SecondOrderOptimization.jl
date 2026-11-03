# API Reference

## Primary Functions

```@docs
optimize
```

## Variable & Retraction Interface

```@docs
nvars
update
update!
```

## Options & Results

```@docs
OptimizationOptions
OptimizationResult
OptimizationWorkspace
OptimizationInternal
```

## Iteration Strategies

`SecondOrderOptimization.jl` supports multiple trust-region and line-search iteration strategies, passed via the `iterator` keyword of [`OptimizationOptions`](@ref):

```@docs
OptimizationIterator
newton
levenbergmarquardt
levenbergmarquardt_dtd
dogleg
gradientdescent
getiterator
NewtonData
NewtonDataRelinearize
DoglegData
LevenbergData
LevenbergDataTrial
MarquardtData
MarquardtDataTrial
GradientDescentData
```

## Linear Solvers

```@docs
OptimizationLinearSolver
auto_solver
cholesky_solver
ldl_solver
qr_solver
solve!
zero!
```

## Callbacks & Diagnostics

```@docs
CostTrajectory
storecostscallback
printoutcallback
nullcallback
```
