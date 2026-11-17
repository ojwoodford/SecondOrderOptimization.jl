"""
    MAX_STATIC_SYSTEM = 16

The largest number of variables for which, without a Hessian prototype, the linear system is
static (stack-allocated `StaticArrays`). See [`usesstaticsystem`](@ref).
"""
const MAX_STATIC_SYSTEM = 16

"""
    usesstaticsystem(x0) -> Bool

Whether `optimize` with `H = nothing` builds a static linear system: when `nvars(x0)` is a static
size (as for a `Number`, a `StaticVector`, or a tuple of them) of at most
[`MAX_STATIC_SYSTEM`](@ref). Otherwise the system is dense, of size `nvars(x0)`. The rule does not
depend on `linsolver`.
"""
# NLLSModel has its own rule for when to build a static single-variable Hessian. It is deliberately
# not shared: neither package depends on the other except weakly (decided 2026-11-16). The two rules
# may differ; when they do, the Hessian is converted to the system's storage, which costs a copy, not
# correctness.
usesstaticsystem(x0) = isstaticsize(nvars(x0))

# For a static `nv` this is resolved at compile time, so `systemsize` is type stable
isstaticsize(nv) = dynamic(is_static(nv)) && nv <= MAX_STATIC_SYSTEM

# The size the linear system is asked for: `nvars(x0)`, the number of degrees of freedom (not
# `length(x0)`, which differs for a vector of multi-dimensional variables), as a `StaticInt` when the
# static rule holds (the system is then static) and an `Int` otherwise
systemsize(x0) = usesstaticsystem(x0) ? nvars(x0) : dynamic(nvars(x0))
