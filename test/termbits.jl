# Test-only helpers for asking why a run stopped, on the exported TERMINATION_* masks (Q8)
using SecondOrderOptimization

# Whether any bit of `mask` is set in the result's termination
hasbits(r, mask::Integer) = (r.termination & mask) != 0
# Whether no bit outside `mask` is set (the reasons are a subset of those in `mask`)
onlybits(r, mask::Integer) = (r.termination & ~mask) == 0

# Names for the bits, for @info records only
const TERMBIT_NAMES = ((TERMINATION_COST_INF, "cost_inf"), (TERMINATION_COST_NAN, "cost_nan"), (TERMINATION_REL_DCOST, "rel_dcost"),
    (TERMINATION_ABS_DCOST, "abs_dcost"), (TERMINATION_STEP_INF, "step_inf"), (TERMINATION_STEP_NAN, "step_nan"),
    (TERMINATION_STEP_SIZE, "step_size"), (TERMINATION_MAX_FAILS, "max_fails"), (TERMINATION_MAX_ITERS, "max_iters"),
    (TERMINATION_MAX_TIME, "max_time"), (TERMINATION_COST_DISCREPANCY, "cost_discrepancy"), (TERMINATION_START_NOT_FINITE, "start_not_finite"),
    (TERMINATION_GRAD_NOT_FINITE, "grad_not_finite"), (TERMINATION_GRAD_TOL, "grad_tol"), (TERMINATION_LINSOLVER_FAILED, "linsolver_failed"))
termnames(r) = join([name for (bit, name) in TERMBIT_NAMES if hasbits(r, bit)], ",") * (r.termination >> 16 != 0 ? ",callback" : "")
