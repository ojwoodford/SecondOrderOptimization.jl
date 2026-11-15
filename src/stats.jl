"""
    TimeCounter(time_ns, count)
    TimeCounter()

Lightweight accumulator for elapsed nanosecond timings and call counts.
Adding a `UInt64` nanosecond duration via `tc + ns` increments `time_ns` by `ns` and `count` by `1`.
"""
struct TimeCounter
    time_ns::UInt64
    count::Int64
end
TimeCounter() = TimeCounter(0, 0)
Base.:+(x::TimeCounter, u::UInt64) = TimeCounter(x.time_ns + u, x.count + 1)

"""
    @elapsed_ns ex

Evaluates expression `ex` and returns the elapsed wall-clock time in nanoseconds as a `UInt64`.
"""
macro elapsed_ns(ex)
    quote
        local t0 = Base.time_ns()
        $(esc(ex))
        Base.time_ns() - t0
    end
end

"""
    StatsCounter(num_allocs, bytes_allocd, time_ns, count)
    StatsCounter()

Accumulator for allocation counts, allocated bytes, elapsed nanoseconds, and call counts.
Adding a [`Stats`](@ref) measurement via `sc + s` accumulates all fields and increments `count` by `1`.
"""
struct StatsCounter
    num_allocs::Int64
    bytes_allocd::Int64
    time_ns::UInt64
    count::Int64
end
StatsCounter() = StatsCounter(0, 0, 0, 0)

"""
    Stats(num_allocs, bytes_allocd, time_ns)
    Stats(init)
    Stats()

Snapshot of allocation count, total allocated bytes, and nanosecond timestamp (or elapsed duration).
- `Stats()` captures the current cumulative GC allocation statistics and `Base.time_ns()`.
- `Stats(init)` initializes all three fields to `init`.
"""
struct Stats
    num_allocs::Int64
    bytes_allocd::Int64
    time_ns::UInt64
end
Stats(init) = Stats(init, init, init)
function Stats()
    gc_data = Base.gc_num()
    return Stats(gc_data.malloc + gc_data.realloc + gc_data.poolalloc + gc_data.bigalloc, Base.gc_total_bytes(gc_data), Base.time_ns())
end
Base.:+(x::StatsCounter, u::Stats) = StatsCounter(x.num_allocs + u.num_allocs, x.bytes_allocd + u.bytes_allocd, x.time_ns + u.time_ns, x.count + 1)

"""
    @stats ex

Evaluates expression `ex` and returns a [`Stats`](@ref) instance containing the number of allocations,
bytes allocated, and elapsed nanoseconds incurred during `ex`.
"""
macro stats(ex)
    quote
        local t0 = Base.time_ns()
        local allocs = Base.gc_num()
        $(esc(ex))
        local diff = Base.GC_Diff(Base.gc_num(), allocs)
        Stats(Base.gc_alloc_count(diff), diff.allocd, Base.time_ns() - t0)
    end
end

"""
    bytesstring(nb) -> String

Formats a byte count `nb` into a human-readable string with units (`" bytes"`, `"KB"`, `"MB"`, `"GB"`, `"TB"`).
"""
function bytesstring(nb)
    bytessuffix = (" bytes", "KB", "MB", "GB", "TB")
    suffix = 1
    while nb >= 1024 && suffix < length(bytessuffix)
        nb /= 1024
        suffix += 1
    end
    return @sprintf("%d%s", nb, bytessuffix[suffix])
end

"""
    timestring(ns) -> String

Formats a nanosecond duration `ns` into a human-readable string with units (`"ns"`, `"μs"`, `"ms"`, `"s"`, `" minutes"`, `" hours"`).
"""
function timestring(ns)
    timesuffix = ("ns", "μs", "ms", "s", " minutes", " hours")
    timeperiod = (1000.0, 1000.0, 1000.0, 60.0, 60.0)
    ns = Float64(ns)
    suffix = 1
    while suffix <= length(timeperiod) && ns >= timeperiod[suffix]
        ns /= timeperiod[suffix]
        suffix += 1
    end
    return @sprintf("%g%s", ns, timesuffix[suffix])
end
