using SecondOrderOptimization, Test
using SecondOrderOptimization: TimeCounter, StatsCounter, Stats, @elapsed_ns, @stats, bytesstring, timestring

@testset "stats.jl" begin
    # timestring
    @test timestring(0) == "0ns"
    @test timestring(500) == "500ns"
    @test timestring(1500) == "1.5μs"
    @test timestring(1500000) == "1.5ms"
    @test timestring(1500000000) == "1.5s"
    @test timestring(90000000000) == "1.5 minutes"
    @test timestring(7200000000000) == "2 hours"
    @test timestring(720000000000000) == "200 hours"

    # bytesstring
    @test bytesstring(0) == "0 bytes"
    @test bytesstring(500) == "500 bytes"
    @test bytesstring(1023) == "1023 bytes"
    @test bytesstring(1024) == "1KB"
    @test bytesstring(2048) == "2KB"
    @test bytesstring(1048576 * 5) == "5MB"
    @test bytesstring(1073741824 * 3) == "3GB"
    @test bytesstring(1099511627776 * 4) == "4TB"
    @test bytesstring(1099511627776 * 4000) == "4000TB"

    # TimeCounter and @elapsed_ns
    tc = TimeCounter()
    @test tc.time_ns == 0 && tc.count == 0
    tc2 = TimeCounter(UInt64(42), 3)
    @test tc2.time_ns == 42 && tc2.count == 3
    tc += UInt64(100)
    tc += UInt64(50)
    @test tc.time_ns == 150 && tc.count == 2
    y = 0
    t = @elapsed_ns y = sum(1:10)
    @test t isa UInt64
    @test y == 55
    tc += t
    @test tc.count == 3 && tc.time_ns == 150 + t

    # StatsCounter, Stats, and @stats
    sc = StatsCounter()
    @test (sc.num_allocs, sc.bytes_allocd, sc.time_ns, sc.count) == (0, 0, 0, 0)
    sc += Stats(1)
    sc += Stats(2, 3, UInt64(4))
    @test (sc.num_allocs, sc.bytes_allocd, sc.time_ns, sc.count) == (3, 4, 5, 2)

    z = nothing
    st = @stats z = zeros(1000)
    @test length(z) == 1000
    @test st.num_allocs >= 1
    @test st.bytes_allocd >= 8000
    @test st.time_ns > 0

    sc += st
    @test sc.count == 3
    @test sc.bytes_allocd >= 8004

    # Snapshot Stats()
    s1 = Stats()
    x = zeros(1000)
    s2 = Stats()
    @test s2.bytes_allocd >= s1.bytes_allocd
    @test s2.time_ns >= s1.time_ns
    @test s2.num_allocs >= s1.num_allocs
end
