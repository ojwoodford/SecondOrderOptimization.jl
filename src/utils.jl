function fast_dot(a::AbstractVector, b::AbstractVector)
    total = zero(promote_type(eltype(a), eltype(b)))
    @inbounds @fastmath @simd for i in eachindex(a, b)
        total += a[i] * b[i]
    end
    return total
end
