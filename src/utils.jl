using SparseArrays, StaticArrays

function fast_dot(a::AbstractVector, b::AbstractVector)
    total = zero(promote_type(eltype(a), eltype(b)))
    @inbounds @fastmath @simd for i in eachindex(a, b)
        total += a[i] * b[i]
    end
    return total
end

function fast_bAb(A::Matrix, b::Vector)
    total = zero(eltype(b))
    @inbounds for i in eachindex(b)
        subtotal = zero(eltype(b))
        @fastmath @simd for j in eachindex(b)
            subtotal += A[j, i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::StaticArray, b::StaticArray)
    total = zero(eltype(b))
    @inbounds for i in eachindex(b)
        subtotal = zero(eltype(b))
        @fastmath @simd for j in eachindex(b)
            subtotal += A[j, i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::SparseMatrixCSC, b::Vector)
    total = zero(eltype(b))
    for i in eachindex(b)
        coltotal = zero(eltype(b))
        for j in nzrange(A, i)
            @inbounds @fastmath coltotal += A.nzval[j] * b[A.rowval[j]]
        end
        @inbounds @fastmath coltotal *= b[i]
        total += coltotal
    end
    return total
end
