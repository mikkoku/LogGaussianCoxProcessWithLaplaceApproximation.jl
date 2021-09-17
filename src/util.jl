# Wrap a Factorization in a AbstractSparseMatrix
struct OptimCholWrapper{T} <: SparseArrays.AbstractSparseMatrix{Float64, Int}
    data::T
end
function Base.similar(::OptimCholWrapper, ::Type{T}, n::Int, m::Int) where T
    @assert n == 0 == m
    zeros(0,0)
end
Base.:\(x::OptimCholWrapper, y::Vector{T}) where T = x.data\y
Base.copy(x::OptimCholWrapper) = x #OptimCholWrapper(copy(x.data))
Base.show(io::IO, X::OptimCholWrapper) = show(io, X.data)
LinearAlgebra.logdet(x::OptimCholWrapper) = logdet(x.data)


function lambertwexp(x)
    if x < 700
        LambertW.lambertw(exp(x))
    else
        x - log(x)
    end
end
function weightstonodes(locationweight)
    weightstonodes!(zeros(size(locationweight)), locationweight)
end
function weightstonodes!(A, locationweight, b=1)
    for (i, w) in locationweight
        # w is the sum of weights for the grid point.
        # The contributions of different datapoints may be combined or kept separate.
        A[i] += w*b
    end
    A
end



# Solution for the problem assuming that Q = qI
function initial_guess(fixed_field, locationweight, q)
    aa = weightstonodes!(zero(fixed_field), locationweight)

    @. aa = aa/q - lambertwexp(fixed_field - log(q) + aa/q)
end
