module MaternQ
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: Symmetric
using OffsetArrays
export Qmatrix_matern2d

include("matern_values.jl")

""" Directly construct a SparseMatrixCSC with a 2d nearest neighbour structure used by
the sparse Matern precision matrix.
"""
function matern_grid2_CSC(nrow, ncol, nu::Int, f)
    colptr = fill(0, nrow*ncol+1)
    # Upper bound
    nnz = cld((nu*2+3)^2,2)*nrow*ncol
    ii = 1
    col = 1 # result matrix column index
    rowval = Vector{Int}(undef, nnz)
    nzval = Vector{Float64}(undef, nnz)
    li = LinearIndices((1:nrow, 1:ncol))
    for j = 1:ncol
        for i = 1:nrow
            colptr[col] = ii
            col += 1
            for l = -(nu+1):(nu+1)
                for k = -(nu+1):(nu+1)
                    if checkindex(Bool, 1:nrow, i+k) && checkindex(Bool, 1:ncol, j+l) && abs(k) + abs(l) <= nu + 1
                        @assert ii <= nnz
                        I = CartesianIndex(i, j)
                        J = CartesianIndex(i+k, j+l)
                        @inbounds nzval[ii] = f(I, J)
                        @inbounds rowval[ii] = li[J]
                        ii += 1
                    end
                end
            end
        end
    end
    colptr[end] = ii
    nnz = ii-1
    resize!(nzval, nnz)
    resize!(rowval, nnz)
    SparseMatrixCSC(ncol*nrow, ncol*nrow, colptr, rowval, nzval)
end
import GaussianRandomFields
dense_cache_matrix_size = (0,0)
dense_cache_valid = falses(1000)
dense_cache = Vector{Matrix{Float64}}(undef, 1000)
""" Constuct a dense precision matrix by inverting a dense covariance matrix
"""
function Qmatrix_matern2d_dense_cached(size, prec, range, nu::Int)
    global dense_cache_matrix_size, dense_cache, dense_cache_valid
    rangei = round(Int, range*100)
    range = rangei/100.0
    if dense_cache_matrix_size != size
        fill!(dense_cache_valid, false)
        dense_cache_matrix_size = size
    end
    if dense_cache_valid[rangei]
        Q = dense_cache[rangei]
    else
        covf = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(range/4, nu, σ=1.0))
        C = GaussianRandomFields.apply(covf, 1:size[1], 1:size[2])
        Q = inv(C)
        dense_cache[rangei] = Q
        dense_cache_valid[rangei] = true
    end
    Symmetric(Q * prec)
end
function Qmatrix_matern2d_dense(size, prec, range, nu::Int)
    covf = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(range/4, nu))
    C = GaussianRandomFields.apply(covf, 1:size[1], 1:size[2])
    Symmetric(inv(C) * prec)
end

dense = Ref(false)
function Qmatrix_matern2d(size, prec, range, nu::Int)
    dense[] && return Qmatrix_matern2d_dense_cached(size, prec, range, nu)
    nrow, ncol = size
    Q = matern_grid2_CSC(nrow, ncol, nu, Q_matern2d_fun(prec, range, nu))
    Symmetric(Q)
end

end # module
