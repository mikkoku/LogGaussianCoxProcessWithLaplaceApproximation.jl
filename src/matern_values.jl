# Precision matrix from
# Lindgren, Rue, Lindtröm (2011) An explicit link ...


# function Q_matern2d(node::Int, nnode, nrow, ncol, prec, range, nu)
#     Q_matern2d([node], [nnode], nrow, ncol, prec, range, nu)[1]
# end

function Q_matern2d_fun(prec, range, nu)
    @assert nu == 2
    #nu = 2
    ν = nu
    ρ = range
    κ = √(8ν) / ρ
    a = 4 + κ^2
    σ² = 1 / (4π * ν * (a - 4)^ν)
    Q0 = OffsetArray([
    a*(a^2+12) -3(a^2+3) 3a -1;
     -3(a^2+3)        6a -3  0;
            3a        -3  0  0;
            -1         0  0  0;
     ], (0:3, 0:3))

     Q0 .*= (prec * σ²)
    (x,y) -> Q_matern2d(x, y, nu, Q0)
end


function Q_matern2d(node1::CartesianIndex, node2, nu, Q0)
    drow, dcol = abs.(Tuple(node1 - node2))

    if drow <= nu+1 && dcol <= nu+1
        @inbounds Q0[drow, dcol]
    else
        0.0
    end
end
