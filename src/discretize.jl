struct Discretization{WT, GT <: AbstractRange}
    size::Tuple{Int,Int}
    gridx::GT
    gridy::GT
    window::WT
end
function Discretization(size, window)
    nx, ny = size
    wx, wy = window
    stepx = (wx[2]-wx[1])/nx
    stepy = (wy[2]-wy[1])/ny
    Discretization(size, wx[1]+stepx/2:stepx:wx[2], wy[1]+stepy/2:stepy:wy[2], window)
end
function Discretization(grid::AbstractRange, window)
    n = length(grid)
    Discretization((n, n), grid, grid, window)
end

function issquare(d::Discretization)
    step(d.gridx) == step(d.gridy)
end

function discretize(x, x0, sx, x1)
    x0 <= x <= x1 || error("Not all points are in the window.")
    xi = 1+floor(Int, (x-x0)/sx)
    if x==x1 xi -= 1 end
    xi
end
# function cellcounts(::Type{SparseArrays.SparseVector}, d::Discretization, xy::PointPattern)
#
# end
function cellcounts(::Type{Vector{Tuple{Int,Int}}}, d::Discretization, xy::PointPattern)
    counts = cellcounts(Matrix, d, xy)

    [(k,v) for (k,v) in enumerate(counts) if v != 0]
end

function cellcounts(::Type{Matrix}, d::Discretization, xy::PointPattern)
    x0, x1 = d.window.x
    y0, y1 = d.window.y
    sx = (x1-x0)/d.size[1]
    sy = (y1-y0)/d.size[2]
    counts = zeros(Int, d.size)
    for (x, y) in xy
        counts[discretize(x, x0, sx, x1), discretize(y, y0, sy, y1)] += 1
    end
    counts
end
