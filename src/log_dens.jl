

function LGCPobs(field, data::AbstractVector{Tuple{Int, Int}}, A)
    # The first sum is over the data points and the second is over the grid.
    logA = log(A)
    s = sum((field[i] + logA)*n for (i, n) in data)
    #s - data[:gridcellsize]*sum(exp, field)
    s - A*sum(exp, field)
end

function PPobs(intensity, data::AbstractVector{Tuple{Int, Int}}, A)
    # The first sum is over the data points and the second is over the grid.
    logA = log(A)
    s = sum((log(intensity[i]) + logA)*n for (i, n) in data)
    #s - data[:gridcellsize]*sum(exp, field)
    s - A*sum(intensity)
end

function Distributions.logpdf(d::Union{LogGaussianCoxProcess, PoissonProcess}, di::Discretization, x)
    logpdf_(d, di, x)[1]
end

function logpdf_(d, di::Discretization, x::PointPattern)
    logpdf_(d, di, cellcounts(Vector{Tuple{Int,Int}}, di, x))
end
function logpdf_(d::PoissonProcess, di::Discretization, x::AbstractVector{Tuple{Int, Int}})
    issquare(di) || throw(ArgumentError("Only square discretizations supported."))
    d.window == di.window && size(d.intensity) == di.size || throw(ArgumentError("Incompatible discretization."))
    scale = step(di.gridx)
    l1 = PPobs(d.intensity, x, scale^2)
    l2 = sum(((cell, count),) -> logabsgamma(count+1)[1], x)
    l1 - l2
end
function logpdf_(d::LogGaussianCoxProcess, di::Discretization, x::AbstractVector{Tuple{Int, Int}})
    issquare(di) || throw(ArgumentError("Only square discretizations supported."))
    d.range > 0 || return -Inf
    d.sd > 0 || return -Inf
    nu = d.nu
    sd = d.sd
    scale = step(di.gridx)
    range = d.range / scale # range in cells
    fixedfield = d.mean
    prec = 1/sd^2
    (l, _, pz), random_field_mode, H = poisson_gp(x, fixedfield, prec, range, nu, scale^2, di.size)
    lr = -Inf
    if (l != -Inf)
        l1 = LGCPobs(fixedfield .+ random_field_mode, x, scale^2)
        l2 = sum(((cell, count),) -> logabsgamma(count+1)[1], x)
        lr = pz + l1 - l2
        l = l + l1 - l2
    end
    l, random_field_mode, H, lr
end

function logpdf__(d, di::Discretization, x::PointPattern, l, random_field_mode, H)
    y = cellcounts(Vector{Tuple{Int,Int}}, di, x)
    logpdf__(d, di, y, l, random_field_mode, H)
end
function logpdf__(d, di::Discretization, x::AbstractVector{Tuple{Int, Int}}, l, random_field_mode, H)
    issquare(di) || throw(ArgumentError("Only square discretizations supported."))
    d.range > 0 || return -Inf
    d.sd > 0 || return -Inf
    scale = step(di.gridx)
    fixedfield = d.mean
    if (l != -Inf)
        l1 = LGCPobs(fixedfield .+ random_field_mode, x, scale^2)
        l2 = sum(((cell, count),) -> logabsgamma(count+1)[1], x)
        l = l + l1 - l2
    end
    l, random_field_mode, H
end


function Distributions.mean(m::LogGaussianCoxProcess, d::Discretization)
    size(m.mean) == d.size || error("Discretizations don't agree")
    sd = m.sd
    fixedfield = m.mean
    step(d.gridx) * step(d.gridy) * @.(exp(fixedfield + 0.5*sd^2))
end
function Base.rand(m::LogGaussianCoxProcess, d::Discretization)
    # pts1, pts2 = @. LinRange(d.windowsize / d.gridsize, d.windowsize, d.gridsize)
    # GaussianRandomFields uses a different parametrization for range.
    cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(m.range/4, m.nu, σ=m.sd))
    grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.CirculantEmbedding(), d.gridx, d.gridy)
    g = GaussianRandomFields.sample(grf)
    intensity = @. exp(m.mean + g)
    rand(PoissonProcess(intensity, d.window), d)
end

function Base.rand(x::PoissonProcess, d::Discretization)#; maxn::Int=typemax(Int))
    size(x.intensity) == d.size || error("Discretizations don't agree")
    x.window == d.window || error("Windows don't agree")
    sx = step(d.gridx)
    sy = step(d.gridy)
    rpois(λ) = rand(Distributions.Poisson(λ * sx*sy))
    counts = rpois.(x.intensity)
    n = sum(counts)
    # if n > maxn
    #     throw(ErrorException)
    # end
    xy = Vector{NTuple{2,Float64}}(undef, n)
    index = 1
    for i in 1:d.size[1], j in 1:d.size[2]
        for _ in 1:counts[i, j]
            xy[index] = i*sx - rand()*sx, j*sy - rand()*sy
            index += 1
        end
    end
    PointPattern(xy, d.window)
end
