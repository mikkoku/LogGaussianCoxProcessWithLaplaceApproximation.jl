# LogGaussianCoxProcessWithLaplaceApproximation

This package implements the marginal log-likelihood approximation and the poisson correction used in [1].

[1] Kuronen, M., Särkkä, A., Vihola, M., Myllymäki, M. Hierarchical log Gaussian Cox process for regeneration in uneven-aged forests. Environ Ecol Stat (2021). [10.1007/s10651-021-00514-3](https://doi.org/10.1007/s10651-021-00514-3)

# Installation

```julia
import Pkg
Pkg.add(url="https://github.com/mikkoku/PointPatternStatistics.jl")
Pkg.add(url="https://github.com/mikkoku/LogGaussianCoxProcessWithLaplaceApproximation.jl")
```

# Usage

```julia
using LogGaussianCoxProcessWithLaplaceApproximation
using PointPatternStatistics
using Distributions: logpdf
using Plots

# Create a random point pattern
win = (x=(0, 40), y=(0, 40))
xy = [(40*rand(), 40*rand()) for _ in 1:6000]
pp = PointPattern(xy, win)

# Mean field for the Gaussian process
mu = zeros(200, 200)
mu[50:150, 20:40] .= 3

# Discretization size should be the same as the mean field
di = Discretization(0.1:0.2:39.9, win)

# Model object for the LGCP
lgcp = LogGaussianCoxProcess(2, 2.5, 1.3, mu)

# Compute log-likelihood
logpdf(lgcp, di, pp)

# Generate and plot a realization from the LGCP
pp2 = rand(lgcp, di)
length(pp2)
plot(pp2)

# Compute log-likelihoods for three sets of parameters
logpdf(lgcp, di, pp2)
logpdf(LogGaussianCoxProcess(2, 2.5, 2.3, mu), di, pp2)
logpdf(LogGaussianCoxProcess(2, 2.5, 1.3, zeros(200, 200)), di, pp2)
```