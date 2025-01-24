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

# Simple Bayesian analysis
```julia
using LogGaussianCoxProcessWithLaplaceApproximation
using PointPatternStatistics
using AdaptiveMCMC
using Distributions
using MCMCChains
using StatsPlots

win = (x=(0, 40), y=(0, 40))
di2 = Discretization((40, 40), win)
# Simulate a LGCP point process and try to infer its parameters
pp3 = rand(LogGaussianCoxProcess(2, 2.5, 1.3, 0.6), di2)

# The log posterior for the problem
function log_p(x)
    range, sd, mean = x

    if range < 1.0 || sd <= 0
        return -Inf
    end
    l = logpdf(LogGaussianCoxProcess(2, range, sd, mean), di2, pp3)
    l += logpdf(Uniform(1, 10), range)
    l += logpdf(Truncated(Normal(1, 5), 0.0, Inf), sd)
    l += logpdf(Normal(0, 5), mean)
    l
end

# Run adaptive MCMC for 1000 iterations
out = adaptive_rwm([2.0, 1.0, 0.0], log_p, 1000)

# Compute MCMC statistics using MCMCChains.jl
# 1000 iterations is not enough for real analysis
chn = Chains(out.X', [:range, :sd, :mean])

plot(chn)
```