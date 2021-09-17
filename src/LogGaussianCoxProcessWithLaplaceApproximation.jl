module LogGaussianCoxProcessWithLaplaceApproximation

import LambertW
import Optim
import LinearAlgebra
import SparseArrays
using LinearAlgebra: cholesky, cholesky!, logdet, norm, Symmetric
using LinearAlgebra: diag
using SpecialFunctions: logabsgamma
using PointPatternStatistics: PointPattern, area
#import FillArrays
import Distributions
#using GaussianRandomFields: CovarianceFunction, Matern, GaussianRandomField, CirculantEmbedding
import GaussianRandomFields

# For Poisson correction
using FFTW
using QuadGK: quadgk


struct PoissonProcess{WT}
    intensity::Matrix{Float64}
    window::WT
end

struct LogGaussianCoxProcess{MT} # with laplace approximation and Matern covariance
    # Parameters for matern covariance
    # mean field
    nu::Int
    range::Float64
    sd::Float64
    mean::MT # Matrix
end
LogGaussianCoxProcess(nu, range, sd, mean) = LogGaussianCoxProcess(nu, float(range), float(sd), mean)

# Could check the length of x here
Distributions.insupport(d::LogGaussianCoxProcess, x) = true

include("discretize.jl")
include("matern_precision.jl")
using .MaternQ

include("util.jl")
include("laplace_approximation.jl")
include("log_dens.jl")
include("poissoncorrection.jl")

export LogGaussianCoxProcess,
    Discretization,
    PoissonProcess


end
