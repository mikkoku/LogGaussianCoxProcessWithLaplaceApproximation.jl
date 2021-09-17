using LogGaussianCoxProcessWithLaplaceApproximation
using PointPatternStatistics
using Distributions: logpdf
using Test

@testset "LogGaussianCoxProcessWithLaplaceApproximation.jl" begin
    win = (x=(0, 40), y=(0, 40))
    di = Discretization(0.1:0.2:39.9, win)
    xy = [(40*rand(), 40*rand()) for _ in 1:1000]
    pp = PointPattern(xy, win)
    @test isfinite(logpdf(LogGaussianCoxProcess(2, 1.5, 2.3, randn(200, 200)), di, pp))
    @test rand(LogGaussianCoxProcess(2, 1.5, 2.3, randn(200, 200)), di) isa PointPattern
end
