using NumericalFreeProbability
using Test, LinearAlgebra

@testset "cauchytransform_point" begin
    M = PointMeasure([-2,-1,1], [0.5, 0.25, 0.25])
    @test isapprox(cauchytransform_point(1+2im, M), complex(37/208, -55/208))
    @test isapprox(cauchytransform_point(0.5, M), -2/15)
end # testset

@testset "invcauchytransform_point" begin
    M = PointMeasure([-1,1], [0.5, 0.5])

    inv = invcauchytransform_point(0.125 - 0.375im, M)
    @test any(isapprox.(inv, -0.2+0.4im))
    @test any(isapprox.(inv, 1+2im))

    inv = invcauchytransform_point(0.375, M)
    @test any(isapprox.(inv, 3))
    @test any(isapprox.(inv, -1/3))
end # testset
    