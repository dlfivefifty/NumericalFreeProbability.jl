using NumericalFreeProbability, Test

@testset "measures.jl" begin
    @testset "ChebyshevUMeasure" begin
        um = ChebyshevUMeasure(-2, 2, x -> 1 + x^2)
        @test isapprox(um[0], 1)
        um = normalize(um)
        @test isapprox(sum(um), 1)
    end

    @testset "JacobiMeasure" begin
        jm = JacobiMeasure(-1, 1, 2, 2, one)
        @test isapprox(jm[0], 1)
        @test isapprox(sum(jm), 16/15)
    end

    @testset "Semicircle" begin
        sc = Semicircle()
        @test isapprox(sc[0], 1/π)
        @test isapprox(sum(sc), 1)
    end

    @testset "PointMeasure" begin
        pm = PointMeasure([-2,-1,1], [0.5, 0.25, 0.25])
        @test isapprox(pm[1], 0.25)
        @test isapprox(sum(pm), 1)
    end
end