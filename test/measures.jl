using NumericalFreeProbability, Test

@testset "measures.jl" begin
    @testset "ChebyshevUMeasure" begin
        um = ChebyshevUMeasure(-2, 2, 1)
        @test isapprox(um[0], 1)
    end

    @testset "JacobiMeasure" begin
        jm = JacobiMeasure(-1, 1, 2, 2, 15/16)
        @test isapprox(jm[0], 15/16)
        @test isapprox(sum(jm), 1)
    end

    @testset "Semicircle" begin
        sc = Semicircle()
        @test isapprox(sc[0], 1/π)
        @test isapprox(sum(sc), 1)
    end
end