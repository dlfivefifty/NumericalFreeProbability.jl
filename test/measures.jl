

@testset "ChebyshevUMeasure" begin
    um = ChebyshevUMeasure(-2, 2, one)
    @test isapprox(um[0], 1/π)
end

@testset "JacobiMeasure" begin
    jm = JacobiMeasure(-1, 1, 2, 2, one)
    @test isapprox(jm[0], 15/16)
end