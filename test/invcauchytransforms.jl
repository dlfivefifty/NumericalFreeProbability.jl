using NumericalFreeProbability, Test, Random

# standard semicircle
sc = Semicircle()

# marchenko-pastur c = 4 
mp4 = ChebyshevUMeasure(1, 9, x -> 2/(π*x))

# jacobi measure
jm = normalize(JacobiMeasure(-2, 2, 1, 2))
jm2 = normalize(JacobiMeasure(-1, 1, 2, 2, x -> 7x^2 + 1))

@testset "invcauchytransform single" begin
    t1 = [0+0.5im, -3]
    for z in t1
        @test isapprox(invcauchytransform(cauchytransform(z, sc), sc)[1], z)
    end
    for z in t1
        @test isapprox(invcauchytransform(cauchytransform(z, mp4), mp4)[1], z)
    end
    for z in t1
        @test isapprox(invcauchytransform(cauchytransform(z, jm), jm)[1], z)
    end
    for z in t1
        @test isapprox(invcauchytransform(cauchytransform(z, jm2), jm2)[1], z)
    end
end # testset

@testset "invcauchytransform multiple" begin
    Random.seed!(153)
    V = randn(ComplexF64, 10)
    @test all(map(isapprox, reduce(hcat, invcauchytransform(cauchytransform(V, sc), sc)), V))
    @test all(map(isapprox, reduce(hcat, invcauchytransform(cauchytransform(V, mp4), mp4)), V))
    # @test all(map(isapprox, reduce(hcat, invcauchytransform(cauchytransform(V, jm), jm)), V))
    # @test_broken all(map(isapprox, reduce(hcat, invcauchytransform(cauchytransform(V, jm2), jm2)), V))
end # testset