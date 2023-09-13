using NumericalFreeProbability, Test


@testset "cauchytransforms.jl" begin
    # standard semicircle
    sc = Semicircle()

    # marchenko-pastur c = 2 
    mp2 = ChebyshevUMeasure(3-2√2, 3+2√2, x -> √2/(π*x))

    # marchenko-pastur c = 4 
    mp4 = ChebyshevUMeasure(1, 9, x -> 2/(π*x))

    # jacobi measure
    jm = JacobiMeasure(-1, 1, 1, 1, 3/4)

    t1 = 1+2im
    t2 = [0+0.5im, -3]
    @testset "cauchytransform" begin
        G_sc = z -> (z - √(z-2) * √(z+2))/2
        @test isapprox(cauchytransform(t1, sc), G_sc(t1))
        @test all(map(isapprox, cauchytransform(t2, sc), G_sc.(t2)))

        G_mp4 = z -> (z - 3 - √(z-1) * √(z-9))/(2*z)
        @test isapprox(cauchytransform(t1, mp4), G_mp4(t1))
        @test all(map(isapprox, cauchytransform(t2, mp4), G_mp4.(t2)))

        G_jm = z -> -3/4 * ((z^2-1)*(log(z+1) - log(z-1)) - 2z)
        @test isapprox(cauchytransform(t1, jm), G_jm(t1))
        @test all(map(isapprox, cauchytransform(t2, jm), G_jm.(t2)))
    end # testset

    @testset "dcauchytransform_sqrt" begin
        dG_sc = z -> 1/2 - z / (2 * √(z-2) * √(z+2))
        @test isapprox(dcauchytransform(t1, sc), dG_sc(t1))
        #@test all(map(isapprox, dcauchytransform(t2, sc), dG_sc.(t2)))
        dG_mp4 = z -> -(5*z - 9 - 3 * √(z-1) * √(z-9))/(2 * z^2 * √(z-1) * √(z-9))
        @test isapprox(dcauchytransform(t1, mp4), dG_mp4(t1))
        #@test all(map(isapprox, dcauchytransform(t2, mp4), G_mp4.(t2)))
        dG_jm = z -> -3/2 * (z * (log(z+1) - log(z-1)) - 2)
        @test isapprox(dcauchytransform(t1, jm), dG_jm(t1))
        #@test all(map(isapprox, dcauchytransform(t2, jm), dG_jm.(t2)))
    end # testset
end


