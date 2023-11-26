using NumericalFreeProbability, Test


@testset "cauchytransforms.jl" begin
    sc = Semicircle()
    mp4 = ChebyshevUMeasure(1, 9, x -> 2/(π*x))
    jm = JacobiMeasure(-1, 1, 1, 1, 3/4)
    pm = PointMeasure([-2,2],[0.5,0.5])

    G_sc = z -> (z - √(z-2) * √(z+2))/2
    G_mp4 = z -> (z - 3 - √(z-1) * √(z-9))/(2*z)
    G_jm = z -> -3/4 * ((z^2-1)*(log(z+1) - log(z-1)) - 2z)
    G_pm = z -> (inv(z-2) + inv(z+2))/2

    t1 = 1+2im
    t2 = [0+0.5im, -3]
    @testset "cauchytransform" begin
        for (G, m) in [(G_sc, sc), (G_mp4, mp4), (G_jm, jm), (G_pm, pm)]
            @test isapprox(Gₘ(t1, m), G(t1))
            @test all(map(isapprox, Gₘ(t2, m), G.(t2)))
        end
    end # testset
    
    dG_sc = z -> 1/2 - z / (2 * √(z-2) * √(z+2))
    dG_mp4 = z -> -(5*z - 9 - 3 * √(z-1) * √(z-9))/(2 * z^2 * √(z-1) * √(z-9))
    dG_pm = z -> -(inv(z-2)^2 + inv(z+2)^2)/2
    dG_jm = z -> -3/2 * (z * (log(z+1) - log(z-1)) - 2)

    @testset "dcauchytransform_sqrt" begin
        for (dG, m) in [(dG_sc, sc), (dG_mp4, mp4), (dG_jm, jm), (dG_pm, pm)]
            @test isapprox(Gₘ′(t1, m), dG(t1))
        end
    end # testset
end


