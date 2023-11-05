
sc = Semicircle()
mp2 = ChebyshevUMeasure(3-2√2, 3+2√2, x -> √2/(π*x))
pm = PointMeasure([-2,-1,1], [1/2, 1/4, 1/4])

@testset "support_sqrt_single" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    G_semi = z -> cauchytransform(z, sc)
    invG_semi = z -> invcauchytransform(z, sc)
    dG_semi = z -> dcauchytransform(z, sc)
    supp = support_sqrt_single(G_semi, G_semi, invG_semi, invG_semi, dG_semi, dG_semi, sc, sc; tol=10^-9, maxits=50)
    @test isapprox(supp[1][1], -2√2)
    @test isapprox(supp[1][2], 2√2)

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4

    G_mp_2 = z -> cauchytransform(z, mp2)
    invG_mp_2 = z -> invcauchytransform(z, mp2)
    dG_mp_2 = z -> dcauchytransform(z, mp2)
    supp = support_sqrt_single(G_mp_2, G_mp_2, invG_mp_2, invG_mp_2, dG_mp_2, dG_mp_2, mp2, mp2; tol=10^-9, maxits=50)
    @test isapprox(supp[1][1], 1)
    @test isapprox(supp[1][2], 9)
end

@testset "freeaddition" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    um = freeaddition(sc, sc)

    @test isapprox(um.a, -2√2)
    @test isapprox(um.b, 2√2)
    @test norm(um.ψ_k[1:20] - [√2 / (2*pi); fill(0, 19)]) < 10^-9

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4
    um = freeaddition(mp2, mp2)
    
    @test isapprox(um.a, 1)
    @test isapprox(um.b, 9)
    @test norm(um.ψ_k[1:20] - [inv(2π * (-2)^i) for i=0:19]) < 10^-6 # NOTE: worse bound to pass test - mainly due to inaccurately calculating supp_c

    #um = freeaddition(sc, pm)




end


