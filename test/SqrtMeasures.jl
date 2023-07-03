using NumericalFreeProbability, Test, ClassicalOrthogonalPolynomials


# standard semicircle chebyshev coefficients
ψ_semi_k = [1/pi]

# marchenko-pastur c = 4 chebyshev coefficients
ψ_x = x -> 2 / (x * pi)
ψ_x_expand = x -> ψ_x(5 + 4 * x)
P = ChebyshevU()
ψ_mp4_k = expand(P, ψ_x_expand).args[2]

@testset "cauchytransform_sqrt" begin

    # standard semicircle

    G_semi = z -> (z - √(z-2) * √(z+2))/2

    for z in (1+2im, 3+0im)
        @test isapprox(cauchytransform_sqrt(z, ψ_semi_k, -2, 2; maxterms=20), G_semi(z))
    end

    # marchenko pastur with c = 4

    G_mp4 = z -> (z - 3 - √(z-1) * √(z-9))/(2*z)

    for z in (1+2im, 0.5+0im)
        @test isapprox(cauchytransform_sqrt(z, ψ_mp4_k, 1, 9; maxterms=40), G_mp4(z))
    end

end # testset

@testset "invcauchytransform_sqrt" begin

    # standard semicircle

    invG_semi = z -> z + 1/z

    for z in (0.1+0.2im, 0.8+0im)
        @test isapprox(invcauchytransform_sqrt(z, ψ_semi_k, -2, 2; maxterms=20), invG_semi(z))
    end

    # marchenko pastur with c = 4

    invG_mp4 = z -> 4/(1-z) + 1/z

    for z in (0.1+0.2im, -0.9+0im)
        @test isapprox(invcauchytransform_sqrt(z, ψ_mp4_k, 1, 9; maxterms=40), invG_mp4(z))
    end

end # testset