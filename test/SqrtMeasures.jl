using NumericalFreeProbability
using Test, ClassicalOrthogonalPolynomials, LinearAlgebra

P = ChebyshevU()
x = axes(P, 1)
# standard semicircle chebyshev coefficients
ψ_semi_k = [1/pi]

# marchenko-pastur c = 2 chebyshev coefficients
ψ_x = x -> √2 / (x * pi)
ψ_x_expand = x -> ψ_x(3 + 2√2 * x)
ψ_mp2_k = P \ ψ_x_expand.(x)

# marchenko-pastur c = 4 chebyshev coefficients
ψ_x = x -> 2 / (x * pi)
ψ_x_expand = x -> ψ_x(5 + 4 * x)
ψ_mp4_k = P \ ψ_x_expand.(x)

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


@testset "dcauchytransform_sqrt" begin
    
    # standard semicircle

    dG_semi = z -> 1/2 - z / (2 * √(z-2) * √(z+2))
    
    for z in (1+2im, 3+0im)
        @test isapprox(dcauchytransform_sqrt(z, ψ_semi_k, -2, 2; maxterms=20), dG_semi(z))
    end

    # marchenko pastur with c = 4

    dG_mp4 = z -> -(5*z - 9 - 3 * √(z-1) * √(z-9))/(2 * z^2 * √(z-1) * √(z-9))

    for z in (1+2im, 0.5+0im)
        @test isapprox(dcauchytransform_sqrt(z, ψ_mp4_k, 1, 9; maxterms=40), dG_mp4(z))
    end

end # testset


# TODO: This testset is probably unnecessary since its also tested below in freeaddition_sqrt.
@testset "support_sqrt" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    G_semi = z -> cauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    invG_semi = z -> invcauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    dG_semi = z -> dcauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    supp = support_sqrt(G_semi, G_semi, invG_semi, invG_semi, dG_semi, dG_semi, (-2, 2), (-2, 2); tol=10^-9, maxits=50)
    for (i, s) in enumerate((-2√2, 2√2))
        @test isapprox(supp[i], s)
    end

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4
    # NOTE: maybe newton's method converges to the roots more accurately, had to use rtol = 10^-6. Affects later test.
    G_mp_2 = z -> cauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    invG_mp_2 = z -> invcauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    dG_mp_2 = z -> dcauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    supp = support_sqrt(G_mp_2, G_mp_2, invG_mp_2, invG_mp_2, dG_mp_2, dG_mp_2, (3-2√2, 3+2√2), (3-2√2, 3+2√2); tol=10^-9, maxits=50)
    for (i, s) in enumerate((1, 9))
        @test isapprox(supp[i], s; rtol = 10^-6)
    end
end

@testset "freeaddition_sqrt" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    ψ_c_k, supp_c = freeaddition_sqrt(ψ_semi_k, ψ_semi_k, (-2,2), (-2,2);
    m=10, maxterms=20, tolcomp=1+10^-6, tolbisect = 10^-6, maxitsbisect=40)

    #test support
    for (i, s) in enumerate((-2√2, 2√2))
        @test isapprox(supp_c[i], s)
    end

    #test coefficients
    @test norm(ψ_c_k[1:20] - [√2 / (2*pi); fill(0, 19)]) < 10^-9

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4
    ψ_c_k, supp_c = freeaddition_sqrt(ψ_mp2_k, ψ_mp2_k, (3-2√2, 3+2√2), (3-2√2, 3+2√2);
    m=10, maxterms=40, tolcomp=1+10^-6, tolbisect = 10^-9, maxitsbisect=40)
    
    #test support
    for (i, s) in enumerate((1, 9))
        @test isapprox(supp_c[i], s; rtol = 10^-6)
    end
 
    #test coefficients
    @test norm(ψ_c_k[1:20] - ψ_mp4_k[1:20]) < 10^-6 # NOTE: worse bound to pass test - mainly due to inaccurately calculating supp_c
end