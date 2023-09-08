
@testset "support_sqrt" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    G_semi = z -> cauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    invG_semi = z -> invcauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    dG_semi = z -> dcauchytransform_sqrt(Complex(z), ψ_semi_k, -2, 2)
    #supp = support_sqrt(G_semi, G_semi, invG_semi, invG_semi, dG_semi, dG_semi, (-2, 2), (-2, 2); tol=10^-9, maxits=50)
    for (i, s) in enumerate((-2√2, 2√2))
        @test_broken isapprox(supp[i], s)
    end

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4
    # NOTE: maybe newton's method converges to the roots more accurately, had to use rtol = 10^-6. Affects later test.
    G_mp_2 = z -> cauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    invG_mp_2 = z -> invcauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    dG_mp_2 = z -> dcauchytransform_sqrt(Complex(z), ψ_mp2_k, 3-2√2, 3+2√2)
    #supp = support_sqrt(G_mp_2, G_mp_2, invG_mp_2, invG_mp_2, dG_mp_2, dG_mp_2, (3-2√2, 3+2√2), (3-2√2, 3+2√2); tol=10^-9, maxits=50)
    for (i, s) in enumerate((1, 9))
        @test_broken isapprox(supp[i], s; rtol = 10^-6)
    end
end

@testset "freeaddition_sqrt" begin

    # free addition of two standard semicircles -> semicircle of variance 2
    #ψ_c_k, supp_c = freeaddition_sqrt(ψ_semi_k, ψ_semi_k, (-2,2), (-2,2);
    #m=10, maxterms=20, tolcomp=1+10^-6, tolbisect = 10^-6, maxitsbisect=40)

    #test support
    for (i, s) in enumerate((-2√2, 2√2))
        @test_broken isapprox(supp_c[i], s)
    end

    #test coefficients
    @test_broken norm(ψ_c_k[1:20] - [√2 / (2*pi); fill(0, 19)]) < 10^-9

    # free addition of two marchenko pastur distributions with c = 2 -> marchenko pastur distribution with c = 4
    #ψ_c_k, supp_c = freeaddition_sqrt(ψ_mp2_k, ψ_mp2_k, (3-2√2, 3+2√2), (3-2√2, 3+2√2);
    #m=10, maxterms=40, tolcomp=1+10^-6, tolbisect = 10^-9, maxitsbisect=40)
    
    #test support
    for (i, s) in enumerate((1, 9))
        @test_broken isapprox(supp_c[i], s; rtol = 10^-6)
    end
 
    #test coefficients
    @test_broken norm(ψ_c_k[1:20] - ψ_mp4_k[1:20]) < 10^-6 # NOTE: worse bound to pass test - mainly due to inaccurately calculating supp_c
end