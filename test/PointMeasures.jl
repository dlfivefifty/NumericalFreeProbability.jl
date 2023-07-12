using NumericalFreeProbability
using Test, LinearAlgebra

@testset "cauchytransform_point" begin
    M = PointMeasure([-2,-1,1], [0.5, 0.25, 0.25])
    @test isapprox(cauchytransform_point(1+2im, M), complex(37/208, -55/208))
    @test isapprox(cauchytransform_point(0.5, M), -2/15)
end # testset

@testset "invcauchytransform_point" begin
    M = PointMeasure([-1,1], [0.5, 0.5])

    inv = invcauchytransform_point(0.125 - 0.375im, M)
    @test any(isapprox.(inv, -0.2+0.4im))
    @test any(isapprox.(inv, 1+2im))

    inv = invcauchytransform_point(0.375, M)
    @test any(isapprox.(inv, 3))
    @test any(isapprox.(inv, -1/3))
end # testset



@testset "support_sqrt_point" begin
    
    G_a = z -> cauchytransform_sqrt(Complex(z), [1/pi], -2, 2)
    M_b = PointMeasure([-2, -1, 1], [0.5, 0.25, 0.25])
    InvG_a = z -> invcauchytransform_sqrt(Complex(z), [1/pi], -2, 2)
    InvG_b = z -> invcauchytransform_point(Complex(z), M_b)
    dG_a = z -> dcauchytransform_sqrt(Complex(z), [1/pi], -2, 2)
    dG_b = z -> dcauchytransform_point(Complex(z), M_b)
    support = support_sqrt_point(G_a, InvG_a, InvG_b, dG_a, dG_b, (-2, 2), M_b; tol=10^-6, maxits=40)
    
    @test isapprox(support[1][1], -3.6261629666616466; rtol = 10^-6)
    @test isapprox(support[1][2], 0.15053138702768765; rtol = 10^-6)
    @test isapprox(support[2][1], 0.37210113565464376; rtol = 10^-6)
    @test isapprox(support[2][2], 2.2420184945851087; rtol = 10^-6)
end


@testset "freeaddition_sqrt_point" begin
    M_b = PointMeasure([-2, -1, 1], [0.5, 0.25, 0.25])
    ψ_c_k_i, supp_c = freeaddition_sqrt_point([1/pi], (-2, 2), M_b)

    @test isapprox(ψ_c_k_i[1][1], 0.2528455940316882; rtol=10^-6)
end
    