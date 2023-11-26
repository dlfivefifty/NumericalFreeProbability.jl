using NumericalFreeProbability, Test, Random

sc = Semicircle()
mp4 = ChebyshevUMeasure(1, 9, x -> 2/(π*x))
jm = normalize(JacobiMeasure(-2, 2, 1, 2))
pm = PointMeasure([-2,-1,1], [0.5,0.25,0.25])
sm = normalize(ChebyshevUMeasure(-3,-1) + ChebyshevUMeasure(1,3))


@testset "invcauchytransforms.jl" begin
    @testset "invcauchytransform single" begin
        t1 = [0+0.5im, -4]
        for z in t1, m in [sc, mp4, jm, pm, sm]
            a = Gₘ(z, m)
            b = Gₘ(Gₘ⁻¹(a, m), m)
            @test all(isapprox.(b, a))
        end
    end # testset
    @testset "invcauchytransform multiple" begin
        Random.seed!(153)
        V = (randn(10) .+ rand(10)*im .+ 0.1) .* rand((-1,1), 10)
        for m in [sc, mp4, jm, pm, sm]
            a = Gₘ(V, m)
            b = Gₘ.(Gₘ⁻¹(a, m), Ref(m))
            @test all([all([isapprox(z, a[i]) for z in b[i]]) for i=1:10])
        end
    end # testset
end


