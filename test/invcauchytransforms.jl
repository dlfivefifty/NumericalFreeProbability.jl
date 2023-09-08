@testset "invcauchytransform" begin
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

end # testset