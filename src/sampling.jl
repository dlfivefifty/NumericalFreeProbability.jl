export sample, gue, haar

function sample(m::AbstractJacobiMeasure{T}, n; c::Real=0) where T<:Real
    ret = Vector{T}(undef, n)
    if c == 0
        c = maximum(m[m.a:0.01:m.b])
    end
    for i=1:n
        ret[i] = sample(m; c)
    end
    ret
end

function sample(m::AbstractJacobiMeasure{T}; c::Real=0) where T<:Real
    x = (m.b - m.a) * rand() + m.a
    if c == 0
        c = maximum(m[m.a:0.01:m.b])
    end
    y = c*rand()
    while m[x] < y
        x = (m.b - m.a) * rand() + m.a
        y = c*rand()
    end
    x
end

function gue(n)
    A = randn(ComplexF64, n, n)
    (A + A')/√(2n)
end

function haar(n)
    A = randn(ComplexF64, n, n)
    Q,_ = qr(A)
    s = Diagonal(cispi.(2 .* rand(n)))
    Q*s
end


function sample(m::PointMeasure, n=1)
    StatsBasesample(m.λ, Weights(m.a))
end