export sample, gue, haar, diag

using StatsBase, SpecialFunctions
import StatsBase: sample
import LinearAlgebra: diag

function sample(m::AbstractJacobiMeasure{T}, n; c::Real=0) where T<:Real
    ret = Vector{T}(undef, n)
    r = m_op(m) * m.ψ_k
    if c == 0
        c = maximum(r[m.a:0.01:m.b])
    end
    for i=1:n
        ret[i] = sample(m; c)
    end
    ret
end

function sample(m::JacobiMeasure{T}; c::Real=0) where T<:Real
    r = jacobi(m.α, m.β, m.a .. m.b) * m.ψ_k
    if c == 0
        c = maximum(r[m.a:0.01:m.b])
    end
    y = (m.b - m.a) * beta_inc_inv.(1+m.β, 1+m.α, rand())[1] + m.a
    u = rand()
    while u > r[y]/c
        y = (m.b - m.a) * beta_inc_inv.(1+m.β, 1+m.α, rand())[1] + m.a
        u = rand()
    end
    y
end

function sample(m::ChebyshevUMeasure{T}; c::Real=0) where T<:Real
    r = chebyshevu(m.a .. m.b) * m.ψ_k
    if c == 0
        c = maximum(r[m.a:0.01:m.b])
    end
    y = (m.b - m.a) * beta_inc_inv.(1.5, 1.5, rand())[1] + m.a
    u = rand()
    while u > r[y]/c
        y = (m.b - m.a) * beta_inc_inv.(1.5, 1.5, rand())[1] + m.a
        u = rand()
    end
    y
end


function sample(m::PointMeasure)
    StatsBase.sample(m.λ, Weights(m.a))
end

function sample(m::PointMeasure, n=1)
    StatsBase.sample(m.λ, Weights(m.a), n)
end

function sample(m::SumMeasure{T})  where T<:Real
    s = length(m.m_k)
    v = StatsBase.sample(1:s, Weights(sum.(m.m_k)))
    sample(m.m_k[v])
end

function sample(m::SumMeasure{T}, n=1) where T<:Real
    s = length(m.m_k)
    v = StatsBase.sample(1:s, Weights(sum.(m.m_k)), n)
    ret = Vector{T}(undef, n)
    for i=1:s
        w = findall(==(i), v)
        ret[w] = sample(m.m_k[i], length(w))
    end
    ret
end

# demonstration functions

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

function diag(m, n)
    Diagonal(sample(m, n))
end

n=1000
Q = haar(n)
A = diag(u, n)
B = diag(u, n)
C = Q*A*Q' + B
histogram(real.(eigvals(C)), normalize=true, bins=50)