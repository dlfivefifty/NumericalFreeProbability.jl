using LinearAlgebra, ClassicalOrthogonalPolynomials, SingularIntegrals, SpecialFunctions
using Random
using BandedMatrices, LazyArrays
using ProfileView
using Profile, Cthulhu, PProf
using Revise
using StaticArrays

struct JacobiMeasure{T1<:Real, T2<:Real}
    P::Jacobi{Float64}
    J::BandedMatrices.AbstractBandedMatrix{Float64}
    f_k::LazyArrays.LazyArray{Float64, 1}
    f::Function
    w::Function
    G::Function
    q_0::Function
    a::T1
    b::T2
    function JacobiMeasure(f::Function, a::T1, b::T2) where {T1<:Real, T2<:Real}
        if a <= -1 || b <= -1
            throw(ArgumentError("Jacobi exponents must be greater than -1"))
        end
        P = Jacobi(a, b)
        J = jacobimatrix(P)
        f_expanded = expand(P, f)
        f_k = f_expanded.args[2]
        Z = beta(a+1,b+1) * 2^(a+b+1)
        w = x -> 1/Z * (1-x)^a * (1+x)^b
        p = x -> f(x) * w(x)
        W = Weighted(P); p_expanded = expand(W, p); w_expanded = expand(W, w);
        x = axes(W, 1)
        function G(z::Number)
            inv.(z .- x') * p_expanded
        end
        function q_0(z::Number)
            inv.(z .- x') * w_expanded
        end
        new{T1, T2}(P, J, f_k, f, w, G, q_0, a, b)
    end
end



struct SquarerootMeasure{}
    P::ChebyshevU{Float64}
    J::BandedMatrices.AbstractBandedMatrix{Float64}
    f_k::LazyArrays.LazyArray{Float64, 1}
    f::Function
    w::Function
    G::Function
    q_0::Function
    function SquarerootMeasure(f::Function)
        P = ChebyshevU()
        J = jacobimatrix(P)
        f_expanded = expand(P, f)
        f_k = f_expanded.args[2]
        w = x -> √(1-x^2) * 2/π
        p = x -> f(x) * w(x)
        W = Weighted(P); p_expanded = expand(W, p); w_expanded = expand(W, w);
        x = axes(W, 1)
        function G(z::Number)
            inv.(z .- x') * p_expanded
        end
        function q_0(z::Number)
            inv.(z .- x') * w_expanded
        end
        new{}(P, J, f_k, f, w, G, q_0)
    end
end



function inversecauchytransform(y::Number, jm, n::Int; radius= 0.8, N = 1000)
    f_k = jm.f_k[1:n+1]
    last = jm.f_k[n+2]
    if iszero(last)
        throw(ArgumentError("n+2 coefficient must be non zero."))
    end
    A = Matrix(jm.J[1:n+1,1:n+1]'); A[end,:] -= f_k ./ last .* jm.J[n+1, n+2]
    b = zeros(ComplexF64, n+1); b[end] = y/last * jm.J[n+1, n+2]; b[1] = 1
    Σ = zeros(n+1); Σ[1] += 1
    A1 = SMatrix{n+2, n+2}([0 Σ';b A])
    A2 = SMatrix{n+2, n+2}(Diagonal([-(i != 0) for i=0:n+1]))
    A3 = SMatrix{n+2, n+2}([0 zeros(n+1)';A*Σ zeros(n+1, n+1)])
    A4 = SMatrix{n+2, n+2}([0 zeros(n+1)';-Σ zeros(n+1, n+1)])
    functionlist = Vector{Function}()
    if imag(y) > 0
        H2(z::Number) = -im * (I + z) * inv(I - z)
        push!(functionlist, H2)
    end
    if imag(y) < 0
        H1(z::Number) = im * (I + z) * inv(I - z)
        push!(functionlist, H1)
    end
    if abs(imag(y)) < 10^-10
        H3(z::Number) = (I + z) * inv(I - z) + 1
        H4(z::Number) = -(I + z) * inv(I - z) - 1
        push!(functionlist, H3)
        push!(functionlist, H4)
    end
    inverses = Vector{ComplexF64}()
    for H in functionlist
        function T(z::Number)
            hz = H(z)
            A1 + hz * A2 + jm.q_0(hz) * (A3 + hz * A4)
        end
        λ = beyn(T, n+2; r=radius, μ=0, N)
        for z in H.(λ)
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end

function beyn(T::Function, m::Int; r=0.8, μ=0, N=1000, svtol=10^-12)
    Random.seed!(163) # my favourite integer
    invf = x -> inv(T(μ + r * x)) * x
    exp2πimjN = [exp(2π*im * j / N) for j=0:N-1]
    invT = invf.(exp2πimjN)
    V̂ = [randn() + randn()*im for i=1:m, j=1:m]
    A_0N = r/N * sum(invT) * V̂
    A_1N = μ * A_0N + r^2/N * sum(invT .* exp2πimjN) * V̂
    V, S, W = svd(A_0N)
    k = m
    while k >= 1 && abs(S[k]) < svtol
        k -= 1
    end
    V_0 = V[1:m, 1:k]
    W_0 = W[1:m, 1:k]
    S_0 = Diagonal(S[1:k])
    B = V_0' * A_1N * W_0 * S_0^-1
    eigvals(B)
end


sqm = SquarerootMeasure(x -> x^3/6 + x/2 + 1); n=2; z=-0.5 + 0.1im; y = sqm.G(z);
@btime inversecauchytransform(y, sqm, n; radius= 0.9, N = 1000)

# julia> sqm = SquarerootMeasure(x -> x^3/6 + x/2 + 1); n=2; z=-0.5 + 0.1im; y = sqm.G(z)
# -1.1331365255107277 - 1.170648976000868im

# julia> @time inversecauchytransform(y, sqm, n; radius= 0.9, N = 1000)
#   0.005652 seconds (22.95 k allocations: 8.010 MiB)
# 1-element Vector{Any}:
#  -0.4999999999999883 + 0.10000000000000556im

jm = JacobiMeasure(x -> (7x^2 + 1)/2, 2, 2); n=1; z=-0.5 + 0.1im; y = jm.G(z);
@btime inversecauchytransform(y, jm, n; radius= 0.9, N = 1000)

# julia> jm = JacobiMeasure(x -> (7x^2 + 1)/2, 2, 2); n=1; z=-0.5 + 0.1im; y = jm.G(z)
# -0.8179061373251617 - 1.8429838453246645im

# julia> @time inversecauchytransform(y, jm, n; radius= 0.9, N = 1000)
#   0.008250 seconds (42.75 k allocations: 6.649 MiB)
# 1-element Vector{Any}:
#  -0.5000000000000075 + 0.09999999999999251im

sqm = SquarerootMeasure(x -> (4x^2+1)/2); n=1; z=0.1im; y = sqm.G(z);
@btime inversecauchytransform(y, sqm, n; radius= 0.9, N = 1000)

# julia> sqm = SquarerootMeasure(x -> (4x^2+1)/2); n=1; z=0.1im; y = sqm.G(z)
# 0.0 - 1.0687880596276051im

# julia> @time inversecauchytransform(y, sqm, n; radius= 0.9, N = 1000)
#   0.005328 seconds (22.97 k allocations: 5.781 MiB)
# 2-element Vector{Any}:
#  -6.145050687937699e-15 + 0.10000000000001238im
#  1.0810235311851757e-14 + 0.3276135241963962im