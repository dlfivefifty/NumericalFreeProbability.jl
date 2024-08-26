export invcauchytransform, precompute_integral, realinvcauchytransform, Gₘ⁻¹


using AMRVW, ToeplitzMatrices


# Joukowski transform
Jinv_p(z) = z - √(z - 1) * √(z + 1)
J(z) = (z + inv(z))/2
dJ(z) = (1 - inv(z^2))/2

inunitdisk(T) = z::Number -> abs(z) ≤ 1+100eps(real(T)) # for rounding errors

struct TrapeziumRulePoints{T<:Number}
    z_j::Vector{T}
    GHz_j::Vector{Vector{T}}
    n::Int64
end

function precompute_integral(m::Measure{T}, N, r) where T<:Number
    supp = support(m); s = length(supp)
    n = inversebound(m) + 1
    z_j = r .* cispi.((0:2:2N-2)./N)
    GHz_j = Vector{Vector{complex(T)}}(undef, s)
    for i=1:s
        if i==1
            H = z -> M_ab(J(z), supp[1][1], supp[end][2])
        else
            H = z -> M_ab(inv(J(z)), supp[i-1][2], supp[i][1])
        end
        GHz_j[i] = cauchytransform(H.(z_j), m)
    end
    TrapeziumRulePoints{complex(T)}(z_j, GHz_j, n)
end

function invcauchytransform(Y::AbstractVector{T}, m::Measure; N=1000, r=0.95, region=1:length(support(m))) where T<:Number
    supp = support(m); s = length(support(m))
    n = inversebound(m) + 1
    z_j = r .* cispi.((0:2:2N-2)./N)
    GHz_j = Vector{Vector{complex(T)}}(undef, s)
    for i=region
        H = conformal_map(i, supp)
        GHz_j[i] = cauchytransform(H.(z_j), m)
    end
    ret = Vector{Vector{complex(T)}}(undef, length(Y))
    for (j,y) in enumerate(Y)
        inverses = Vector{complex(T)}()
        for i=region
            H = conformal_map(i, supp)
            ksv_method!(r, N, z_j, GHz_j[i], y, n, inverses, H)
            filterinverses!(inverses, y, m)
        end
        ret[j] = inverses
    end
    ret
end

function invcauchytransform(y::T, m::Measure; N=1000, r=0.9, region=1:length(support(m))) where T<:Number
    supp = support(m)
    n = inversebound(m) + 1
    z_j = r .* cispi.((0:2:2N-2)./N)
    inverses = Vector{complex(T)}()
    for i=region
        H = conformal_map(i, supp)
        GHz_j = cauchytransform(H.(z_j), m)
        ksv_method!(r, N, z_j, GHz_j, y, n, inverses, H)
    end
    filterinverses!(inverses, y, m)
end

# precompute all the function evaluations

function invcauchytransform(y::T, m::Measure, TG::TrapeziumRulePoints; region=1:length(support(m))) where T<:Number
    supp = support(m)
    inverses = Vector{complex(T)}()
    for i=region
        H = conformal_map(i, supp)
        ksv_method!(real(TG.z_j[1]), length(TG.z_j), TG.z_j, TG.GHz_j[i], y, TG.n, inverses, H)
    end
    filterinverses!(inverses, y, m)
end

function invcauchytransform(Y::AbstractVector{T}, m::Measure, TG::TrapeziumRulePoints; region=1:length(support(m))) where T<:Number
    supp = support(m)
    ret = Vector{Vector{complex(T)}}(undef, length(Y))
    for (j,y) in enumerate(Y)
        inverses = Vector{complex(T)}()
        for i=region
            H = conformal_map(i, supp)
            ksv_method!(real(TG.z_j[1]), length(TG.z_j), TG.z_j, TG.GHz_j[i], y, TG.n, inverses, H)
            filterinverses!(inverses, y, m)
        end
        ret[j] = inverses
    end
    ret
end


function conformal_map(i::Int64, supp::Vector{Vector{T}}) where T<:Real
    if i==1
        H = z -> M_ab(J(z), supp[1][1], supp[end][2])
    else
        H = z -> M_ab(inv(J(z)), supp[i-1][2], supp[i][1])
    end
    H
end

function ksv_method!(r::Real, N::Int64, z_j::Vector{T}, GHz_j::Vector{T}, y::Number, n::Int64, inverses::Vector{T2}, H::Function) where {T<:Number, T2<:Number}
    a_i = r/N * [sum(z_j.^(k+1) .* inv.(GHz_j .- y)) for k=0:2n-1]
    H0 = Hankel(a_i[1:end-1])
    H1 = Hankel(a_i[2:end])
    for z in H.(eigvals(H1, H0))
        if all(abs.(z .- inverses) .> 10^-9)
            append!(inverses, z)
        end
    end
end

function filterinverses!(inverses::Vector{T}, y::Number, m::Measure, tol=10^-12) where T<:Number
    testinverse = z::T -> abs(cauchytransform(z, m) - y) < tol
    filter!(testinverse, inverses)
end


# specialised method for pure point measures

function invcauchytransform(y::Number, m::PointMeasure; region=nothing)
    n = length(m.a)
    P = -y .* productlinearfactors(m.λ)
    for i=1:n
        P += [m.a[i] * productlinearfactors([m.λ[1:i-1]; m.λ[i+1:end]]);0]
    end
    if y == 0
        P = P[1:end-1]
    end
    inverses = invertpolynomial(P, 0)
    if region !== nothing
        sort!(inverses; by=real)
        if isa(y, Real) && y ≥ 0
            return region == 1 ? inverses[end] : inverses[region-1]
        else
            return inverses[region]
        end
    end
    inverses
end

function invcauchytransform(Y::AbstractVector{T}, m::PointMeasure; region=nothing) where T<:Number
    n = length(m.a)
    P1 = productlinearfactors(m.λ)
    P2 = zeros(n+1)
    for i=1:n
        P2 += [m.a[i] * productlinearfactors([m.λ[1:i-1]; m.λ[i+1:end]]);0]
    end
    inverses = Vector{Vector{T}}(undef, length(Y))
    for (i, y) in enumerate(Y)
        P = P2 - y .* P1
        if y == 0
            P = P[1:end-1]
        end
        ans = invertpolynomial(P, 0)
        if region !== nothing
            sort!(ans; by=real)
            if isa(y, Real) && y ≥ 0
                inverses[i] = [region == 1 ? ans[end] : ans[region-1]]
            elseif isa(y, Real)
                inverses[i] = [ans[region]]
            else
                inverses[i] = ans
            end
        else
            inverses[i] = ans
        end
    end
    inverses
end





function productlinearfactors(terms)
    P = zeros(length(terms)+1)
    P[1] = 1.0
    for i in terms
        P[2:end] = P[1:end-1]
        P[1] = 0.0
        P[1:end-1] += P[2:end] * -i
    end
    P
end

function invertpolynomial(P, z::Number)
    P1 = P; P1[1] -= z
    AMRVW.roots(P1)
end


# specialised method for square root measures

function invcauchytransform(y::T, m::ChebyshevUMeasure; maxterms=100, tol=10^-15, region=nothing) where T<:Number
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    P = Complex.(vcat([0], m.ψ_k[1:n]))
    P1 = P[1:end-1]/P[end]
    P1[1] -= y/π/P[end]
    C = SpecialMatrices.Companion(P1)
    s = eigvals(C)
    M_ab.(J.(filter!(inunitdisk(T), s)), m.a, m.b)
end

function invcauchytransform(Y::AbstractVector{T}, m::ChebyshevUMeasure; maxterms=100, tol=10^-15, region=nothing) where T<:Number
    ans = Vector{Vector{complex(T)}}(undef, length(Y))
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    P = Complex.(vcat([0], m.ψ_k[1:n]))
    P1 = P[1:end-1]/P[end]
    C = SpecialMatrices.Companion(P1)
    for (i,y) in enumerate(Y)
        C.c[1] = -y/π/P[end]
        s = eigvals(C)
        ans[i] = M_ab.(J.(filter!(inunitdisk(T), s)), m.a, m.b)
    end
    ans
end



# bounding number of inverses

function inversebound(m::AbstractJacobiMeasure{T}) where T<:Number
    xv = LinRange(support(m)[1][1], support(m)[1][2], 123)
    yv = m[xv]
    t = 0
    for i=1:121
        if yv[i] < yv[i+1] > yv[i+2] || yv[i] > yv[i+1] < yv[i+2]
            t += 1 
        end
    end
    (t+1)÷2
end

function inversebound(m::PointMeasure{T}) where T<:Number
    length(m.a)
end

function inversebound(m::SumMeasure{T}) where T<:Number
    sum(inversebound, m.m_k)
end


# bisection methods for inverses on the real line, due to monotonicity

function realinvcauchytransform(y::Real, m::Measure, region::Int)
    supp = support(m)
    G = z::Real -> cauchytransform(z, m)
    ε = 100eps()
    if region == 1
        if y ≤ 0
            inverse = bisection(z::Real -> G(inv(z)+1+supp[1][1]) - y, -1+ε, -ε, tol=eps(), maxits = 60, forcereturn=true)
            inverse = inv(inverse)+1+supp[1][1]
        else
            inverse = bisection(z::Real -> G(inv(z)-1+supp[end][2]) - y, ε, 1-ε, tol=eps(), maxits = 60, forcereturn=true)
            inverse = inv(inverse)-1+supp[end][2]
        end
    else
        inverse = bisection(z::Real -> G(z) - y, supp[region-1][2]+ε, supp[region][1]-ε, tol=eps(), maxits = 60, forcereturn=true)
    end
    inverse
end

function realinvcauchytransform(y::Real, m::ChebyshevUMeasure, region::Int)
    invcauchytransform(y, m)[1]
end

function realinvcauchytransform(y::Real, m::PointMeasure, region::Int)
    real(invcauchytransform(y, m; region))
end



const Gₘ⁻¹ = invcauchytransform
