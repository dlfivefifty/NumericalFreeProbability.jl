export freeaddition, recovermeasure_singlyssupportedsqrt, pointcloud_sqrt, support_sqrt_single, ⊞,
                NumberOrVectorNumber, pairsum, prunepoints_multivalued

const NumberOrVectorNumber = Union{Number, AbstractVector{T}} where T<:Number
const NOVN = NumberOrVectorNumber
unitcirclenodes(T, n) = [cispi(convert(T, 2k)/n-1) for k=0:n-1]


function pointcloud_sqrt(G_a::Function, G_b::Function, supp_c::Vector{Vector{T}}, invG_b::Function; m = 10) where T<:Real
    d_M = vec(unitcirclenodes(Float64, m)*[x for x in ChebyshevGrid{1}(2m+1)[1:m] if x > eps()]') # temporary Float64
    z_μ_M = [M_ab(J(x), supp_c[1][1], supp_c[end][2]) for x in d_M if imag(x) > eps()]
    y_M = G_a(z_μ_M)
    invcache = invG_b(y_M)
    y_M = [y for (i,y) in enumerate(y_M) if length(invcache[i]) > 0 && isapprox(G_b(invcache[i][1]), y)] 
end


function prunepoints_multivalued(points::Vector{T}, invG_a::Function, invG_b::Function) where T<:Number
    invG_c = z::Union{T, Vector{T}} -> combine_invcauchytransform(z, invG_a, invG_b)
    preimages = Vector{T}(); images = Vector{T}()
    inv_cache = invG_c(points)
    for (i,y) in enumerate(points)
        for x in inv_cache[i]
            if sign(imag(x)) != sign(imag(y))
                push!(preimages, x)
                push!(images, y)
            end
        end
    end
    preimages, images
end





function combine_invcauchytransform(y::Number, invG_a::Function, invG_b::Function)
    ans = pairsum(invG_a(y), invG_b(y)) .- inv(y)
    filter!(z -> sign(imag(y)) != sign(imag(z)), ans)
end

function combine_invcauchytransform(y::Vector{T}, invG_a::Function, invG_b::Function) where T<:Number
    invg_a = invG_a(y); invg_b = invG_b(y)
    ans = Vector{Vector{T}}(undef, length(y))
    for i in eachindex(y)
        ans[i] = pairsum(invg_a[i], invg_b[i]) .- inv(y[i])
        filter!(z -> sign(imag(y[i])) != sign(imag(z)), ans[i])
    end
    ans
end

function pairsum(u::Vector{T}, v::Vector{T}) where T<:Number
    result = T[]
    for i in u
        for j in v
            push!(result, i + j)
        end
    end
    return result
end


function freeaddition(m_a::Measure, m_b::Measure; m=20, tolbisect = 10^-13, maxitsbisect=60, N::Int=20)
    G_a = z::NOVN -> cauchytransform(z, m_a)
    G_b = z::NOVN -> cauchytransform(z, m_b)
    dG_a = z::NOVN -> dcauchytransform(z, m_a)
    dG_b = z::NOVN -> dcauchytransform(z, m_b)
    if isa(m_a, Union{ChebyshevUMeasure, PointMeasure})
        invG_a = v -> z::NOVN -> invcauchytransform(z, m_a; region=v)
    else
        TG_a = precompute_integral(m_a, 2000, 0.95)
        invG_a = v -> z::NOVN -> invcauchytransform(z, m_a, TG_a; region=v)
    end
    if isa(m_b, Union{ChebyshevUMeasure, PointMeasure})
        invG_b = v -> z::NOVN -> invcauchytransform(z, m_b; region=v)
    else
        TG_b = precompute_integral(m_b, 2000, 0.98)
        invG_b = v -> z::NOVN -> invcauchytransform(z, m_b, TG_b; region=v)
    end
    supp_c = additive_support(G_a, G_b, dG_a, dG_b, m_a, m_b; tol=tolbisect, maxits=maxitsbisect)
    
    y_M = pointcloud_sqrt(G_a, G_b, supp_c, invG_b(1:length(support(m_b))); m)
    preimages, images = prunepoints_multivalued(y_M, invG_a(1:length(support(m_a))), invG_b(1:length(support(m_b))))
    
    if length(supp_c) == 1
        ψ_c_k = recovermeasure(supp_c, preimages, images, N)
        return ChebyshevUMeasure(supp_c[1][1], supp_c[1][2], vcat(ψ_c_k[1], zeros(∞)))
    end
    ψ_c_k_i = recovermeasure(supp_c, preimages, images, N)
    SumMeasure([ChebyshevUMeasure(supp_c[i][1], supp_c[i][2], vcat(ψ_c_k_i[i], zeros(∞))) for i in eachindex(supp_c)])
end

⊞(m_a::Measure, m_b::Measure) = freeaddition(m_a::Measure, m_b::Measure)

function recovermeasure(supp_c::Vector{Vector{T2}}, preimages::Vector{T}, images::Vector{T}, N::Int=20)  where {T<:Number, T2<:Real}
    n = length(images)
    A = zeros(complex(T), n, N*length(supp_c))
    for (i, s) in enumerate(supp_c)
        A[:,N*(i-1)+1:N*i] = [Jinv_p(M_ab_inv(preimages[j],s[1],s[2]))^k for j=1:n, k=1:N]
    end
    V = [real.(A);imag.(A)]
    f = [real.(images);imag.(images)]
    Q, R̂ = qr(V)
    Q̂ = Q[:,1:length(supp_c) * N]
    sol = R̂ \ Q̂'f ./ pi
    [sol[(i-1)*N+1:i*N] for i=1:length(supp_c)]
end
