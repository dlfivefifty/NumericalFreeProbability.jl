export ACMeasure, AbstractJacobiMeasure,
            ChebyshevUMeasure, JacobiMeasure, Semicircle, SquareRootMeasure, SumMeasure, PointMeasure,
            normalize, m_op, support

abstract type Measure{T} end
abstract type ACMeasure{T} <: Measure{T} end
abstract type OPMeasure{T} <: ACMeasure{T} end
abstract type AbstractJacobiMeasure{T} <: OPMeasure{T} end
axes(m::AbstractJacobiMeasure) = m.a..m.b



struct ChebyshevUMeasure{T<:Real} <: AbstractJacobiMeasure{T} 
    a::T
    b::T
    ψ_k::LazyArray{T, 1}
    ChebyshevUMeasure{T}(a, b, ψ_k) where T<:Real = new{T}(a, b, ψ_k)
end

const SquareRootMeasure = ChebyshevUMeasure

ChebyshevUMeasure(a::T, b::T, ψ_k::LazyArray{T, 1}) where {T<:Real} = ChebyshevUMeasure{T}(a, b, ψ_k)
function ChebyshevUMeasure(a::Real, b::Real, ψ_k::LazyArray{T, 1}) where {T<:Real}
    U = promote_type(typeof(a), typeof(b), eltype(ψ_k))
    ChebyshevUMeasure(convert(U, a), convert(U, b), convert(LazyArray{U, 1}, ψ_k))
end

function ChebyshevUMeasure(a::Real, b::Real, f::Function)
    P = chebyshevu(a..b)
    ψ_k = P \ f.(axes(P, 1))
    ChebyshevUMeasure(a, b, ψ_k)
end
ChebyshevUMeasure(a::Real, b::Real, k::Real) = ChebyshevUMeasure(a, b, vcat([1], zeros(promote_type(typeof(a), typeof(b), typeof(k)), ∞)) * k)
ChebyshevUMeasure(a::Real, b::Real) = ChebyshevUMeasure(a, b, one)








# jacobi measure on a single interval
struct JacobiMeasure{T<:Real} <: AbstractJacobiMeasure{T} 
    a::T
    b::T
    α::T
    β::T
    ψ_k::LazyArray{T, 1}
    function JacobiMeasure{T}(a, b, α, β, ψ_k) where {T<:Real}
        if !(α > -1 && β > -1)
            error("jacobi exponents must be > -1")
        end
        new{T}(a, b, α, β, ψ_k)
    end
end

JacobiMeasure(a::T, b::T, α::T, β::T, ψ_k::LazyArray{T, 1}) where {T<:Real} = JacobiMeasure{T}(a, b, α, β, ψ_k)
function JacobiMeasure(a::Real, b::Real, α::Real, β::Real, ψ_k::LazyArray{T, 1}) where {T<:Real}
    U = promote_type(typeof(a), typeof(b), typeof(α), typeof(β), eltype(ψ_k))
    JacobiMeasure(convert(U, a), convert(U, b), convert(U, α), convert(U, β), convert(LazyArray{U, 1}, ψ_k))
end

function JacobiMeasure(a::Real, b::Real, α::Real, β::Real, f::Function)
    P = jacobi(α, β, a..b)
    ψ_k = P \ f.(axes(P, 1))
    JacobiMeasure(a, b, α, β, ψ_k)
end

JacobiMeasure(a::Real, b::Real, α::Real, β::Real, k::Real) = JacobiMeasure(a, b, α, β, vcat([1], zeros(promote_type(typeof(a), typeof(b), typeof(k), typeof(α), typeof(β)), ∞)) * k)
JacobiMeasure(a::Real, b::Real, α::Real, β::Real) = JacobiMeasure(a, b, α, β, one)



Semicircle(R::Real) = ChebyshevUMeasure(-R, R, 2*vcat([1], zeros(∞))/(π*R))
Semicircle() = Semicircle(2)



struct SumMeasure{T<:Real} <: Measure{T}
    m_k::Vector{Measure{T}}
    SumMeasure{T}(m_k) where T<:Real = new{T}(m_k)
end

SumMeasure(m_k::Vector{M}) where {T<:Real, M<:Measure{T}} = SumMeasure{T}(m_k)

+(m1::Measure, m2::Measure) = SumMeasure([m1, m2])
+(m1::Measure, m2::SumMeasure) = SumMeasure(vcat([m1], m2.m_k))
+(m1::SumMeasure, m2::Measure) = SumMeasure(vcat(m1.m_k, [m2]))
+(m1::SumMeasure, m2::SumMeasure) = SumMeasure(vcat(m1.m_k, m2.m_k))




m_op(m::ChebyshevUMeasure) = chebyshevu(m.a..m.b)
m_op(m::JacobiMeasure) = jacobi(m.α, m.β, m.a..m.b)

+(m::ChebyshevUMeasure, x::Real) = ChebyshevUMeasure(m.a+x, m.b+x, m.ψ_k)
+(m::JacobiMeasure, x::Real) = JacobiMeasure(m.a+x, m.b+x, m.α, m.β, m.ψ_k)

*(m::ChebyshevUMeasure, x::Real) = ChebyshevUMeasure(x * m.a, x * m.b, m.ψ_k ./ x)
*(m::JacobiMeasure, x::Real) = JacobiMeasure(x * m.a, x * m.b, m.α, m.β, m.ψ_k ./ x)



+(x::Real, m::Measure) = m+x
-(x::Real, m::Measure) = x + (-1)*m
-(m::Measure, x::Real) = m + -x
*(x::Real, m::Measure) = m*x
/(m::Measure, x::Real) = m * inv(x)





struct PointMeasure{T<:Real} <: Measure{T}
    λ::Vector{T}
    a::Vector{T}
    function PointMeasure{T}(λ, a) where {T<:Real}
        if length(λ) != length(a)
            throw(ArgumentError("Arrays must have the same size."))
        end
        new{T}(λ, a)
    end
end

PointMeasure(λ::Vector{T}, a::Vector{T}) where {T<:Real} = PointMeasure{T}(λ, a)

function PointMeasure(λ::Vector{T1}, a::Vector{T2}) where {T1<:Real, T2<:Real}
    U = promote_type(eltype(a), eltype(λ))
    PointMeasure(convert(Vector{U}, λ), convert(Vector{U}, a))
end

+(m::PointMeasure, x::Real) = PointMeasure(m.λ .+ x, m.a)
*(m::PointMeasure, x::Real) = PointMeasure(m.λ .* x, m.a)


sum(m::ChebyshevUMeasure) = sum(orthogonalityweight(chebyshevu(m.a..m.b))) * m.ψ_k[1]
sum(m::JacobiMeasure) = sum(orthogonalityweight(jacobi(m.α, m.β, m.a..m.b))) * m.ψ_k[1]
sum(f, m::PointMeasure) = sum(f, m.a)
sum(m::SumMeasure) = sum(map(sum, m.m_k))

# normalize to total measure 1, or anything really
normalize(m::ChebyshevUMeasure, n::Real = 1) = ChebyshevUMeasure(m.a, m.b, m.ψ_k/sum(m) * n)
normalize(m::JacobiMeasure, n::Real = 1) = JacobiMeasure(m.a, m.b, m.α, m.β, m.ψ_k/sum(m) * n)
normalize(m::PointMeasure, n::Real = 1) = PointMeasure(m.λ, m.a/sum(m) * n)
normalize(m::SumMeasure, n::Real = 1) = SumMeasure([normalize(μ, n * sum(μ) / sum(m)) for μ in m.m_k])

support(m::AbstractJacobiMeasure) = [[m.a, m.b]]
support(m::PointMeasure) = [[i,i] for i in m.λ]
function support(m::SumMeasure{T}) where T<:Real
    t = Vector{Vector{T}}()
    for μ in m.m_k
        t = [t; support(μ)]
    end
    sort!(t, by = x -> x[1])
    supp = Vector{Vector{T}}()
    for s in t
        if length(supp) == 0 || supp[end][2] < s[1]
            append!(supp, Ref(s))
        else
            supp[end][2] = max(supp[end][2], s[2])
        end
    end
    supp
end



function getindex(m::ChebyshevUMeasure{T}, x::Real) where T<:Real
    if x ∈ m.a .. m.b
        return (Weighted(chebyshevu(m.a .. m.b)) * m.ψ_k)[x]
    end
    zero(T)
end
function getindex(m::JacobiMeasure{T}, x::Real) where T<:Real
    if x ∈ m.a .. m.b
        return (Weighted(jacobi(m.α, m.β, m.a .. m.b)) * m.ψ_k)[x]
    end
    zero(T)
end
function getindex(m::SumMeasure, x::Real)
    sum(μ -> μ[x],m.m_k)
end
function getindex(m::PointMeasure{T}, x::Real) where T <: Real
    t = findfirst(==(x), m.λ)
    if t !== nothing
        return m.a[t]
    end
    zero(T)
end

function getindex(m::Measure{T}, x::AbstractVector{T2}) where {T<:Real, T2<:Real}
    [m[i] for i in x]
end # lazy

