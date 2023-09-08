export ChebyshevUMeasure, JacobiMeasure, Semicircle, ACMeasure, CompactACMeasure

abstract type ACMeasure{T} <: AbstractQuasiArray{T,1} end
abstract type CompactACMeasure{T} <: ACMeasure{T} end


abstract type AbstractJacobiMeasure{T} <: CompactACMeasure{T} end
axes(m::AbstractJacobiMeasure) = m.a..m.b

const e0inf = vcat([1], zeros(∞))

# square root measure on single interval
struct ChebyshevUMeasure{T<:Real} <: AbstractJacobiMeasure{T} 
    a::T
    b::T
    Z::T
    ψ_k::LazyArray{T, 1}
    ChebyshevUMeasure{T}(a, b, ψ_k) where T<:Real = new{T}(a, b, π * (b - a)/4, ψ_k)
end

ChebyshevUMeasure(a::T, b::T, ψ_k::LazyArray{T, 1}) where {T<:Real} = ChebyshevUMeasure{T}(a, b, ψ_k)
function ChebyshevUMeasure(a::Real, b::Real, ψ_k::LazyArray{T, 1}) where {T<:Real}
    U = promote_type(typeof(a), typeof(b), eltype(ψ_k))
    ChebyshevUMeasure(convert(U, a), convert(U, b), convert(LazyArray{U, 1}, ψ_k))
end

function ChebyshevUMeasure(a::Real, b::Real, f::Function)
    P = ChebyshevU()
    f_ab(x::Real) = f(M_ab(x, a, b))
    ψ_k = P \ f_ab.(axes(P, 1))
    ChebyshevUMeasure(a, b, ψ_k)
end

getindex(m::ChebyshevUMeasure, x) = (Weighted(ChebyshevU()) * m.ψ_k)[M_ab_inv.(x, m.a, m.b)] / m.Z



# jacobi measure on a single interval
struct JacobiMeasure{T<:Real} <: AbstractJacobiMeasure{T} 
    a::T
    b::T
    α::T
    β::T
    Z::T
    ψ_k::LazyArray{T, 1}
    function JacobiMeasure{T}(a, b, α, β, ψ_k) where {T<:Real}
        if !(α > -1 && β > -1)
            error("jacobi exponents must be > -1")
        end
        Z = SpecialFunctions.beta(α+1,β+1) * 2^(α+β) * (b-a)
        new{T}(a, b, α, β, Z, ψ_k)
    end
end

JacobiMeasure(a::T, b::T, α::T, β::T, ψ_k::LazyArray{T, 1}) where {T<:Real} = JacobiMeasure{T}(a, b, α, β, ψ_k)
function JacobiMeasure(a::Real, b::Real, α::Real, β::Real, ψ_k::LazyArray{T, 1}) where {T<:Real}
    U = promote_type(typeof(a), typeof(b), typeof(α), typeof(β), eltype(ψ_k))
    JacobiMeasure(convert(U, a), convert(U, b), convert(U, α), convert(U, β), convert(LazyArray{U, 1}, ψ_k))
end

function JacobiMeasure(a::Real, b::Real, α::Real, β::Real, f::Function)
    P = Jacobi(α, β)
    f_ab(x::Real) = f(M_ab_inv(x, a, b))
    ψ_k = P \ f_ab.(axes(P, 1))
    JacobiMeasure(a, b, α, β, ψ_k)
end

getindex(m::JacobiMeasure, x) = (Weighted(Jacobi(m.α, m.β)) * m.ψ_k)[M_ab_inv.(x, m.a, m.b)] / m.Z

const Semicircle(R) = ChebyshevUMeasure(-R, R, e0inf)
const Semicircle() = Semicircle(2)

