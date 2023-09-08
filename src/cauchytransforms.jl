export cauchytransform, dcauchytransform

const x_axes = axes(ChebyshevU(), 1)

function cauchytransform(z::Number, m::ChebyshevUMeasure)
    inv.(M_ab_inv(z,m.a,m.b) .- x_axes') * Weighted(ChebyshevU()) * m.ψ_k / m.Z
end

function cauchytransform(z::Vector{T}, m::ChebyshevUMeasure) where T<:Number
    (inv.(M_ab_inv.(z,m.a,m.b) .- x_axes') * Weighted(ChebyshevU()) * m.ψ_k / m.Z)[:]
end

function cauchytransform(z::Number, m::JacobiMeasure)
    inv.(M_ab_inv(z,m.a,m.b) .- x_axes') * Weighted(Jacobi(m.α, m.β)) * m.ψ_k / m.Z
end

function cauchytransform(z::Vector{T}, m::JacobiMeasure) where T<:Number
    (inv.(M_ab_inv.(z,m.a,m.b) .- x_axes') * Weighted(Jacobi(m.α, m.β)) * m.ψ_k / m.Z)[:]
end


function dcauchytransform(z::Number, m::AbstractJacobiMeasure)
    G(z::Number) = cauchytransform(z, m)
    function G_real(x)
        z = x[1] + x[2] * im
        w = G(z)
        [real(w), imag(w)]
    end
    if isa(z, Real)
        return ForwardDiff.derivative(G, z)
    end
    w = ForwardDiff.jacobian(G_real, [real(z), imag(z)])
    w[1,1] + w[2,1] * im
end
