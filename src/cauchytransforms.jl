export cauchytransform, dcauchytransform

function cauchytransform(z::Number, m::OPMeasure)
    inv.(z .- axes(m_op(m), 1)') * Weighted(m_op(m)) * m.ψ_k
end

function cauchytransform(z::AbstractVector{T}, m::OPMeasure) where T<:Number
    (inv.(z .- axes(m_op(m), 1)') * Weighted(m_op(m)) * m.ψ_k)[:]
end

function cauchytransform(z::Number)
    f(m) = cauchytransform(z, m)
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

function dcauchytransform(z::Number)
    f(m) = dcauchytransform(z, m)
end

function cauchytransform(z::Number, m::SumOPMeasure)
    sum(map(cauchytransform(z), m.m_k))
end

function dcauchytransform(z::Number, m::SumOPMeasure)
    sum(map(dcauchytransform(z), m.m_k))
end