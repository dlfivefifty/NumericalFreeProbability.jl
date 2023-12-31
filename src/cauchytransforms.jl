export cauchytransform, dcauchytransform, Gₘ, Gₘ′

function cauchytransform(z::Number, m::OPMeasure)
    inv.(z .- axes(m_op(m), 1)') * Weighted(m_op(m)) * m.ψ_k
end

function cauchytransform(z::AbstractVector{T}, m::OPMeasure) where T<:Number
    (inv.(z .- axes(m_op(m), 1)') * Weighted(m_op(m)) * m.ψ_k)[:]
end

function cauchytransform(z)
    f(m::Measure) = cauchytransform(z, m)
end



function cauchytransform(z::Number, m::SumMeasure)
    sum(map(cauchytransform(z), m.m_k))
end

function cauchytransform(z::AbstractVector{T}, m::SumMeasure) where T<:Number
    sum(map(cauchytransform(z), m.m_k))
end



function cauchytransform(z::Number, pm::PointMeasure)
    n = length(pm.a)
    ans = 0
    for i=1:n
        ans += pm.a[i]/(z-pm.λ[i])
    end
    ans
end

function cauchytransform(z::AbstractVector{T}, pm::PointMeasure) where T<:Number
    n = length(pm.a)
    ans = zeros(complex(T), length(z))
    for i=1:n
        ans .+= pm.a[i]./(z.-pm.λ[i])
    end
    ans
end


# derivative of cauchy transform, only evaluated on single points

function dcauchytransform(z::Number, pm::PointMeasure)
    n = length(pm.a)
    ans = 0
    for i=1:n
        ans -= pm.a[i]/(z-pm.λ[i])^2
    end
    ans
end


function dcauchytransform(z::Number, m::SumMeasure)
    sum(map(dcauchytransform(z), m.m_k))
end


function dcauchytransform(z::Number, m::Measure)
    G(z::Number) = cauchytransform(z, m)
    function G_real(x::Vector{T}) where T<:Number
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


const Gₘ = cauchytransform
const Gₘ′ = dcauchytransform