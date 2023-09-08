export invcauchytransform

# Joukowski transform
Jinv_p(z) = z - √(z - 1) * √(z + 1)
J(z) = (z + 1/z)/2

inunitdisk(z::Number) = abs(z) ≤ 1+10^-9    # for rounding errors

function invertpolynomial(P, z::Number)
    P1 = P[1:end-1]/P[end]
    P1[1] -= z/P[end]
    eigvals(SpecialMatrices.Companion(P1))
end


function invcauchytransform(z::Number, m::ChebyshevUMeasure; maxterms=100, tol=10^-15)::Vector{Complex}
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    s = invertpolynomial(Complex.(vcat([0], m.ψ_k[1:n])), (m.b-m.a)/4 * z)
    M_ab.(J.(filter!(inunitdisk, s)), m.a, m.b)
end

function invcauchytransform(y::T, m::JacobiMeasure; maxterms=20, tol=10^-9, N=1000, r=0.9) where T<:Number
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    if n == 1
        return invcauchytransform_1(y, m; N, r)
    end
    f_k = m.ψ_k[1:n-1]
    last = m.ψ_k[n]
    J = jacobimatrix(Jacobi(m.α,m.β))
    A = Matrix(J[1:n-1,1:n-1]'); A[end,:] -= f_k ./ last .* J[n, n-1]
    b = zeros(T, n-1); b[end] = y/last * J[n, n-1]; b[1] = 1
    Σ = zeros(n-1); Σ[1] += 1

    A1 = SMatrix{n, n, T}([0 Σ';b A])
    A2 = SMatrix{n, n, T}(Diagonal([-(i != 0) for i=0:n-1]))
    A3 = SMatrix{n, n, T}([0 zeros(n-1)';A*Σ zeros(n-1, n-1)])
    A4 = SMatrix{n, n, T}([0 zeros(n-1)';-Σ zeros(n-1, n-1)])

    function q_0(z::Vector{T}) where {T<:Number}
        (inv.(M_ab_inv.(z,m.a,m.b).- x_axes') * Weighted(Jacobi(m.α, m.β)) * e0inf / m.Z)[:]
    end

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
        H3(z::Number) = (I + z) * inv(I - z) + m.b
        H4(z::Number) = -(I + z) * inv(I - z) + m.a
        push!(functionlist, H3)
        push!(functionlist, H4)
    end
    inverses = Vector{Complex}()
    for H in functionlist
        function T_nep(z::Vector{T}) where T <: Number
            hz = map(H, z)
            q_0hz = q_0(hz)
            [A1 + hz[i] * A2 + q_0hz[i] * (A3 + hz[i] * A4) for i=1:length(z)]
        end
        λ = beyn(T_nep, n; r, N)
        for z in H.(λ)
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end

function beyn(T::Function, m::Int; r=0.8, N=1000, svtol=10^-12)
    Random.seed!(163) # my favourite integer
    exp2πimjN = [exp(2π*im * j / N) for j=0:N-1]

    invT = inv.(T(r .* exp2πimjN)) .* exp2πimjN

    V̂ = randn(ComplexF64,m,m)
    A_0N = r/N * sum(invT) * V̂
    A_1N = r^2/N * sum(invT .* exp2πimjN) * V̂
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

function invcauchytransform_1(y::T, m::JacobiMeasure; N=1000, r=0.9) where T<:Number
    function q_0(z::Vector{T}) where {T<:Number}
        (inv.(M_ab_inv.(z,m.a,m.b).- x_axes') * Weighted(Jacobi(m.α, m.β)) * e0inf / m.Z)[:]
    end
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
    inverses = Vector{Complex}()
    for H in functionlist
        T_nep(z::Vector{T}) where T <: Number = cauchytransform(H.(z), m) .- y
        λ = beyn(T_nep, 1; r, N)
        for z in H.(λ)
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end