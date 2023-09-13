export invcauchytransform

# Joukowski transform
Jinv_p(z) = z - √(z - 1) * √(z + 1)
J(z) = (z + 1/z)/2

inunitdisk(T) = z::Number -> abs(z) ≤ 1+100eps(real(T)) # for rounding errors

function invcauchytransform(y::T, m::ChebyshevUMeasure; maxterms=100, tol=10^-15) where T<:Number
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

function invcauchytransform(y::AbstractVector{T}, m::ChebyshevUMeasure; maxterms=100, tol=10^-15) where T<:Number
    ans = Vector{Vector{complex(T)}}(undef, length(y))
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    P = Complex.(vcat([0], m.ψ_k[1:n]))
    P1 = P[1:end-1]/P[end]
    C = SpecialMatrices.Companion(P1)
    for i=1:length(y)
        C.c[1] = -y[i]/π/P[end]
        s = eigvals(C)
        ans[i] = M_ab.(J.(filter!(inunitdisk(T), s)), m.a, m.b)
    end
    ans
end

function getconformalmaps(y, supp) # TODO: make work for multiple interval support
    functionlist = Vector{Function}()
    if imag(y) > 0
        push!(functionlist, z::Number -> -im * (I + z) * inv(I - z))
    end
    if imag(y) < 0
        push!(functionlist, z::Number -> im * (I + z) * inv(I - z))
    end
    if abs(imag(y)) < 10^-10
        push!(functionlist, z::Number -> (I + z) * inv(I - z) + supp[2])
        push!(functionlist, z::Number -> -(I + z) * inv(I - z) + supp[1])
    end
    functionlist
end

function invcauchytransform(y::T, m::OPMeasure; maxterms=20, tol=10^-9, N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    if n == 1
        return invcauchytransform_1(y, m; N, r)
    end
    f_k = m.ψ_k[1:n-1]; last = m.ψ_k[n]
    A1, A2, A3, A4, K = creatematrices(y, OP, f_k, last, n)
    function q_0(z::AbstractVector{T}) where {T<:Number}
        (inv.(z .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(T, ∞)))[:]
    end
    functionlist = getconformalmaps(y, (m.a, m.b))
    inverses = Vector{Complex{real(T)}}()
    for H in functionlist
        λ = beyn(A1, A2, A3, A4, OP, H; r, N, svtol=10^-10)
        for z in λ
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end

function invcauchytransform_1(y::T, m::OPMeasure; N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    last = m.ψ_k[1]
    function q_0(z::AbstractVector{T}) where {T<:Number}
        (inv.(z .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(T, ∞)))[:]
    end
    functionlist = getconformalmaps(y, (m.a, m.b))
    inverses = Vector{Complex{real(T)}}()
    for H in functionlist
        λ = beyn(y, convert(T,last), OP, H; r, N, svtol=10^-12)
        for z in λ
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end

function invcauchytransform_1(y::AbstractVector{T}, m::OPMeasure; N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    last = m.ψ_k[1]
    function q_0(z::AbstractVector{T}) where {T<:Number}
        (inv.(z .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(T, ∞)))[:]
    end
    H(z::Number) = im * (I + z) * inv(I - z)
    beyn_multi(y, last, OP, H; r, N, svtol=10^-12)
end


# at the moment, jacobi(2,2,-1..1) isa OrthogonalPolynomial is false...

function beyn(A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12) where T <: AbstractArray
    Random.seed!(163) # my favourite integer
    m = size(A1)[1]; C = 0:N-1
    V̂ = SMatrix{m,m}(randn(ComplexF64,m,m))
    exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
    Hrexp2πimjN = @. H(r * exp2πimjN)
    q_0hz = (inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = A1 + Hrexp2πimjN[n] * A2 + q_0hz[n] * (A3 + Hrexp2πimjN[n] * A4)
    array_T = LazyArray(@~ @. T_nep(1:N))
    invTV = BroadcastArray(\, array_T, Ref(V̂)) .* exp2πimjN
    A_0N = r/N * sum(invTV)
    A_1N = r^2/N * sum(invTV .* exp2πimjN)
    H.(beynsvd(A_0N, A_1N, svtol))
end

function beyn(y::T, last::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12) where T <: Number
    Random.seed!(163) # my favourite integer
    C = 0:N-1
    exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
    Hrexp2πimjN = @. H(r * exp2πimjN)
    q_0hz = (inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = -y + q_0hz[n] * last
    invTV = LazyArray(@~ @. inv(T_nep(1:N))) .* exp2πimjN
    A_0N = sum(invTV)
    A_1N = r * sum(invTV .* exp2πimjN)
    B = A_1N/A_0N
    abs(r / N * A_0N) < svtol ? Vector{T}[] : [H(B)]
end



function invcauchytransform(y::AbstractVector{T}, m::OPMeasure; maxterms=20, tol=10^-9, N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    if n == 1
        return invcauchytransform_1(y, m; N, r)
    end
    testval = -one(eltype(y))im
    f_k = m.ψ_k[1:n-1]; last = m.ψ_k[n]
    A1, A2, A3, A4, K = creatematrices(testval, OP, f_k, last, n)
    function q_0(z::AbstractVector{T}) where {T<:Number}
        (inv.(z .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(T, ∞)))[:]
    end
    H = z::Number -> im * (2 * inv(1-z)-1)
    beyn_multi(y, A1, A2, A3, A4, OP, H; r, N, svtol=10^-10, testval, K)
end

# function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12, testval=-im, K::Real=1) where {T3<: Number, T2<:Complex, T<:AbstractArray{T3}}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = randn(ComplexF64,m,m)
#     exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
#     Hrexp2πimjN = LazyArray(@~ @. H(r * exp2πimjN))

#     q_0hz = Vector(inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))

#     T_nep(n::Int) = A1 + Hrexp2πimjN[n] * A2 + q_0hz[n] * (A3 + Hrexp2πimjN[n] * A4)

#     invT = Vector{Matrix{T3}}(undef, N)
#     invT = inv.(T_nep.(1:N))

#     ans = Vector{Vector{T3}}(undef, length(y))
#     rank1invT = Vector{Matrix{T3}}(undef, N)
#     for i in eachindex(y)
#         yp = y[i]
#         rank1invT = deepcopy(invT)
#         f(M::Matrix) = invbl!(M, (yp - testval)*K)
#         map(f, rank1invT)
#         rank1invT .*= exp2πimjN
#         A_0N = r/N * sum(rank1invT) * V̂
#         rank1invT .*= exp2πimjN
#         A_1N = r^2/N * sum(rank1invT) * V̂
#         ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
#     end
#     ans
# end
m_f(x::Int) = x < 6 ? x : round(log2(x))+3

function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12, testval=-im, K::Real=1) where {T3<: Number, T2<:Complex, T<:AbstractArray{T3}}
    Random.seed!(163) # my favourite integer
    m = size(A1)[1]; C = 0:N-1
    V̂ = randn(ComplexF64,m,m_f(m))
    exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
    Hrexp2πimjN = LazyArray(@~ @. H(r * exp2πimjN))

    q_0hz = Vector(inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))

    T_nep(n::Int) = (A1 + Hrexp2πimjN[n] * A2 + q_0hz[n] * (A3 + Hrexp2πimjN[n] * A4))
    invT = Vector{Matrix{T3}}(undef, N)
    invT = inv.(T_nep.(1:N))

    ans = Vector{Vector{T3}}(undef, length(y))
    yv_a = ApplyArray(vcat, y[1] - testval, Diff(y)) .* K
    for (i,yv) in enumerate(yv_a)
        f(M::AbstractMatrix) = invbl!(M, yv)
        map(f, invT)
        A_0N = r/N * sum(invT .* exp2πimjN) * V̂
        A_1N = r^2/N * sum(invT .* exp2πimjN .^ 2) * V̂
        ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
    end
    ans
end



# function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12, testval=-im, K::Real=1) where {T3<: Number, T2<:Complex, T<:AbstractArray{T3}}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = randn(ComplexF64,m,m)
#     exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
#     Hrexp2πimjN = LazyArray(@~ @. H(r * exp2πimjN))

#     q_0hz = Vector(inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))

#     T_nep(n::Int) = A1 + Hrexp2πimjN[n] * A2 + q_0hz[n] * (A3 + Hrexp2πimjN[n] * A4)

#     invT = Array{T3}(undef, m, m, N)
#     rank1invT = Array{T3}(undef, m, m, N)

#     invT = cat(inv.T_nep.(1:N)..., dims=3);
#     ans = Vector{Vector{T3}}(undef, length(y))


#     for i in eachindex(y)
#         yp = y[i]
#         copy!(rank1invT, invT)
#         for j=1:N
#             A = @view rank1invT[:,:,j]
#             invbl!(A, (yp - testval)*K)
#         end
#         rank1invT .*= exp2πimjN
#         A_0N = r/N * sum(rank1invT) * V̂
#         rank1invT .*= exp2πimjN
#         A_1N = r^2/N * sum(rank1invT) * V̂
#         ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
#     end
#     ans
# end


# function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12, testval=-im, K=1) where {T<:AbstractArray, T2<:Complex}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = SMatrix{m,m}(randn(ComplexF64,m,m))
#     exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
#     Hrexp2πimjN = LazyArray(@~ @. im * (2 * inv(1-r * exp2πimjN)-1))
#     q_0hz = SVector{N}(inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))
#     T_nep(n::Int) = A1 + Hrexp2πimjN[n] * A2 + q_0hz[n] * (A3 + Hrexp2πimjN[n] * A4)
#     invT = inv.(Ref(A1) .+ BroadcastArray(*, Hrexp2πimjN, Ref(A2)) + BroadcastArray(*, q_0hz, Ref(A3)) + BroadcastArray(*, Hrexp2πimjN, Ref(A4)) .* q_0hz)
#     ans = Vector{Vector{eltype(A1)}}(undef, length(y))
#     for i=1:length(y)
#         yp = y[i]
#         test1 = MMatrix{m,m}.(invT)
#         rank1invT = invbl!.(test1, (yp - testval)*K) .* exp2πimjN
#         A_0N = r/N * sum(rank1invT) * V̂
#         A_1N = r^2/N * sum(rank1invT .* exp2πimjN) * V̂ 
#         ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
#     end
#     ans
# end

function beyn_multi(y::AbstractVector{T}, last::T2, OP::AbstractQuasiMatrix, H::Function; r=0.9, N=1000, svtol=10^-12) where {T2<:Number, T<:Complex}
    Random.seed!(163) # my favourite integer
    C = 0:N-1
    exp2πimjN = LazyArray(@~ @. cispi(2*C/N))
    Hrexp2πimjN = @. H(r * exp2πimjN)
    q_0hz = (inv.(Hrexp2πimjN .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = -q_0hz[n] * last
    arrayT = LazyArray(@~ @. T_nep(1:N)) # is now a concrete array
    ans = Vector{Vector{T}}(undef, length(y))
    @simd for i=1:length(y)
        yp = y[i]
        invT = inv.(arrayT .+ yp) .* exp2πimjN
        A_0N = sum(invT)
        A_1N = sum(invT .* exp2πimjN)
        B = r*A_1N/A_0N
        ans[i] = abs(r / N * A_0N) < svtol ? Vector{T}[] : [H(B)]
    end
    ans
end

function creatematrices(y::T, OP::AbstractQuasiArray, f_k::AbstractVector{T2}, last::T2, n::Int) where {T<:Number, T2<:Real}
    J = jacobimatrix(OP)
    A = Matrix(J[1:n-1,1:n-1]'); A[end,:] -= f_k ./ last .* J[n, n-1]
    b = zeros(T, n-1); b[end] = y/last * J[n, n-1]; b[1] = sum(orthogonalityweight(OP))
    Σ = zeros(n-1); Σ[1] += 1
    U = complex(T)
    A1 = MMatrix{n,n,U}([0 Σ';b A])
    A2 = MMatrix{n,n,U}(Diagonal([-(i != 0) for i=0:n-1]))
    A3 = MMatrix{n,n,U}([0 zeros(n-1)';A*Σ zeros(n-1, n-1)])
    A4 = MMatrix{n,n,U}([0 zeros(n-1)';-Σ zeros(n-1, n-1)])
    A1, A2, A3, A4, J[n, n-1]/last
end

function beynsvd(A_0N::AbstractMatrix, A_1N::AbstractMatrix, svtol=10^-10)
    V, S, W = svd(A_0N)
    k = m = size(A_0N)[1]
    while k >= 1 && abs(S[k]) < svtol
        k -= 1
    end
    V_0 = V[1:m, 1:k]
    W_0 = W[1:m, 1:k]
    S_0 = Diagonal(S[1:k])
    eigvals(V_0' * A_1N * W_0 * S_0^-1)
end

function invbl!(Ainv::AbstractMatrix{T}, y::Number) where T<:Number
    v = @view Ainv[1,:]; u = @view Ainv[:,end];
    y = convert(T,y)
    Ainv .-= y ./ (1 .+ y * v[end]) .* u*transpose(v)
end;

function invbl!(Ainv::AbstractMatrix{T}, y::Number) where T<:LinearAlgebra.BlasFloat
    v = Ainv[1,:]; u = Ainv[:,end];
    y = convert(T,y)
    BLAS.ger!(-y/(1 + y * v[end]), u, conj!(v), Ainv)
end;

function invertpolynomial(P, z::Number)
    P1 = P[1:end-1]/P[end]
    P1[1] -= z/P[end]
    eigvals(SpecialMatrices.Companion(P1))
end

