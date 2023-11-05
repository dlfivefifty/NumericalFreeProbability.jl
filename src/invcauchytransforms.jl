export invcauchytransform, invcauchytransform_fft, weightmoments

using AMRVW, FFTW, ToeplitzMatrices, FastGaussQuadrature

# Joukowski transform
Jinv_p(z) = z - √(z - 1) * √(z + 1)
J(z) = (z + inv(z))/2
dJ(z) = (1 - inv(z^2))/2

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

#############################################################################


function invcauchytransform(y::T, m::JacobiMeasure; maxterms=20, tol=10^-9, N=1000, r=0.9) where T<:Number
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
    inverses = Vector{Complex{real(T)}}()
    moments = weightmoments(m, n+1)
    λ = beyn(A1, A2, A3, A4, OP, z::Number -> M_ab(J(z), m.a, m.b), z::Number -> (m.b - m.a)/2 * dJ(z), (m.a,m.b), moments, true; r, N, svtol=10^-14)
    for z in λ
        if all(abs.(z .- inverses) .> 10^-10)
            push!(inverses, z)
        end
    end
    filterinverses!(inverses, y, m)
end


function invcauchytransform_1(y::T, m::JacobiMeasure; N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    last = m.ψ_k[1]
    moments = weightmoments(m, 1)
    inverses = beyn(y, convert(T,last), OP, z::Number -> M_ab(J(z), m.a, m.b), z::Number -> (m.b - m.a)/2 * dJ(z), (m.a,m.b), moments; r, N, svtol=10^-14)
    filterinverses!(inverses, y, m)
end

function invcauchytransform_1(y::AbstractVector{T}, m::JacobiMeasure; N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    last = m.ψ_k[1]
    moments = weightmoments(m, 1)
    inverses = beyn_multi(y, convert(T,last), OP, z::Number -> M_ab(J(z), m.a, m.b), z::Number -> (m.b - m.a)/2 * dJ(z), (m.a,m.b), moments; r, N, svtol=10^-14)
    filterinverses!(inverses, y, m)
end

# function beyn(A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function, dH::Function, support, moments, res::Bool; r=0.9, N=1000, N2=100, svtol=10^-14) where {T2<: Number, T <: Union{AbstractArray{T2}, T2}}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = randn(complex(eltype(A1)),m,m_f(m))

#     z = LazyArray(@~ @. cispi(2*C/N))
#     hz =  @. H(r * z)
#     dhz =  @. dH(r * z)
#     q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]

    
#     invT = nep_arrays(A1, A2, A3, A4, hz, q_0hz, N)
#     A_0N = r/N * dotu_mv(invT, z, N)
#     A_1N = r^2/N * dotu_mv(invT, z.^2, N)

#     # r2 = 0.1
#     # C2 = 0:N2-1
#     # z_2 = LazyArray(@~ @. r2 * cispi(2*C2/N2))
#     # hz_2 = LazyArray(@~ @. H(z_2))
#     # q_0hz_2 = Vector(inv.(hz_2 .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))
#     # invT_2 = nep_arrays(A1, A2, A3, A4, hz_2, q_0hz_2, N2)
#     # res_0N = 1/N2 * dotu_mv(invT_2, z_2, N2) 
#     # res_1N = 1/N2 * dotu_mv(invT_2, z_2.^2, N2)

#     c = (support[1] + support[2])/2
#     w = (support[2] - support[1])/4

#     if res
#         res_0N = zeros(T2, m, m)
#         res_1N = zeros(T2, m, m)

#         A = A1[2:end, 2:end]; Σ = A1[1, 2:end]
#         y_i = vcat([A1[2:end,1] - moments[1] * Σ], [A*Σ*moments[i] - Σ*moments[i+1] for i=1:2m-3])
#         B_i = create_B_i(A, y_i, Σ, m)
#         for n=1:m-1
#             res_0N += B_i[m-n] * sum([trinomial(n,n-k,(k+1)÷2) * w^k * c^(n-k) for k=1:2:n])
#         end
#         for n=2:m-1
#             res_1N += B_i[m-n] * sum([trinomial(n,n-k,(k+2)÷2) * w^k * c^(n-k) for k=2:2:n])
#         end
#         return H.(beynsvd((A_0N - res_0N) * V̂ , (A_1N - res_1N) * V̂, svtol))
#     end
#     H.(beynsvd(A_0N * V̂ , A_1N * V̂, svtol))
# end

# function beyn(A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function, dH::Function, support, moments, res::Bool; r=0.9, N=1000, N2=1000, svtol=10^-14) where {T2<: Number, T <: Union{AbstractArray{T2}, T2}}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = randn(complex(eltype(A1)),m,m_f(m))

#     z = LazyArray(@~ @. cispi(2*C/N))
#     hz =  @. H(r * z)
#     dhz =  @. dH(r * z)
#     q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]

#     invT = nep_arrays(A1, A2, A3, A4, hz, q_0hz, N)
#     A_0N = r/N * dotu_mv(invT, z .* dhz, N)
#     A_1N = r/N * dotu_mv(invT, z .* dhz .* hz, N)

#     if res
#         res_0N = zeros(T2, m, m)
#         res_1N = zeros(T2, m, m)

#         A = A1[2:end, 2:end]; Σ = A1[1, 2:end]
#         y_i = vcat([A1[2:end,1] - moments[1] * Σ], [A*Σ*moments[i] - Σ*moments[i+1] for i=1:2m])
#         B_i = create_B_i(A, y_i, Σ, m)


#         # function itnep(z)
#         #     hz = H(z)
#         #     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#         #     inv(A1 + hz * A2 + qhz * (A3 + hz * A4))
#         # end
#         # function itnep2(z)
#         #     hz = 1/z
#         #     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#         #     inv(A1 + hz * A2 + qhz * (A3 + hz * A4))
#         # end
#         # function tnep(z)
#         #     hz = H(z)
#         #     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#         #     A1 + hz * A2 + qhz * (A3 + hz * A4)
#         # end
#         # function tnep2(z)
#         #     hz = 1/z
#         #     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#         #     A1 + hz * A2 + qhz * (A3 + hz * A4)
#         # end
#         # println("grhgregergregeri")
#         # display(norm(cres1(-1, itnep2)-B_i[1]))
#         # display(norm(cres1(0, itnep2)-B_i[2]))
#         # display(norm(cres1(1, itnep2)-B_i[3]))
#         # display(norm(cres1(2, itnep2)-B_i[4]))
#         # println("grhgregergregeri")
#         # display(norm(cres1(2, itnep2) + res_0N))
#         # display(norm(cres1(3, itnep2) + res_1N))


#         # display(norm(B_i[1]*A_i[1]))
#         # display(norm(A_i[1]*B_i[1]))
#         # display(norm(B_i[1]*A_i[2] + B_i[2]*A_i[1]))
#         # display(norm(A_i[1]*B_i[2] + A_i[2]*B_i[1]))

#         res_0N = -B_i[end-1]
#         res_1N = -B_i[end]
#         return beynsvd((A_0N - res_0N) * V̂ , (A_1N - res_1N) * V̂, svtol)
#     end
#     H.(beynsvd(A_0N * V̂ , A_1N * V̂, svtol))
# end

function beyn(A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function, dH::Function, support, moments, res::Bool; r=0.9, N=1000, N2=1000, svtol=10^-14) where {T2<: Number, T <: Union{AbstractArray{T2}, T2}}
    Random.seed!(163) # my favourite integer
    m = size(A1)[1]; C = 0:N-1
    V̂ = randn(complex(eltype(A1)),m,m_f(m))

    z = LazyArray(@~ @. cispi(2*C/N))
    hz =  @. H(r * z)
    dhz =  @. dH(r * z)
    q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]

    invT = nep_arrays(A1, A2, A3, A4, hz, q_0hz, N)
    A_0N = r/N * dotu_mv(invT, z .* dhz, N)
    A_1N = r/N * dotu_mv(invT, z .* dhz .* hz, N)
    if res
        res_0N = zeros(T2, m, m)
        res_1N = zeros(T2, m, m)
        A = A1[2:end, 2:end]; Σ = A1[1, 2:end]
        y_i = vcat([A1[2:end,1] - moments[1] * Σ], [A*Σ*moments[i] - Σ*moments[i+1] for i=1:2m])
        B_i = create_B_i(A, y_i, Σ, m, m+3)

        res_0N = B_i[end-1]
        res_1N = B_i[end]
        return beynsvd((A_0N + res_0N) * V̂ , (A_1N + res_1N) * V̂, svtol)
    end
    H.(beynsvd(A_0N * V̂ , A_1N * V̂, svtol))
end

function B_i_tlentry_solve(A::Matrix{T}, y_i::Vector{Vector{T}}, m) where T<:Number
    tv = y_i[1]
    j = 1
    α_i = T[]
    for _=1:m-1
        push!(α_i, tv[1])
        tv = A*tv
        j += 1
        tv += y_i[j]
    end
    i = m-1
    while abs(α_i[end-m+2]) <10^-14
        push!(α_i, tv[1])
        tv = A*tv
        j += 1
        tv += y_i[j]
        i -= 1
    end
    M = LowerTriangularToeplitz(α_i[end-m+2: end])
    v = zeros(size(M)[1]); v[1] += 1 # v[i] += 1
    M \ v
end

function create_B_i(A::Matrix{T}, y_i::Vector{Vector{T}},Σ::Vector{T}, m::Int, n::Int=m) where T<:Number
    α_i = B_i_tlentry_solve(A, y_i, n)
    A_i = vcat([[0 Σ';y_i[1] A]], [[zeros(m)'; y_i[i] zeros(m-1,m-1)] for i=2:n-2]) # laurent coefficients A_0 to A_M-1
    B_i = [zeros(T,m,m) for _=1:n-1]
    B_i[1][1] += α_i[1]
    for j=2:n-1
        B = dotmatu(A_i, B_i, m, j-1, true)
        B[2:end, 1] = dotmatu(A_i, B_i, m, j-1, false)[2:end, 1]
        if j == m+1
            B -= I
        end
        B[1] = α_i[j]
        B_i[j] = B
    end
    B_i
end

function dotmatu(a::Vector{Matrix{T}}, b::Vector{Matrix{T}}, m::Int, k::Int, right::Bool) where T<:Number
    x = zeros(T, m, m)
    if right
        for i=1:k
            x += b[k+1-i] * a[i]
        end
    else
        for i=1:k
            x += a[i] * b[k+1-i]
        end
    end
    x
end


function cres1(m, f; r=0.1)
    function dotu_mv2(A::Vector{U}, B::AbstractVector{T}, N::Int) where {T<:Number, U<:Any}
        total = A[1] * B[1]
        for i=2:N
            total += A[i] * B[i]
        end
        total
    end
    N=2000
    z = cispi.(0:2//N:2-eps())*r
    dotu_mv2(f.(z), z.^(1-m), N) / N
end





function nep_arrays(A1::T, A2::T, A3::T, A4::T, hz::AbstractVector{T2}, qhz::AbstractVector{T2}, N::Int) where {T2<: Number, T <: AbstractArray{T2}}
    ret = Vector{T}(undef, N)
    for i=1:N
        ret[i] = inv(A1 + hz[i] * A2 + qhz[i] * (A3 + hz[i] * A4))
    end
    ret
end

function dotu_mv(A::Vector{Matrix{T}}, B::AbstractVector{T}, N::Int) where T<:Number
    total = A[1] * B[1]
    for i=2:N
        total += A[i] * B[i]
    end
    total
end




function beyn(y::T, last::T, OP::AbstractQuasiArray, H::Function, dH::Function, support, moments; r=0.9, N=1000, svtol=10^-14) where T <: Number
    Random.seed!(163) # my favourite integer
    C = 0:N-1
    z = LazyArray(@~ @. cispi(2*C/N))

    hz = @. H(r * z)
    dhz = @. dH(r * z)

    q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = -y + q_0hz[n] * last
    invTV = LazyArray(@~ @. inv(T_nep(1:N))) .* dhz .* z

    res_0N = last * moments[1] / y^2
    res_1N = last * moments[2] / y^2 + last^2 * moments[1]^2 / y^3

    A_0N = r/N * sum(invTV) - res_0N
    A_1N = r/N * sum(invTV .* hz) - res_1N
    B = A_1N/A_0N


    abs(r / N * A_0N) < svtol ? Vector{T}() : [B]
end



function invcauchytransform(Y::AbstractVector{T}, m::JacobiMeasure; maxterms=20, tol=10^-9, N=1000, r=0.9) where T<:Number
    OP = m_op(m)
    n = maxterms
    while abs(m.ψ_k[n]) < tol
        n -= 1
    end
    if n == 1
        return invcauchytransform_1(Y, m; N, r)
    end
    testval = -one(eltype(Y))im
    f_k = m.ψ_k[1:n-1]; last = m.ψ_k[n]
    A1, A2, A3, A4, K = creatematrices(testval, OP, f_k, last, n)
    H = z::Number -> im * (2 * inv(1-z)-1)
    inverses = beyn_multi(Y, A1, A2, A3, A4, OP, H; r, N, svtol=10^-14, testval, K)
    filterinverses!(inverses, Y, m)
end

m_f(x::Int) = (x + 1) ÷ 2

function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function; r=0.9, N=1000, svtol=10^-14, testval=-im, K::Real=1) where {T3<: Number, T2<:Complex, T<:AbstractArray{T3}}
    Random.seed!(163) # my favourite integer
    m = size(A1)[1]; C = 0:N-1
    V̂ = randn(ComplexF64,m,m_f(m))
    z = LazyArray(@~ @. cispi(2*C/N))
    hz = LazyArray(@~ @. H(r * z))

    q_0hz = Vector(inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))

    T_nep(n::Int) = (A1 + hz[n] * A2 + q_0hz[n] * (A3 + hz[n] * A4))

    invT = inv.(T_nep.(1:N))

    ans = Vector{Vector{T3}}(undef, length(y))
    yv_a = ApplyArray(vcat, y[1] - testval, Diff(y)) .* K
    for (i,yv) in enumerate(yv_a)
        f(M::AbstractMatrix) = invbl!(M, yv)
        map(f, invT)
        A_0N = r/N * sum(Vector(invT .* z)) * V̂     # TODO: make this not bad
        A_1N = r^2/N * sum(Vector(invT .* z .^ 2)) * V̂
        ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
    end
    ans
end

function beynupdate!(invT::Vector{Matrix{T1}}, yv_a::AbstractVector{T2}, m::Int, ans::Vector{Vector{T1}}, svtol=10^-14) where {T1<:Number, T2<:Complex}
    V̂ = randn(ComplexF64,m,m_f(m))
    for (i,yv) in enumerate(yv_a)
        f(M::AbstractMatrix) = invbl!(M, yv)
        map(f, invT)
        A_0N = r/N * sum(invT .* z) * V̂
        A_1N = r^2/N * sum(invT .* z .^ 2) * V̂
        ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
    end
end

# function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function; r=0.9, N=1000, svtol=10^-14, testval=-im, K::Real=1) where {T3<: Number, T2<:Complex, T<:AbstractArray{T3}}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = randn(ComplexF64,m,m)
#     z = LazyArray(@~ @. cispi(2*C/N))
#     hz = LazyArray(@~ @. H(r * z))

#     q_0hz = Vector(inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))

#     T_nep(n::Int) = A1 + hz[n] * A2 + q_0hz[n] * (A3 + hz[n] * A4)

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
#         rank1invT .*= z
#         A_0N = r/N * sum(rank1invT) * V̂
#         rank1invT .*= z
#         A_1N = r^2/N * sum(rank1invT) * V̂
#         ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
#     end
#     ans
# end


# function beyn_multi(y::AbstractVector{T2}, A1::T, A2::T, A3::T, A4::T, OP::AbstractQuasiArray, H::Function; r=0.9, N=1000, svtol=10^-14, testval=-im, K=1) where {T<:AbstractArray, T2<:Complex}
#     Random.seed!(163) # my favourite integer
#     m = size(A1)[1]; C = 0:N-1
#     V̂ = SMatrix{m,m}(randn(ComplexF64,m,m))
#     z = LazyArray(@~ @. cispi(2*C/N))
#     hz = LazyArray(@~ @. im * (2 * inv(1-r * z)-1))
#     q_0hz = SVector{N}(inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))
#     T_nep(n::Int) = A1 + hz[n] * A2 + q_0hz[n] * (A3 + hz[n] * A4)
#     invT = inv.(Ref(A1) .+ BroadcastArray(*, hz, Ref(A2)) + BroadcastArray(*, q_0hz, Ref(A3)) + BroadcastArray(*, hz, Ref(A4)) .* q_0hz)
#     ans = Vector{Vector{eltype(A1)}}(undef, length(y))
#     for i=1:length(y)
#         yp = y[i]
#         test1 = MMatrix{m,m}.(invT)
#         rank1invT = invbl!.(test1, (yp - testval)*K) .* z
#         A_0N = r/N * sum(rank1invT) * V̂
#         A_1N = r^2/N * sum(rank1invT .* z) * V̂ 
#         ans[i] = H.(beynsvd(A_0N, A_1N, svtol))
#     end
#     ans
# end

function beyn_multi(y::AbstractVector{T}, last::T2, OP::AbstractQuasiArray, H::Function; r=0.9, N=1000, svtol=10^-14) where {T2<:Number, T<:Complex}
    Random.seed!(163) # my favourite integer
    C = 0:N-1
    z = LazyArray(@~ @. cispi(2*C/N))
    hz = @. H(r * z)
    q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = -q_0hz[n] * last
    arrayT = LazyArray(@~ @. T_nep(1:N)) # is now a concrete array
    ans = Vector{Vector{T}}(undef, length(y))
    @simd for i=1:length(y)
        yp = y[i]
        invT = inv.(arrayT .+ yp) .* z
        A_0N = sum(invT)
        A_1N = sum(invT .* z)
        B = r*A_1N/A_0N
        ans[i] = abs(r / N * A_0N) < svtol ? Vector{T}[] : [H(B)]
    end
    ans
end

function beyn_multi(Y::AbstractVector{T}, last::T2, OP::AbstractQuasiArray, H::Function, dH::Function, support, moments; r=0.9, N=1000, svtol=10^-14) where {T2<:Number, T<:Complex}
    Random.seed!(163) # my favourite integer
    C = 0:N-1
    z = LazyArray(@~ @. cispi(2*C/N))

    hz = @. H(r * z)
    dhz = @. dH(r * z)

    q_0hz = (inv.(hz .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))[:]
    T_nep(n::Int) = q_0hz[n] * last

    res_0N(y) = last * moments[1] / y^2
    res_1N(y) = last * moments[2] / y^2 + last^2 * moments[1]^2 / y^3

    arrayT = T_nep.(1:N)
    ans = Vector{Vector{T}}(undef, length(Y))
    @simd for i=1:length(Y)
        y = Y[i]
        invT = inv.(arrayT .- y) .* dhz .* z
        A_0N = r/N * sum(invT) - res_0N(y)
        A_1N = r/N * sum(invT .* hz) - res_1N(y)
        B = A_1N/A_0N
        ans[i] = abs(r / N * A_0N) < svtol ? Vector{T}() : [B]
    end
    ans
end



function creatematrices(y::T, OP::AbstractQuasiArray, f_k::AbstractVector{T2}, last::T2, n::Int) where {T<:Number, T2<:Real}
    J = jacobimatrix(OP)
    A = Matrix(J[1:n-1,1:n-1]'); A[end,:] .-= f_k ./ last .* J[n, n-1]
    b = zeros(T, n-1); b[end] = y/last * J[n, n-1]; b[1] += sum(orthogonalityweight(OP))
    Σ = zeros(n-1); Σ[1] += 1
    U = complex(T)
    # A1 = MMatrix{n,n,U}([0 Σ';b A])
    # A2 = MMatrix{n,n,U}(Diagonal([-(i != 0) for i=0:n-1]))
    # A3 = MMatrix{n,n,U}([0 zeros(n-1)';A*Σ zeros(n-1, n-1)])
    # A4 = MMatrix{n,n,U}([0 zeros(n-1)';-Σ zeros(n-1, n-1)])
    A1 = U.([0 Σ';b A])
    A2 = U.(diagm([-(i != 0) for i=0:n-1]))
    A3 = U.([0 zeros(n-1)';A*Σ zeros(n-1, n-1)])
    A4 = U.([0 zeros(n-1)';-Σ zeros(n-1, n-1)])
    # A1 = U.([b A[:,2:end]])
    # A2 = U.(diagm([-(i != 0) for i=0:n-2]))
    # A3 = U.([A*Σ zeros(n-1, n-2)])
    # A4 = U.([-Σ zeros(n-1, n-2)])
    # print("A1 = ")
    # show(A1)
    # print(";A2 = ")
    # show(A2)
    # print(";A3 = ")
    # show(A3)
    # print(";A4 = ")
    # show(A4)
    A1, A2, A3, A4, J[n, n-1]/last
end

function beynsvd(A_0N::AbstractMatrix, A_1N::AbstractMatrix, svtol=10^-14)
    V, S, W = svd(A_0N)
    m = size(A_0N)[1]
    k = m_f(m)
    while k >= 1 && abs(S[k]) < svtol
        k -= 1
    end
    V_0 = V[:, 1:k]
    W_0 = W[:, 1:k]
    S_0 = Diagonal(S[1:k])
    ans = eigvals(V_0' * A_1N * W_0 * inv(S_0))
    #filter!(inunitdisk(eltype(S_0)), ans)
end

#updating bottom left entry of inverse functions
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

#required for inverting point measure
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
    P1 = P
    P1[1] -= z
    AMRVW.roots(P1)
    # P1 = P[1:end-1]/P[end]
    # P1[1] -= z/P[end]
    # eigvals(SpecialMatrices.Companion(P1))
end


function filterinverses!(inverses::Vector{T}, y::Number, m::OPMeasure, tol=10^-2) where T<:Number
    testinverse = z::T -> abs(cauchytransform(z, m) - y) < tol
    filter!(testinverse, inverses)
end

function filterinverses!(inverses::Vector{Vector{T}}, y::Vector{T}, m::OPMeasure, tol=10^-2) where T<:Number
    testinverse(y) = z::T -> abs(cauchytransform(z, m) - y) < tol
    n = length(y)
    for i=1:n
        filter!(testinverse(y[i]), inverses[i])
    end
    inverses
end

#alternative function for inverting via fft
function invcauchytransform_fft(z::T, m::OPMeasure; r=0.9, n = 200) where T<:Number
    a, b = minimum(axes(m)), maximum(axes(m))
    g = θ -> cauchytransform(M_ab(J(r * exp(im * θ)), a, b), m)
    θ = range(0, 2π; length=2*n+2)[1:end-1]
    fc = FFTW.fft(g.(θ))/(2*n+1)
    g_k = fc[1:n+1]
    P = g_k[1:n]
    while abs(g_k[n]) < 10^-9
        n -= 1
    end
    s = invertpolynomial(P[1:n], z)
    filter!(inunitdisk(T), s)
    M_ab.(J.(r .* s), a, b)
end


function invcauchytransform(y::Number, m::PointMeasure)
    n = length(m.a)
    P = -y .* productlinearfactors(m.λ)
    for i=1:n
        P += [m.a[i] * productlinearfactors([m.λ[1:i-1]; m.λ[i+1:end]]);0]
    end
    if y == 0
        P = P[1:end-1]
    end
    ans = sort!(invertpolynomial(P, 0); by=real)
    if y == 0
        ans = [Inf;ans]
    elseif isa(y, Real) && (y > 0)
        ans = [ans[end];ans[1:end-1]]
    end
    ans
end

function invcauchytransform(y::AbstractVector{T}, m::PointMeasure) where T<:Number
    n = length(m.a)
    P1 = productlinearfactors(m.λ)
    P2 = zeros(n+1)
    for i=1:n
        P2 += [m.a[i] * productlinearfactors([m.λ[1:i-1]; m.λ[i+1:end]]);0]
    end
    inverses = Vector{Vector{T}}(undef, length(y))
    for (i, yi) in enumerate(y)
        P = P2 - yi .* P1
        if yi == 0
            P = P[1:end-1]
        end
        ans = invertpolynomial(P, 0)
        if yi == 0
            inverses[i] = [Inf;ans]
        elseif isa(yi, Real) && (yi > 0)
            inverses[i] = ans
            circshift!(inverses[i], 1)
        else
            inverses[i] = ans
        end
    end
    inverses
end

function weightmoments(m::JacobiMeasure, n)
    nodes, weights = gaussjacobi(n, m.α, m.β)
    nodes = M_ab.(nodes, m.a, m.b)
    [(m.b - m.a)/2 * dot(nodes.^i, weights) for i=0:2n-1] # lazy
end



# function itnep(z)
#     hz = H(z)
#     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#     inv(A1 + hz * A2 + qhz * (A3 + hz * A4))
# end
# function itnep2(z)
#     hz = 1/z
#     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#     inv(A1 + hz * A2 + qhz * (A3 + hz * A4))
# end
# function tnep(z)
#     hz = H(z)
#     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#     A1 + hz * A2 + qhz * (A3 + hz * A4)
# end
# function tnep2(z)
#     hz = 1/z
#     qhz = inv.(hz.- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞))
#     A1 + hz * A2 + qhz * (A3 + hz * A4)
# end
# display(res_1N)
# display(cres1(-1, itnep))

# display(res_0N)
# display(cres1(0, itnep))
# println("A-1    ##############")
# display(cres1(0, tnep2))
# println("A 0    ##############")
# display(cres1(1, tnep2))
# println("A 1    ##############")
# display(cres1(2, tnep2))
# println("A 2    ##############")
# display(cres1(3, tnep2))
# println("A 3    ##############")
# display(cres1(4, tnep2))
# println("########################################")
# res_1N = zeros(eltype(A1), axes(A1))
# res_0N = zeros(eltype(A1), axes(A1))




# m0 = 16/15; m1 = 0; m2 = 1/7 * 16/15; m3 = 0

# A = A1[2:end, 2:end]
# Σ = A1[1, 2:end]
# y = A1[2:end,1] - m0 * Σ
# y1 = A*Σ*m0 - Σ*m1
# y2 = A*Σ*m1 - Σ*m2
# α = inv(Σ'*y1 +  Σ'*A * y)
# # res_1N[1] = α/4
# # β = -α*(Σ'*y2 + Σ'*A*y1 + Σ'*A*A*y)/(Σ'*A*y + Σ'*y1)
# # res_0N = [β α*Σ'; α*y zeros(2,2)] / 2

# α = inv(Σ'*y)
# # res_0N[1] = α/2

# display(α/2)
# display(res_1N)
# display(cres1(-1, itnep))

# display(res_0N)
# display(cres1(0, itnep))

#display([(norm(cres1(i, itnep)),i) for i=-4:4])


            # for i in eachindex(B_i)
            #     display(norm(B_i[i] - cres1(i-m+1, itnep2)))
            # end
            #display(abs.(B_i[4]-cres1(4-m+1, itnep2)))
            #display(m)

            
            
            # z_2 = LazyArray(@~ @. r2 * cispi(2*C2/N2))
            # hz_2 = LazyArray(@~ @. H(z_2))
            # q_0hz_2 = Vector(inv.(hz_2 .- axes(OP, 1)') * Weighted(OP) * vcat([1], zeros(∞)))
            # invT_2 = nep_arrays(A1, A2, A3, A4, hz_2, q_0hz_2, N2)
            # res_0N = 1/N2 * dotu_mv(invT_2, z_2, N2) 
            # res_1N = 1/N2 * dotu_mv(invT_2, z_2.^2, N2)
            # println("begin tesat")
            # display(norm(B_i[1]/16 - cres1(-3, itnep)))
            # display(norm(B_i[2]/8 - cres1(-2, itnep)))
            # display(norm(B_i[1]/4+B_i[3]/4 - cres1(-1, itnep)))
            # #display(norm(3B_i[2]/8+B_i[4]/2 - cres1(0, itnep)))

            # display(norm(B_i[1]/8 - cres1(-2, itnep)))
            # display(norm(B_i[2]/4 - cres1(-1, itnep)))
            # display(norm(3B_i[1]/8+B_i[3]/2 - cres1(0, itnep)))


function trinomial(n::Integer, k::Integer, l::Integer)
    binomial(n,k) * binomial(n-k,l)
end


function integrate(f, γ, dγ; N=1000)
    T(t) = f(γ(t)) * dγ(t)
    tj = (0:N-1)/N * 2π
    2π * sum(T, tj)/N
end
function residue(f::Function, c; r=0.1, N=200)
    if c == Inf
        fg(z) = -1/z^2 * f(1/z)
        return residue(fg, 0; r, N)
    end
    γ(t) = r * exp(im * t) + c
    dγ(t) = r * im * exp(im * t)
    integrate(f, γ, dγ; N) / (2π * im)
end