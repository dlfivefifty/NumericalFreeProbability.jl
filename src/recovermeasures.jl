export freeaddition, recovermeasure_sqrt, pointcloud_sqrt, support_sqrt_single, ⊞,
                NumberOrVectorNumber, pairsum, prunepoints_multivalued

const NumberOrVectorNumber = Union{Number, AbstractVector{T}} where T<:Number

unitcirclenodes(T, n) = [cispi(convert(T, 2k)/n-1) for k=0:n-1]

"""
Compute the point cloud required to approximate the Cauchy transform G_a⊞b, where the output measure is square root decaying
and has support on a compact interval.

Parameters:
    G_a, G_b, InvG_b: Function
        - Respectively, the Cauchy transform of μ_a, μ_b and the inverse Cauchy transform of μ_b.

    supp_c: Tuple{Real}
        - Indicates the support of the convolution measure.

    m: Int
        - Controls the number of points that are used to sample. In general, increasing m will lead to more sampling points.
        - However, if m is too large then the expansion in terms of Chebyshev U polynomials will lead to large errors in the final output measure.
"""
function pointcloud_sqrt(G_a::Function, G_b::Function, supp_c::Vector{Tuple{T,T}}, InvG_b::Function; m = 10) where T<:Real
    d_M = vec(unitcirclenodes(Float64, m)*[x for x in ChebyshevGrid{1}(2m+1)[1:m] if x > eps()]') # temporary Float64
    z_μ_M = [M_ab(J(x), supp_c[1][1], supp_c[1][2]) for x in d_M if imag(x) >= eps()]
    y_M = G_a(z_μ_M)
    invcache = InvG_b(y_M)
    y_M = [y for (i,y) in enumerate(y_M) if length(invcache[i]) > 0 && isapprox(G_b(invcache[i][1]), y)] # filter out points which have no inverse
end


# function pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m = 10)
#     d_M = vec(unitcirclenodes(Float64, m)*[x for x in ChebyshevGrid{1}(2m+1)[1:m] if x > eps()]') # temporary Float64
#     filter!(x::Complex -> imag(x) ≥ eps(), d_M)
#     map!(x::Complex -> M_ab(J(x), supp_c[1], supp_c[2]), d_M, d_M)
#     d_M = G_a(d_M)
#     filter!(y::Complex -> length(InvG_b(y)) > 0, d_M) # filter out points which have no inverse
#     filter!(y::Complex -> isapprox(G_b(InvG_b(y)[1]), y), d_M) # filter out points which are not their own inverses.
#     d_M
# end

"""
Prune the points used to evaluate the inverse Cauchy transform of G_a⊞b.

Parameters:
    points: Array{Complex}
        - Point cloud created from the function pointcloud_sqrt() (May be used for other measures later...)
    
    InvG_a, InvG_b: Function
        - Respectively, the inverse Cauchy transform of μ_a and μ_b.
"""
function prunepoints_univalent!(points, InvG_a, InvG_b)
    InvG_c = y -> InvG_a(y)[1] .+ InvG_b(y)[1] .- 1 ./ y
    filter!(y::Complex -> length(InvG_c(y)) == 1, points)
    filter!(y::Complex -> sign(imag(y)) != sign(imag(InvG_c(y))), points)
end # TODO: this function is kind of universal for any convolution, not just sqrt. Move to somewhere else later?

function bisection(f::Function, x_l::Real, x_r::Real; tol=10^-12, maxits = 40)
    y_l = f(x_l)
    if !(y_l > 0) ⊻ (f(x_r) > 0)
        return nothing # cauchy transform is monotone
    end
    for _=1:maxits
        x_m = (x_l + x_r)/2
        y_m = f(x_m)
        if abs(y_m) < tol
            return x_m
        end
        if (y_m > 0) ⊻ (y_l > 0)
            x_r = x_m
        else
            x_l = x_m
            y_l = y_m
        end
    end
end

function prunepoints_multivalued(points::Vector{T}, InvG_a::Function, InvG_b::Function) where T<:Number
    InvG_c = z::Union{T, Vector{T}} -> combine_invcauchytransform(z, InvG_a, InvG_b)
    preimages = Vector{T}(); images = Vector{T}()
    inv_cache = InvG_c(points)
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

"""
Compute the support of the additive convolution of two measures.
This is done using a Bisection method.

Parameters:
    G_a, G_b, InvG_a, InvG_b, dG_a, dG_b: Function
        - Respectively, the Cauchy transform of μ_a, μ_b, the inverse Cauchy transform of μ_a and μ_b and the
          derivatives of the Cauchy transform ofs μ_a and μ_b.

    m_a, m_b: ChebyshevUMeasure

    tol: Float
            - Tolerance for bisection.

    maxits: Int
            - Maximum number of iterations for bisection.
"""
function support_sqrt_single(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a, m_b; tol=10^-13, maxits=60)
    a_0 = max(G_a(m_a.a), G_b(m_b.a))
    b_0 = min(G_a(m_a.b), G_b(m_b.b))
    dInvG_a = z::Number -> inv.(dG_a(InvG_a(z)[1]))
    dInvG_b = z::Number -> inv.(dG_b(InvG_b(z)[1]))
    dInvG_c = z::Number -> dInvG_a(z) .+ dInvG_b(z) .+ (1/z^2)
    dInvG_c_real = z::Real -> iszero(z) ? -Inf : real.(dInvG_c(z))
    ξ_a = bisection(dInvG_c_real, a_0, 0; tol, maxits)
    ξ_b = bisection(dInvG_c_real, 0, b_0; tol, maxits)
    InvG_c = y -> InvG_a(y) .+ InvG_b(y) .- 1/y
    [(real(InvG_c(ξ_a)[1]), real(InvG_c(ξ_b)[1]))]
end






"""
Recover the coeffiencients of the Chebyshev U expansion of a square root decaying measure.
This is done using a Least-squares method.

Parameters:
    InvG_a, InvG_b: Function
        - Respectively, the inverse Cauchy transform of μ_a and μ_b.
    
    supp_c: Tuple{Real}
        - Support of the measure of the convolution μ_c. Can be calculated from support_sqrt_single()

    y_m: Array{Complex}
        - Sampling points for solving the Least-squares problem.
"""
# function recovermeasure_sqrt(InvG_a, InvG_b, supp_c, y_m)
#     InvG_c = z -> combine_invcauchytransform(z, InvG_a, InvG_b)
#     n = length(y_m)
#     inv_cache = InvG_c(y_m)
#     display(inv_cache)

#     A = [Jinv_p(M_ab_inv(inv_cache[j],supp_c[1],supp_c[2]))^k for j=1:n, k=1:n]
#     V = [real.(A);imag.(A)]
#     f = [real.(y_m);imag.(y_m)]
#     Q, R̂ = qr(V)
#     Q̂ = Q[:,1:n]
#     R̂ \ Q̂'f / π
# end

function recovermeasure_sqrt(supp_c::Vector{Tuple{T2,T2}}, preimages::Vector{T}, images::Vector{T}) where {T<:Number, T2<:Real}
    n = length(images)
    A = zeros(Complex, n, n)
    for (i, s) in enumerate(supp_c)
        A[:,n*(i-1)+1:n*i] = [Jinv_p(M_ab_inv(preimages[j],s[1],s[2]))^k for j=1:n, k=1:n]
    end
    V = [real.(A);imag.(A)]
    f = [real.(images);imag.(images)]
    Q, R̂ = qr(V)
    Q̂ = Q[:,1:n]
    R̂ \ Q̂'f ./ pi
end

function combine_invcauchytransform(y::Number, InvG_a::Function, InvG_b::Function)
    ans = pairsum(InvG_a(y), InvG_b(y)) .- inv(y)
    filter!(z -> sign(imag(y)) != sign(imag(z)), ans)
end

function combine_invcauchytransform(y::Vector{T}, InvG_a::Function, InvG_b::Function) where T<:Number
    invg_a = InvG_a(y); invg_b = InvG_b(y)
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


"""
Compute the Free additive convolution of two square root decaying measures μ_a and μ_b, where 
each measure is individually supported on a compact interval.

Parameters:
    m_a, m_b: ChebyshevUMeasure

    m: Int
        - Controls the number of points that are used to sample. In general, increasing m will lead to more sampling points.
        - However, if m is too large then the expansion in terms of Chebyshev U polynomials will lead to large errors in the final output measure.
    
    maxterms: Int
        - Max number of terms ψ_a_k, ψ_b_k used in expansion of ψ(x) in Chebyshev U polynomials.

     tol: Float
        - How close eigenvalues of the companion matrix need to be to the origin in order to count as a valid inverse Cauchy transform.

    tolbisect: Float
        - Tolerance for bisection.

    maxitsbisect: Int
        - Maximum number of iterations for bisection.
    

returns:
    ChebyshevUMeasure
"""
function freeaddition(m_a::ChebyshevUMeasure, m_b::ChebyshevUMeasure; m=10, maxterms=100, tolbisect = 10^-9, maxitsbisect=40, tol=10^-15)
    G_a = z::NumberOrVectorNumber -> cauchytransform(z, m_a)
    G_b = z::NumberOrVectorNumber -> cauchytransform(z, m_b)
    InvG_a = z::NumberOrVectorNumber -> invcauchytransform(z, m_a; maxterms, tol)
    InvG_b = z::NumberOrVectorNumber -> invcauchytransform(z, m_b; maxterms, tol)
    dG_a = z::NumberOrVectorNumber -> isapprox(z, m_a.a) || isapprox(z, m_a.b) ? real(dcauchytransform(z+eps()^2 *im, m_a)) : dcauchytransform(z, m_a)
    dG_b = z::NumberOrVectorNumber -> isapprox(z, m_b.a) || isapprox(z, m_b.b) ? real(dcauchytransform(z+eps()^2 *im, m_b)) : dcauchytransform(z, m_b)
    
    supp_c = support_sqrt_single(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a, m_b; tol=tolbisect, maxits=maxitsbisect)
    y_M = pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m)
    preimages, images = prunepoints_multivalued(y_M, InvG_a, InvG_b)
    ψ_c_k = recovermeasure_sqrt(supp_c, preimages, images)
    ChebyshevUMeasure(supp_c[1][1], supp_c[1][2], vcat(ψ_c_k, zeros(∞)))
end

⊞(m_a::Measure, m_b::Measure) = freeaddition(m_a::Measure, m_b::Measure)


function recovermeasure_multiplysupportedsqrt(supp_c, preimages, images, N=20)
    n = length(images)
    A = Complex.(zeros(n, N*length(supp_c))) # lazy convert to complex
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




function findallroots(f, x_l, x_r; tol=10^-6, maxits = 40, step=0.01)
    points = []
    sign_previous = 2
    sgn_p = x -> x > 0
    for x = x_l:step:x_r
        if sign_previous != sgn_p(f(x))
            sign_previous = sgn_p(f(x))
            push!(points, x)
        end
    end
    roots = []
    n = length(points) - 1
    for i=1:n
        push!(roots, bisection(f, points[i], points[i+1]; tol, maxits))
    end
    roots
end

# a is the square root measure, b is the point measure
function support_multiplysupportedsqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a::Measure, m_b::Measure; tol=10^-6, maxits=30)
    dInvG_a = z -> 1 ./ (dG_a.(InvG_a(z)))
    dInvG_b = z -> 1 ./ (dG_b.(InvG_b(z)))
    InvG_c = z -> InvG_a(z) .+ InvG_b(z) .- 1/Complex(z)
    dInvG_c = z -> dInvG_a(z) .+ dInvG_b(z) .+ 1/Complex(z)^2 # TODO: subject to catastrophic cancellation when z is close to 0 in multi cut case...
    dInvG_c_real = z -> real.(dInvG_c(z))

    number_of_atoms = length(m_b.a)

    a_0 = real(G_a(minimum(axes(m_a))))
    b_0 = real(G_a(maximum(axes(m_a))))
    support_points = []

    for i=1:number_of_atoms
        if i == 1
            test = z -> begin
                a = dInvG_c_real(z)[1]
                if isnan(a)
                    return -1/z^2
                end
                a
            end
        else
            test = z -> dInvG_c_real(z)[i]
        end
        s=10^-3
        roots = findallroots(test, a_0, -s; tol, maxits)
        for k in roots
            if k !== nothing
                push!(support_points, InvG_c(k)[i])
            end
        end
        roots = findallroots(test, s, b_0; tol, maxits)
        for k in roots
            if k !== nothing
                push!(support_points, InvG_c(k)[i])
            end
        end
    end
    support_points = sort(real.(support_points))
    [(support_points[2*i-1], support_points[2*i]) for i=1:length(support_points)÷2]
end


# some performance issues

function freeaddition(m_a::OPMeasure, m_b::PointMeasure; m=40, tolbisect = 10^-6, maxitsbisect=30, N=20)
    G_a = z -> cauchytransform(z, m_a)
    G_b = z -> cauchytransform(z, m_b)
    InvG_a = z -> invcauchytransform(z, m_a)
    InvG_b = z -> invcauchytransform(z, m_b)
    dG_a = z -> dcauchytransform(z, m_a)
    dG_b = z -> dcauchytransform(z, m_b)
    supp_c = support_multiplysupportedsqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a, m_b; tol=tolbisect, maxits=maxitsbisect)

    y_M = pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m)
    preimages, images = prunepoints_multivalued(y_M, InvG_a, InvG_b)
    ψ_c_k_i = recovermeasure_multiplysupportedsqrt(supp_c, preimages, images, N)
    
    SumOPMeasure([ChebyshevUMeasure(supp_c[i][1], supp_c[i][2], vcat(ψ_c_k_i[i], zeros(∞))) for i in eachindex(supp_c)])
end