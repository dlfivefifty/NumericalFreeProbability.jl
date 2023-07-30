export PointMeasure
export cauchytransform_point, dcauchytransform_point, d2cauchytransform_point, invcauchytransform_point, support_sqrt_point
export pointcloud_sqrt_point, prunepoints_multivalued
export support_sqrt_point, recovermeasure_multiplysupportedsqrt, freeaddition_sqrt_point


using FFTW, SpecialMatrices

# Mobius transformation
mobius(x, a, b, c, d) = (a*x+b)/(c*x+d)


# Conformal map from unit disk to ℂ \ ((-∞,-1] U [1, ∞))
H(z) = mobius(mobius(z, 1, 1, -1, 1)^2, 1, -1, 1, 1)


"""
PointMeasure type.
Parameters:
    λ: Vector{Real}
        - Vector of distinct reals representing the locations of the atoms of the measure.
    a: Vector{Real}
        - Vector of reals representing the measures at the atoms. Must sum to 1.
"""
struct PointMeasure{T1<:Real, T2<:Real}
    λ::Vector{T1}
    a::Vector{T2}
    function PointMeasure(λ::Vector{T1}, a::Vector{T2}) where {T1<:Real, T2<:Real}
        if length(λ) != length(a)
            throw(ArgumentError("Arrays must have the same size."))
        elseif !isapprox(sum(a), 1)
            throw(ArgumentError("Point measure must have total measure 1."))
        end
        new{T1, T2}(λ, a)
    end
end


"""
Compute the Cauchy transform G_μ of a point measure with finitely many atoms.

Parameters:
    z: Complex
        - Point at which G_μ is evaluated at.
    M: PointMeasure
        - PointMeasure type.
Returns:
    G_μ(z): Complex
        - Evaluation of Cauchy Transform.
"""
function cauchytransform_point(z, pm::PointMeasure)
    n = length(pm.a)
    ans = 0
    for i=1:n
        ans += pm.a[i]/(z-pm.λ[i])
    end
    ans
end

function dcauchytransform_point(z, pm::PointMeasure)
    n = length(pm.a)
    ans = 0
    for i=1:n
        ans -= pm.a[i]/(z-pm.λ[i])^2
    end
    ans
end

function d2cauchytransform_point(z, pm::PointMeasure)
    n = length(pm.a)
    ans = 0
    for i=1:n
        ans += pm.a[i]/(z-pm.λ[i])^3
    end
    2 * ans
end




function invcauchytransform_point(z, M)
    n = length(M.a)
    P = -z .* productlinearfactors(M.λ)
    for i=1:n
        P += [M.a[i] * productlinearfactors([M.λ[1:i-1]; M.λ[i+1:end]]);0]
    end
    if z == 0
        P = P[1:end-1]
    end
    ans = invertpolynomial(P, 0)
    if z == 0
        ans = [Inf;ans]
    elseif isa(z, Real) && (z > 0)
        ans = [ans[end];ans[1:end-1]]
    end
    ans
end

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


# function invcauchytransform_point(z, pm::PointMeasure; maxterms = 200, tol = 1, r=0.9, n = 200)
#     a, b = minimum(pm.λ), maximum(pm.λ)
#     g = θ -> cauchytransform_point(M_ab(J(r * exp(im * θ)), a, b), pm)
#     θ = range(0, 2π; length=2*n+2)[1:end-1]
#     fc = FFTW.fft(g.(θ))/(2*n+1)
#     g_k = fc[1:n+1]

#     n = min(n,maxterms)
#     P = g_k[1:n]
#     while abs(g_k[n]) < 10^-9
#         n -= 1
#     end

#     s = invertpolynomial(P[1:n], z)
#     s = [z for z in s if abs(z) < tol]
    
#     x = M_ab.(J.(r .* s), a, b)
    
#     number_of_atoms = length(pm.a)
#     for i=1:number_of_atoms-1
#         if length(x) >= number_of_atoms
#             return x
#         end
#         for i in invcauchytransform_point_real2(z, pm, pm.λ[i], pm.λ[i+1]; maxterms, tol, r, n)
#             if !any(isapprox.(x,i))
#                 push!(x, i)
#             end
#         end
#     end
#     x
# end

# # not exported
# function invcauchytransform_point_real2(z, pm::PointMeasure, a, b; maxterms=200, tol=1, r=0.9, n=200)
#     g = θ -> cauchytransform_point(M_ab(H(r * exp(im * θ)), a, b), pm)
#     θ = range(0, 2π; length=2*n+2)[1:end-1]
#     fc = FFTW.fft(g.(θ))/(2*n+1)
#     g_k = fc[1:n+1]
#     n = min(n,maxterms)
#     P = g_k[1:n]
#     while abs(g_k[n]) < 10^-9
#         n -= 1
#     end
#     s = invertpolynomial(P[1:n], z)
#     s = [z for z in s if abs(z) < tol]
#     if length(s) == 1
#         return M_ab.(H.(r .* s), a, b)[1]
#     else
#         return M_ab.(H.(r .* s), a, b)
#     end
# end

function invertpolynomial(P, z)
    P1 = P[1:end-1]/P[end]
    P1[1] -= z/P[end]
    eigvals(SpecialMatrices.Companion(P1))
end


function findallroots(f, x_l, x_r; tol=10^-6, maxits = 40, step=0.001)
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
function support_sqrt_point(G_a, InvG_a, InvG_b, dG_a, dG_b, supp_a, pm_b::PointMeasure; tol=10^-6, maxits=30)
    dInvG_a = z -> 1 ./ (dG_a.(InvG_a(z)))
    dInvG_b = z -> 1 ./ (dG_b.(InvG_b(z)))
    InvG_c = z -> InvG_a(z) .+ InvG_b(z) .- 1/Complex(z)
    dInvG_c = z -> dInvG_a(z) .+ dInvG_b(z) .+ 1/Complex(z)^2
    dInvG_c_real = z -> real.(dInvG_c(Complex(z)))

    number_of_atoms = length(pm_b.a)
    
    a_0 = real(G_a(supp_a[1]))
    b_0 = real(G_a(supp_a[2]))
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
        roots = findallroots(test, a_0, -0.0; tol, maxits)
        for k in roots
            if k !== nothing
                push!(support_points, InvG_c(k)[i])
            end
        end
        roots = findallroots(test, 0.0, b_0; tol, maxits)
        for k in roots
            if k !== nothing
                push!(support_points, InvG_c(k)[i])
            end
        end
    end
    support_points = sort(real.(support_points))
    [(support_points[2*i-1], support_points[2*i]) for i=1:length(support_points)÷2]
end

"""
Returns a pair of vectors that form the point cloud:
    preimages: Vector{Complex}
        - Vector of sample points.
    images: Vector{Complex}
        - The cauchy transforms of the sample points.
"""


function pointcloud_sqrt_point(G_a, supp_c, InvG_b; m = 10)
    d_M = vec(unitcirclenodes(Float64, m)*[x for x in ChebyshevGrid{1}(2m+1)[1:m] if x > eps()]') # temporary Float64
    z_μ_M = [M_ab(J(x), supp_c[1][1], supp_c[end][2]) for x in d_M if imag(x) >= eps()]
    y_M = G_a.(z_μ_M)
    y_M = [y for y in y_M if length(InvG_b(y)) != 0] # all points which are in the image of G_b
end


function prunepoints_multivalued(points, InvG_a, InvG_b)
    InvG_c = z -> InvG_a(z) .+ InvG_b(z) .- 1/Complex(z)
    preimages = Vector{ComplexF64}()
    images = Vector{ComplexF64}()
    for y in points
        inv_y = InvG_c(y)
        for x in inv_y
            if sign(imag(x)) != sign(imag(y))
                push!(preimages, x)
                push!(images, y)
            end
        end
    end
    preimages, images
end

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

    ψ_c_k_i = [sol[(i-1)*N+1:i*N] for i=1:length(supp_c)]
end

# some performance issues

function freeaddition_sqrt_point(ψ_a_k, supp_a, pm_b::PointMeasure; m=40, maxterms=20, tolcomp=1, tolbisect = 10^-6, maxitsbisect=30, N=20)
    G_a = z -> cauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms)
    #G_b = z -> cauchytransform_point(Complex(z), pm_b)
    InvG_a = z -> invcauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms, tol=tolcomp)
    InvG_b = z -> invcauchytransform_point(Complex(z), pm_b)
    dG_a = z -> dcauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms)
    dG_b = z -> dcauchytransform_point(Complex(z), pm_b)
    supp_c = support_sqrt_point(G_a, InvG_a, InvG_b, dG_a, dG_b, supp_a, pm_b; tol=tolbisect, maxits=maxitsbisect)
    y_M = pointcloud_sqrt_point(G_a, supp_c, InvG_b; m)
    preimages, images = prunepoints_multivalued(y_M, InvG_a, InvG_b)
    ψ_c_k_i = recovermeasure_multiplysupportedsqrt(supp_c, preimages, images, N)
    ψ_c_k_i, supp_c
end



