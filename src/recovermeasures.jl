export freeaddition, recovermeasure_sqrt, pointcloud_sqrt, prunepoints, support_sqrt

unitcirclenodes(T, n) = [exp(π * (convert(T, 2k)/n-1)im) for k=0:n-1]

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
function pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m = 10)
    d_M = vec(unitcirclenodes(Float64, m)*[x for x in ChebyshevGrid{1}(2m+1)[1:m] if x > eps()]') # temporary Float64
    z_μ_M = [M_ab(J(x), supp_c[1], supp_c[2]) for x in d_M if imag(x) >= eps()]

    y_M = G_a.(z_μ_M)

    y_M = [y for y in y_M if length(InvG_b(y)) == 1] # filter out points which have multivalued inverses
    y_M = [y for y in y_M if isapprox(G_b(InvG_b(y)), y)] # filter out points which are not their own inverses.
    # TODO: Check if the last line is necessary, in all the cases I tested, it's not needed.
end


"""
Prune the points used to evaluate the inverse Cauchy transform of G_a⊞b.

Parameters:
    points: Array{Complex}
        - Point cloud created from the function pointcloud_sqrt() (May be used for other measures later...)
    
    InvG_a, InvG_b: Function
        - Respectively, the inverse Cauchy transform of μ_a and μ_b.
"""
function prunepoints(points, InvG_a, InvG_b)
    InvG_c = y -> InvG_a(y) .+ InvG_b(y) .- 1 ./ y
    y_m = [y for y in points if length(InvG_c(y)) == 1]
    [y for y in y_m if sign(imag(y)) != sign(imag(InvG_c(y)))]
end # TODO: this function is kind of universal for any convolution, not just sqrt. Move to somewhere else later?

function bisection(f, x_l, x_r; tol=10^-6, maxits = 40)
    y_l = f(x_l)
    if !(y_l > 0) ⊻ (f(x_r) > 0)
        return nothing # cauchy transform is monotone
    end
    for i=1:maxits
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
function support_sqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a, m_b; tol=10^-9, maxits=40)
    a_0 = max(G_a(m_a.a), G_b(m_b.a))
    b_0 = min(G_a(m_a.b), G_b(m_b.b))
    dInvG_a = z -> 1 ./ (dG_a.(InvG_a(z)[1]))
    dInvG_b = z -> 1 ./ (dG_b.(InvG_b(z)[1])) # TODO: temporary
    dInvG_c = z -> dInvG_a(z) .+ dInvG_b(z) .+ 1/z^2
    dInvG_c_real(z::Real) = iszero(z) ? -Inf : real.(dInvG_c(Complex(z)))

    ξ_a = bisection(dInvG_c_real, a_0, 0; tol, maxits)
    ξ_b = bisection(dInvG_c_real, 0, b_0; tol, maxits)
    InvG_c = y -> InvG_a(y) .+ InvG_b(y) .- 1/y
    (real(InvG_c(ξ_a)), real(InvG_c(ξ_b))) # TODO: temporary
end


   




"""
Recover the coeffiencients of the Chebyshev U expansion of a square root decaying measure.
This is done using a Least-squares method.

Parameters:
    InvG_a, InvG_b: Function
        - Respectively, the inverse Cauchy transform of μ_a and μ_b.
    
    supp_c: Tuple{Real}
        - Support of the measure of the convolution μ_c. Can be calculated from support_sqrt()

    y_m: Array{Complex}
        - Sampling points for solving the Least-squares problem.
"""
function recovermeasure_sqrt(InvG_a, InvG_b, supp_c, y_m)
    InvG_c = z -> InvG_a(z) .+ InvG_b(z) .- 1/z;
    n = length(y_m)
    A = [Jinv_p(M_ab_inv(InvG_c(y_m[j]),supp_c[1],supp_c[2]))^k for j=1:n, k=1:n]
    V = [real.(A);imag.(A)]
    f = [real.(y_m);imag.(y_m)]
    Q, R̂ = qr(V)
    Q̂ = Q[:,1:n]
    R̂ \ Q̂'f
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
    G_a = z -> cauchytransform(z, m_a)
    G_b = z -> cauchytransform(z, m_b)
    InvG_a = z -> invcauchytransform(z, m_a; maxterms, tol)[1]
    InvG_b = z -> invcauchytransform(z, m_b; maxterms, tol)[1]
    dG_a = z -> isapprox(z, m_a.a) || isapprox(z, m_a.b) ? 0 : dcauchytransform(z, m_a)
    dG_b = z -> isapprox(z, m_b.a) || isapprox(z, m_b.b) ? 0 : dcauchytransform(z, m_b)
    
    supp_c = support_sqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, m_a, m_b; tol=tolbisect, maxits=maxitsbisect)
    y_M = pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m)
    y_m = prunepoints(y_M, InvG_a, InvG_b)
    ψ_c_k = recovermeasure_sqrt(InvG_a, InvG_b, supp_c, y_m)

    ChebyshevUMeasure(supp_c[1], supp_c[2], vcat(ψ_c_k * (supp_c[2] - supp_c[1])/4, zeros(∞)))
end

