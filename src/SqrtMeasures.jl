export cauchytransform_sqrt, invcauchytransform_sqrt, dcauchytransform_sqrt, d2cauchytransform_sqrt
export pointcloud_sqrt, prunepoints
export support_sqrt, recovermeasure_sqrt, freeaddition_sqrt

# TODO: turn the measures into special types

# Joukowski transform
Jinv_p = z -> z - √(z - 1) * √(z + 1)
J = w -> 1/2 * (w + 1/w)
function dJinv_p(z)
    if z == 1 || z == -1
        return Inf * -1 * z
    end
    w = √(z + 1)/√(z - 1)
    1 - w/2 - 1/(2*w)
end

function d2Jinv_p(z)
    if z == 1 || z == -1
        return Inf * z
    end
    w = √(z - 1)/√(z + 1)
    -1/4 * ((w - 1/w)/(z-1) + (1/w - w)/(z+1))
end

# Affine maps to transform support
M_ab = (x,a,b) -> (a + b)/2 + (b - a) * x /2 # maps from (-1, 1) to (a, b)
M_ab_inv = (y,a,b) -> (2*y - (a + b))/(b - a) # maps from (a, b) to (-1, 1)
dM_ab_inv = (y,a,b) -> 2/(b - a)

# function to generate chebyshev roots, transformed to any interval
function chebyshevnodes(a,b,n)
    [(a+b)/2 + (b-a)/2 * cos((2k-1)/2n * pi) for k = 1:n]
end

# function to generate points on unit circle in complex plane
function unitcirclenodes(n)
    [exp(pi * (-1 + 2/n * k)im) for k=0:n-1]
end



"""
Compute the Cauchy transform G_μ of a square root decaying measure supported on one compact interval.

Parameters:
    z: Complex
        - Point at which G_μ is evaluated at.

    ψ_k: 1 x n array, n ∈ ℕ ∪ {∞}
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials.

    a, b: Real, a < b
        - Indicates the support of the measure.

    maxterms: Int
        - Max number of terms ψ_k used in expansion of ψ(x) in Chebyshev U polynomials.
Returns:
    G_μ(z): Complex
        - Evaluation of Cauchy Transform.
"""
function cauchytransform_sqrt(z, ψ_k, a, b; maxterms=20)
    ans = 0
    w_0 = Jinv_p(M_ab_inv(z, a, b))
    w = w_0
    n = min(maxterms, length(ψ_k))
    for k=1:n
        ans += ψ_k[k] * w
        w *= w_0
    end
    pi * ans
end


"""
Compute the Inverse Cauchy transform G_μ of a square root decaying measure supported on one compact interval.
This is done using a Companion Matrix method.

Parameters:
    z: Complex
        - Point at which G_μ is evaluated at.

    ψ_k: 1 x n array, n ∈ ℕ ∪ {∞}
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials.

    a, b: Real, a < b
        - Indicates the support of the measure.

    maxterms: Int
        - Max number of terms ψ_k used in expansion of ψ(x).

    tol: Float
        - How close eigenvalues need to be to the origin in order to count as a valid inverse to the polynomial problem.
        - To be honest, I don't know if this actually helps or not. Setting it just above 1 seems to be right.
        - If it is equal to 1, it sometimes misses solutions due to floating point errors.
Returns:
    G_μ⁻¹(z): Complex, if single valued.
            : Array{Complex}, if multi valued.
        - Evaluation of Inverse Cauchy Transform.
"""
function invcauchytransform_sqrt(z, ψ_k, a, b; maxterms=20, tol=1+10^-6)
    # find largest index i which ψ_k[i] is non-zero
    P = ψ_k[1:min(length(ψ_k),maxterms)]
    n = min(length(ψ_k),maxterms)
    while abs(ψ_k[n]) < 10^-9
        n -= 1
    end
    P ./= P[n] # make monic
    C = zeros(Complex, n, n)
    for i=1:n-1
        C[i+1,i] = 1
        C[i+1,n] = -P[i]
    end
    C[1, n] = z/(pi * ψ_k[n])
    s = eigvals(C)
    if length(s) > 1
        s = [z for z in s if abs(z) < tol] # select the eigenvalues whos abs() < 1
    end

    if length(s) == 1
        return M_ab.(J.(s), a, b)[1]
    elseif length(s) == 0
        error("Input z outside domain of G_μ⁻¹.")
    else
        return M_ab.(J.(s), a, b)
    end
    # is this bad practice? it returns a complex if single and a vector if not single.
end



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
    d_M = vec(unitcirclenodes(m)*[x for x in chebyshevnodes(-1,1,2*m+1) if x > eps()]')
    z_μ_M = [M_ab(J(x), supp_c[1], supp_c[2]) for x in d_M if imag(x) >= eps()]
    y_M = [G_a(y) for y in z_μ_M]
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
    for i=1:maxits
        x_m = (x_l + x_r)/2
        if abs(f(x_m)) < tol
            return x_m
        end
        if (f(x_m) > 0) ⊻ (f(x_l) > 0)
            x_r = x_m 
        else
            x_l = x_m
        end
    end
    error("failed to converge in maxits")
end


"""
Compute the derivative of the Cauchy transform G_μ of a square root decaying measure supported on a compact interval.

Parameters:
    z: Complex
        - Point at which G_μ' is evaluated at.

    ψ_k: 1 x n array, n ∈ ℕ ∪ {∞}
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials.

    a, b: Real, a < b
        - Indicates the support of the measure.

    maxterms: Int
        - Max number of terms ψ_k used in expansion of ψ(x).
"""
function dcauchytransform_sqrt(z, ψ_k, a, b; maxterms=20)
    ans = 0
    w_0 = Jinv_p(M_ab_inv(z, a, b))
    w = 1
    for k=1:min(maxterms, length(ψ_k))
        ans += ψ_k[k] * w * k * dJinv_p(M_ab_inv(z, a, b)) * dM_ab_inv(z, a, b)
        w *= w_0
    end
    pi * ans
end


# NO UNIT TESTING YET FOR d2cauchytransform_sqrt
"""
Compute the second derivative of the Cauchy transform G_μ of a square root decaying measure supported on a compact interval.

Parameters:
    z: Complex
        - Point at which G_μ'' is evaluated at.

    ψ_k: 1 x n array, n ∈ ℕ ∪ {∞}
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials.

    a, b: Real, a < b
        - Indicates the support of the measure.

    maxterms: Int
        - Max number of terms ψ_k used in expansion of ψ(x).
"""
function d2cauchytransform_sqrt(z, ψ_k, a, b; maxterms=20)
    ans = 0
    w_0 = Jinv_p(M_ab_inv(z, a, b))
    w = 1
    for k=1:min(maxterms, length(ψ_k))
        dw = dJinv_p(M_ab_inv(z, a, b)) * dM_ab_inv(z, a, b)
        d2w = d2Jinv_p(M_ab_inv(z, a, b)) * dM_ab_inv(z, a, b)^2
        ans += ψ_k[k] * k * ((k-1) * w/w_0 * dw^2 + w*d2w)
        w *= w_0
    end
    pi * ans
end


"""
Compute the support of the additive convolution of two measures.
This is done using a Bisection method.

Parameters:
    G_a, G_b, InvG_a, InvG_b, dG_a, dG_b: Function
        - Respectively, the Cauchy transform of μ_a, μ_b, the inverse Cauchy transform of μ_a and μ_b and the
          derivatives of the Cauchy transform ofs μ_a and μ_b.

supp_a, supp_b: Tuple{Real}
        - Pairs of Reals representing the supports of μ_a and μ_b.

tol: Float
        - Tolerance for bisection.

maxits: Int
        - Maximum number of iterations for bisection.
"""
function support_sqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, supp_a, supp_b; tol=10^-6, maxits=40)
    a_0 = max(real(G_a(Complex(supp_a[1]))), real(G_b(Complex(supp_b[1]))))
    b_0 = min(real(G_a(Complex(supp_a[2]))), real(G_b(Complex(supp_b[2]))))  
    dInvG_a = z -> 1 / (dG_a(InvG_a(z)))
    dInvG_b = z -> 1 / (dG_b(InvG_b(z)))
    dInvG_c = z -> dInvG_a(z) + dInvG_b(z) + 1/z^2
    dInvG_c_real = z -> iszero(z) ? -Inf : real(dInvG_c(Complex(z)))
    ξ_a = bisection(dInvG_c_real, a_0, 0; tol, maxits)
    ξ_b = bisection(dInvG_c_real, 0, b_0; tol, maxits)
    InvG_c = y -> InvG_a(y) + InvG_b(y) - 1/y
    (real(InvG_c(ξ_a)), real(InvG_c(ξ_b)))
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
    InvG_c = z -> InvG_a(z) .+ InvG_b(z) - (1/z);
    n = length(y_m)
    A = [Jinv_p(M_ab_inv(InvG_c(y_m[j]),supp_c[1],supp_c[2]))^k for j=1:n, k=1:n]
    V = [real.(A);imag.(A)]
    f = [real.(y_m);imag.(y_m)]
    Q, R̂ = qr(V)
    Q̂ = Q[:,1:n]
    R̂ \ Q̂'f ./ pi
end


"""
Compute the Free additive convolution of two square root decaying measures μ_a and μ_b, where 
each measure is individually supported on a compact interval.

Parameters:
    ψ_a_k, ψ_a_k: 1 x n array, n ∈ ℕ ∪ {∞}
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials for measures μ_a and μ_b.

    supp_a, supp_b: Tuple{Real}
        - Indicates the supports of the measures.

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
    

Returns
    ψ_a_k: 1 x n array, n ∈ ℕ
        - Coefficients of expansion of ψ(x) in terms of Chebyshev U polynomials for output measure.
    supp_c: Tuple{Real}
        - Support of the output measure.
"""
function freeaddition_sqrt(ψ_a_k, ψ_b_k, supp_a, supp_b; m=10, maxterms=20, tolcomp=1+10^-6, tolbisect = 10^-6, maxitsbisect=40)
    G_a = z -> cauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms)
    G_b = z -> cauchytransform_sqrt(Complex(z), ψ_b_k, supp_b[1], supp_b[2]; maxterms)
    InvG_a = z -> invcauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms, tol=tolcomp)
    InvG_b = z -> invcauchytransform_sqrt(Complex(z), ψ_b_k, supp_b[1], supp_b[2]; maxterms, tol=tolcomp)
    dG_a = z -> isapprox(z, supp_a[1]) || isapprox(z, supp_a[1]) ? 0 : dcauchytransform_sqrt(Complex(z), ψ_a_k, supp_a[1], supp_a[2]; maxterms)
    dG_b = z -> isapprox(z, supp_b[1]) || isapprox(z, supp_b[1]) ? 0 : dcauchytransform_sqrt(Complex(z), ψ_b_k, supp_b[1], supp_b[2]; maxterms)
    supp_c = support_sqrt(G_a, G_b, InvG_a, InvG_b, dG_a, dG_b, supp_a, supp_b; tol=tolbisect, maxits=maxitsbisect)
    y_M = pointcloud_sqrt(G_a, G_b, supp_c, InvG_b; m)
    y_m = prunepoints(y_M, InvG_a, InvG_b)
    ψ_c_k = recovermeasure_sqrt(InvG_a, InvG_b, supp_c, y_m)
    ψ_c_k, supp_c
end