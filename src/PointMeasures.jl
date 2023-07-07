export PointMeasure
export cauchytransform_point, invcauchytransform_point

using FFTW

# Joukowski transform
function J(z)
    1/2 * (z + 1/z)
end

# Affine maps to transform support
function M_ab(x,a,b)
    (a + b)/2 + (b - a) * x /2 # maps from (-1, 1) to (a, b)
end

# Mobius transformation
function mobius(x, a, b, c, d)
    (a*x+b)/(c*x+d)
end

# Conformal map from unit disk to ℂ \ ((-∞,-1] U [1, ∞))
function H(z)
    mobius(mobius(z, 1, 1, -1, 1)^2, 1, -1, 1, 1)
end

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

"""
Compute the inverse Cauchy transform G_μ of a point measure with finitely many atoms.
This is done by using a conformal map to a unit disk and applying the Fourier transform, followed by polynomial inversion.
Note that the inverse Cauchy transform here is multivalued unless the point measure consists of only one atom.

Parameters:
    z: Complex
        - Point at which G_μ⁻¹ is evaluated at.
    M: PointMeasure
        - PointMeasure type.
    maxterms: Int
        - Max number of terms used in fourier transform.
    tol: Float
        - How close eigenvalues need to be to the origin in order to count as a valid inverse to the polynomial problem.
        - Theoretically, this is always 1 due to conformally mapping to the unit disk.
    r: Real
        - Parameter dictating the region of validity of the inverse. See https://arxiv.org/pdf/1203.1958v2.pdf for more details.
    n: Int
        - Number of fourier coefficients.
Returns:
    G_μ(z): Array{Complex}
        - Evaluation of Cauchy Transform.
"""
function invcauchytransform_point(z, pm::PointMeasure; maxterms = 200, tol = 1, r=0.9, n = 200)
    a, b = minimum(pm.λ), maximum(pm.λ)
    g = θ -> cauchytransform_point(M_ab(J(r * exp(im * θ)), a, b), pm)
    θ = range(0, 2π; length=2*n+2)[1:end-1]
    fc = FFTW.fft(g.(θ))/(2*n+1)
    g_k = fc[1:n+1]

    n = min(n,maxterms)
    P = g_k[1:n]
    while abs(g_k[n]) < 10^-9
        n -= 1
    end

    s = invertpolynomial(P[1:n], z)
    s = [z for z in s if abs(z) < tol]
    
    x = M_ab.(J.(r .* s), a, b)
    
    number_of_atoms = length(pm.a)
    for i=1:number_of_atoms-1
        if length(x) >= number_of_atoms
            return x
        end
        x = [x; invcauchytransform_point_real(z, pm, pm.λ[i], pm.λ[i+1]; maxterms, tol, r, n)]
    end
    x
end

# not exported
function invcauchytransform_point_real(z, pm::PointMeasure, a, b; maxterms=200, tol=1, r=0.9, n=200)
    g = θ -> cauchytransform_point(M_ab(H(r * exp(im * θ)), a, b), pm)
    θ = range(0, 2π; length=2*n+2)[1:end-1]
    fc = FFTW.fft(g.(θ))/(2*n+1)
    g_k = fc[1:n+1]
    n = min(n,maxterms)
    P = g_k[1:n]
    while abs(g_k[n]) < 10^-9
        n -= 1
    end
    s = invertpolynomial(P[1:n], z)
    s = [z for z in s if abs(z) < tol]
    if length(s) == 1
        return M_ab.(H.(r .* s), a, b)[1]
    else
        return M_ab.(H.(r .* s), a, b)
    end
end

function invertpolynomial(P, z)
    last = P[end]
    n = length(P) - 1
    P = P / last
    C = zeros(Complex, n, n)
    for i=1:n-1
        C[i+1,i] = 1
        C[i,n] = -P[i]
    end
    C[n,n] = -P[n]
    C[1, n] += z/last
    eigvals(C)
end