export cauchytransform_sqrt, invcauchytransform_sqrt

# TODO: turn the measures into special types

# Joukowski transform
Jinv_p = z -> z - √(z - 1) * √(z + 1)
J = w -> 1/2 * (w + 1/w)

# Affine maps to transform support
M_ab = (x,a,b) -> (a + b)/2 + (b - a) * x /2 # maps from (-1, 1) to (a, b)
M_ab_inv = (y,a,b) -> (2*y - (a + b))/(b - a) # maps from (a, b) to (-1, 1)

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
        - Max number of terms ψ_k used in expansion of ψ(x) in Chebysev U polynomials.
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