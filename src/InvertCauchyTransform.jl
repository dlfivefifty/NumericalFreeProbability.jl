export inversecauchytransform

using LinearAlgebra
using NonlinearEigenproblems, ClassicalOrthogonalPolynomials, FastGaussQuadrature, SingularIntegrals

"""
    Let μ be a measure with density expressed in the form f(x)w(x), where w(x) is the weight
    function which the Jacobi matrix is wrt to, and f(x) is a "nice" function that is bounded and strictly positive.


    y: solve for G(z) = y

    P: orthogonal polynomials

    w: weight function

    f: function f(x)
    
"""
function inversecauchytransform(y, P, w, f, n; radius= 0.8, N = 2000)
    J = jacobimatrix(P)'
    f_expanded = expand(P, f)
    f_k = f_expanded.args[2][1:n+1]
    last = f_expanded.args[2][n+2]
    A = Matrix(J[1:n+1,1:n+1])
    A[end,:] -= f_k ./ last .* J[n+1, n+2]
    b = zeros(n+1) .+ 0im
    b[end] = y/last * J[n+1, n+2]
    b[1] = 1
    Σ = zeros(n+1)
    Σ[1] += 1

    A1 = [0 Σ';b A]
    A2 = Diagonal([-(i != 0) for i=0:n+1])
    A3 = [0 zeros(n+1)';A*Σ zeros(n+1, n+1)]
    A4 = [0 zeros(n+1)';-Σ zeros(n+1, n+1)]

    function q_0(z)
        W = Weighted(P); p_expanded = expand(W, w);
        x = axes(W, 1)
        inv.(z .- x') * p_expanded
    end
    
    function H1(z)
        -((I + z) * inv(I - z))^2 - I
    end
    function H2(z)
        ((I + z) * inv(I - z))^2 + I
    end
    functionlist = [H1, H2]
    
    inverses = []
    for H in functionlist
        f1 = z -> one(z)
        f2 = z -> H(z)
        f3 = z -> q_0(H(z))
        f4 = z -> H(z) * q_0(H(z))
                
        AA = [A1, A2, A3, A4]
        fii = [f1, f2, f3, f4]
        nep=SPMF_NEP(AA, fii, check_consistency = false);

        (λ,v)= contour_beyn(nep; radius, σ=0, N);
        for z in H.(λ)
            if all(abs.(z .- inverses) .> 10^-10)
                push!(inverses, z)
            end
        end
    end
    inverses
end




P = ChebyshevU()
f = x -> x^3/6 + x/2 + 1
w = x -> √(1-x^2) * 2/π
p = x -> (x^3/6 + x/2 + 1) * 2/pi * √(1-x^2)
W = Weighted(P); p_expanded = expand(W, p);
x = axes(W, 1)
function G(z)
    inv.(z .- x') * p_expanded
end

n=2
z=-1.2 + 0.0im
y = G(z)

display(inversecauchytransform(y, P, w, f, n))




P = Jacobi(2,2)
f = x -> exp(x) / 1.0734430519421176
w = x -> 15/16 * (1-x)^2 * (x+1)^2

p = x -> exp(x) * 15/16 * (x-1)^2 * (x+1)^2 / 1.0734430519421176
W = Weighted(P); p_expanded = expand(W, p);
x = axes(W, 1)
function G(z)
    inv.(z .- x') * p_expanded
end

n=5
z=-1.3 + 0.0im
y = G(z)

display(inversecauchytransform(y, P, w, f, n))

