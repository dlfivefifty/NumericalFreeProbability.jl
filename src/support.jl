export support_sqrt_single, additive_support, support_multiplysupportedsqrt

function bisection(f::Function, x_l::Real, x_r::Real; tol=10^-12, maxits = 40, info=false, forcereturn=false)
    y_l = f(x_l)
    if !(y_l > 0) ⊻ (f(x_r) > 0)
        return nothing
    end
    for i=1:maxits
        if info
            println("iteration")
            println(i)
            println((x_l, x_r))
            println(abs(x_l - x_r))
            println((y_l, f(x_r)))
        end
        x_m = (x_l + x_r)/2; y_m = f(x_m)
        if abs(y_m) < tol return x_m end
        if (y_m > 0) ⊻ (y_l > 0) x_r = x_m else x_l = x_m; y_l = y_m end
    end
    if forcereturn
        return (x_l+x_r)/2
    end
end



function findallroots(f, x_l::T, x_r::T; tol=10^-6, maxits = 40, step=0.01) where T<:Real
    points = T[]; sign_previous = 2; sgn_p = x -> x > 0
    for x = x_l:step:x_r
        if sign_previous != sgn_p(f(x))
            sign_previous = sgn_p(f(x)); push!(points, x)
        end
    end
    roots = T[]; n = length(points) - 1
    for i=1:n
        a = bisection(f, points[i], points[i+1]; tol, maxits)
        if a !== nothing
            push!(roots, a)
        end
    end
    roots
end


function additive_support(G_a, G_b, dG_a, dG_b, m_a::Measure{T}, m_b::Measure{T}; tol=10^-13, maxits=60) where T<:Real
    ε = 1000eps()
    supp_a = support(m_a); supp_b = support(m_b)
    
    invG_a = v -> z::Number -> realinvcauchytransform(z, m_a, v)
    invG_b = v -> z::Number -> realinvcauchytransform(z, m_b, v)

    support_points = T[]
    dinvG_a(v) = z::Number -> inv(dG_a(invG_a(v)(z)))
    dinvG_b(v) = z::Number -> inv(dG_b(invG_b(v)(z)))
    invG_c(v1, v2) = z::Number -> invG_a(v1)(z)+ invG_b(v2)(z) - inv(z)
    dinvG_c(v1, v2) = z::Number -> real(dinvG_a(v1)(z) + dinvG_b(v2)(z) + inv(z^2))

    max_bisection_range = 10 # for safety
    a = max(G_a(supp_a[1][1] - ε), G_b(supp_b[1][1] - ε), -max_bisection_range)
    b = min(G_a(supp_a[end][2] + ε), G_b(supp_b[end][2] + ε), max_bisection_range)

    ξ_a = bisection(dinvG_c(1,1), a, 0-ε; tol, maxits, forcereturn=true)
    ξ_b = bisection(dinvG_c(1,1), 0+ε, b; tol, maxits, forcereturn=true)

    push!(support_points, real(invG_c(1,1)(ξ_a)))
    push!(support_points, real(invG_c(1,1)(ξ_b)))
    for i = 2:length(supp_a)
        a = max(G_a(supp_a[i][1] - ε), G_b(supp_b[1][1] - ε), -max_bisection_range)
        b = min(G_a(supp_a[i-1][2] + ε), G_b(supp_b[end][2] + ε), max_bisection_range)
        ξ = findallroots(dinvG_c(i,1), a+ε, b-ε; tol, maxits)
        if length(ξ) == 2
            push!(support_points, real(invG_c(i,1)(ξ[1])))
            push!(support_points, real(invG_c(i,1)(ξ[2])))
        end
    end
    for i = 2:length(supp_b)
        a = max(G_a(supp_a[1][1] - ε), G_b(supp_b[i][1] - ε), -max_bisection_range)
        b = min(G_a(supp_a[end][2] + ε), G_b(supp_b[i-1][2] + ε), max_bisection_range)
        ξ = findallroots(dinvG_c(1,i), a+ε, b-ε; tol, maxits)
        if length(ξ) == 2
            push!(support_points, real(invG_c(1,i)(ξ[1])))
            push!(support_points, real(invG_c(1,i)(ξ[2])))
        end
    end
    sort!(support_points)
    [[support_points[2*i-1], support_points[2*i]] for i=1:length(support_points)÷2]
end


