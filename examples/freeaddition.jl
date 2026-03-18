using NumericalRepresentationTheory, CairoMakie


## Marchenko-Pastur distribution
m,n = 100, 500
X = randn(m,n)
A = X*X'/n
λ = m/n; a,b = (1-√λ)^2, (1+√λ)^2
mp = ChebyshevUMeasure(a, b, x -> (b-a)/(4π*λ*x))
@test sum(mp)  ≈ 1
# (sqrt.(max.(0, (b .- 𝐱).*(𝐱 .- a))))./(2π*λ*𝐱)
fig = Figure()
ax = Axis(fig[1, 1]; title = "Marchenko–Pastur distribution")
hist!(ax, eigvals(Symmetric(A)); bins=0:0.1:3, normalization=:pdf); fig
𝐱 = range(0,3,100)
lines!(ax, 𝐱, mp[𝐱]; color=:black, alpha=0.8, label=nothing)
fig


## Point distribution

fig = Figure()
ax = Axis(fig[1, 1]; title = "Discrete sample")
pm = PointMeasure([-2,-1,1], [1/2, 1/4, 1/4])
hist!(ax, sample(pm,100); bins=-2:0.1:3, normalization=:probability); fig



## Free addition
μ = freeaddition(mp, pm)
fig = Figure()
ax = Axis(fig[1, 1]; title = "Free addition")
hist!(ax, eigvals(Symmetric(A + Diagonal(sample(pm,100)))); bins=-2:0.25:3, normalization=:pdf);fig
𝐱 = range(-2,3,1000)
lines!(ax, 𝐱, μ[𝐱]; color=:black, alpha=0.8, label=nothing); fig