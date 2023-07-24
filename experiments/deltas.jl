using RandomMatrices, Plots

n = 2000

@time Q = qr(randn(n,n)).Q * Diagonal(rand([-1,1],n))
# @time Q = rand(Haar(1), n);
a = 1; histogram(eigvals(Hermitian(Q * Diagonal(rand([-1,1/2],n)) * Q' + Diagonal(rand([-a/3,a],n)))); nbins = 100, normalized=true)