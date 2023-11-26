# NumericalFreeProbability.jl
 A Julia package for computing free convolutions.

 Currently free additive convolution of linear combinations of Jacobi and Point measures is supported, provided that the output measure is square-root decaying on multiple intervals.



<!-- ## Measures
### Square-root & Jacobi measures
A Jacobi measure supported on $[a,b]$ with exponents $\alpha$ and $\beta$ is a measure of the form

$\mathrm{d} \mu = r(x) \frac{2(b-x)^{\beta}(x-a)^{\alpha}}{b-a} \mathrm{d} x$

where $r(x)$ is a function. A Square-root measure is the special case when $\alpha = \beta = \frac 12$.

To create a measure, we specify the support and the modification function. In the Jacobi case, we also specify the powers. -->

```julia
using NumericalFreeProbability, Plots
sc = Semicircle()
pm = PointMeasure([-2,-1,1], [1/2, 1/4, 1/4])
u = sc ⊞ pm
xv = -4:0.01:4
plot(xv, sc[xv])
plot!(xv, pm[xv])
plot!(xv, u[xv])
```
![image](https://github.com/dlfivefifty/NumericalFreeProbability.jl/assets/100195398/3c3b9e14-93ea-467c-a110-7e3ac54c8bb5)


<!-- jm = normalize(JacobiMeasure(-1,2, 0.6, 0.4, x -> 1 + x^2)) -->


```julia
sm = normalize(ChebyshevUMeasure(-3,-1) + ChebyshevUMeasure(1,3))
a = sm
xv = -15:0.01:15
plot(xv, a[xv])
for _=1:10
    a = a ⊞ sm
    plot!(xv, a[xv])
end
```

![image](https://github.com/dlfivefifty/NumericalFreeProbability.jl/assets/100195398/debea82d-7a3e-43df-b292-9f7feb9084c9)
