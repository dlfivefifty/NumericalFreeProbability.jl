module NumericalFreeProbability

using LinearAlgebra, QuasiArrays, LazyArrays, ClassicalOrthogonalPolynomials, InfiniteArrays, SpecialFunctions,
                SingularIntegrals, StaticArrays, Random, ForwardDiff, SpecialMatrices

import Base: @_inline_meta, axes, getindex, unsafe_getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, setindex,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum, cumsum,
                to_indices, tail, getproperty, inv, show, isapprox, summary,
                findall, searchsortedfirst, diff

# affine transformations
M_ab(x::Number,a::Number,b::Number) = (a + b)/2 + (b - a) * x/2 # maps from (-1, 1) to (a, b)
M_ab_inv(y::Number,a::Number,b::Number) = (2*y - (a + b))/(b - a) # maps from (a, b) to (-1, 1)

include("measures.jl")
include("cauchytransforms.jl")
include("invcauchytransforms.jl")
include("recovermeasures.jl")

#include("SqrtMeasures.jl")
#include("PointMeasures.jl")

end # module NumericalFreeProbability
