module NumericalFreeProbability

using LinearAlgebra, QuasiArrays, LazyArrays, ClassicalOrthogonalPolynomials, InfiniteArrays, StatsBase,
                SingularIntegrals, Random, ForwardDiff, SpecialMatrices, IntervalSets

using ClassicalOrthogonalPolynomials: _p0, orthogonalityweight

import Base: @_inline_meta, axes, getindex, unsafe_getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy, setindex,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum, cumsum,
                to_indices, tail, getproperty, inv, show, isapprox, summary,
                findall, searchsortedfirst, diff, promote_rule
import LinearAlgebra: normalize
import StatsBase: sample

# affine transformations
M_ab(x::Number,a::Real,b::Real) = (a + b)/2 + (b - a) * x/2 # maps from (-1, 1) to (a, b)
M_ab_inv(y::Number,a::Real,b::Real) = (2y - (a + b))/(b - a) # maps from (a, b) to (-1, 1)
dM_ab(x::Number,a::Real,b::Real) = (b - a) /2

include("measures.jl")
include("cauchytransforms.jl")
include("invcauchytransforms.jl")
include("support.jl")
include("recovermeasures.jl")
include("sampling.jl")

end # module NumericalFreeProbability
