using NumericalFreeProbability, Test, LinearAlgebra

@testset "NumericalFreeProbability" begin
	include("measures.jl")
	include("cauchytransforms.jl")
	include("invcauchytransforms.jl")
	#include("recovermeasures.jl")
	
	#include("SqrtMeasures.jl")
	#include("PointMeasures.jl")
end # testset
