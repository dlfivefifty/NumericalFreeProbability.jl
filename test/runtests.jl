using NumericalFreeProbability, Test, LinearAlgebra

@testset "NumericalFreeProbability" begin
	include("measures.jl")
	include("cauchytransforms.jl")
	include("invcauchytransforms.jl")
	# include("recovermeasures.jl")
end # testset
