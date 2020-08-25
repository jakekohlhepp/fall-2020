using JLD2 
using Random 
using LinearAlgebra 
using Statistics 
using CSV 
using DataFrames 
using FreqTables

function q1()
	cd("C:/Users/jakek/Documents/econ_phd/structural_modeling_course/forked_class/ProblemSets/PS1-julia-intro")
	### 1.
	# a.
	Random.seed!(1234)
	A = 15 .*rand(10,7) .-5
	B = sqrt(15) .*randn(10,7) .-2
	C = [A[1:5, 1:5] B[1:5,6:7]]
	D = [max(elem,0) for elem in A]

	# b.
	length(A)
	# c.
	length(unique(D))
	# d
	E = reshape(B, (1, length(B)))'
	E_alt = vec(B)
	# e
	F = cat(A,B; dims=3)
	# f
	F = permutedims(F, [3,1,2])
	# g
	G = kron(B,C)
	# error opccurs when we try kron(F,G) due to differences in dimensions.
	# h
	@save "matrixpractice.jld" A B C D E F G 
	# i
	@save "firstmatrix.jld" A B C D
	# j
	C_csv = convert(DataFrame,C)
	CSV.write("Cmatrix.csv",C_csv)
	# k 
	D_dat = convert(DataFrame,D)
	CSV.write("Dmatrix.dat",D_dat, delim="\t")
	return(A,B,C,D)
end

A,B,C,D = q1()

