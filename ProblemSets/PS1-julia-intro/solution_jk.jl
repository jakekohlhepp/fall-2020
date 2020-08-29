using JLD2 
using Random 
using LinearAlgebra 
using Statistics 
using CSV 
using DataFrames 
using FreqTables
using Distributions

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

## 2.
# a.
function q2(A,B,C,D)
	AB = [A[i,j]*A[i,j] for i=1:size(A,1),j=1:size(A,2)]
	AB2 = A .*B
	# b.
	Cprime = Float64[]
	for i in 1:length(C)
		if abs(C[i])<5
		append!(Cprime, C[i])
		end
	end
	Cprime2 = C[abs.(C) .<5]
	Cprime == Cprime2

	# c
	X = Array{Float64}(undef, 15169, 6,5)
	for t in 1:size(X,3)
	X[1:15169,1,t] .= 1
	X[1:15169,2,t] .= rand(Bernoulli(0.75*(6-t)/5),15169)
	X[1:15169,3,t] .= vec((5*(t-1)) .*randn(15169,1) .+(15+t-1))
	X[1:15169,4,t] .= vec(1/MathConstants.e .*randn(15169,1) .+(MathConstants.pi*(6-t)/3))
	X[1:15169,5,t] .= vec(rand(Binomial(20,0.6),15169))
	X[1:15169,6,t] .= vec(rand(Binomial(20,0.5),15169))
	end

	# d.
	beta = [(0.25*t +0.75 , log(t),-t^(1/2),exp(t)-exp(t+1),t,t/3) for t in 1:5]
	beta = reduce(hcat, getindex.(beta,i) for i in eachindex(beta[1]))'

	# e
	Y = [ X[n,:,t]'*beta[:,t] + randn(1,1)[1]*0.36 for t in 1:5, n in 1:15169]'
end

q2(A,B,C,D)

function q3()
	## 3. 
	# a.
	data = CSV.read("nlsw88.csv")
	@save "nlsw88.jld" data
	# b.
	mean(data.never_married)
	mean(data.collgrad)
	# c
	prop(freqtable(data.race))
	# d 
	summarystats=describe(data)
	# e
	freqtable(data.industry, data.occupation)
	# f split apply combine part.
	subset = select(data, :wage, :occupation, :industry)
	combine(groupby(subset, All(:industry, :occupation)), :wage => mean)
end
q3()