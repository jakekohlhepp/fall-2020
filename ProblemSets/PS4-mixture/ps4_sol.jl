using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using Distributions
using CSV
using FreqTables
using CategoricalArrays
using Tables
using StatsModels
cd("C:/Users/jakek/Documents/econ_phd/structural_modeling_course/forked_class/ProblemSets/PS4-mixture/")

# 1

df = CSV.read("nlsw88t.csv")
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

function mlogit(alpha, X, d,Z)
	T = promote_type(eltype(X),eltype(alpha), eltype(Z), eltype(d))
	Zbar = zeros(T, size(Z,1),1)
	u = zeros(T, size(Z,1),7)
	help = zeros(T, size(Z,1),7)
	loglike = zeros(T, 1)
	Zbar = Z .- Z[:,8]
	u = reduce(hcat,[X*alpha[(3*i-2):(3*i)] .+ Zbar[:,i] .*alpha[22] for i in 1:7])
	help = reduce(hcat,[d.==i for i in 1:7])
	loglike = sum(help .*u) - sum(log.(1 .+sum(exp.(u), dims=2)))
	return -loglike
end

# starting values - pull from last homework
start_ps3 = CSV.read("../PS3-gev/q1.csv")
starting = vec(reduce(vcat,[start_ps3[:,i] for i in 1:8])[Not([23 24]),:])

td = TwiceDifferentiable(b -> mlogit(b, X, y,Z), vec(starting); autodiff = :forward)
beta_hat_mlogit = optimize(td,starting, Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
H  = Optim.hessian!(td, beta_hat_mlogit.minimizer)
beta_hat_mlogit_se = sqrt.(diag(inv(H)))

res = vcat(beta_hat_mlogit.minimizer', beta_hat_mlogit_se' )
CSV.write("q1.csv", convert(DataFrame,res))

#2.  now the coefficient on wage is positive, reflecting that increased wages results in higher utility. This makes more sense than before.

#########################################
### 3.
#########################################
### a. Qudrature learning
include("lgwt.jl")
d = Normal(0,1)
nodes, weights = lgwt(7,-4,4)
sum(weights.*pdf.(d,nodes)) # pdf integration should be near 1.
sum(weights.*nodes.*pdf.(d,nodes)) # expectation shoudl bear near 0
### b. quadrature practice
d2 = Normal(0,2)
nodes, weights = lgwt(7,-10,10)
sum(weights.*nodes.^2 .*pdf.(d2,nodes))
nodes, weights = lgwt(10,-10,10)
sum(weights.*nodes.^2 .*pdf.(d2,nodes))
# the second option is much closer to the true variance, 2.

### c.
a= -10
b =10
d=10000000
nds = rand(d,1) .*(b-a) .+a
print((b-a)/d * sum(pdf.(d2,nds))) # should be close to 1.
print(sum(nds .^2 .* ((b-a)/d .* pdf.(d2,nds))))
print(sum(nds .* ((b-a)/d .* pdf.(d2,nds))))

#### 4. Quadrature.
function mixlogit(alpha, X, d,Z)
	include("lgwt.jl")
	T = promote_type(eltype(X),eltype(alpha), eltype(Z), eltype(d))
	Zbar = zeros(T, size(Z,1),1)
	u = repeat(zeros(T, size(Z,1),7),1,1,7)
	help = zeros(T, size(Z,1),7)
	loglike = zeros(T, 1)
	Zbar = Z .- Z[:,8]
	sig = exp(alpha[23])
	dist = Normal(alpha[22],sig)
	nodes, weights = lgwt(7,-5*sig+alpha[22],5*sig+alpha[22])
	# make 3d array, with quad points.
	for j in 1:7
		u[:,:,j] = exp.(reduce(hcat,[X*alpha[(3*i-2):(3*i)] .+ Zbar[:,i] .*nodes[j] for i in 1:7]))
	end
	usum =  sum(u,dims=2)
	help = repeat(reduce(hcat,[d.==i for i in 1:7]),1,1,7)
	premix =  prod(u .^help, dims=2) ./usum 
	mix = reduce(hcat,[weights[i]*pdf(dist,nodes[i]) .* premix[:,:,i] for i in 1:7])
	loglike = sum(log.(sum(mix,dims=2)))
	return -loglike
end

starting = append!(vec(starting), 0)
td = TwiceDifferentiable(b -> mixlogit(b, X, y,Z), starting; autodiff = :forward)
beta_mixlogit = optimize(td,starting, Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
H  = Optim.hessian!(td, beta_mixlogit.minimizer)
#beta_hat_mlogit_se = sqrt.(diag(inv(H)))
# res = vcat(beta_mixlogit.minimizer', beta_hat_mlogit_se' )
CSV.write("q4.csv", convert(DataFrame,beta_mixlogit.minimizer'))

#### 5. simulation
function mixlogit_mc(alpha, X, d,Z)
	include("lgwt.jl")
	T = promote_type(eltype(X),eltype(alpha), eltype(Z), eltype(d))
	Zbar = zeros(T, size(Z,1),1)
	u = repeat(zeros(T, size(Z,1),7),1,1,1000000)
	help = zeros(T, size(Z,1),7)
	loglike = zeros(T, 1)
	Zbar = Z .- Z[:,8]
	dist = Normal(alpha[22],sig)
	sig = exp(alpha[23])
	nodes = rand(1000000,1) .*(2*sig*5) .+ alpha[22]
	weights = (2*sig*5)/1000000
	# make 3d array, with quad points.
	for j in 1:1000000
		u[:,:,j] = exp.(reduce(hcat,[X*alpha[(3*i-2):(3*i)] .+ Zbar[:,i] .*nodes[j] for i in 1:7]))
	end
	usum =  sum(u,dims=2)
	help = repeat(reduce(hcat,[d.==i for i in 1:7]),1,1,10)
	premix =  prod(u .^help, dims=2) ./usum 
	mix = reduce(hcat,[weights[i]*pdf(dist,nodes[i]) .* premix[:,:,i] for i in 1:1000000])
	loglike = sum(mix)
	return -loglike
end
