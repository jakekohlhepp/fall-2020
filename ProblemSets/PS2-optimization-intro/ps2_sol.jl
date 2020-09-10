using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using CategoricalArrays
using Tables
using StatsModels
cd("C:/Users/jakek/Documents/econ_phd/structural_modeling_course/forked_class/ProblemSets/PS2-optimization-intro")

function fullquestions()
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# question 1
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
	minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
	startval = rand(1)   # random starting value
	result = optimize(minusf, startval, BFGS())


	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# question 2
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
	df = CSV.read(HTTP.get(url).body)
	X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
	y = df.married.==1

	function ols(beta, X, y)
		ssr = (y.-X*beta)'*(y.-X*beta)
		return ssr
	end

	beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
	println(vec(beta_hat_ols.minimizer))

	bols = inv(X'*X)*X'*y
	df.white = df.race.==1
	bols_lm = lm(@formula(married ~ age + white + collgrad), df)

	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# question 3
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	function logit(alpha, X, d)
		# your turn
		loglike = sum(X*alpha.*d .- log.(1 .+exp.(X*alpha)))
		return -loglike
	end

	beta_hat_logit = optimize(b -> logit(b, X, y),beta_hat_ols.minimizer , LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
	println(beta_hat_logit.minimizer)

	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# question 4
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# check it
	df.raceind = df.race .==1
	beta_glm = glm(@formula(married ~ age + raceind+collgrad), df, Binomial(), LogitLink())
	if !all(round.(coef(beta_glm) .- beta_hat_logit.minimizer,digits=5).==0)
		error()
	end

	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	# question 5
	#:::::::::::::::::::::::::::::::::::::::::::::::::::
	freqtable(df, :occupation) # note small number of obs in some occupations
	df = dropmissing(df, :occupation)
	df[df.occupation.==8 ,:occupation] .= 7
	df[df.occupation.==9 ,:occupation] .= 7
	df[df.occupation.==10,:occupation] .= 7
	df[df.occupation.==11,:occupation] .= 7
	df[df.occupation.==12,:occupation] .= 7
	df[df.occupation.==13,:occupation] .= 7
	freqtable(df, :occupation) # problem solved

	X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
	y = df.occupation

	# set base occ to be first one.
	function mlogit(alpha, X, d)
		u = reduce(hcat,[X*alpha[(4*i-3):(4*i)] for i in 1:6])
		help = reduce(hcat,[d.==i for i in 2:7])
		loglike = sum(help .*u) - sum(log.(1 .+sum(exp.(u), dims=2)))
		return -loglike
	end

	# starting values - set intercepts to fraction present in data.
	starting = Vector(prop(freqtable(df, :occupation)))
	starting = repeat(starting[Not(1)], inner = 4)
	starting[Not((1:6) .*4 .-3)] .= 0

	beta_hat_mlogit = optimize(b -> mlogit(b, X, y),starting, BFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
	println(vec(beta_hat_mlogit.minimizer))
end

fullquestions()
