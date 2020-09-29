using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM


# 1. read in and reshape
cd("C:/Users/jakek/Documents/econ_phd/structural_modeling_course/forked_class/ProblemSets/PS5-ddc/")
include("create_grids.jl")
df = CSV.read("busdataBeta0.csv")
df = @transform(df, bus_id = 1:size(df,1))
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

##### 2. static model.
beta_glm = glm(@formula(Y ~ Odometer+Branded ), df_long, Binomial(), LogitLink())
println(beta_glm)

###### 3.
beta = 0.9
data = CSV.read("busdata.csv")
Y = @select(data, :Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20)
Y = convert(Matrix, Y)
O = @select(df, :Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
O = convert(Matrix, O)
X = data[:, 43:62]
Z = Vector(df[:,:RouteUsage])
X = convert(Matrix, X)
Zst = convert(Vector, data[:,:Zst])
B = convert(Vector, data[:,:Branded])
# create xst

## b
zval,zbin,xval,xbin,xtran = create_grids()

### c.
@views @inbounds function zurcher(theta, Y, X, O,Zst,B,beta,zval,zbin,xval,xbin,xtran)
	T = promote_type(eltype(X),eltype(theta), eltype(Y), eltype(O), eltype(B), eltype(Zst),eltype(beta))
	Fut = repeat(zeros(T,size(xtran,1),2 ),1,1,21)
	for t=reverse(1:20)
		for b=0:1
			for z=1:zbin
				for x=1:xbin
					row = x+(z-1)*xbin
					v1 = theta[1]+theta[2]*xval[x] + theta[3]*b+(xtran[row,:]'*Fut[(z-1)*xbin+1:z*xbin,b+1,t+1])
					v0 = xtran[1+(z-1)*xbin,:]'*Fut[(z-1)*xbin+1:z*xbin,b+1,t+1]
					Fut[row,b+1,t] = beta*log(exp(v1)+exp(v0))
				end
			end
		end
	end
	loglik = 0
	for i=1:size(Y,1)
		for t=1:20
			row0 = 1+(Zst[i]-1)*xbin
			row1 = X[i,t]+(Zst[i]-1)*xbin
			vflow = theta[1]+theta[2]*O[i,t]+ theta[3]*B[i]
			vdiff = vflow + (xtran[row1,:].-xtran[row0,:])'*Fut[row0:row0+xbin-1,B[i]+1,t+1]
			loglik = loglik + (Y[i,t]==1)*vdiff - log(1+exp(vdiff))
		end
	end
	return -loglik
end

starting = vec(coef(beta_glm))
td = TwiceDifferentiable(b -> zurcher(b, Y, X, O, Zst,B,beta,zval,zbin,xval,xbin,xtran), θ_true; autodiff = :forward)
beta_zurcher = optimize(td,θ_true, Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
H  = Optim.hessian!(td, beta_zurcher.minimizer)
se= sqrt.(diag(inv(H)))
res = vcat(beta_zurcher.minimizer', se' )
CSV.write("q5.csv", convert(DataFrame,res'))
