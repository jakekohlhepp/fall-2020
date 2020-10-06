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

# transform to long.
df = CSV.read("busdata.csv")
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

Zst = convert(Vector, df[:,:Zst])
B = convert(Vector, df[:,:Branded])
X = df[:, 43:62]

## 2. Run flexible logit
beta_glm = glm(@formula(Y ~ Odometer*Odometer*RouteUsage*RouteUsage*Branded*time*time ), df_long, Binomial(), LogitLink())
println(beta_glm)

### 3
## a. state transitions
zval,zbin,xval,xbin,xtran = create_grids()


## b. 
Helper = DataFrame(Odometer = kron(ones(zbin),xval),RouteUsage = kron(ones(xbin),zval), Branded = vec(zeros(size(xtran,1),1)),time = vec(zeros(size(xtran,1),1)))
beta=0.9

function ccp_estim(glm_est, helper, zval,zbin,xval,xbin,xtran,beta,df_long,df, B, Zst,X)
	FV = repeat(zeros(size(xtran,1),2 ),1,1,21)
	FVT1 = zeros(size(df,1),20 )
	for t=2:20
		for b=0:1
			helper[:,:time] = t .*(ones(size(helper,1)))
			helper[:,:Branded] = b.* ones(size(helper,1))
			p0 = 1 .- predict(glm_est,helper)
			FV[:,b+1,t] = -beta .* log.(p0)
		end
	end

	for i=1:size(df,1)
		for t=1:20
			row0 = 1+(Zst[i]-1)*xbin
			row1 = X[i,t]+(Zst[i]-1)*xbin
			FVT1[i,t] = (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1,B[i]+1,t+1]
		end
	end
	FVT1 = @transform(convert(DataFrame, FVT1), bus_id = 1:size(df,1))
	FVT1 = DataFrames.stack(FVT1, Not([:bus_id]))
	FVT1 = @transform(FVT1, time = kron(collect([1:20]...),ones(size(df,1))))
	sort!(FVT1,[:bus_id,:time])
	return FVT1
end

fvt1 = ccp_estim(beta_glm, Helper, zval,zbin,xval,xbin,xtran,beta,df_long,df,B,Zst, X)  

fv=convert(Vector, fvt1[:,:value])
df_long = @transform(df_long,fv=fv )

## final estimation

theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset=df_long.fv)
