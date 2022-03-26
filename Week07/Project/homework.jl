using BenchmarkTools
using Distributions
using Random
using StatsBase
using Roots
using QuadGK
using DataFrames
using Plots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV

include("gbsm.jl")
include("bt_american.jl")
include("../../Week05/RiskStats.jl")
include("../../Week05/simulate.jl")
include("../../Week05/fitted_model.jl")

s = 165
x = 165
ttm = (Date(2022,4,15)-Date(2022,3,13)).value/365
rf = 0.0025
b = 0.0053

#Calculate the GBSM Values.  Return Struct has all values
bsm_call = gbsm(true,s,x,ttm,rf,b,.2,includeGreeks=true)
bsm_put = gbsm(false,s,x,ttm,rf,b,.2,includeGreeks=true)

outTable = DataFrame(
    :Valuation => ["GBSM","GBSM"],
    :Type => ["Call", "Put"],
    :Method => ["Closed Form","Closed Form"],
    :Delta => [bsm_call.delta, bsm_put.delta],
    :Gamma => [bsm_call.gamma, bsm_put.gamma],
    :Vega => [bsm_call.vega, bsm_put.vega],
    :Theta => [bsm_call.theta, bsm_put.theta],
    :Rho => [missing, missing],
    :CarryRho => [bsm_call.cRho, bsm_put.cRho]
)

#Differential Library call the calculate the gradient
_x = [s,x,ttm,rf,b,.2]
f(_x) = gbsm(true,_x...).value
call_grad = ForwardDiff.gradient(f,_x)

f(_x) = gbsm(false,_x...).value
put_grad = ForwardDiff.gradient(f,_x)

#Derivative of Delta = Gamma
f(_x) = gbsm(true,_x...;includeGreeks=true).delta
call_gamma = ForwardDiff.gradient(f,_x)[1]
f(_x) = gbsm(false,_x...;includeGreeks=true).delta
put_gamma = ForwardDiff.gradient(f,_x)[1]

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["GBSM","GBSM"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[6], put_grad[6]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [call_grad[5], put_grad[5]]
    )
)

# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
divDate = Date(2022,04,11)
divDays = (divDate - Date(2022,3,13)).value
ttmDays = ttm*365
NPoints = convert(Int64,ttmDays*3)
divPoint = divDays*3
divAmt = 0.88

#Values
am_call = bt_american(true, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)
am_put = bt_american(false, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)

_x = [s,x,ttm,rf,.2]
function f(_x)
    _in = collect(_x)
    bt_american(true, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
call_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 1 #Need to play with the offset value to get a good derivative.  EXTRA 0.5 point if they do this
call_gamma = (bt_american(true, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(true, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
call_div = (bt_american(true, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_call)/(δ)


function f(_x)
    _in = collect(_x)
    bt_american(false, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
put_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 10
put_gamma = (bt_american(false, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(false, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
put_div = (bt_american(false, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_put)/(δ)

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["BT","BT"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[5], put_grad[5]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [missing, missing]
    )
)

sort!(outTable,[:Type, :Valuation, :Method])
println(outTable)
println("Call Derivative wrt Dividend: $call_div")
println("Put  Derivative wrt Dividend: $put_div")

#Problem #2
portfolio = CSV.read("Project/problem2.csv",DataFrame)
currentDate = Date(2022,2,25)
divDate = Date(2022,3,15)
divAmt =1.00
currentS=164.85
mult = 5
daysDiv = (divDate - currentDate).value


returns = CSV.read("Project/DailyReturn.csv",DataFrame)[!,:AAPL]
sd = std(returns)

portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

#Implied Vols
# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
portfolio[!, :ImpVol] = [
    portfolio.Type[i] == "Option" ?
    find_zero(x->bt_american(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-currentDate).value/365,
                        rf,
                        [divAmt],[daysDiv*mult],x,convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))
                -portfolio.CurrentPrice[i],.2)    : missing     
    for i in 1:size(portfolio,1)
]

#Delta function for BT American
function bt_delta(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

    f(_x) = bt_american(call::Bool, _x,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
    FiniteDiff.finite_difference_derivative(f, underlying)
end

#Position Level Deltas needed for DN VaR
portfolio[!, :Delta] = [
    portfolio.Type[i] == "Option" ?  (
            bt_delta(portfolio.OptionType[i]=="Call",
                currentS, 
                portfolio.Strike[i], 
                (portfolio.ExpirationDate[i]-currentDate).value/365, 
                rf, 
                [divAmt],[daysDiv*mult],
                portfolio.ImpVol[i],convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))*portfolio.Holding[i]    
    ) : portfolio.Holding[i]     
    for i in 1:size(portfolio,1)
]

#Simulate Returns
nSim = 10000
fwdT = 10
_simReturns = rand(Normal(0,sd),nSim*fwdT)

#collect 10 day returns
simPrices = Vector{Float64}(undef,nSim)
for i in 1:nSim
    r = 1.0
    for j in 1:fwdT
        r *= (1+_simReturns[fwdT*(i-1)+j])
    end
    simPrices[i] = currentS*r
end

iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)

#Precalculate the fwd TTM
values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-currentDate-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

#Valuation
simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
Threads.@threads for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*bt_american(values.OptionType[i]=="Call",
                                                simprice,
                                                values.Strike[i],
                                                values.fwd_ttm[i],
                                                rf,
                                                [divAmt],[(daysDiv-fwdT)*mult],
                                                values.ImpVol[i],
                                                convert(Int64,values.fwd_ttm[i]*mult*365)
                                            )
    elseif values.Type[i] == "Stock"
        simulatedValue[i] = values.Holding[i]*simprice
    end
    pnl[i] = simulatedValue[i] - currentValue[i]
end

values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
values[!,:currentValue] = currentValue


#Calculation of Risk Metrics
function aggRisk(df,aggLevel::Vector{Symbol})
    gdf = []
    if !isempty(aggLevel)
        gdf = groupby(df,vcat(aggLevel,[:iteration]))

        agg = combine(gdf,
            :currentValue => sum => :currentValue,
            :simulatedValue => sum => :simulatedValue,
            :pnl => sum => :pnl
        )
        
        gdf = groupby(agg,aggLevel)
    else
        gdf = groupby(values,:iteration)

        gdf = combine(gdf,
            :currentValue => sum => :currentValue,
            :simulatedValue => sum => :simulatedValue,
            :pnl => sum => :pnl
        )
    end

    risk = combine(gdf, 
        :currentValue => (x-> first(x,1)) => :currentValue,
        :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
        :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
        :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
        :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
        :pnl => std => :Standard_Dev,
        :pnl => (x -> [extrema(x)]) => [:min, :max],
        :pnl => mean => :mean
    )
    return risk
end

#Calculate Simulated Risk Values
risk = aggRisk(values,[:Portfolio])

#Calculate the Portfolio Deltas
gdf = groupby(portfolio, [:Portfolio])
portfolioDelta = combine(gdf,
    :Delta => sum => :PortfolioDelta    
)

#Delta Normal VaR is just the Portfolio Delta * quantile * current Underlying Price
portfolioDelta[!,:DN_VaR] = abs.(quantile(Normal(0,sd),.05)*sqrt(10)*portfolioDelta.PortfolioDelta*currentS)
portfolioDelta[!,:DN_ES] = abs.((sqrt(10)*sd*pdf(Normal(0,1),quantile(Normal(0,1),.05))/.05)*portfolioDelta.PortfolioDelta*currentS)

leftjoin!(risk,portfolioDelta[!,[:Portfolio, :DN_VaR, :DN_ES]],on=:Portfolio)

println(risk)



###Problem 3 ###
#Read All Data
ff3 = CSV.read("Project/F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("Project/F-F_Momentum_Factor_daily.CSV",DataFrame)
returns = CSV.read("Project/DailyReturn.csv",DataFrame)

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = Date.(returns.Date,dateformat"mm/dd/yyyy")

# Our 20 stocks
stocks = [:AAPL, :FB, :UNH, :MA, :MSFT, :NVDA, :HD, :PFE, :AMZN, Symbol("BRK-B"), :PG, :XOM, :TSLA, :JPM, :V, :DIS, :GOOGL, :JNJ, :BAC, :CSCO]

# Data set of all stock returns and FF3+1 returns
to_reg = innerjoin(returns[!,vcat(:Date,stocks)], ffData, on=:Date)

println("Max RF value is: $(max(to_reg.RF...))")
#since the value is always 0, no need to difference the stock returns.

xnames = [Symbol("Mkt-RF"), :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))
Y = Matrix(to_reg[!,stocks])

Betas = (inv(X'*X)*X'*Y)'

#Calculate the means of the last 10 years of factor returns
#adding the 0.0 at the front to 0 out the fitted alpha in the next step
means = vcat(0.0,mean.(eachcol(ffData[ffData.Date .>= Date(2012,01,31),xnames])))

#Discrete Returns, convert to Log Returns and scale to 1 year
stockMeans =log.(1 .+ Betas*means)*255
covar = cov(log.(1.0 .+ Y))*255

#optimize.  Directly find the max SR portfolio.  Can also do this like in the notes and
#   build the Efficient Frontier

function sr(w...)
    _w = collect(w)
    m = _w'*stockMeans - .0025
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = length(stocks)

m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = round.(value.(w),digits=4)

OptWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :Er => stockMeans)
println(OptWeights)
println("Expected Retrun = $(stockMeans'*w)")
println("Expected Vol = $(sqrt(w'*covar*w))")
println("Expected SR = $(sr(w...)) ")