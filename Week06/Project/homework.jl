using CSV
using DataFrames
using Dates
using Roots
using Distributions
using Plots
using Random

include("gbsm.jl")

#Problem #1
currentPrice = 165
currentDate=Date("02/25/2022",dateformat"mm/dd/yyyy")
rf = 0.0025
dy = 0.0053
DaysYear = 365

expirationDate = Date("03/18/2022",dateformat"mm/dd/yyyy")
ttm = (expirationDate - currentDate).value/DaysYear

strike = 165
iv = [i/100 for i in 10:2:80]
#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
call_vals = gbsm.(true,currentPrice,strike,ttm,rf,rf-dy,iv)
put_vals  = gbsm.(false,currentPrice,strike,ttm,rf,rf-dy,iv)

plot(
    plot(call_vals,iv, label="Call Values"),
    plot(put_vals,iv, label="Put Values",linecolor=:red),
    layout=(1,2)
)


#Problem #2
currentPrice = 164.85
options = CSV.read("Project/AAPL_Options.csv",DataFrame)

options[!,:Expiration] = Date.(options.Expiration,dateformat"mm/dd/yyyy")


n = length(options.Expiration)

#list comprehension for TTM
options[!,:ttm] = [(options.Expiration[i] - currentDate).value / DaysYear for i in 1:n]

#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
iv = [find_zero(x->gbsm(options.Type[i]=="Call",currentPrice,options.Strike[i],options.ttm[i],rf,rf-dy,x)-options[i,"Last Price"],.2) for i in 1:n]
options[!,:ivol] = iv
options[!,:gbsm] = gbsm.(options.Type.=="Call",currentPrice,options.Strike,options.ttm,rf,rf-dy,options.ivol)


calls = options.Type .== "Call"
puts = [!calls[i] for i in 1:n]

plot(options.Strike[calls],options.ivol[calls],label="Call Implied Vol",title="Implied Volatilities")
plot!(options.Strike[puts],options.ivol[puts],label="Put Implied Vol",linecolor=:red)
vline!([currentPrice],label="Current Price",linestyle=:dash,linecolor=:purple)


#problem 3
include("../../Week05/RiskStats.jl")
currentS=164.85
returns = CSV.read("Project/DailyReturn.csv",DataFrame)[!,:AAPL]
sd = std(returns)
current_dt = Date(2022,2,25)

portfolio = CSV.read("Project/problem3.csv", DataFrame)
portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

portfolio[!, :ImpVol] = [
    portfolio.Type[i] == "Option" ?
    find_zero(x->gbsm(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-current_dt).value/365,
                        rf,rf-dy,x)
                -portfolio.CurrentPrice[i],.2)    : missing     
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

values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-current_dt-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*gbsm(values.OptionType[i]=="Call",simprice,values.Strike[i],values.fwd_ttm[i],rf,rf-dy,values.ImpVol[i])
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
    risk[!,:VaR95_Pct] =  risk.VaR95 ./ risk.currentValue
    risk[!,:VaR99_Pct] =  risk.VaR99 ./ risk.currentValue
    risk[!,:ES95_Pct] =  risk.ES95 ./ risk.currentValue
    risk[!,:ES99_Pct] =  risk.ES99 ./ risk.currentValue
    return risk
end

risk = aggRisk(values,[:Portfolio])

CSV.write("problem3_risk.csv",risk)