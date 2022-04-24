using CSV
using Distributions
using Plots
using StatsPlots
using QuadGK
using DataFrames
using Ipopt
using JuMP
using LoopVectorization
using StatsBase
using LinearAlgebra

#Problem #1
# sd = .05
# nu = 5
# ts = sqrt( sd^2*(nu-2)/nu )
# td = TDist(5)*ts

# x = rand(td,500)
# CSV.write("Project/problem1.csv",DataFrame(:x=>x))

x = CSV.read("Project/problem1.csv",DataFrame).x

include("../fitted_model.jl")
include("../RiskStats.jl")

tFit = fit_general_t(x)
nFit = fit_normal(x)

nVaR = VaR(nFit.errorModel; alpha=.05)
tVaR = VaR(tFit.errorModel; alpha=.05)

nES = ES(nFit.errorModel,alpha=0.05)
tES = ES(tFit.errorModel; alpha=0.05)

println("Normal  : VaR $(nVaR) ES $(nES)")
println("Fitted T: VaR $(tVaR) ES $(tES)")

minX, maxX = extrema(x)
df = DataFrame(:x=>[i for i in minX:.001:maxX])
df[!,:tPDF] = pdf.(tFit.errorModel,df.x)
df[!,:nPDF] = pdf.(nFit.errorModel,df.x)

plot(df.x,df.nPDF, label="",color=:red)
plot!(df.x,df.tPDF, label="",color=:blue)
vline!([-nVaR], label="Normal VaR",color=:red)
vline!([-tVaR],label="T VaR",color=:blue)
vline!([-nES], label="Normal ES",color=:red)
vline!([-tES],label="T ES",color=:blue)

#Problem 3
include("../simulate.jl")
include("../../Week04/return_calculate.jl")
portfolio = CSV.read("Project/portfolio.csv",DataFrame)
prices = CSV.read("DailyPrices.csv",DataFrame)

#filter portfolio for testing
# portfolio = portfolio[
#                 [portfolio.Stock[i] ∈ ["AAPL", "IBM"] for i in 1:length(portfolio.Stock)]
#                     ,:]

#current Prices
current_prices = prices[size(prices,1),:]


#discrete returns
returns = return_calculate(prices,dateColumn="Date")

stocks = names(returns)
intersect!(stocks,portfolio.Stock)

fittedModels = Dict{String,FittedModel}()

for stock in stocks
    fittedModels[stock] = fit_general_t(returns[!,stock])
end

#construct the copula:
#Start the data frame with the U of the SPY - we are assuming normallity for SPY
U = DataFrame()
for nm in stocks
    U[!,nm] = fittedModels[nm].u
end

R = corspearman(Matrix(U))


#what's the rank of R
evals = eigvals(R)
if min(evals...) > -1e-8
    println("Matrix is PSD")
else
    println("Matrix is not PSD")
end

#simulation
NSim = 5000
simU = DataFrame(
            #convert standard normals to U
            cdf(Normal(),
                simulate_pca(R,NSim)  #simulation the standard normals
            )   
            , stocks
        )

simulatedReturns = DataFrame()
for stock in stocks
    simulatedReturns[!,stock] = fittedModels[stock].eval(simU[!,stock])
end


#Protfolio Valuation
iteration = [i for i in 1:NSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = current_prices[values.Stock[i]]
    currentValue[i] = values.Holding[i] * price
    simulatedValue[i] = values.Holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

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
temp = aggRisk(values,Vector{Symbol}(undef,0))
temp[!,:Portfolio] .= "Total"
risk = vcat(risk,temp)

# CSV.write("problem3_out.csv", risk)