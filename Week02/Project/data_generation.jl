using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using JuMP
using Ipopt
using SpecialFunctions

#save for week 3

n = Normal(0,1)
t = TDist(5)
mv = MvNormal([0.0,0.0], [[1.0, .5] [.5, 1.0]])
sim = rand(mv,100)
x = sim[1,:]
y = quantile.(t,cdf(n,sim[2,:]))


Plots.plot(x,y, seriestype=:scatter)
cor(x,y)
corspearman(x,y)

#Problem 1
# Compare 
mv = MvNormal([0.0,0.0], [[1.0, .5] [.5, 1.0]])
# sim = rand(mv,100)
# df = DataFrame(:x => sim[1,:], :y=>sim[2,:])
# CSV.write("Project/problem1.csv",df)

prob1 = CSV.read("Project/problem1.csv",DataFrame)

mu = mean.(eachcol(prob1))
sigma = cov(Matrix(prob1))

n = size(prob1,1)
X = [ones(n) prob1.x]
Y = prob1.y

B = inv(X'*X)*X'*Y
e = Y - X*B
s2_e = e'*e/(n-1)

#E(y|x=?)
e_y = B[1] + 1.0*B[2]


#MVN 
y_hat = mu[2] + sigma[1,2]*(1.0-mu[1])/sigma[1,1]
s2_y = sigma[2,2] - (sigma[1,2]^2)/sigma[1,1]



#problem 2
n = Normal(0,1)
t = TDist(5)
mv = MvNormal([0.0,0.0], [[1.0, .5] [.5, 1.0]])
sim = rand(mv,100)
x = sim[1,:]
y = quantile.(t,cdf(n,sim[2,:]))
# df = DataFrame(:x => x, :y=>y)
# CSV.write("Project/problem2.csv",df)

prob2 = CSV.read("Project/problem2.csv",DataFrame)

Y = prob2.y
X = [ones(100) prob2.x]

B = inv(X'*X)*X'*Y
e = Y - X*B
kurtosis(e)

function normal_ll(s, b...)
    n = size(Y,1)
    beta = collect(b)
    xm = Y - X*beta
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - xm'*xm/(2*s2)
    return ll
end

function __T_loglikelihood(mu,s,nu,x)
    n = size(x,1)
    np12 = (nu + 1.0)/2.0

    mess = loggamma(np12) - loggamma(nu/2.0) - log(sqrt(π*nu)*s)
    xm = ((x .- mu)./s).^2 * (1/nu) .+ 1
    innerSum = sum(log.(xm))
    ll = n*mess - np12*innerSum
    return ll
end

function t_ll(s,nu, b...)
    td = TDist(nu)
    beta = collect(b)
    xm = (Y - X*beta)

    ll = __T_loglikelihood(0.0,s,nu,xm)
    return ll
end

mle = Model(Ipopt.Optimizer)
set_silent(mle)

@variable(mle, beta[i=1:2],start=0)
@variable(mle, σ >= 0.0, start = 1.0)
@variable(mle, ν >= 0.0, start = 10.0)

register(mle,:normLL,3,normal_ll;autodiff=true)

register(mle,:tLL,4,t_ll;autodiff=true)

@NLobjective(
    mle,
    Max,
    normLL(σ,beta...)
)
optimize!(mle)
normal_beta = value.(beta)
normal_s = value(σ)
normalLL = normal_ll(normal_s,normal_beta...)
nAIC = 6 - 2*normalLL
println("Normal Betas: ", normal_beta)
println("Normal S: ", normal_s)
println("Normal LL:", normalLL)
println("Normal AIC: ",nAIC)



mle = Model(Ipopt.Optimizer)
set_silent(mle)

@variable(mle, beta[i=1:2],start=0)
@variable(mle, σ >= 0.0, start = 1.0)
@variable(mle, 100 >= ν >= 3.0, start = 10.0)

register(mle,:tLL,4,t_ll;autodiff=true)

@NLobjective(
    mle,
    Max,
    tLL(σ,ν, beta...)
)

optimize!(mle)
t_beta = value.(beta)
t_s = value(σ)
t_nu = value(ν)
tLL = t_ll(t_s, t_nu, t_beta...)
tAIC = 8 - 2*tLL
println("T Betas: ", t_beta)
println("T S: ", t_s)
println("T df: ", t_nu)
println("T LL:", tLL)
println("T AIC: ", tAIC)


#Problem 3









