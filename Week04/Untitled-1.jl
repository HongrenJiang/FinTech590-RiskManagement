
function bsc(K, S, r, sig, T, call)
    d = Normal()
    d1 = (log(S/K) + (r + sig^2/2)*T) / (sig*sqrt(T))
    d2 = d1 - sig*sqrt(T)
    value = 0.0
    if call_put == true
        value = S*cdf(d,d1) - K*exp(-r*T)*cdf(d,d2)
    else
        value =  (K*exp(-r*T)*cdf(d,-d2) - S*cdf(d,-d1))
    end
    return value
end

Po=bsc(50,50,.001,.03,100,1)

ep = 1e-6
delta = ( bsc(50,50+ep,.001,.03,100,1) - bsc(50,50-ep,.001,.03,100,1))/(2*ep)

dpct = bsc(50,50*(1+.01),.001,.03,100,1)/Po - 1

dval = Vector{Float64}(undef,10000)
d = Normal(0,.01)
r = rand(d,10000) .+ 1.0
dval = bsc.(50,50*r,.001,.03,100,1)

dens = InterpKDE(kde(dval))
pdf(dens,8)

domain = min(dval...):.001:max(dval...)

cdens = Vector{Float64}(undef,length(domain))
cdens[1] = pdf(dens,domain[1])
for i in 2:length(domain)
    cdens[i] = cdens[i-1] + pdf(dens,domain[i])
end
cdens = cdens ./ cdens[length(domain)]
interp_linear = LinearInterpolation(domain, cdens)