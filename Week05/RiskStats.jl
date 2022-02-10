
function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end

function ES(a; alpha=0.05)
    x = sort(a)
    v = -VaR(a,alpha=alpha)
    
    es = mean(x[x.<=v])
    return -es
end
