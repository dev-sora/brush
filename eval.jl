using Flux: reshape
using Flux
using Zygote
using BSON: @load

# NN Setting
model = Chain(
    Conv((2,2), 1=>8, relu, pad=1, stride=1),
    MaxPool((2,2)),
    Conv((2,2), 8=>16, relu, pad=1, stride=1),
    MaxPool((2,2)),
    Conv((2,2), 16=>32, relu, pad=1, stride=1),
    MaxPool((2,2)),
    x->reshape(x, :, size(x, 4)),
    Dense(64,8),
    softmax
)

@load "model.bson" weights
Flux.loadparams!(model, weights)

# Data Loading
f = open("evalData.txt","r")
Acce = Array{Float64}(undef, 3, 0)
for line in eachline(f)
    x,y,z = parse.(Float64, split(line))
    Acce = hcat(Acce, [x, y, z])
end

area = []
for idx in 1:11:length(Acce[1,:])-9
    x = reshape(Acce[:,idx:idx+9], (3,10,1,1))
    y = model(x)
    push!(area, argmax(y)[1])
end

areaName = ["左上(前)","右上(前)","左下(前)","右下(前)","左上(裏)","右上(裏)","左下(裏)","右下(裏)"]

len = length(area)
ratio = []
for idx=1:8
    push!(ratio, count(i->i==idx, area)/len)
end

display(ratio)
mostUnbrushed = areaName[argmin(ratio)]
ratio[argmin(ratio)] = Inf
secondUnbrushed = areaName[argmin(ratio)]

println(mostUnbrushed, "と", secondUnbrushed, "が他の部分に比べて磨けていないようです。")