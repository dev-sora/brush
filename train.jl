using Flux
using LinearAlgebra
using Statistics
using Flux:
    onehotbatch,
    @epochs,
    train!,
    crossentropy,
    throttle,
    onecold,
    onehotbatch,
    mse
using Flux.Data: DataLoader

using Random
function partitionTrainTest(x, y, at = 0.7)
    n = size(x)[4]
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n)+1):n)
    x[:, :, :, train_idx], x[:, :, :, test_idx], y[:, train_idx], y[:, test_idx]
end

# Data Loading
f = open("trainData.txt","r")
Acce = Array{Float64}(undef, 3, 0)
Id = Vector{Int}(undef,0)
for line in eachline(f)
    x,y,z,id = parse.(Float64, split(line))
    id = Int(id)
    Acce = hcat(Acce, [x, y, z])
    Id = vcat(Id, id)
end

len = length(Id)

# Dataset Setting
x = Array{Float64}(undef, (3, 10, 1, len-9))
y = Array{Int}(undef, (8, len-9))
for i = 1:len-9
    if all(a -> a==Id[i], Id[i:i+9])
        block = Acce[:,i:i+9]
        x[:, :, :, i] = reshape(block, (3,10,1))
        y[:, i] = onehotbatch(Id[i], 1:8)
    end
end

train_x, test_x, train_y, test_y = partitionTrainTest(x, y)
train_data = DataLoader((train_x, train_y), batchsize = 32)

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

loss(x,y)= crossentropy(model(x),y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
optimizer = ADAM()
cb_eval = () -> @show(accuracy(test_x, test_y))
@epochs 500 train!(loss, params(model), train_data, optimizer, cb=throttle(cb_eval,10))

weights = params(model)
using BSON: @save
@save "model.bson" weights