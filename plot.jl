using Plots
using LinearAlgebra
using DelimitedFiles

acce = readdlm("trainData.txt", ' ', Float64, '\n')

plot(acce[:,3])