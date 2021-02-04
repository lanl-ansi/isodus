# Code to reproduce Figures 2-4

using Distributed
using IterTools

addprocs(10)
@everywhere begin
    using JuMP
    using Ipopt
    using Statistics
    using QuadGK
    using SpecialFunctions
    using Distributions
    using LinearAlgebra
    
    include("common.jl")

end


num_dims = 10
arr = Matrix(Hermitian(rand(10, 10)))

for i in 1:num_dims
    arr[i, i] = 3
        for j in (i+1):num_dims
                arr[i,j]=arr[j,i]
        end
end

S = Matrix(Hermitian(inv(arr)))

# get distribution
dist = MvNormal(S)
nus = [2.0]
deltas = [2.0]
nsamples = [100, 500, 1000, 5000, 10000, 50000, 100000]
num_iters = 1

ise_err = []
pl_err = []

# set lambda = 0: no sparsity regularization
lambda = 0.

f(x) = sum(abs.(x - arr))/num_dims^2

for n in nsamples
    println(n)
    gaussian_samples = [rand(dist, n) for x in 1:num_iters]
    is = pmap(solve_ise, Iterators.product(gaussian_samples, [lambda], nus, deltas))
    tmp_is = [f(x) for x in dropdims(is, dims=(2, 3, 4))]
    pl = pmap(solve_lasso, Iterators.product(gaussian_samples, [lambda]))
    tmp_pl = [f(x) for x in dropdims(pl, dims=(2))]
    push!(ise_err, mean(tmp_is))
    push!(pl_err, mean(tmp_pl))
end

# Grid search
nus =range(0.01, 10.0, length=50)
epsilons = range(0.01, 3.0, length=50)
nsamples = [10^5]
iters = 80
samples = [rand(dist, n) for x in 1:num_iters]
grid_err = pmap(solve_ise, Iterators.product(samples, [0.], nus, epsilons))
mean_gridd_err = mean(grid_err, dims=(1, 2))