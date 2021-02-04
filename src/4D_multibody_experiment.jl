using Distributed
using NPZ
using ProgressMeter
addprocs(5)

@everywhere begin
    include("common.jl")

    using StatsBase
    using JLD
    using ProgressBars
    
    function combine_sampling(
            grid_points, discrete_pdf_1, discrete_pdf_2, num_samples)
        
        num_samples = floor(Int, num_samples)
       
        sample_pdf_1 = StatsBase.sample(
                        grid_points, StatsBase.Weights(discrete_pdf_1), num_samples, replace=true, ordered=false)

        sample_pdf_2 = StatsBase.sample(
                grid_points, StatsBase.Weights(discrete_pdf_2), num_samples, replace=true, ordered=false)

        samples = vcat(hcat(sample_pdf_1...), hcat(sample_pdf_2...))
        
        return samples
    end

    function ise(samples)        
    
        # if grid search then these become params
        nu = 2.
        epsilon = 2.
        
        time = @elapsed reconstruction =  learn_ise_non_gaussian(samples, nu, epsilon, 4, 1e-8)
        
        return reconstruction, time
    end
end

@everywhere sample_list = [floor(Int, x) for x in [10^3, 10^3.5, 10^4, 10^4.5, 10^5, 10^5.5, 10^6]]

# Combine two independent 2D Models to make a 4D Model
# Data for Figures 10 and 11 

reconstruction_error = []
reconstruction_times = []

discrete_pdf_1 = load("../data/2d_pdf_1.jld")["pdf"]
discrete_pdf_2 = load("../data/2d_pdf_2.jld")["pdf"]

factor_graph = load("../data/combined_4d_factor_graph.jld")["factor_graph"]

lower_bounds= [-2, -2, -2, -2]
upper_bounds = [2., 2., 2., 2.]

grid_points = create_grid(2, 5000, lower_bounds, upper_bounds)

f(x) = get_reconstruction_error(factor_graph, x, false)

iters = range(1, 45, step=1)

for num_samples in sample_list
    
    samples = repeat([zeros(4, num_samples)], length(iters))
    
    p = Progress(length(iters))

    Threads.@threads for i in iters
        samples[i] = combine_sampling(
                grid_points,  discrete_pdf_1, discrete_pdf_2, num_samples)
        next!(p)
    end

    println("Sampling Complete!")
    
    println("Reconstructing")
    
    results = pmap(ise, samples)
    reconstructions = [x[1] for x in results]
    times = [x[2] for x in results]
    errors = f.(reconstructions)
    mean_elem_error = mean(errors)/length(reconstructions[1])
    mean_time = mean(times)
    push!(reconstruction_times, mean_time)
    push!(reconstruction_error, mean_elem_error)
    
    println("Mean time for ", num_samples, " = ", mean_time)
    println("Mean error for ", num_samples, " = ", mean_elem_error)
end


