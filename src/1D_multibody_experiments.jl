using Distributed
using NPZ
using ProgressMeter
addprocs(2)

@everywhere begin
    include("common.jl")

    using StatsBase
    using JLD
    using ProgressBars
    using Plots

    function ise(samples)
        """
        Utility function to return reconstruction and time
        """
    
        # if grid search then these become params
        nu = 2.
        epsilon = 2.
        
        time = @elapsed reconstruction =  learn_ise_non_gaussian(
            samples, nu, epsilon, 1e-7)
        
        return reconstruction, time
    end
    
    function pl(samples)
        """
        Utility function to return reconstruction and time
        """
        time = @elapsed reconstruction = learn_pseudolikelihood(samples, 4, 2.3)
        return reconstruction, time
    end
    
    function energy_function_1D(x, interactions_dict)
        """
        Function to evaluate multi-body energy function 
        """
        term = 0.
        for (key, interaction) in interactions_dict
            term += prod([interaction; [x[1] for i in 1:length(key)]])
        end

        val = exp(-term)
        return val
    end
    
    function sample_from_pdf(
            grid_points, discrete_pdf, num_samples)
        """
        Sample num_samples samples from grid using probability density function
        """

            num_samples = floor(Int, num_samples)

            sample_pdf = StatsBase.sample(
                grid_points, StatsBase.Weights(discrete_pdf),
                num_samples, replace=true, ordered=false)

            samples = hcat(sample_pdf...)

            return samples
    end
end

@everywhere sample_list = [floor(Int, x) for x in [10^3, 10^3.5, 10^4, 10^4.5, 10^5, 10^5.5, 10^6]]

reconstruction_error = []
reconstruction_times = []
pl_err = []
pl_times = []

factor_graph = load(
    "../data/1d_factor_graph.jld")["factor_graph"]

lower_bounds= [-2.1]
upper_bounds = [2.1]

# generate pdf to sample from

grid_points = create_grid(1, 5000, lower_bounds, upper_bounds)

to_sum = []
for x in grid_points
    push!(to_sum, exp_energy_function(x, factor_graph))
end

discrete_pdf = to_sum/sum(to_sum)

iters = range(1, 45, step=1)

f(x) = get_reconstruction_error(factor_graph, x, false)

## Figure 7 and Figure 8

for num_samples in sample_list
    
    samples = repeat([zeros(1, num_samples)], length(iters))
    
    p = Progress(length(iters))

    Threads.@threads for i in iters
        sample_pdf = StatsBase.sample(
            grid_points, StatsBase.Weights(discrete_pdf),
            num_samples, replace=true, ordered=false)
        samples[i] = hcat(sample_pdf...)
        next!(p)
    end
    
    pl_results = pmap(pl, samples)
    pl_reconstructions = [x[1] for x in pl_results]
    pl_times = [x[2] for x in pl_results]
    pl_errors = f.(pl_reconstructions)
    
    results = pmap(ise, samples)
    reconstructions = [x[1] for x in results]
    times = [x[2] for x in results]
    errors = f.(reconstructions)
    
    mean_elem_error = mean(errors)/length(reconstructions[1])
    mean_time = mean(times)
    pl_mean_elem_error = mean(pl_errors)/length(reconstructions[1])
    pl_mean_time = mean(pl_times)
    push!(reconstruction_times, mean_time)
    push!(reconstruction_error, mean_elem_error)
    push!(pl_times, pl_time)
    push!(pl_error, pl_mean_elem_error)
    
    println("Mean time for ", num_samples, " = ", mean_time)
    println("Mean error for ", num_samples, " = ", mean_elem_error)
end

# Interaction order runtime ratio --> Figure 9
# NOTE: bounds may have to be adjusted for a given factor graph, especially as order increases.

pl = []
is = []

for order in 3:9

    fg = get_factor_graph(1, order)
    
    f(x) = energy_function_1D(x, fg)
    
    bound = 2.1

    grid_points = create_grid(1, Int64(1e6), [-bound], [bound])

    to_sum = []

    for x in tqdm(grid_points)
        push!(to_sum, energy_function_1D(x, fg))
    end

    discrete_pdf = to_sum/sum(to_sum)

    samples = sample_from_pdf(grid_points, discrete_pdf, 1000)

    pl_time = @elapsed reconstruction = learn_pseudolikelihood(
        samples, order, bound)
    
    println("PL time = ", pl_time)
    
    push!(pl, pl_time)
    
    is_time = @elapsed reconstruction = learn_ise_non_gaussian(
        samples, 0.5, 2.0, order, 1e-7)
    
    println("IS time = ", is_time)
    
    push!(is, is_time)
end

