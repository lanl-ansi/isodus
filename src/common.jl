using StatsBase
using LinearAlgebra
using Distributions
using JLD
using NPZ
using LightGraphs
using JuMP
using Ipopt
using QuadGK
using SpecialFunctions
using Combinatorics
using DataStructures

function exp_energy_function(
        x::Array{Float64, 1},
        interactions_dict::Dict{Tuple, Float64})

    """
    x is a vector length D where D is the dimensionality of the problem.
    In this case 4.
    """

    two_body_terms = []
    three_body_terms = []
    four_body_terms = []
    
    for (key, interaction) in interactions_dict
        if length(key) == 2
            push!(two_body_terms, prod([interaction, x[key[1]], x[key[2]]]))
            
        elseif length(key) == 3
            push!(three_body_terms, prod([interaction, x[key[1]], x[key[2]], x[key[3]]]))
            
        elseif length(key) == 4
            push!(four_body_terms, prod([interaction, x[key[1]], x[key[2]], x[key[3]], x[key[4]]]))
        end
    end
    exponential_energy_func = exp(-(sum(two_body_terms)) - (sum(three_body_terms)) - (sum(four_body_terms)))

    return exponential_energy_func
                        
end


function get_reconstruction_error(factor_graph, reconstruction,
        get_elementwise::Bool)
    err = Dict()
    for (key, val) in reconstruction
        if key in collect(keys(factor_graph))
            err[key] = abs(factor_graph[key] - reconstruction[key])
        else
            err[key] = abs(reconstruction[key])
        end
    end
    if get_elementwise
        return err
    else
        return sum(collect(values(err)))
    end
end



function lasso(samples, num_dims, dim_index, coeff_l1)

    num_samples = size(samples, 2)
    sigma_emp_full = (1/num_samples)*samples*samples';

    theta_i_temp = zeros(num_dims,1);
    sigma_vec_i_temp = zeros(num_dims,1);

    sigma_emp = sigma_emp_full[
                        1:size(sigma_emp_full,1).!=dim_index,1:size(sigma_emp_full,2).!=dim_index]  

    #removing row and column i from the covariance matrix
    sigma_vec_i_temp = sigma_emp_full[:,dim_index];
    splice!(sigma_vec_i_temp, dim_index);
    sigma_vec_i = sigma_vec_i_temp;

    # remove row i from samples
    samples_tmp = samples[1:end .!= dim_index, :]

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => 0,  "tol" => 1e-7, "max_iter" => 10000))


    @variable(m, theta_i[j=1:num_dims-1], start = 0); #couplings
    @variable(m, rho[j=1:num_dims-1] >= 0); #slack variables for the regularizer

    @objective(m, Min, sum(theta_i[e]^2*sigma_emp[e,e] for e=1:num_dims-1) + 
              2*sum(theta_i[k]*theta_i[j]*sigma_emp[k,j] for k=1:num_dims-1, j=k+1:num_dims-1) + 
              2*sum(theta_i[k]*sigma_vec_i[k] for k=1:num_dims-1) +  
                                          coeff_l1*sqrt(log(num_dims)/num_samples)*sum(rho[j] for j=1:num_dims-1))
    
    if coeff_l1 > 0.
        @constraint(m, constr1[k=1:num_dims-1], theta_i[k] <= rho[k]);
        @constraint(m, constr2[k=1:num_dims-1], theta_i[k] >= -rho[k]);
    end

    status = optimize!(m);

    beta_i = vec(JuMP.value.(theta_i));

    # estimate theta_ii
    theta_ii = 1/(1/num_samples * sum(
        (samples[dim_index, k] + sum(
            beta_i[j] * samples_tmp[j, k] for j=1:num_dims-1))^2 for k in 1:num_samples))

    theta_i_vec = theta_ii .* beta_i
    insert!(theta_i_vec, dim_index, theta_ii)

    return theta_i_vec
end


function solve_lasso(params)
    samples, coeff_l1 = params
    num_dims = size(samples, 1)
    thetas = zeros(num_dims,num_dims);

    for dim_idx in 1:num_dims

        theta_i_temp = lasso(samples, num_dims, dim_idx, coeff_l1);

        thetas[dim_idx,:] = theta_i_temp;
    end

    for i in 1:size(thetas)[1]
        for j in i+1:size(thetas)[2]
            thetas[i,j] = sqrt(abs(thetas[i,j]*thetas[j,i]));
            thetas[j,i] = thetas[i,j];
        end
    end

    return thetas
end


function get_centering_term(nu::Float64, epsilon::Float64, z::Int64, k::Int64)
    """
    calculates centering term for even powered terms
    """
    f(x) = exp(-(nu*abs(x)^(z+epsilon)))
    z_i_p =  quadgk(f, -100, 100, rtol=1e-5)[1]
    return 2/(z_i_p[1]*(z + epsilon)) * nu^-((k+1)/(z+epsilon))*gamma((k+1)/(z+epsilon))
end


function interaction_screening(samples::Array{Float64,2},
    dim_index::Int64, num_dims::Int64, epsilon::Float64, nu::Float64, coeff_l1)
    """
    Function to use JuMP and Ipopt to solve for a row of the precision matrix (theta)
    """
    num_samples = size(samples, 2)

    theta_i_temp = zeros(num_dims,1);

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => 0,  "tol" => 1e-7, "max_iter" => 10000))
    
    if nu == 0
        centering_constant = 0
    else
        centering_constant = get_centering_term(
               nu, epsilon, 2, 2)
    end

    @variable(m, theta_i[j=1:num_dims], start=0); #couplings
    @variable(m, rho[j=1:num_dims] >= 0); #slack variables for the regularizer

    @NLobjective(m, Min, 1/num_samples * sum(
        exp(
            0.5*theta_i[dim_index]*(samples[dim_index, k]^2 - centering_constant) + sum(
                theta_i[j]*samples[dim_index, k]*samples[j, k] for j=1:num_dims if j!=dim_index) - nu*abs(
                samples[dim_index, k])^(2 + epsilon)) for k in 1:num_samples) +  
                                          coeff_l1*sqrt(log(num_dims)/num_samples)*sum(rho[j] for j=1:num_dims if j!=dim_index));
    
    if coeff_l1 > 0.
        

    @constraint(m, constr1[k=1:num_dims, k!= dim_index], theta_i[k] <= rho[k]);
    @constraint(m, constr2[k=1:num_dims, k!= dim_index], theta_i[k] >= -rho[k]);
        
    end
    status = optimize!(m);

    return vec(JuMP.value.(theta_i));
end


function solve_ise(params)
    samples, coeff_l1, nu, epsilon = params
    num_dims = size(samples, 1)
    thetas = zeros(num_dims,num_dims);

    for dim_idx in 1:num_dims

        theta_i_temp = interaction_screening(
            samples, dim_idx, num_dims,
            epsilon, nu, coeff_l1);

        thetas[dim_idx,:] = theta_i_temp;
    end
    
    for i in 1:size(thetas)[1]
        for j in i+1:size(thetas)[2]
            thetas[i,j] = sqrt(abs(thetas[i,j]*thetas[j,i]));
            thetas[j,i] = thetas[i,j];
        end
    end

    return thetas
end


function learn_ise_non_gaussian(
        samples, nu::Float64, epsilon::Float64, inter_order, tol)
    
    num_samples = size(samples, 2)    
    
    num_dims = size(samples, 1)

    reconstruction = Dict{Tuple,Real}()
        
    centering = Dict()
    
    for order in 1:inter_order
        if isodd(order)
            centering[order] = 0.
        else
            centering[order] = get_centering_term(
                nu, epsilon, inter_order, order)
        end
    end

    for current_spin = 1:num_dims
        
        nodal_stat = DataStructures.SortedDict{Tuple,Array{Real,1}}()
        neighbours = [i for i=1:num_dims]
        centering_coefs = DataStructures.SortedDict()
        
        for p in 2:inter_order
            idxs = Combinatorics.with_replacement_combinations(neighbours, p-1)
            terms = [tuple(x...) for x in idxs]

            nodal_keys =  [(current_spin, terms[i]...) for i=1:length(terms)]
            
            #countmap doesn't work on tuples so create second array containing arrays instead of tuples
            nodal_count =  [[current_spin, terms[i]...] for i=1:length(terms)]
            
           for index = 1:length(nodal_keys)
                variable_counts = countmap(nodal_count[index])
                current_spin_power = variable_counts[current_spin]
                other_spins = [x for x in nodal_keys[index] if x != current_spin]

                # case where x_i^4
                if isempty(other_spins)
                    nodal_stat[nodal_keys[index]] = [prod(
                          [samples[i, k] for i in nodal_keys[index]]) - centering[current_spin_power] for k=1:num_samples]

                else        

                    nodal_stat[nodal_keys[index]] = [
                        prod([samples[i, k] for i in nodal_keys[index]]) for k=1:num_samples] - [prod(
                        [samples[j, k] for j in other_spins])*centering[current_spin_power] for k=1:num_samples]
                end
            end
        end

        model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))
        
        set_optimizer_attributes(model, "tol" => tol, "max_iter" => 5000)
        
        @variable(model, x[1:length(nodal_stat)], start=0)
        
        @NLobjective(model, Min,
            log(sum(1/num_samples*exp(
                sum(x[idx]*stat[k] for (idx, (inter,stat)) in enumerate(nodal_stat)) - nu*abs(
                samples[current_spin, k])^(inter_order + epsilon)) for k in 1:num_samples)))

        JuMP.optimize!(model)
        #println(JuMP.termination_status(model))
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        
        nodal_reconstruction = JuMP.value.(x)
        for (idx, inter) in enumerate(keys(nodal_stat))
            reconstruction[inter] = deepcopy(nodal_reconstruction[idx])
        end
    end
   
    reconstruction_list = Dict{Tuple,Vector{Real}}()
    for (k,v) in reconstruction
        key = tuple(sort([i for i in k])...)
        if !haskey(reconstruction_list, key)
            reconstruction_list[key] = Vector{Real}()
        end
        push!(reconstruction_list[key], v)
    end

    symmetrised_reconstruction = Dict{Tuple,Real}()
    for (k,v) in reconstruction_list
        mean_sign = sign(mean(v))
        symmetrised_reconstruction[k] = mean_sign*exp((1/length(v))*sum(log.(abs.(v))))
        
    end
    
    return symmetrised_reconstruction

end


function create_grid(d::Integer, n::Integer, lower, upper)
    """
    creates a grid for d dimensions from lower to upper.
    """

    @assert d >= 1 ("d (number of dimensions) must be a positive integer")
    @assert n >= 2 ("n (number of points) must be a at least 2")
    
    ranges = []
    
    for i in 1:d
        push!(ranges, range(lower[i], upper[i], length=n))
    end

    iter = Iterators.product(ranges...)

    return vec([collect(i) for i in iter])
end


function pl_exp_energy_function(x, var, nodal_stat)
    
    term = 0.
    for (idx, (key, _)) in enumerate(nodal_stat)
        term += prod([var[idx]; [x[key[i]] for i in 1:length(key)]])
    end
    
    val = exp(-term)
    return val
end

function local_partition_function(
        dim_idx::Int, sample::Array{Float64}, var, nodal_stat, bound)
    """
    local partition function, need to be evaluated for each sample 
    """
    to_integrate = deepcopy(sample)
    
    #for quadgk output must be better way to do this???
    function f(x)
        to_integrate = deepcopy(sample)
        splice!(to_integrate, dim_idx, x)
        return pl_exp_energy_function(to_integrate, var, nodal_stat)
    end

    return QuadGK.quadgk(f, -bound, bound, rtol=1e-8)[1]
end


function pseudo_obj(var, nodal_stat, samples, bound)
    """
    nodal_stat is now a sorted dict
    var is an array ordered by the keys of nodal_stat (need this to pass to JuMP)
    
    """
    num_dims, num_samples = size(samples)

    log_conditional_term = [sum(var[idx]*stat[k] for (idx, (inter,stat)) in enumerate(nodal_stat)
            ) for k in 1:num_samples]

    # keys are sorted + loop through by dimension so all first elements should be equal
    dim_idx = [x[1] for x in keys(nodal_stat)][1]
    
    local_partition_term = [local_partition_function(
            dim_idx, samples[:, k], var, nodal_stat, bound) for k in 1:num_samples]
    
    return mean(log_conditional_term + log.(local_partition_term))
end



function learn_pseudolikelihood(samples, inter_order, bound)     
    
    num_samples = size(samples, 2)    
    
    num_dims = size(samples, 1)

    reconstruction = DataStructures.SortedDict{Tuple,Real}()

    for current_spin = 1:num_dims
        nodal_stat = DataStructures.SortedDict{Tuple,Array{Real,1}}()
        neighbours = [i for i=1:num_dims]
        
        for p in 2:inter_order
            idxs = Combinatorics.with_replacement_combinations(neighbours, p-1)
            terms = [tuple(x...) for x in idxs]
            nodal_keys = [(current_spin, terms[i]...) for i=1:length(terms)]

            for index = 1:length(nodal_keys)
                nodal_stat[nodal_keys[index]] =  [prod(
                      [samples[i, k] for i=nodal_keys[index]]) for k=1:num_samples]
            end
        end
        
                           
        model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))                    
        
        set_optimizer_attributes(model, "tol" => 1e-7, "max_iter" => 5000)
        
        @variable(model, x[1:length(nodal_stat)])
        

        
        obj(x...) = pseudo_obj(x, nodal_stat, samples, bound)     
        
        JuMP.register(model, :obj, length(nodal_stat),
            obj, autodiff=true)

        @NLobjective(model, Min, obj(x...))

        JuMP.optimize!(model)
        println(JuMP.termination_status(model))
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED

        nodal_reconstruction = JuMP.value.(x)
        for (idx, inter) in enumerate(keys(nodal_stat))
            reconstruction[inter] = deepcopy(nodal_reconstruction[idx])
        end
    end
   
    reconstruction_list = Dict{Tuple,Vector{Real}}()
    for (k,v) in reconstruction
        key = tuple(sort([i for i in k])...)
        if !haskey(reconstruction_list, key)
            reconstruction_list[key] = Vector{Real}()
        end
        push!(reconstruction_list[key], v)
    end

    symmetrised_reconstruction = Dict{Tuple,Real}()
    for (k,v) in reconstruction_list
        mean_sign = sign(mean(v))
        symmetrised_reconstruction[k] = mean_sign*exp((1/length(v))*sum(log.(abs.(v))))
    end
    
    return symmetrised_reconstruction

end