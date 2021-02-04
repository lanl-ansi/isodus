using Distributed
addprocs(2)

using ProgressBars

@everywhere begin
    include("common.jl")
    using Statistics
    using Distributions
    using LinearAlgebra

    
    function get_samples(mat, num_samples)
        """
        Utility function to generate samples from a precision matrix
        """
        cov_mat = Matrix(Hermitian(inv(mat)))
        gaussian = MvNormal(cov_mat)
        return rand(gaussian, num_samples)
    end
    
    

    function threshold(M::AbstractMatrix, tol)
        for ι in eachindex(M)
            if abs(M[ι]) ≤ tol
                M[ι] = 0
            else
                M[ι] = 1
            end

        end
        return M
    end


    function get_kappa_mat(precision_mat)
        kappas = zeros(size(precision_mat))
        for i in range(1, stop=size(precision_mat)[1])
            for j in range(1, stop=size(precision_mat)[1])
                kappas[i, j] = sqrt(
                  precision_mat[i, j]*precision_mat[j, i]/(precision_mat[i, i]*precision_mat[j, j]))

            end
        end
        return kappas
    end

    
    
    function get_random_regular_mat(n::Int64, k=3, val=0.25)
        graph = LightGraphs.SimpleGraphs.random_regular_graph(n, k)
        result = convert(
            Array{Float64, 2},Matrix(LightGraphs.LinAlg.adjacency_matrix(graph)))

        result[result .==1 ].= val
        result[diagind(result)] .= 1
        check_vals = sum(result, dims=1)
        @assert sum(any.([x != (k*val+1) for x in check_vals])) == 0
        return result
    end
    
    optimal_ise(samples) = solve_ise([samples, 0.2, 2.0, 2.0])
    optimal_lasso(samples) = solve_lasso([samples, 2.3])
    
    function evaluate_structure(samples, mat, func)
        """
        Perform reconstruction, threshold at kappa/2, for these experiments
        We set kappa = 0.25. If structure is correctly recovered returns true.
        """
        reconstruction = func(samples)
        kappas = get_kappa_mat(reconstruction)
        precision_thresh = threshold(mat, 0.125)
        kappa_thresh = threshold(kappas, 0.125)
        structure_reconstructed = (precision_thresh == kappa_thresh) 
        return structure_reconstructed
    end
    
    
    function search_iteration(mat, estimate, iters, func)
        """
        A single iteration of the sequential search. Generates samples and
        evaluates whether the structure is correctly recovered.
        """
        estimate = Int64(floor(estimate))
        samples = [get_samples(mat, estimate) for i in 1:iters]
        h(x) = evaluate_structure(x, mat, func)
        results = pmap(h, samples)
        success = (sum(results .== 1) == iters)
        return success
    end
    
end

function sequential_search(mat, estimate, iters, search_func)
    """
    For a given estimate (number of samples), attempt to recover the structure 45 times. If unsuccessful increment the estimate by +25 until successful recovery.
    If the structure is successfully recovered, decrease the estimate by 10 until recovery is no longer successful and return this value.
    """
        current_estimate = estimate
        complete = false
        decreasing_flag = false
        previous_estimate = 0
        while complete != true
            is_success = search_iteration(mat, current_estimate, iters, search_func)

            if is_success == true 
                println("Success for ", current_estimate, ", decreasing M*")
                decreasing_flag = true
                previous_estimate = current_estimate
                current_estimate = current_estimate-10

            else
                if decreasing_flag == true
                    println("Returning last succesful value of M* = ", previous_estimate)
                    return previous_estimate
                    complete=true
                end

                current_estimate = current_estimate+25
                println("Increasing M* to ", current_estimate)
            end
        end    

end

# lasso ise lambda sweep --> Figure 5
num_dims = 100
f(x) = sum(abs.(x - mat))/num_dims^2
precision_mat = get_random_regular_mat(num_dims, 3, 0.25)
samples = [get_samples(precision_mat, 10000) for x in range(1, stop=1)]
ise_coeffs = range(0, stop=0.5, step=0.01)
lasso_coeffs = range(0, stop=5, step=0.1)
lasso_sweep = pmap(solve_lasso, Iterators.product(samples, lasso_coeffs))
ise_sweep = pmap(solve_ise, Iterators.product(samples, ise_coeffs, [2.0], [2.0]))


# n* search --> Figure 6.
ise_sequential_iters = []
for i in 1:5
    sequential_vals = []
    estimate = 900
    for p in tqdm(range(10, stop=100, step=10))
        mat = get_random_regular_mat(p)
        m = sequential_search(mat, estimate, 45, optimal_ise)
        estimate = m
        push!(sequential_vals, m)
    end
    push!(ise_sequential_iters, sequential_vals)
end

lasso_sequential_iters = []
for i in 1:5
    sequential_vals = []
    estimate = 900
    for p in tqdm(range(10, stop=100, step=10))
        mat = get_random_regular_mat(p)
        m = sequential_search(mat, estimate, 45, optimal_ise)
        estimate = m
        push!(sequential_vals, m)
    end
    push!(sequential_iters, sequential_vals)
end