using CPLEX, Distributions, JuMP

# Share CPLEX environment to speed up
const cpx = CPLEX.Optimizer()

# Cost coefficients
const h = 1.0
const c = 0.5
const p = 4.0

# Distribution of the demand
const mu = [100.0, 200, 150, 170, 180, 170, 170]
const sigma = [20.0, 50, 30, 50, 40, 30, 50]

const d_dist = product_distribution(truncated.(Normal.(mu, sigma), mu - 3 * sigma, mu + 3 * sigma))

function build_subproblem(s::Vector{Float64}, d::Vector{Float64})
    """
    Constructs the dual of transshipment subproblem given the first stage
    decision (order-up-to quantity) and a realization of random variable
    (demand)
    """

    # Sanity check: length of s and d should match.
    @assert(length(s) == length(d))
    
    N = length(s)

    # Bind the new problem
    model = direct_model(CPLEX.Optimizer(cpx.env))

    # Clear the model 
    # empty!(model)

    # Add variables
    @variables(model, begin
    B[1:N]
    M[1:N]
    R
    E[1:N]
    end)

    # Add constraints
    @constraint(model, con1[i=1:N], B[i] + E[i] <= h)
    @constraint(model, con2[i=1:N], B[i] + M[i] <= 0)
    @constraint(model, con3[(i, j) = ((i, j) for i in 1:N for j in 1:N if i != j)], B[i] + M[j] <= c)
    @constraint(model, con4[i=1:N], M[i] + R <= p)
    @constraint(model, con5[i=1:N], R + E[i] <= 0)

    # Objective 
    @objective(model, Max, sum(s[i] * B[i] + d[i] * M[i] + d[i] * R + s[i] * E[i] for i in 1:N))

    return model
end

function get_dtc(s::Vector{Float64}, d::Vector{Float64})
    """
    Solve one instance of the dual model given first stage decision
    and the realized demand, returning the
    named tuple (objective_value, subgradient).
    """
    model = build_subproblem(s, d)
    # Suppress output to avoid flooding the screen
    set_silent(model)

    optimize!(model)
    
    if termination_status(model) != OPTIMAL
        @warn("Instance Not Optimal s=$s, d=$d")
    end

    dtc = value.(model[:B] + model[:E])
    obj = objective_value(model)

    return (objective_value = obj, subgradient = dtc)
end

using Random
# Seed the Random number generator
const rng = MersenneTwister(1234)

function get_average_dtc(s::Vector{Float64}, M = 10)
    """
    Solve M instances given first stage decision.
    Returning the objective value estimates and subgradient estimates.
    """
    sum_dtc = zeros(length(s))
    sum_obj = 0.0

    for i in 1:M
        # Generate a random demand, call get_dtc
        d = rand(rng, d_dist)
        obj, g = get_dtc(s, d)

        # objective value
        sum_obj += obj

        # subgrad estimate
        sum_dtc .= sum_dtc + g
    end

    # Return 1/M * sum(obj), 1/M * sum(grad)
    return (objective_value = sum_obj / M, subgradient = sum_dtc / M)
end


function work(repl)
    """
    Run stoch approx algorithm with replication number repl.
    """
    open("output_stoch$repl.txt", "w") do io
        s = copy(mu)
        max_iter = 3000
        for k in 1:max_iter
            # Mean objective and Subgradient estimate
            obj, grad = get_average_dtc(s, 1000)

            alpha = 1000 / k
            s .= s - alpha * grad

            if k % 10 == 0
                write(io, "Iteration $k: obj = $obj, grad = $grad, s = $s\n")
            end
        end
    end
end
# get_average_dtc(zeros(7), 10)

# Run 4 replications
for rep in 1:4
    work(rep)
end