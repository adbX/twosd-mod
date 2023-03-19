# Scenario generation for the transshipment problem
# 
# This show an example of how to
# discretize the random variables in Julia
# using Distributions.jl. The result can
# be used to generate the scenarios.

using Distributions

# The mean and stddev for the demand at each location
# Note that we assume independence
mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

# For each location, we discretize the levels into
# low, medium, high, using the 25, 50 and 75 quantile.
q = [
    quantile(Normal(mu[i], sigma[i]), [0.25, 0.5, 0.75])
    for i in 1:7
]

# Use the "splat" operator to treat each q_i as iterators,
# take the cartesian product and pull into a vector.
# Now this results in a 3^7-vector of tuples. Each element
# of the vector represents the demand at each location
# in that scenario.
scenarios = vec(collect(Iterators.product(q...)))

# Show the first and second scenario for example.
println(scenarios[1])
# -> (86.51, 166.27, 129.76, 136.27, 153.02, 149.76, 136.27)

println(scenarios[2])
# -> (100.0, 166.27, 129.76, 136.27, 153.02, 149.76, 136.27)
