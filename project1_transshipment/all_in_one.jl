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
D = vec(collect(Iterators.product(q...)))

SC = length(D)
N = length(mu)

using JuMP, CPLEX
model = direct_model(CPLEX.Optimizer())

@variables(model, begin
    s[1:N] >= 0
    e[1:SC, 1:N] >= 0
    f[1:SC, 1:N] >= 0
    q[1:SC, 1:N] >= 0
    r[1:SC, 1:N] >= 0
    t[1:SC, 1:N, 1:N] >= 0
end)

@constraint(model, con1[sc=1:SC, i=1:N],
    f[sc, i] + sum(t[sc, i, j] for j in 1:N if j != i) + e[sc, i] == s[i]
)

@constraint(model, con2[sc=1:SC, i=1:N],
    f[sc, i] + sum(t[sc, j, i] for j in 1:N if j != i) + r[sc, i] == D[sc][i]
)

@constraint(model, con3[sc=1:SC], sum(r[sc, :]) + sum(q[sc, :]) == sum(D[sc]))

@constraint(model, con4[sc=1:SC, i=1:N], e[sc, i] + q[sc, i] == s[i])

h = 1.0
c = 0.5
p = 4.0
# We set t_ii = 0 to avoid issues. Implicitly this should set t_ii = 0
@objective(model, Min, 1.0/SC*(sum(h*e) + sum(c*t) + sum(p*r)))

@info "Start Solving"
optimize!(model)

@show termination_status(model)

@show objective_value(model)
@show value.(s)
