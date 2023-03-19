using Distributions

mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

dist = product_distribution(truncated.(Normal.(mu, sigma), mu-3*sigma, mu+3*sigma))

SC = 20000
D = rand(dist,  SC)'
#Now D is SC * N
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
    f[sc, i] + sum(t[sc, j, i] for j in 1:N if j != i) + r[sc, i] == D[sc, i]
)

@constraint(model, con3[sc=1:SC], sum(r[sc, :]) + sum(q[sc, :]) == sum(D[sc, :]))

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