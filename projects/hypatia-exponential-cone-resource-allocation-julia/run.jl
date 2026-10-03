using JuMP
using Hypatia
using LinearAlgebra
import MathOptInterface as MOI

alpha = [1.40, 1.15, 0.90, 0.70]
cost = [4.0, 3.2, 2.2, 1.4]
budget = 8.0
capacity = 2.8
lower = 0.05

n = length(alpha)
model = Model(Hypatia.Optimizer)
set_silent(model)

@variable(model, x[1:n] >= lower)
@variable(model, t[1:n])

@constraint(model, sum(cost[i] * x[i] for i in 1:n) <= budget)
@constraint(model, sum(x) <= capacity)
@constraint(model, [i = 1:n], [t[i], 1.0, x[i]] in MOI.ExponentialCone())

@objective(model, Max, sum(alpha[i] * t[i] for i in 1:n))

optimize!(model)

status = termination_status(model)
status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
    error("Hypatia did not return an optimal solution: $status")

x_star = value.(x)
weighted_log_utility = sum(alpha .* log.(x_star))

println("status               = ", status)
println("objective            = ", round(objective_value(model), digits = 8))
println("weighted log utility = ", round(weighted_log_utility, digits = 8))
println("budget used          = ", round(dot(cost, x_star), digits = 8))
println("capacity used        = ", round(sum(x_star), digits = 8))
println("allocations          = ", round.(x_star, digits = 6))
