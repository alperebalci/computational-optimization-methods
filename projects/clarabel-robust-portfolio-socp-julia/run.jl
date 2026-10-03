using JuMP
using Clarabel
using LinearAlgebra
import MathOptInterface as MOI

mu = [0.105, 0.092, 0.081, 0.067, 0.055]
L = [
    0.110  0.020  0.010  0.000  0.005
    0.000  0.095  0.025  0.010  0.005
    0.000  0.000  0.085  0.020  0.010
]
w0 = fill(0.20, length(mu))
risk_cap = 0.055
turnover_penalty = 0.006

n = length(mu)
m = size(L, 1)

model = Model(Clarabel.Optimizer)
set_silent(model)

@variable(model, w[1:n] >= 0)
@variable(model, z[1:n] >= 0)
@expression(model, factor_exposure[j = 1:m], sum(L[j, i] * w[i] for i in 1:n))

@constraint(model, sum(w) == 1.0)
@constraint(model, [risk_cap; factor_exposure] in SecondOrderCone())
@constraint(model, [i = 1:n], z[i] >= w[i] - w0[i])
@constraint(model, [i = 1:n], z[i] >= w0[i] - w[i])

@objective(
    model,
    Max,
    sum(mu[i] * w[i] for i in 1:n) - turnover_penalty * sum(z),
)

optimize!(model)

status = termination_status(model)
status == MOI.OPTIMAL || error("Clarabel did not return an optimal solution: $status")

w_star = value.(w)
portfolio_return = dot(mu, w_star)
portfolio_risk = norm(L * w_star)
turnover = sum(abs.(w_star .- w0))

println("status           = ", status)
println("objective        = ", round(objective_value(model), digits = 8))
println("expected return  = ", round(portfolio_return, digits = 8))
println("risk             = ", round(portfolio_risk, digits = 8))
println("turnover         = ", round(turnover, digits = 8))
println("weights          = ", round.(w_star, digits = 6))
