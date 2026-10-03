using JuMP
using COSMO
using LinearAlgebra

C = [
     1.00   0.92   0.78   0.55
     0.92   1.00  -0.82   0.48
     0.78  -0.82   1.00   0.88
     0.55   0.48   0.88   1.00
]

n = size(C, 1)
model = Model(COSMO.Optimizer)
set_silent(model)

@variable(model, X[1:n, 1:n], Symmetric)
@constraint(model, X in PSDCone())
@constraint(model, [i = 1:n], X[i, i] == 1.0)
@objective(model, Min, sum((X[i, j] - C[i, j])^2 for i in 1:n, j in 1:n))

optimize!(model)

status = termination_status(model)
status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
    error("COSMO did not return an optimal solution: $status")

X_star = value.(X)
lambda_min = eigmin(Symmetric(X_star))

println("status            = ", status)
println("objective         = ", round(objective_value(model), digits = 8))
println("minimum eigenvalue= ", round(lambda_min, digits = 10))
println("repaired matrix:")
display(round.(X_star, digits = 5))
