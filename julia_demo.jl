using Flux, DifferentialEquations, DiffEqFlux, Plots, Printf

# Generate true free-fall data with drag
g = 9.8         # gravitational acceleration (m/s^2)
m = 1.0         # mass (kg)
k = 0.5         # drag coefficient (kg/s)
μ = k / m       # simplified constant

function freefall!(du, u, p, t)
    du[1] = u[2]            # dy/dt = v
    du[2] = g - μ * u[2]      # dv/dt = g - μ*v
end

u0 = [0.0, 0.0]
tspan = (0.0, 5.0)
t = range(0.0, stop=5.0, length=100)

# Solve the ODE to generate true position data
prob = ODEProblem(freefall!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=t)
y_data = sol[1, :]           # extract position data

# Define Neural ODE model using Flux and DiffEqFlux
nn_model = Chain(Dense(2, 16, tanh), Dense(16, 2))
n_ode = NeuralODE(nn_model, tspan, Tsit5(), saveat=t)

# Prediction and loss functions
predict_positions() = n_ode(u0)[1, :]    # predicted positions from NeuralODE
loss_fn() = sum((predict_positions() .- y_data).^2)

# Train the NeuralODE model
opt = ADAM(0.05)
num_epochs = 300
loss_list = Float64[]

for epoch in 1:num_epochs
    grads = gradient(Flux.params(nn_model)) do
        loss_fn()
    end
    Flux.Optimise.update!(opt, Flux.params(nn_model), grads)
    push!(loss_list, loss_fn())
    if epoch % 50 == 0
        @printf("Epoch %d, loss = %.6f\n", epoch, loss_fn())
    end
end

trained_positions = predict_positions()
final_loss = loss_fn()
println("Final training loss: ", final_loss)
println("True vs Predicted positions at t=5s: ", y_data[end], " vs ", trained_positions[end])

# Plot true vs predicted positions and training loss
p1 = plot(t, y_data, lw=2, label="True positions", color=:blue)
plot!(p1, t, trained_positions, lw=2, ls=:dash, label="Predicted positions", color=:red)
xlabel!(p1, "Time (s)")
ylabel!(p1, "Position (m)")
title!(p1, "Freefall Dynamics: True vs Predicted")

true_val = y_data[end]
pred_val = trained_positions[end]
error_val = abs((true_val - pred_val) / true_val)
annot_text = @sprintf("At t=5s:\nTrue = %.2f\nPredicted = %.2f\nError = %.2f%%", true_val, pred_val, error_val*100)
annotate!(p1, 5, maximum(y_data)*0.2, text(annot_text, 10))

p2 = plot(1:num_epochs, loss_list, lw=2, label="Training Loss", color=:green)
xlabel!(p2, "Epoch")
ylabel!(p2, "Loss")
title!(p2, "Training Loss Curve")

plot(p1, p2, layout=(1,2), size=(1200,500))