import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Generate true free-fall data with drag
g = 9.8        # gravitational acceleration (m/s^2)
m = 1.0        # mass (kg)
k = 0.5        # drag coefficient (kg/s)
mu = k / m     # simplified constant

def freefall(t, u):
    y, v = u[0], u[1]
    dy_dt = v
    dv_dt = g - mu * v
    return torch.stack([dy_dt, dv_dt])

u0 = torch.tensor([0.0, 0.0])
t = torch.linspace(0, 5, 100)
true_solution = odeint(freefall, u0, t)  # shape: (100, 2)
y_data = true_solution[:, 0]             # extract position data

# Define Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
    
    def forward(self, t, u):
        return self.net(u)

odefunc = ODEFunc()

def predict_positions(model):
    sol = odeint(model, u0, t)
    return sol[:, 0]

def loss_fn(model):
    y_pred = predict_positions(model)
    return torch.sum((y_pred - y_data) ** 2)

# Train Neural ODE model
optimizer = optim.Adam(odefunc.parameters(), lr=0.05)
num_epochs = 300
loss_list = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss_val = loss_fn(odefunc)
    loss_val.backward()
    optimizer.step()
    
    loss_list.append(loss_val.item())
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, loss = {loss_val.item():.6f}")

trained_positions = predict_positions(odefunc)
final_loss = loss_fn(odefunc).item()
print("Final training loss:", final_loss)
print("True vs Predicted positions at t=5s:", y_data[-1].item(), "vs", trained_positions[-1].item())

# Plot true vs predicted positions and training loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t.numpy(), y_data.numpy(), 'b-', label='True positions')
plt.plot(t.numpy(), trained_positions.detach().numpy(), 'r--', label='Predicted positions')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Freefall Dynamics: True vs Predicted")
plt.legend(loc='upper left')

true_val = y_data[-1].item()
pred_val = trained_positions[-1].item()
error_val = abs((true_val - pred_val) / true_val)
annotation_text = (f"At t=5s:\nTrue = {true_val:.2f}\nPredicted = {pred_val:.2f}\n"
                   f"Error = {error_val*100:.2f}%")
plt.text(0.95, 0.2, annotation_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8))

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), loss_list, 'g-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.tight_layout()

plt.show()