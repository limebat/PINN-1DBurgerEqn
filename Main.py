import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, NeuronCount):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(NeuronCount) - 1):
            self.layers.append(nn.Linear(NeuronCount[i], NeuronCount[i+1]))
        
        self.network = nn.Sequential(*self.layers)
        
    def forward(self, x): #Note: x contains both the spatial X and time T, as these are inputs
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x)) #TANH HIDDEN LAYERS
        x = self.layers[-1](x)   #FINAL LAYER, NO ACTIVATION
        return x
        
        
def data_generate(N_res):
    x_ic = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    x_res = np.random.rand(N_res, 2).astype(np.float32) * np.array([[1.0, 1.0]], dtype=np.float32)
    
    return x_ic,  x_res

def residual(model, x, v):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    t_tensor = x_tensor[:, 1]
    spatial_tensor = x_tensor[:, 0]
    u_pred = model(x_tensor)
    f_plus = 2 
    f_minus = 0 
    c = (f_plus + f_minus) / 2.0
    residual = u_pred - c + (f_plus - f_minus) / 2.0 * torch.tanh((f_plus - f_minus) / (4.0 * v) * (spatial_tensor - c*t_tensor))

    return residual
    
def analytical_solution(x, t, v):
    return 2 / (1 + np.exp((x - t) / v))
    

def IC_loss(model, x_ic_tensor):
    u_ic_pred = model(x_ic_tensor).squeeze()
    u_ic_true = torch.tensor([1.0, 0.0], dtype=torch.float32) #IC of 1, at time t=0  
    loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
    
    return loss_ic

def residual_loss(model, x_res_tensor, v):
    residual_values = residual(model, x_res_tensor, v)
    loss_residual = torch.mean(residual_values**2)
    return loss_residual

def PDE_loss(model, v):
    x_spatial = np.linspace(0, 1, 100).reshape((-1, 1))
    x_shape = np.shape(x_spatial)
    t_analytical = np.zeros(x_shape)
    u_analytical = torch.tensor(analytical_solution(x_spatial.flatten(), 0, v), dtype=torch.float32)
    xt = np.hstack((x_spatial, t_analytical))
    xt_tensor = torch.tensor(xt, dtype=torch.float32)
    u_pred_analytical = model(xt_tensor).squeeze()
    loss_PDE = torch.mean((u_pred_analytical - u_analytical)**2)
    
    return loss_PDE

def loss(model, x_ic, x_res, epoch_max, v):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        x_ic_tensor = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True)
        x_res_tensor = torch.tensor(x_res, dtype=torch.float32, requires_grad=True)
        
        loss_ic = IC_loss(model, x_ic_tensor)
        loss_residual = residual_loss(model, x_res_tensor, v)
        loss_PDE = PDE_loss(model, v)
        loss_tot = loss_ic + loss_residual + loss_PDE        
        loss_tot.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss IC: {loss_ic.item()}, Loss Residual: {loss_residual.item()}, Loss Analytical: {loss_PDE.item()}")

    return model

def plot(model):
    x = np.linspace(0, 1, 100).reshape((-1, 1))
    t = np.zeros_like(x)
    xt = np.hstack((x, t))
    u_plot = model(torch.tensor(xt, dtype=torch.float32)).detach().numpy().flatten()
    u_analytical = analytical_solution(x.flatten(), 0, v)

    plt.figure(figsize=(8, 6))
    plt.plot(x, u_plot, label='Supervised Extended PINN Prediction', color='blue')
    plt.plot(x, u_analytical, label='Analytical Solution', color='red', linestyle='--')
    plt.xlabel('x [m]')
    plt.ylabel('U [m/s]')
    plt.title('PINN Prediction vs Analytical Solution of U(x) at t=0')
    plt.grid(True)
    plt.legend()
    plt.show()
    

NeuronCount = [2, 5, 5, 5, 1]
N_ic, N_res = 2, 500
epoch_max = 4000
v = 0.1
model = PINN(NeuronCount)
x_ic, x_res = data_generate(N_res)

model = loss(model, x_ic, x_res, epoch_max, v)

plot(model)