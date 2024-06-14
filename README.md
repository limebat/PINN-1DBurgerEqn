# Global Objective {#global-objective .unnumbered}

The prediction and simulation of fluid partial differential equations
(PDEs) are inherently challenging due to the nonlinear nature of the
simulations and their design. In this study, we demonstrate the
application of PINNs to the 1D Burgers' PDE to further explore the topic
of neural networks in the context of physics-informed predictions.

Direct solving of the PDE is compared and given to the resolution
methods of lines proposed by Biazar et al. [@BiazarBERGER]. The accuracy
and efficiency are directly compared to the feasibility and prediction
of the PINN [@MENG2023116172]. The direct solution is compared and
estimated from the residuals of the epochs of the machine learning
process, and the data is feed-forwarded from the PDE residuals to
epochs. The final prediction for the 1D Burgers' equation is solved and
compared to the exact solution.

<!--
![Equation](https://latex.codecogs.com/png.latex?\frac{\partial%20u}{\partial%20t}%20+%20u\frac{\partial%20u}{\partial%20x}%20=%20\nu%20\frac{\partial^20u}{\partial%20x^2})
-->

![image](https://github.com/limebat/PINN-1DBurgerEqn/assets/86577233/3281351c-d16b-4d38-a9ab-6983543f5b2b)


Derived by Harry Bateman [@harryBateman1915], the solution for $f^{+}=2, f^{-}=0, c=1$ results to the following analytical
solution to the PDE.

![Equation](https://latex.codecogs.com/png.latex?u(x,t)%20=%20\frac{2}{1+e^{\frac{x-t}{\nu}}})

# Pre-processing strategem

## Standardization

Standardize the distribution of data to a mean of zero and standard
deviation of 1. Improves the model learning speed, alongside
distributing equal importance to all features.

## Normalization

Normalize the data between 0 and 1, allowing the features to not
disproportionately affect the model in bias.

## Sampling Strategies

Uniform grid is used to avoid sensitivity comparisons of non-uniform
bias grids.

# Models

## Standard unsupervised PINN

In unsupervised learning, the CNN of the model is based on PDE's without
labeled data (initial and boundary conditions for CFD applications). The
loss is only derived from the residuals to base a solution on learned
behaviors, eg. input-output pairs.

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{PDE}}%20=%20\frac{1}{M}%20\sum_{j=1}^{M}%20\left(%20u_t(x_j,%20t_j)%20+%20u(x_j,%20t_j)%20u_x(x_j,%20t_j)%20-%20\nu%20u_{xx}(x_j,%20t_j)%20\right)^2)

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}%20=%20\mathcal{L}_{\text{PDE}})

## Standard supervised PINN

In supervised learning, the CNN of the model is based on labeled data,
and when coupled with the residual function, the model will fit and form
properly.

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{data}}%20=%20\frac{1}{N}%20\sum_{i=1}^{N}%20\left(%20u_{\text{pred}}(x_i,%20t_i)%20-%20u_{\text{data}}(x_i,%20t_i)%20\right)^2)
K is the number of boundary condition points, which are in the
beginning and specified. M is the number of collocation points specified
for the PDE. N is the number of data points

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}%20=%20\mathcal{L}_{\text{data}}%20+%20\mathcal{L}_{\text{PDE}})

## Supervised Extended PINN

The loss function is a combination of the data loss (difference between
predictions and labeled data) and the PDE residuals:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{BC}}%20=%20\frac{1}{K}%20\sum_{k=1}^{K}%20\left(%20u_{\text{pred}}(x_{b_k},%20t_{b_k})%20-%20u_{b_k}%20\right)^2)

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}%20=%20\mathcal{L}_{\text{data}}%20+%20\mathcal{L}_{\text{PDE}}%20+%20\mathcal{L}_{\text{BC}})

# Validation and Verification

Comparison with Analytical Solutions - Available from Harry Bateman
[@harryBateman1915].

Sampling and Convergence Analysis - Collocation is important to capture
regions of high complexity such as regions where significant changes or
features in the solution occur.

Sensitivity Analysis - With regards to PINN this includes how number of
layers and neurons effect the accuracy and convergence rate.

# Expected Results

The expected results of using a PINN to solve the Burgers equation are
as follows: Training should be smooth, showing the model is learning
well. The model should focus more on complex areas without needing too
much computation. The model handles variations in initial and boundary
conditions, balancing accuracy and computation.

# Goals

**1. Analysis of Burgers Equation Using PINN:** Analyze and reconstruct
the flow field of the Burgers equation using standard supervised, supervised extended, and
standard unsupervised PINNs.

**2. Accurate Solution Capture:** Ensure the PINN model accurately
replicates the analytical solutions of the Burgers equation

**3. Smooth Convergence:** Achieve smooth and efficient training of the
model, indicating effective learning and proper convergence to the
solution.

**4. Robustness to Variations:** Ensure the model remains accurate even
with uncertainties in initial and boundary conditions

**5. Optimized Training:** Balance accuracy and computational efficiency
by optimizing the network architecture and training parameters.

**6. Visualization and Comparison:** Visualize the flowfield solution,
comparing it with analytical solutions to differences within numerical
simulation.
