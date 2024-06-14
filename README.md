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

$\begin{gathered}
    {\displaystyle {\frac {\partial u}{\partial t}}+u{\frac {\partial u}{\partial x}}=\nu {\frac {\partial ^{2}u}{\partial x^{2}}}.}
\end{gathered}$

$$\begin{gathered}
    {\displaystyle {\frac {\partial u}{\partial t}}+u{\frac {\partial u}{\partial x}}=\nu {\frac {\partial ^{2}u}{\partial x^{2}}}.}
\end{gathered}$$ Derived by Harry Bateman [@harryBateman1915], the
solution for $f^{+}=2, f^{-}=0, c=1$ results to the following analytical
solution to the PDE.

$${\displaystyle u(x,t)={\frac {2}{1+e^{\frac {x-t}{\nu }}}}}$$

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
\[mathcal{L}_{\text{PDE}} = \frac{1}{M} \sum_{j=1}^{M} \left( u_t(x_j, t_j) + u(x_j, t_j) u_x(x_j, t_j) - \nu u_{xx}(x_j, t_j) \right)^2$$
$$\mathcal{L} = \mathcal{L}_{\text{PDE}}]\

##  Standard supervised PINN

In supervised learning, the CNN of the model is based on labeled data,
and when coupled with the residual function, the model will fit and form
properly.

$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left( u_{\text{pred}}(x_i, t_i) - u_{\text{data}}(x_i, t_i) \right)^2$$
K is thhe number of boundary condition points, which are in the
beginning and specified. M is the number of collocation points specified
for the PDE. N is the number of data points

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}}$$

## Supervised Extended PINN

The loss function is a combination of the data loss (difference between
predictions and labeled data) and the PDE residuals:
$$\mathcal{L}_{\text{BC}} = \frac{1}{K} \sum_{k=1}^{K} \left( u_{\text{pred}}(x_{b_k}, t_{b_k}) - u_{b_k} \right)^2$$
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}$$

$$
\mathcal{L}_{\text{BC}} = \frac{1}{K} \sum_{k=1}^{K} \left( u_{\text{pred}}(x_{b_k}, t_{b_k}) - u_{b_k} \right)^2
$$

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}
$$



# Validation and Verification

\
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
the flow field of the Burgers equation using both supervised and
unsupervised PINNs.

**2. Accurate Solution Capture:**Ensure the PINN model accurately
replicates the analytical solutions of the Burgers equation

**3. Effective Sampling:** Use smart sampling techniques, such as
adaptive and weighted sampling, to focus on complex regions with
significant changes

**4. Smooth Convergence:** Achieve smooth and efficient training of the
model, indicating effective learning and proper convergence to the
solution.

**5. Robustness to Variations:** Ensure the model remains accurate even
with uncertainties in initial and boundary conditions

**6. Optimized Training:** Balance accuracy and computational efficiency
by optimizing the network architecture and training parameters.

**7. Visualization and Comparison:**Visualize the flowfield solution,
comparing it with analytical solutions to differences within numerical
simulation.
