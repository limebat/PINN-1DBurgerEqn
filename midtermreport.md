---
title: PINN -- Midterm Report of Supervised and Unsupervised Predictions of 1D Burger Equation
author: He. Anwen, Felicity. N, I. Aziz, William E. Zhang
date: May 2024
---

## Introduction/Background

The prediction and simulation of fluid partial differential equations (PDEs) are inherently challenging due to their nonlinear nature. In this study, we demonstrate the application of PINNs to the 1D Burgers' PDE to explore neural networks in physics-informed predictions.

Direct solving of the PDE is compared with the resolution methods of lines proposed by Biazar et al. [BiazarBERGER]. The accuracy and efficiency are directly compared to the feasibility and prediction of the PINN [MENG2023116172]. The direct solution is compared and estimated from the residuals of the epochs of the machine learning process, and the data is feed-forwarded from the PDE residuals to epochs. The final prediction for the 1D Burgers' equation is solved and compared to the exact solution.

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}.
$$

Derived by Harry Bateman [harryBateman1915], the solution for \( f^{+} = 2, f^{-} = 0, c = 1 \) results in the following analytical solution to the PDE:

$$
u(x,t) = \frac{2}{1 + e^{\frac{x-t}{\nu}}}
$$

## Problem Definition and Motivation

Solving systems of turbulent flow is a persistent and challenging issue in the field of fluid dynamics. Understanding these systems is important for many problems including those in aerodynamics, fluid spray dynamics, and atmospheric modeling.

Physics-informed neural networks (PINNs) are a new way to solve tough problems that follow physical laws, especially when there is a lack of available data. To make predictions, PINNs embed prior knowledge of physical laws in the learning process. Traditional high fidelity methods like computational fluid dynamics (CFD) or FEM are computationally expensive, but PINNs offer an alternative method that is much faster and requires less data. These methods can be applied in many real-world scenarios to make predictions and facilitate design and decision-making processes, including problems in thermodynamics, climate sciences, and CFD. In our project, we apply PINN models to solve the 1D Burger's equation. The 1D Burger's equation is a simplification of the Navier-Stokes equation, and solving this simplified system using a neural network is a preliminary step in applying machine learning methods to more complex representations of fluid dynamical systems.

## Methods

We use a Supervised Extended Physics-Informed Neural Network to solve the Burger's Equation.

### Data Pre-processing

The data is simplified from the full Burger's equation under steady propagating wave solutions [harryBateman1915].

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}.
$$

Which simplifies to the below expression:

$$
u(x,t) = c - \frac{f^{+} - f^{-}}{2}\tanh\left[\frac{f^{+} - f^{-}}{4\nu}(x - ct)\right].
$$

For the analytical solution comparison with the extended supervised model, additional parameters of \( f^+ = 2 \), \( f^- = 0 \), and \( c = 1 \) are used to compare with the analytical solution below; the simplification with these given parameters is in accordance with Bateman's [harryBateman1915] solutions simplifications.

$$
u(x,t) = \frac{2}{1 + e^{\frac{x-t}{\nu}}}
$$

### Dataset Generation

The analytical training dataset included within the PDE generation of Berger's Equation via the analytical function below.

$$
u(x,t) = \frac{2}{1 + e^{\frac{x-t}{\nu}}}
$$

And the residual dataset from the PDE is generated via the function below for residual losses; this function is equivalent to the analytical solution given \( f^+ = 2 \), \( f^- = 0 \), and \( c = 1 \).

$$
u(x,t) = c - \frac{f^{+} - f^{-}}{2}\tanh\left[\frac{f^{+} - f^{-}}{4\nu}(x - ct)\right].
$$

The grid is uniformly sampled for MSE PDE analytical solution comparisons (N points), and the grid is randomly sampled for residual losses (M points).

#### Training and Validation Data

- **Training Data**:
  - **Initial Condition Points (K)**: Specific points at the boundaries of the domain (\( x=0 \) and \( x=1 \) at \( t=0 \)).
  - **Residual Points (M)**: Randomly selected points within the domain to sample residuals throughout the domain.
  - **Analytical Data Points (N)**: Uniformly selected points within the domain to sample analytical velocity to compare against predicted velocity.

- **Validation Data**: Analytical solutions within the domain are used to validate the model's accuracy in the visualization of data as a baseline.

#### Initial and Boundary Conditions

The initial condition is set at \( t=0 \), where the boundary conditions of \( U(x=0,t=0)=1 \) and \( U(x=1,t=0)=0 \) are given by the Bateman solutions for the supervised extended learning.

- Initial Condition: At \( t=0 \), \( u(x=0, t=0) = 1 \) and \( u(x=1, t=0) = 0 \). This is included in the loss function as the initial condition loss.
- Boundary Condition: Boundaries are set at \( x=0 \) and \( x=1 \). The actual values of \( u \) at these boundaries are implicitly handled by the model through the learning process.

### Model - Supervised Extended PINN

The supervised extended PINN model uses a loss function that is a combination of the data loss (difference between predictions and labeled data) and the PDE residuals:

$$ \mathcal{L}_{\text{PDE}} = \frac{1}{M} \sum_{j=1}^{M} \left( u_t(x_j, t_j) + u(x_j, t_j) u_x(x_j, t_j) - \nu u_{xx}(x_j, t_j) \right)^2 $$


Where:
- \( K \) is the number of boundary condition points,
- \( M \) is the number of collocation points specified for the PDE (residual losses), which are randomly selected within the grid,
- \( N \) is the number of grid points uniformly sampled within the grid.

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}
$$

#### Neural Network Architecture

The architecture consists of three layers:
- **Input Layer**: Two inputs, \( x \) (spatial coordinate) and \( t \) (time coordinate).
- **Hidden Layer**: Contains 5 neurons per layer with Tanh activation functions.
- **Output Layer**: Produces the final prediction of the velocity field \( u(x, t) \).

#### Activation Function: Tanh

Tanh is used as the activation function. It handles non-linear relationships well and outputs values between -1 and 1, which are centered at 0. Tanh provides stronger gradients compared to the sigmoid activation function at points away from -1 and 1, making it suitable for stabilizing and speeding up the learning process.

#### Optimizer: Adam

The Adam optimizer is used to train the model. This optimizer adjusts the learning rate for each parameter, helping the model converge faster by combining the benefits of AdaGrad and RMSProp for more efficient learning.

## Results and Discussion

In this section, the results of the Physics-Informed Neural Network (PINN) approach to solve the 1-D Burgers' equation are presented. The model's predictions are compared with the analytical solutions provided by Harry Bateman [harryBateman1915]. The discussion covers the performance of the model under various combinations of residual points (N) and analytical points (M), varying epochs, and different hidden layer configurations.

### N and M points variance in Extended Supervised Learning Model

#### Baseline N=3, M=3

In this setup, the number of sample or residual points \( N \) and analytical points \( M \) are both set to 3. The training time is approximately 2.82 seconds, which serves as the baseline for comparison with other configurations.

<embed src="PINN_Midterm_Report.pdf" type="application/pdf" width="100%" height="600px" />

#### Effect of variation of N and M points extended configuration
