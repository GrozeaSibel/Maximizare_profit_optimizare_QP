# Maximizing Company Profit - Optimization Problem

## Description
This project focuses on an **optimization problem with constraints**, aiming to maximize a company's profit from selling two products, **A** and **B**. Each product requires different amounts of resources and generates different profits per unit. The company operates under **limited material and human resources**, making it necessary to determine the optimal production quantities for maximizing overall profit.

## Problem Definition
- **Product A:**  
  - Profit per unit: **10**  
  - Required material: **2 units**  
  - Required human resources: **1 worker**  

- **Product B:**  
  - Profit per unit: **15**  
  - Required material: **3 units**  
  - Required human resources: **2 workers**  

- **Resource Constraints:**  
  - **Material availability:** **100 units**  
  - **Total employees:** **80 workers**  

Additionally, the cost per unit increases quadratically at higher production levels:  
- **Product A:** \(0.1x^2\)  
- **Product B:** \(0.2y^2\)  
- **Interaction term:** \(0.05xy\)  

## Mathematical Formulation
The optimization problem is formulated as:

$$
F(x, y) = 10x + 15y - \frac{1}{2} (0.1x^2 + 0.2y^2 + 2 \times 0.05xy)
$$

where:
- **x** = Quantity of **Product A**
- **y** = Quantity of **Product B**
- The objective is to **maximize** \( F(x, y) \).  
- To apply optimization algorithms, we transform it into a **minimization problem**:  
   Z = -F(x, y) 

### **Constraints**
$$
2x + 3y \leq 100  \quad  (\text{Material constraint})
$$
$$
x + 2y \leq 80 \quad  (\text{Human resource constraint})
$$
$$
x \geq 0, \quad y \geq 0
$$

## Solution Approach
To solve this **Quadratic Programming** problem, the **Projected Gradient Descent** method was implemented. This method is suitable since the objective function is **continuous and differentiable**. However, due to constraints, a **projection step** is required to ensure solutions remain within feasible limits.

### **Gradient Calculation**
The gradient of the objective function is given by:

$$
\text{grad}(F) =
\begin{bmatrix}
-(10 - 0.1x - 0.05y) \\
-(15 - 0.2y - 0.05x)
\end{bmatrix}
$$

### **Projection on Constraints**
- **Box constraints projection**:  
  ( x' = min(max(x, l), u) ) where **l** = lower bound, **u** = upper bound.

- **Linear constraint projection:**  
  The constraints are projected onto the feasible region.

### **Algorithm Parameters**
- **Max iterations:** **10,000**
- **Tolerance:** **1e-6**
- **Learning rate (step size α):** **0.1**

## Implementation
The problem was implemented in **Python** using the following key libraries:
- `NumPy` for matrix operations.
- `SciPy.optimize` for the built-in optimization method (`SLSQP`).
- `Matplotlib` for visualizing the solution evolution.

### **Optimization Using SciPy**
To compare performance, the **SciPy `minimize` function** was used with the `SLSQP` method:
```python
from scipy.optimize import minimize

result_scipy = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
