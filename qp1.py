import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import time

# Definirea functiei obiectiv si a gradientului acesteia
def objective(x):
    x_A, x_B = x
    return -(10 * x_A + 15 * x_B - 0.5 * (0.1 * x_A**2 + 0.2 * x_B**2 + 2 * 0.05 * x_A * x_B))

def gradient(x):
    x_A, x_B = x
    grad_xA = -(10 - 0.1 * x_A - 0.05 * x_B)
    grad_xB = -(15 - 0.2 * x_B - 0.05 * x_A)
    return np.array([grad_xA, grad_xB])

# Definirea constrangerilor
def constraint1(x):
    return 100 - (2 * x[0] + 3 * x[1])  # Constraint pentru resursa 1

def constraint2(x):
    return 80 - (1 * x[0] + 2 * x[1])  # Constraint pentru resursa 2

constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2}
]

bounds = [(0, None), (0, None)]

# Functie pentru proiectia pe multimea de constrangeri de tipul Ax <= b
def project_onto_bounds(x, bounds):
    return np.maximum(np.minimum(x, [b[1] if b[1] is not None else np.inf for b in bounds]),
                      [b[0] if b[0] is not None else -np.inf for b in bounds])

# Functie pentru proiectia pe multimea de constrangeri de tipul a_i^T x <= b_i
def project_onto_constraints(x, constraints):
    for constraint in constraints:
        if constraint['type'] == 'ineq' and constraint['fun'](x) < 0:
            # Adjust x to satisfy the constraint
            grad_c = approx_fprime(x, constraint['fun'], epsilon=1e-8)
            correction = (constraint['fun'](x) / np.dot(grad_c, grad_c)) * grad_c
            x = x - correction
    return x

# Functie pentru aproximarea gradientului
def approx_fprime(xk, f, epsilon=1e-8):
    f0 = f(xk)
    grad = np.zeros_like(xk)
    ei = np.zeros_like(xk)
    for k in range(len(xk)):
        ei[k] = 1.0
        grad[k] = (f(xk + ei * epsilon) - f0) / epsilon
        ei[k] = 0.0
    return grad

# Metoda gradientului proiectat
def projected_gradient_descent(objective, grad, x0, constraints, bounds, alpha=0.01, tol=1e-6, max_iter=10000):
    x = x0
    history = [x0]
    for _ in range(max_iter):
        grad_val = grad(x)
        x_new = x - alpha * grad_val
        
        # Proiectia pe multimea de constrangeri
        x_new = project_onto_constraints(x_new, constraints)
        x_new = project_onto_bounds(x_new, bounds)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        history.append(x)
    return x, history

# Definirea valorilor initiale si rezolvarea problemei de optimizare
x0 = np.array([1, 1])

def plot_solution_evolution(history):
    '''
    Functie pentru afisarea evolutiei solutiei in spatiu
    '''
    history = np.array(history)
    fig, ax = plt.subplots()
    ax.plot(history[:, 0], history[:, 1], 'o-')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Evolutia solutiei in spatiu')
    plt.show()

# Rezolvare folosind metoda gradientului proiectat
t_start = time()
result_pgd, history = projected_gradient_descent(objective, gradient, x0, constraints, bounds)
t_end = time()
print("Time taken:", t_end - t_start)
# Grafic evolutie solutie
plot_solution_evolution(history)

# Rezolvare folosind SciPy minimize
t_start = time()
result_scipy = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
t_end = time()
print("Time taken:", t_end - t_start)

print("Projected Gradient Descent result:", result_pgd)
print("Objective value:", -objective(result_pgd))
print("SciPy minimize result:", result_scipy.x)


