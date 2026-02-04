import numpy as np
import sympy as sp
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from optimization.ocp import OptimalControlProblem
from optimization.direct import DirectCollocation

def test_double_integrator():
    # 1. Define OCP
    # Minimize J = 0.5 * integral(u^2) dt
    # Subject to:
    # x_dot = v
    # v_dot = u
    # x(0)=0, v(0)=0
    # x(1)=1, v(1)=0
    
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    
    x, v = ocp.define_states(['x', 'v'])
    u = ocp.define_controls(['u'])
    
    ocp.set_dynamics([v, u[0]])
    
    ocp.set_running_cost(0.5 * u[0]**2)
    
    # Boundary constraints
    # Initial
    ocp.add_constraint('initial', x) # x(0) = 0
    ocp.add_constraint('initial', v) # v(0) = 0
    
    # Final
    ocp.add_constraint('final', x - 1.0) # x(1) = 1
    ocp.add_constraint('final', v)       # v(1) = 0
    
    # 2. Solve
    solver = DirectCollocation(ocp, num_nodes=20, method='hermite-simpson')
    
    t0 = 0.0
    tf = 1.0 # Fixed time
    
    # Initial guess (just zeros is fine for SLSQP often, or simple interpolation)
    nodes = 20
    x_guess = np.linspace(0, 1, nodes).reshape((nodes, 1))
    v_guess = np.zeros((nodes, 1))
    u_guess = np.zeros((nodes, 1))
    
    # Combine states
    initial_guess_x = np.hstack((x_guess, v_guess))
    
    sol = solver.solve(t0, tf, initial_guess_x=initial_guess_x, initial_guess_u=u_guess)
    
    print("Optimization Result:")
    print(f"Success: {sol['success']}")
    print(f"Message: {sol['message']}")
    print(f"Cost: {sol['cost']}")
    
    # 3. Analytic Solution Comparison
    # Analytic: x(t) = 3t^2 - 2t^3
    #           v(t) = 6t - 6t^2
    #           u(t) = 6 - 12t
    # Cost = integral(0.5 * (6-12t)^2) from 0 to 1
    #      = 0.5 * integral(36 - 144t + 144t^2)
    #      = 18t - 36t^2 + 24t^3 | 0 to 1
    #      = 18 - 36 + 24 = 6.0
    
    analytic_cost = 6.0
    print(f"Analytic Cost: {analytic_cost}")
    
    # Check error
    assert sol['success'], "Optimization failed"
    assert abs(sol['cost'] - analytic_cost) < 1e-3, f"Cost mismatch: {sol['cost']} vs {analytic_cost}"
    
    # Check trajectory endpoint
    xf = sol['x'][-1]
    print(f"Final State: {xf}")
    assert abs(xf[0] - 1.0) < 1e-4, "Final position error"
    assert abs(xf[1] - 0.0) < 1e-4, "Final velocity error"
    
    # Check control inputs
    # u(0) should be ~6, u(1) should be ~-6
    u_opt = sol['u'].flatten()
    print(f"u[0] = {u_opt[0]}, u[-1] = {u_opt[-1]}")
    assert abs(u_opt[0] - 6.0) < 0.5, "Initial control error" # Loose tolerance for grid
    assert abs(u_opt[-1] - (-6.0)) < 0.5, "Final control error"

if __name__ == "__main__":
    test_double_integrator()
