import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.ocp import OptimalControlProblem
from src.optimization.indirect import IndirectSolver

def solve_simple_double_integrator():
    """
    Problem:
    Minimize J = 0.5 * integral(u^2) dt from t=0 to t=1
    Subject to:
    x_dot = v
    v_dot = u
    BCs:
    x(0) = 0, v(0) = 0
    x(1) = 1, v(1) = 0
    
    Analytical Solution:
    u(t) = 12t - 6
    x(t) = 2t^3 - 3t^2 + t? No wait.
    The Hamiltonian is H = 0.5*u^2 + lam_x * v + lam_v * u
    dH/du = u + lam_v = 0 => u = -lam_v
    
    lam_x_dot = -dH/dx = 0 => lam_x = c1
    lam_v_dot = -dH/dv = -lam_x => lam_v = -c1*t + c2
    
    u = c1*t - c2
    v_dot = c1*t - c2 => v = 0.5*c1*t^2 - c2*t + c3
    x_dot = v => x = ...
    """
    print("Setting up Optimal Control Problem...")
    
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    
    # States: Position (x), Velocity (v)
    x, v = ocp.define_states(['x', 'v'])
    
    # Control: Acceleration (u)
    u_ctrl = ocp.define_controls(['u'])[0]
    
    # Dynamics: x_dot = v, v_dot = u
    ocp.set_dynamics([v, u_ctrl])
    
    # Cost: L = 0.5 * u^2
    ocp.set_running_cost(0.5 * u_ctrl**2)
    
    # Initialize Solver
    solver = IndirectSolver(ocp)
    
    print("Deriving necessary conditions symbolically...")
    solver.derive_conditions()
    
    print("Costate Equations:")
    for eq in solver.costate_eqs:
        print(eq)
        
    print("\nControl Equation:")
    print(solver.control_eqs)
    
    print("Compiling numerical functions...")
    solver.lambdify_system()
    
    # Define Boundary Conditions for solve_bvp
    # y = [x, v, lam_x, lam_v]
    def bc(ya, yb):
        # ya = y(0), yb = y(tf)
        # x(0)=0, v(0)=0
        # x(tf)=1, v(tf)=0
        
        # Residuals must be 0
        return np.array([
            ya[0] - 0.0,
            ya[1] - 0.0,
            yb[0] - 1.0,
            yb[1] - 0.0
        ])
    
    # Initial Guess
    t_span = np.linspace(0, 1, 100)
    # y shape: (4, 100)
    # Guess zeros
    y_guess = np.zeros((4, len(t_span)))
    # Maybe linear guess for x?
    y_guess[0, :] = t_span 
    
    print("Solving BVP...")
    res = solver.solve(t_span, y_guess, bc)
    
    if res.success:
        print("Success!")
        print(res.message)
        
        # Plot results
        t = res.x
        x_sol = res.y[0]
        v_sol = res.y[1]
        lam_x = res.y[2]
        lam_v = res.y[3]
        
        # Calculate control u = -lam_v
        u_sol = -lam_v
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(t, x_sol, label='Position (x)')
        plt.plot(t, v_sol, label='Velocity (v)')
        plt.ylabel('States')
        plt.legend()
        plt.grid(True)
        plt.title('Double Integrator Bang-Bang (Continuous) Solution')
        
        plt.subplot(3, 1, 2)
        plt.plot(t, u_sol, 'r-', label='Control (u)')
        plt.ylabel('Control')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, lam_x, label='lambda_x')
        plt.plot(t, lam_v, label='lambda_v')
        plt.ylabel('Costates')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Optimization Failed:")
        print(res.message)

def solve_with_homotopy_demo():
    print("\n--- Homotopy Demo ---")
    print("Problem: Minimize J = 0.5 * alpha * u^2, varying alpha.")
    
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    x, v = ocp.define_states(['x', 'v'])
    u_ctrl = ocp.define_controls(['u'])[0]
    
    # Define parameter alpha
    alpha = ocp.define_parameters(['alpha'])[0]
    
    ocp.set_dynamics([v, u_ctrl])
    ocp.set_running_cost(0.5 * alpha * u_ctrl**2)
    
    solver = IndirectSolver(ocp)
    solver.derive_conditions()
    solver.lambdify_system()
    
    # BCs: Rest-to-rest from 0 to 1
    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[0]-1.0, yb[1]])

    t_span = np.linspace(0, 1, 50)
    y_guess = np.zeros((4, len(t_span)))
    y_guess[0, :] = t_span
    
    # Homotopy: Decrease alpha from 10 to 1
    alphas = [10.0, 5.0, 2.0, 1.0]
    
    # Initial solve with alpha=10
    solver.set_parameter_value('alpha', 10.0)
    base_res = solver.solve(t_span, y_guess, bc)
    
    if not base_res.success:
        print("Base solve failed.")
        return

    # Run homotopy
    final_res = solver.solve_with_homotopy(base_res.x, base_res.y, bc, 'alpha', alphas, verbose=0)
    
    if final_res and final_res.success:
        print("Homotopy Success!")
        # Check explicit control eq: u = -lambda_v / alpha
        # For alpha=1, should match previous result roughly.
    else:
        print("Homotopy Failed.")

def solve_constrained_demo():
    print("\n--- Constrained Demo ---")
    print("Problem: Minimize Energy with Velocity Constraint v <= 0.3")
    
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    x, v = ocp.define_states(['x', 'v'])
    u_ctrl = ocp.define_controls(['u'])[0]
    
    penalty_weight = ocp.define_parameters(['rho'])[0]
    
    ocp.set_dynamics([v, u_ctrl])
    ocp.set_running_cost(0.5 * u_ctrl**2)
    
    # Constraint: v <= 1.2  =>  v - 1.2 <= 0
    # Note: Unconstrained max v is 1.5. Distance=1 in Time=1 requires v_avg=1.
    # v <= 1.2 is feasible and active.
    ocp.add_path_constraint(v - 1.2, penalty_weight)
    
    solver = IndirectSolver(ocp)
    solver.derive_conditions()
    solver.lambdify_system()
    
    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[0]-1.0, yb[1]])

    t_span = np.linspace(0, 1, 100)
    y_guess = np.zeros((4, len(t_span)))
    y_guess[0, :] = t_span
    
    # Increase penalty weight
    rhos = [0.0, 10.0, 100.0, 1000.0, 5000.0]
    
    solver.set_parameter_value('rho', 0.0)
    res = solver.solve(t_span, y_guess, bc)
    
    if not res.success:
        print("Initial solve failed")
        return
        
    print(f"Unconstrained Max Velocity: {np.max(res.y[1]):.4f}")

    # Homotopy
    final_res = solver.solve_with_homotopy(res.x, res.y, bc, 'rho', rhos, verbose=0)
    
    if final_res and final_res.success:
        print("Constrained Optimization Success!")
        v_max = np.max(final_res.y[1])
        print(f"Max Velocity: {v_max:.4f} (Constraint: <= 1.2)")
    else:
        print("Constrained Optimization Failed.")

def solve_orbit_transfer_demo():
    print("\n--- Non-Linear Orbit Transfer Demo (1D) ---")
    print("Problem: Vertical ascent with Gravity mu/r^2 (Non-linear).")
    print("Minimize J = 0.5 * integral(u^2) dt")
    print("Dynamics: r_dot = v, v_dot = u - 1/r^2")
    
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    r, v = ocp.define_states(['r', 'v'])
    u = ocp.define_controls(['u'])[0]
    
    # Dynamics (Canonical units: mu=1)
    # 1D gravity: -1/r^2
    ocp.set_dynamics([v, u - 1/r**2])
    
    ocp.set_running_cost(0.5 * u**2)
    
    solver = IndirectSolver(ocp)
    solver.derive_conditions()
    solver.lambdify_system()
    
    # BCs: Rest to Rest, r=1 to r=1.5
    def bc(ya, yb):
        return np.array([
            ya[0] - 1.0, # r0 = 1
            ya[1] - 0.0, # v0 = 0
            yb[0] - 1.5, # rf = 1.5
            yb[1] - 0.0  # vf = 0
        ])
    
    # Fixed time tf = 5.0 (needs to be long enough to reach 1.5 with reasonable thrust)
    tf = 5.0
    t_span = np.linspace(0, tf, 50)
    
    # Initial Guess
    y_guess = np.zeros((4, len(t_span)))
    y_guess[0, :] = np.linspace(1.0, 1.5, len(t_span)) # Linear r guess
    
    print("Solving Non-linear BVP...")
    res = solver.solve(t_span, y_guess, bc, tol=1e-8)
    
    if res.success:
        print("Orbit Transfer Success!")
        print(f"Iterations: {res.niter}") # Hopefully > 1
        print(f"Max Residual: {np.max(res.rms_residuals)}")
    else:
        print("Orbit Transfer Failed.")
        print(res.message)

if __name__ == "__main__":
    solve_simple_double_integrator()
    solve_with_homotopy_demo()
    solve_constrained_demo()
    solve_orbit_transfer_demo()
