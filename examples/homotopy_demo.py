import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.optimization.ocp import OptimalControlProblem
from src.optimization.indirect import IndirectSolver

def run_homotopy_demo():
    print("========================================")
    print("   Smoothing Homotopy Demo")
    print("   Double Integrator Rest-to-Rest (Min Fuel)")
    print("========================================")

    # 1. Define OCP: Double Integrator
    ocp = OptimalControlProblem()
    ocp.set_time_variable('t')
    
    x, v = ocp.define_states(['x', 'v'])
    u = ocp.define_controls(['u'])[0]
    
    # Dynamics
    ocp.set_dynamics([v, u])

    # 2. Setup Solver
    solver = IndirectSolver(ocp)
    
    # 3. Define BVP Conditions
    tf = 2.0
    x0 = np.array([0.0, 0.0])
    xf = np.array([1.0, 0.0])
    
    # Manual BC function
    def bc_func_manual(ya, yb, p=None):
        return np.array([
            ya[0] - x0[0],
            ya[1] - x0[1],
            yb[0] - xf[0],
            yb[1] - xf[1]
        ])
    
    # 4. Initial Guess
    n_nodes = 20
    t_guess = np.linspace(0, tf, n_nodes)
    y_guess = np.zeros((4, n_nodes))
    y_guess[0, :] = np.linspace(0, 1, n_nodes) # x
    y_guess[1, :] = 0.5 # dummy v
    y_guess[2, :] = -0.1 # lam_x
    y_guess[3, :] = 0.1 # lam_v
    
    # 5. Run Homotopy
    p_schedule = [2.0, 1.8, 1.6, 1.4, 1.2]
    
    current_t = t_guess
    current_y = y_guess
    last_res = None
    
    print("\nStarting p-Norm Homotopy Loop...")
    
    for val in p_schedule:
        print(f"\nOptimization Step: p = {val}")
        
        # 1. Update Cost
        # L = (u**2)**(val/2) / val
        cost_expr = (u**2)**(val/2) / val
        ocp.set_running_cost(cost_expr)
        
        # 2. Manual Control Derivation (Real Solution)
        # Init costates if needed
        if not solver.costates:
            solver.costates = [sp.Function(f'lambda_{s.func.name}')(ocp.t) for s in ocp.states]
            
        lam_v = solver.costates[1]
        # u = -sign(lam) * |lam|^(1/(p-1))
        # Use sp.Abs to ensure it treats it as real magnitude
        u_real = -sp.sign(lam_v) * sp.Abs(lam_v)**(1.0/(val - 1.0))
        
        solver.control_eqs = {u: u_real} # Force override
        
        # 3. Derive (will use u_real)
        solver.derive_conditions()
        solver.lambdify_system()
        
        # 4. Solve
        try:
             # Increased max_nodes to 10000 to handle "flat" gradients near 0
             res = solver.solve(current_t, current_y, bc_func_manual, verbose=0, tol=1e-4, max_nodes=10000)
             if res.success:
                 print(f"  Converged.")
                 last_res = res
                 current_t = res.x
                 current_y = res.y
             else:
                 print(f"  Failed. {res.message}")
                 break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    res = last_res # for plotting

    if res and res.success:
        print("\nHomotopy Successful!")
        
        t_sol = res.x
        lam_v = res.y[3]
        last_p = p_schedule[-1]
        
        try:
            u_expr = solver.control_eqs[u]
            print(f"Final Control Equation: {u_expr}")
        except Exception as e:
            print(f"Could not retrieve control eq: {e}")

        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        
        ax[0].plot(t_sol, res.y[0], label='x')
        ax[0].plot(t_sol, res.y[1], label='v')
        ax[0].set_ylabel('States')
        ax[0].legend()
        ax[0].grid(True)
        ax[0].set_title("States")
        
        ax[1].plot(t_sol, res.y[3], 'r', label='lambda_v (Switching Func)')
        ax[1].set_ylabel('Costate')
        ax[1].grid(True)
        ax[1].legend()
        ax[1].set_title("Switching Function (-lambda_v)")
        
        ax[2].set_title(f"Control Approximation (p={last_p})")
        # For p-norm, u = -sign(lam) * |lam|^(1/(p-1))
        u_vals = -np.sign(lam_v) * np.abs(lam_v)**(1.0/(last_p - 1.0))
        ax[2].plot(t_sol, u_vals, label='u (derived)')
        ax[2].grid(True)
        ax[2].legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_homotopy_demo()
