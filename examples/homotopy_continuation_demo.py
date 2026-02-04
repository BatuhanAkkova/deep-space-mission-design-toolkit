import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Adjust path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.ocp import OptimalControlProblem
from optimization.indirect import IndirectSolver

def run_homotopy_demo():
    """
    Demonstrates Indirect Optimization with Parameter Continuation (Homotopy).
    We solve for an Earth-Mars transfer, starting with a long duration (easy)
    and reducing the Transfer Time (tf) to find faster trajectories.
    
    The Transfer Time 'TF' is treated as a parameter in the OCP dynamics terms (via time scaling)
    or simply updated in the loop if we use fixed-time formulation.
    
    Here we use the Solver's built-in `solve_with_homotopy` method which iterates over a parameter value.
    For this to work, we must define the problem such that a Parameter affects the system.
    
    We'll use a Free-Time formulation where t is normalized [0, 1], and real time = t * TF.
    Dynamics: dx/dt = f * TF.
    So TF is a parameter we can vary.
    """
    print("Setting up Homotopy Problem (Varying Time of Flight)...")
    
    ocp = OptimalControlProblem()
    
    # --- 1. Define Variables ---
    r, theta, vr, vt = ocp.define_states(['r', 'theta', 'vr', 'vt'])
    ur, ut = ocp.define_controls(['ur', 'ut'])
    
    # Define Parameter 'TF'
    # dynamics will be scaled by TF
    TF = ocp.define_parameters(['TF'])[0] 
    
    # --- 2. Dynamics (Normalized Time) ---
    # Real dynamics:
    # dr/dtau = vr * TF
    # ...
    
    dynamics = [
        vr * TF,
        (vt / r) * TF,
        (vt**2 / r - 1/r**2 + ur) * TF,
        (-vr * vt / r + ut) * TF
    ]
    ocp.set_dynamics(dynamics)
    
    # --- 3. Objective ---
    # Min Energy: Integral 0 to 1 of (0.5*(u^2)*TF) dt  (since dt_real = TF * dt_norm)
    ocp.set_running_cost(0.5 * (ur**2 + ut**2) * TF)
    
    # --- 4. Boundary Conditions ---
    r_earth = 1.0
    v_earth = 1.0
    r_mars = 1.524
    v_mars = 1.0 / np.sqrt(r_mars)
    
    x0 = np.array([r_earth, 0.0, 0.0, v_earth])
    xf = np.array([r_mars, np.nan, 0.0, v_mars])
    
    # --- 5. Solver Setup ---
    solver = IndirectSolver(ocp)
    solver.derive_conditions()
    solver.lambdify_system()
    
    # BC Function
    # x0 is fixed, xf is fixed (except theta)
    bc_func = solver.create_bc_function(x0, xf, free_time=False) 
    # Note: free_time=False in create_bc_func means "Standard BCs" (not adding H(tf)=0 condition).
    # Since we are setting TF as a parameter and varying it, this is correct. 
    # (We are not optimizing TF, we are scanning through it).
    
    # --- 6. Initial Guess ---
    # Start with TF = 6.0 (Easy, slow)
    initial_tf = 6.0
    
    # Normalized Grid [0, 1]
    t_guess = np.linspace(0, 1.0, 50)
    
    # Guess values
    y_guess = np.zeros((8, len(t_guess)))
    y_guess[0, :] = np.linspace(r_earth, r_mars, len(t_guess))
    y_guess[1, :] = np.linspace(0, 3.5, len(t_guess)) 
    y_guess[2, :] = 0.0
    y_guess[3, :] = np.linspace(v_earth, v_mars, len(t_guess))
    y_guess[4:, :] = 0.01

    # Set initial parameter
    solver.set_parameter_value('TF', initial_tf)
    
    # --- 7. Run Homotopy ---
    # Target values for TF: 6.0 -> 3.0
    tf_values = [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.2]
    
    # We solve first one manually to get a valid 'res' object if we want, 
    # but solver.solve_with_homotopy handles the loop.
    # However, solve_with_homotopy uses the *current* solver state.
    
    final_res = solver.solve_with_homotopy(
        t_guess, 
        y_guess, 
        bc_func, 
        param_name='TF', 
        param_values=tf_values,
        verbose=1
    )
    
    if final_res and final_res.success:
        print("\nHomotopy Sequence Successful!")
        plot_results(final_res, solver)
    else:
        print("\nHomotopy Finished (with failures).")
        if final_res:
            plot_results(final_res, solver)

def plot_results(res, solver):
    t_norm = res.x
    y = res.y
    r = y[0]
    
    # Retrieve final parameter value used
    tf_val = solver.parameter_values['TF']
    t_real = t_norm * tf_val
    
    # Reconstruct controls
    u_syms = solver.ocp.controls
    u_exprs = [solver.control_eqs[u] for u in u_syms]
    args = [solver.ocp.t] + solver.ocp.states + solver.costates + solver.ocp.parameters
    u_funcs = [sp.lambdify(args, expr, modules='numpy') for expr in u_exprs]
    
    yp = [y[i] for i in range(y.shape[0])]
    # parameters: [TF]
    params = [tf_val]
    
    ur_val = u_funcs[0](t_norm, *yp, *params)
    ut_val = u_funcs[1](t_norm, *yp, *params)
    
    if np.ndim(ur_val) == 0: ur_val = np.full_like(t_norm, ur_val)
    if np.ndim(ut_val) == 0: ut_val = np.full_like(t_norm, ut_val)
    
    thrust_mag = np.sqrt(ur_val**2 + ut_val**2)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    theta = y[1]
    plt.plot(r*np.cos(theta), r*np.sin(theta), label=f'Transfer (TF={tf_val})')
    # Orbits
    th = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(th), np.sin(th), 'b--', alpha=0.3)
    plt.plot(1.524*np.cos(th), 1.524*np.sin(th), 'r--', alpha=0.3)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Trajectory')
    
    plt.subplot(1, 2, 2)
    plt.plot(t_real, thrust_mag, 'k-', label='Thrust')
    plt.plot(t_real, ur_val, 'g--', label='Radial')
    plt.plot(t_real, ut_val, 'b--', label='Tangential')
    plt.xlabel('Time (TU)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Control Profile (TF={tf_val})')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_homotopy_demo()
