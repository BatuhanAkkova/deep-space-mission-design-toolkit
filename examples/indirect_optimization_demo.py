import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to import src if needed, assuming run from root
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.ocp import OptimalControlProblem
from optimization.indirect import IndirectSolver

def run_earth_mars_transfer():
    """
    Demonstrates an indirect optimization for a 2D Earth-Mars Heliocentric Transfer.
    Unit System: Canonical (mu_sun = 1, r_earth = 1 AU)
    """
    print("Setting up Optimal Control Problem...")
    
    ocp = OptimalControlProblem()
    
    # --- 1. Define Variables ---
    # State: [r, theta, vr, vt]
    # r: radius (AU)
    # theta: polar angle (rad)
    # vr: radial velocity
    # vt: tangential velocity
    r, theta, vr, vt = ocp.define_states(['r', 'theta', 'vr', 'vt'])
    
    # Control: [ur, ut] (Radial and Tangential acceleration)
    ur, ut = ocp.define_controls(['ur', 'ut'])
    
    # Parameters
    # None for this simple case, use fixed constants
    
    # --- 2. Dynamics (2D Polar Two-Body Problem + Control) ---
    # r_dot = vr
    # theta_dot = vt / r
    # vr_dot = vt^2 / r - 1/r^2 + ur  (mu=1)
    # vt_dot = -vr*vt / r + ut
    
    dynamics = [
        vr,
        vt / r,
        vt**2 / r - 1/r**2 + ur,
        -vr * vt / r + ut
    ]
    ocp.set_dynamics(dynamics)
    
    # --- 3. Objective ---
    # Minimize Energy: J = 0.5 * integral(ur^2 + ut^2) dt
    metrics_scaling = 1.0 
    ocp.set_running_cost(0.5 * (ur**2 + ut**2))
    
    # --- 4. Boundary Conditions ---
    # Start: Earth (Circular Orbit, r=1, v=1)
    # End: Mars (Circular Orbit, r=1.524, v=1/sqrt(1.524))
    
    r_earth = 1.0
    v_earth = 1.0 # sqrt(mu/r) = sqrt(1/1)
    
    r_mars = 1.524
    v_mars = 1.0 / np.sqrt(r_mars)
    
    # Time of flight guess (Hohmann transfer is approx pi * sqrt(a_trans^3))
    # a_trans = (1 + 1.524)/2 = 1.262
    # tof_hohmann = pi * 1.262^1.5 approx 4.49 canonical time units
    # We will fix time for this demo to ensure convergence
    tf_fixed = 4.5
    
    print(f"Targeting Transfer time: {tf_fixed:.2f} TU")
    
    x0 = np.array([r_earth, 0.0, 0.0, v_earth])
    
    # Target state: [r_mars, free_theta, 0.0, v_mars]
    xf = np.array([r_mars, np.nan, 0.0, v_mars]) 
    
    # --- 5. Indirect Solution ---
    solver = IndirectSolver(ocp)
    
    print("Deriving Necessary Conditions (Symbolic)...")
    solver.derive_conditions()
    
    print("Compiling Numerical Functions...")
    solver.lambdify_system()
    
    print("Generating Boundary Condition Function...")
    # Create BC function. Free time = False for now.
    bc_func = solver.create_bc_function(x0, xf, free_time=False)
    
    # --- 6. Initial Guess ---
    # Linear guess for states, small constant for costates
    print("Solving BVP...")
    t_guess = np.linspace(0, tf_fixed, 50)
    
    # Naive initialization
    # r goes 1 -> 1.5
    # theta goes 0 -> pi (approx)
    # vr goes 0 -> 0
    # vt goes 1 -> 0.8
    y_guess = np.zeros((8, len(t_guess)))
    
    y_guess[0, :] = np.linspace(r_earth, r_mars, len(t_guess)) # r
    y_guess[1, :] = np.linspace(0, 3.14, len(t_guess))         # theta
    y_guess[2, :] = 0.0                                        # vr
    y_guess[3, :] = np.linspace(v_earth, v_mars, len(t_guess)) # vt
    
    # Costates guess:
    y_guess[4:, :] = 0.01 
    
    # Solve
    res = solver.solve(t_guess, y_guess, bc_func, verbose=2, tol=1e-4) # Fixed time in this call as free_time=False default
    
    if res.success:
        print("Optimization Successful!")
        plot_results(res)
    else:
        print("Optimization Failed.")
        print(res.message)

def plot_results(res):
    t = res.x
    y = res.y
    r = y[0]
    theta = y[1]
    vr = y[2]
    vt = y[3]
    
    # Reconstruct Controls
    lam_vr = y[6]
    lam_vt = y[7]
    ur = -lam_vr
    ut = -lam_vt
    
    # Convert polar to cartesian for plot
    x_pos = r * np.cos(theta)
    y_pos = r * np.sin(theta)
    
    # 1. Trajectory Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    # Plot Sun
    plt.plot(0, 0, 'yo', markersize=10, label='Sun')
    
    # Plot Earth Orbit (approx)
    theta_draw = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta_draw), np.sin(theta_draw), 'b--', alpha=0.3, label='Earth Orbit')
    
    # Plot Mars Orbit (approx)
    r_m = 1.524
    plt.plot(r_m*np.cos(theta_draw), r_m*np.sin(theta_draw), 'r--', alpha=0.3, label='Mars Orbit')
    
    # Transfer
    plt.plot(x_pos, y_pos, 'k-', linewidth=2, label='Transfer')
    plt.plot(x_pos[0], y_pos[0], 'bo', label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'rx', label='End')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Earth-Mars Low-Thrust Transfer (Min Energy)')
    
    # 2. Control Profile
    plt.subplot(1, 2, 2)
    thrust_mag = np.sqrt(ur**2 + ut**2)
    plt.plot(t, thrust_mag, 'r-', label='Combined Thrust')
    plt.plot(t, ur, 'g--', label='Radial Control')
    plt.plot(t, ut, 'b--', label='Tangential Control')
    
    plt.xlabel('Time (Canonical Units)')
    plt.ylabel('Control Magnitude (accel)')
    plt.grid(True)
    plt.legend()
    plt.title('Control Profile')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_earth_mars_transfer()
