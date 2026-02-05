import numpy as np
import matplotlib.pyplot as plt

# Adjust path to import src if needed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.ocp import OptimalControlProblem
from optimization.direct import DirectCollocation

def run_earth_mars_transfer_direct():
    """
    Demonstrates Direct Collocation optimization for a 2D Earth-Mars Heliocentric Transfer.
    Unit System: Canonical (mu_sun = 1, r_earth = 1 AU)
    """
    print("Setting up Optimal Control Problem (Direct Method)...")
    
    ocp = OptimalControlProblem()
    
    # --- 1. Define Variables ---
    r, theta, vr, vt = ocp.define_states(['r', 'theta', 'vr', 'vt'])
    ur, ut = ocp.define_controls(['ur', 'ut'])
    
    # --- 2. Dynamics ---
    dynamics = [
        vr,
        vt / r,
        vt**2 / r - 1/r**2 + ur,
        -vr * vt / r + ut
    ]
    ocp.set_dynamics(dynamics)
    
    # --- 3. Objective ---
    # Min Energy: J = 0.5 * integral(ur^2 + ut^2)
    ocp.set_running_cost(0.5 * (ur**2 + ut**2))
    
    # --- 4. Constraints ---
    r_earth = 1.0
    v_earth = 1.0
    r_mars = 1.524
    v_mars = 1.0 / np.sqrt(r_mars)
    
    # Initial State Constraints (Fixed)
    # x(0) - x0 = 0
    # Note: DirectCollocation expects expressions that equal 0
    # In OCP.add_constraint, we pass the expression.
    
    # Initial: r=1, theta=0, vr=0, vt=1
    ocp.add_constraint('initial', r - r_earth)
    ocp.add_constraint('initial', theta - 0.0)
    ocp.add_constraint('initial', vr - 0.0)
    ocp.add_constraint('initial', vt - v_earth)
    
    # Final State Constraints
    # r=1.524, vr=0, vt=v_mars
    # Theta is FREE (we don't constrain theta at tf)
    ocp.add_constraint('final', r - r_mars)
    ocp.add_constraint('final', vr - 0.0)
    ocp.add_constraint('final', vt - v_mars)
    
    # --- 5. Direct Solution ---
    # Use fewer nodes for speed in demo, or 50 for accuracy
    solver = DirectCollocation(ocp, num_nodes=50, method='hermite-simpson')
    
    tf_fixed = 4.5
    print(f"Targeting Transfer time: {tf_fixed:.2f} TU")
    
    # --- 6. Initial Guess ---
    # Linear interpolation
    N = solver.num_nodes
    t_guess = np.linspace(0, tf_fixed, N)
    
    x_guess = np.zeros((N, 4))
    x_guess[:, 0] = np.linspace(r_earth, r_mars, N)
    x_guess[:, 1] = np.linspace(0, 3.14, N)
    x_guess[:, 2] = 0.0
    x_guess[:, 3] = np.linspace(v_earth, v_mars, N)
    
    u_guess = np.zeros((N, 2)) # zeros start
    
    # Solve
    # We pass t0=0, tf_guess=tf_fixed.
    # Note: The current DirectCollocation implementation in direct.py assumes fixed time tf_guess
    # unless logic for variable time is added (which usually requires p[0] = tf).
    # Based on reading direct.py, it uses tf_guess as fixed tf.
    
    print("Solving Nonlinear Programming Problem...")
    res = solver.solve(t0=0.0, tf_guess=tf_fixed, initial_guess_x=x_guess, initial_guess_u=u_guess)
    
    if res['success']:
        print("Optimization Successful!")
        print(f"Cost: {res['cost']}")
        plot_results(res)
    else:
        print("Optimization Failed.")
        print(res['message'])

def plot_results(res):
    t = res['t']
    x = res['x']
    u = res['u']
    
    r = x[:, 0]
    theta = x[:, 1]
    # vr = x[:, 2]
    # vt = x[:, 3]
    
    ur = u[:, 0]
    ut = u[:, 1]
    
    # Cartesian
    x_pos = r * np.cos(theta)
    y_pos = r * np.sin(theta)
    
    # 1. Trajectory
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plt.plot(0, 0, 'yo', markersize=10, label='Sun')
    
    theta_draw = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta_draw), np.sin(theta_draw), 'b--', alpha=0.3, label='Earth Orbit')
    r_m = 1.524
    plt.plot(r_m*np.cos(theta_draw), r_m*np.sin(theta_draw), 'r--', alpha=0.3, label='Mars Orbit')
    
    plt.plot(x_pos, y_pos, 'k-', linewidth=2, label='Transfer (Direct)')
    plt.plot(x_pos[0], y_pos[0], 'bo', label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'rx', label='End')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Earth-Mars Transfer (Direct Collocation)')
    
    # 2. Controls
    plt.subplot(1, 2, 2)
    thrust_mag = np.sqrt(ur**2 + ut**2)
    plt.plot(t, thrust_mag, 'r-', label='Combined Thrust')
    plt.plot(t, ur, 'g--', label='Radial Control')
    plt.plot(t, ut, 'b--', label='Tangential Control')
    
    plt.xlabel('Time (TU)')
    plt.ylabel('Control Magnitude')
    plt.grid(True)
    plt.legend()
    plt.title('Control Profile')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_earth_mars_transfer_direct()
