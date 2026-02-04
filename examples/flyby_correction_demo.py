import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import numpy as np

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.flyby import FlybyCorrector, state_to_bplane, compute_aiming_radius
from src.trajectory.lambert import LambertSolver

def run_test_case(name, nbody, corrector, state0, epoch, target_br, target_bt, dt_flight, mu_flyby):
    print(f"\n--- {name} ---")
    print(f"Target: B_R = {target_br:.2f} km, B_T = {target_bt:.2f} km")
    
    try:
        corrected_state, success = corrector.target_b_plane(
            state0, epoch, 
            target_br, target_bt, 
            dt_max=dt_flight, 
            tol=1.0,
            flyby_body='MOON'
        )
    except Exception as e:
        print(f"Correction failed with error: {e}")
        return None

    if not success:
        print("Correction failed to converge.")
        return None

    # Verify
    sol = nbody.propagate(corrected_state, (epoch, epoch + dt_flight))
    final = sol.y[:, -1]
    t_final = sol.t[-1]
    
    state_moon = spice_manager.get_body_state('MOON', 'EARTH', t_final, 'J2000')
    r_moon = state_moon[0:3]
    v_moon = state_moon[3:6]
    
    r_rel = final[0:3] - r_moon
    v_rel = final[3:6] - v_moon
    
    br, bt = state_to_bplane(r_rel, v_rel, mu_flyby)
    
    print(f"Achieved: B_R = {br:.2f} km, B_T = {bt:.2f} km")
    print(f"Error:    dR  = {br - target_br:.2f} km, dT  = {bt - target_bt:.2f} km")
    
    # Calculate Periapsis Altitude approx
    v_inf = np.linalg.norm(v_rel) # approx at distance
    b_mag = np.sqrt(br**2 + bt**2)
    # b = rp * sqrt(1 + 2mu/rp vinf^2) -> solve for rp
    
    alpha = 2*mu_flyby / v_inf**2
    rp = (-alpha + np.sqrt(alpha**2 + 4*b_mag**2))/2
    
    radii_moon = spice_manager.get_body_constant('MOON', 'RADII', 3)
    r_moon_surf = radii_moon[0]
    alt = rp - r_moon_surf
    print(f"Approx Periapsis Altitude: {alt:.2f} km")
    
    return {
        "name": name,
        "sol": sol,
        "br_target": target_br,
        "bt_target": target_bt,
        "br_actual": br,
        "bt_actual": bt,
        "r_moon_final": r_moon
    }

def plot_results(results):
    if not results:
        print("No results to plot.")
        return

    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Earth-Moon Transfer Trajectories (J2000)")
    
    # Plot Earth
    ax.scatter(0, 0, 0, color='blue', s=100, label='Earth')
    
    # Plot Moon at arrival
    r_moon = results[0]['r_moon_final']
    ax.scatter(r_moon[0], r_moon[1], r_moon[2], color='gray', s=50, label='Moon')
    
    colors = ['r', 'g', 'b', 'm', 'c']
    
    for i, res in enumerate(results):
        sol = res['sol']
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color=colors[i % len(colors)], label=res['name'])
        
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    
    # 2. B-Plane Plot
    ax2 = fig.add_subplot(122)
    ax2.set_title("B-Plane Targets (Moon)")
    ax2.set_aspect('equal')
    ax2.grid(True)
    
    # Draw Moon Disk (approx projection of radius)
    moon_radius = 1737.4
    circle = plt.Circle((0, 0), moon_radius, color='gray', alpha=0.3, label='Moon Disk')
    ax2.add_patch(circle)
    
    for i, res in enumerate(results):
        # Target
        ax2.scatter(res['br_target'], res['bt_target'], marker='x', color=colors[i % len(colors)], s=100, label=f"{res['name']} Target")
        # Actual
        ax2.scatter(res['br_actual'], res['bt_actual'], marker='.', color=colors[i % len(colors)], s=100)
        # Link
        ax2.plot([res['br_target'], res['br_actual']], [res['bt_target'], res['bt_actual']], '--', color=colors[i % len(colors)], alpha=0.5)

    ax2.set_xlabel('B_R [km]')
    ax2.set_ylabel('B_T [km]')
    
    plt.tight_layout()
    plt.show()


from src.trajectory.lambert import LambertSolver

def optimize_initial_guess(nbody, r0, target_body, t0, tf):
    """
    Finds a v0 that results in a close approach to target_body at tf.
    Uses Lambert Solver.
    """
    print(f"Refining initial guess for {target_body} encounter at T+{tf-t0:.1f}s...")
    
    # Get target position
    state_target = spice_manager.get_body_state(target_body, nbody.central_body, tf, nbody.frame)
    r_target = state_target[0:3]
    
    dt = tf - t0
    mu = nbody.mus[nbody.central_body]
    
    # Solve Lambert
    try:
        # Assuming prograde transfer (standard for Earth-Moon)
        v0, vf = LambertSolver.solve(r0, r_target, dt, mu, prograde=True)
        print("Lambert Solution Found.")
        return np.concatenate((r0, v0))
    except RuntimeError as e:
        print(f"Lambert Solver Failed: {e}")
        direction = (r_target - r0) / np.linalg.norm(r_target - r0)
        v0 = direction * 10.9 # sub-optimal but handles the crash
        return np.concatenate((r0, v0))

def main():
    print("Initializing SPICE...")
    if not spice_manager._kernels_loaded:
        spice_manager.load_standard_kernels()
        
    mu_moon = spice_manager.get_mu('MOON')
    radii_moon = spice_manager.get_body_constant('MOON', 'RADII', 3)
    r_moon_avg = radii_moon[0]
    
    print(f"Moon MU: {mu_moon:.2f}, Radius: {r_moon_avg:.2f} km")
    
    start_time = 0.0 # J2000
    
    # Transfer time
    dt_transfer = 3.5 * 86400.0 # 3.5 days
    
    r0 = np.array([6500.0, 0.0, 0.0]) # 6500 km LEO (approx) in equatorial plane
    
    nbody = NBodyDynamics(bodies=['EARTH', 'MOON', 'SUN'], frame='J2000', central_body='EARTH')
    corrector = FlybyCorrector(nbody)
    
    # Refine Initial Guess
    state0 = optimize_initial_guess(nbody, r0, 'MOON', start_time, start_time + dt_transfer)
    
    results = []
    
    # --- Test Case 1: Polar Flyby ---
    # B_R = 2000 km (miss distance), B_T = 0
    # Just checking convergence
    res1 = run_test_case(
        "Case 1: Vertical Offset (Polar)", 
        nbody, corrector, state0, start_time, 
        target_br=2000.0 + r_moon_avg, # 2000 km altitude roughly
        target_bt=0.0, 
        dt_flight=dt_transfer * 1.05,
        mu_flyby=mu_moon
    )
    if res1: results.append(res1)
    
    # --- Test Case 2: Equatorial Flyby ---
    # B_R = 0, B_T = 5000 km
    res2 = run_test_case(
        "Case 2: Horizontal Offset (Equatorial)", 
        nbody, corrector, state0, start_time, 
        target_br=0.0, 
        target_bt=5000.0, 
        dt_flight=dt_transfer * 1.05,
        mu_flyby=mu_moon
    )
    if res2: results.append(res2)

    # --- Test Case 3: Specific Altitude and Angle ---
    target_alt = 100.0 # km
    target_theta = np.deg2rad(45.0) # 45 degrees
    
    # We need to guess V_inf to convert Alt to B-mag
    v_inf_guess = 0.8 # km/s
    rp_target = r_moon_avg + target_alt
    b_mag_target = compute_aiming_radius(v_inf_guess, mu_moon, rp_target)
    
    target_br_3 = b_mag_target * np.cos(target_theta)
    target_bt_3 = b_mag_target * np.sin(target_theta)
    
    res3 = run_test_case(
        f"Case 3: Altitude ~{target_alt} km, Theta=45 deg (Approx)", 
        nbody, corrector, state0, start_time, 
        target_br=target_br_3, 
        target_bt=target_bt_3, 
        dt_flight=dt_transfer * 1.05,
        mu_flyby=mu_moon
    )
    if res3: results.append(res3)
    
    plot_results(results)

if __name__ == "__main__":
    main()
