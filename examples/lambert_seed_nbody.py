import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.lambert import LambertSolver

def main():
    print("Lambert Solver Seeded N-Body Simulation")
    
    # 1. Load Kernels
    try:
        spice_manager.load_standard_kernels()
        print("Kernels loaded.")
    except Exception as e:
        print(f"Error loading kernels: {e}")
        return

    # 2. Define Scenario: Earth-Mars Transfer
    # 2020 window
    launch_date_utc = "2020-07-30T12:00:00"
    arrival_date_utc = "2021-02-18T12:00:00"
    
    et_launch = spice_manager.utc2et(launch_date_utc)
    et_arrival = spice_manager.utc2et(arrival_date_utc)
    dt = et_arrival - et_launch
    print(f"Transfer time: {dt/86400:.2f} days")
    
    # 3. Get Positions from SPICE
    # Positions relative to SUN in ECLIPJ2000
    r_earth_launch_state = spice_manager.get_body_state("EARTH", "SUN", et_launch, "ECLIPJ2000")
    r_mars_arrival_state = spice_manager.get_body_state("MARS BARYCENTER", "SUN", et_arrival, "ECLIPJ2000")
    
    r1 = r_earth_launch_state[0:3]
    r2 = r_mars_arrival_state[0:3]
    v_earth_launch = r_earth_launch_state[3:6]
    v_mars_arrival = r_mars_arrival_state[3:6]
    
    print(f"R1 (Earth): {r1}")
    print(f"R2 (Mars) : {r2}")
    
    # 4. Solve Lambert Problem
    mu_sun = spice_manager.get_mu("SUN")
    
    try:
        v1_lambert, v2_lambert = LambertSolver.solve(r1, r2, dt, mu_sun, prograde=True)
        print(f"V1 (Departure): {v1_lambert}")
        print(f"V2 (Arrival)  : {v2_lambert}")
    except Exception as e:
        print(f"Lambert solver failed: {e}")
        return
    
    # Calculate C3 and V_inf
    v_inf_dep = v1_lambert - v_earth_launch
    c3 = np.dot(v_inf_dep, v_inf_dep)
    print(f"Departure V_inf: {np.linalg.norm(v_inf_dep):.4f} km/s")
    print(f"C3: {c3:.4f} km^2/s^2")
    
    # 5. Propagate with N-Body Dynamics
    print("Propagating N-Body Trajectory...")

    # [FIX] Avoid singularity at Earth center (r1) by stepping forward analytically
    # using the Keplerian solution for a short duration (e.g., 1 day)
    # This places us away from the infinite gravity well of the point-mass Earth.
    
    def kepler_step(r0, v0, mu, dt_step):
        # Using Lagrangian coefficients f and g (approximate for short step or exact)
        # Here we use a simple symplectic Euler or RK4 for just one step, 
        # OR better: use the exact f/g formulation.
        
        r_mag = np.linalg.norm(r0)
        # f/g series for small dt
        # f = 1 - 0.5 * (mu/r^3) * dt^2
        # g = dt - (1/6) * (mu/r^3) * dt^3
        
        # Better: use scipy solve_ivp with just 2-body for robustness
        def two_body(t, y):
            rr = y[0:3]
            vv = y[3:6]
            rm = np.linalg.norm(rr)
            acc = -mu * rr / rm**3
            return np.concatenate((vv, acc))
            
        from scipy.integrate import solve_ivp
        res = solve_ivp(two_body, (0, dt_step), np.concatenate((r0, v0)), rtol=1e-12, atol=1e-12)
        return res.y[:, -1]

    # Step forward 1 day (86400s) using only Sun gravity
    dt_safe = 86400.0 * 2
    state_safe = kepler_step(r1, v1_lambert, mu_sun, dt_safe)
    
    t_start_nbody = et_launch + dt_safe
    t_span = (t_start_nbody, et_arrival) # Reduced time span
    
    bodies = ['SUN', 'EARTH', 'MARS BARYCENTER', 'JUPITER BARYCENTER']
    nbody = NBodyDynamics(bodies, frame='ECLIPJ2000', central_body='SUN')
    
    sol = nbody.propagate(state_safe, t_span, rtol=1e-9, atol=1e-12)
    
    if not sol.success:
        print("Propagation failed!")
    else:
        print(f"Propagation success! Steps: {len(sol.t)}")
        
    final_state = sol.y[:, -1]
    r_final = final_state[0:3]
    
    # 6. Analyze Results
    # Compare final position with Mars actual position at arrival
    mars_state_now = spice_manager.get_body_state("MARS BARYCENTER", "SUN", t_span[1], "ECLIPJ2000")
    r_mars_actual = mars_state_now[0:3]
    
    miss_distance = np.linalg.norm(r_final - r_mars_actual)
    print(f"Miss distance at arrival: {miss_distance:.2f} km")
    
    # 7. Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Trajectory
    r_hist = sol.y[0:3, :].T
    ax.plot(r_hist[:,0], r_hist[:,1], r_hist[:,2], label='Spacecraft (Lambert Seed)', color='lime')
    
    # Plot Planets at key times
    ax.scatter(r1[0], r1[1], r1[2], color='blue', s=50, label='Earth Launch')
    ax.scatter(r2[0], r2[1], r2[2], color='red', s=50, label='Mars Target')
    ax.scatter(r_mars_actual[0], r_mars_actual[1], r_mars_actual[2], color='orange', marker='x', s=50, label='Mars Actual')
    ax.scatter(0, 0, 0, color='yellow', s=100, label='Sun')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f'Lambert Seeded N-Body: Miss {miss_distance:.0f} km')
    ax.legend()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'lambert_nbody_result.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
