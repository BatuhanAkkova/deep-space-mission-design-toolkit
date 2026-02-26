"""
Mars Hohmann Transfer Example
-----------------------------
This script demonstrates a classic Earth-to-Mars Hohmann transfer.
It calculates the required Delta-V and compares the analytic Hohmann result 
with a Lambert solver solution for realistic planetary dates.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.spice.manager import spice_manager
from src.trajectory.lambert import LambertSolver

def mars_hohmann_demo():
    # 1. Setup SPICE
    spice_manager.load_standard_kernels()
    
    # Constants
    mu_sun = spice_manager.get_mu('SUN')
    au = 1.495978707e8 # km
    r_earth = 1.0 * au
    r_mars = 1.524 * au # Average Mars orbit
    
    # 2. Analytic Hohmann Transfer (Circular Approx)
    print("--- Analytic Hohmann Transfer (Circular Approximation) ---")
    
    v_e = np.sqrt(mu_sun / r_earth)
    v_m = np.sqrt(mu_sun / r_mars)
    
    a_trans = (r_earth + r_mars) / 2.0
    v_p = np.sqrt(mu_sun * (2.0/r_earth - 1.0/a_trans))
    v_a = np.sqrt(mu_sun * (2.0/r_mars - 1.0/a_trans))
    
    dv1 = v_p - v_e
    dv2 = v_m - v_a
    total_dv = dv1 + dv2
    tof_days = np.pi * np.sqrt(a_trans**3 / mu_sun) / (86400.0)
    
    print(f"Earth Velocity: {v_e:.4f} km/s")
    print(f"Mars Velocity: {v_m:.4f} km/s")
    print(f"Departure Delta-V: {dv1:.4f} km/s")
    print(f"Arrival Delta-V: {dv2:.4f} km/s")
    print(f"Total Hohmann Delta-V: {total_dv:.4f} km/s")
    print(f"Time of Flight: {tof_days:.2f} days")
    
    # 3. Realistic Lambert Solver Comparison
    # Pick a good window (roughly 2026 window)
    utc_dep = "2026-10-20T00:00:00"
    tof_actual_days = 210
    et_dep = spice_manager.utc2et(utc_dep)
    et_arr = et_dep + tof_actual_days * 86400
    
    print(f"\n--- Realistic Lambert Solver (Dep: {utc_dep}, TOF: {tof_actual_days}d) ---")
    
    state_e = spice_manager.get_body_state('EARTH BARYCENTER', 'SUN', et_dep, 'ECLIPJ2000')
    state_m = spice_manager.get_body_state('MARS BARYCENTER', 'SUN', et_arr, 'ECLIPJ2000')
    
    r1 = state_e[0:3]
    v1_e = state_e[3:6]
    r2 = state_m[0:3]
    v2_m = state_m[3:6]
    
    v1_trans, v2_trans = LambertSolver.solve(r1, r2, tof_actual_days * 86400, mu_sun)
    
    dv_dep = np.linalg.norm(v1_trans - v1_e)
    dv_arr = np.linalg.norm(v2_trans - v2_m)
    
    print(f"Lambert Departure Delta-V: {dv_dep:.4f} km/s")
    print(f"Lambert Arrival Delta-V: {dv_arr:.4f} km/s")
    print(f"Total Lambert Delta-V: {dv_dep + dv_arr:.4f} km/s")
    
    # 4. Visualization
    from scipy.integrate import solve_ivp
    def two_body(t, y):
        r = y[0:3]
        v = y[3:6]
        r_norm = np.linalg.norm(r)
        a = -mu_sun * r / (r_norm**3)
        return np.concatenate((v, a))

    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, 'yo', markersize=10, label='Sun')
    
    # Plot Earth and Mars Orbits using 2-body propagation for 1 full period
    t_earth = np.linspace(0, 365.25 * 86400, 200)
    sol_earth = solve_ivp(two_body, [0, 365.25 * 86400], np.concatenate((r1, v1_e)), t_eval=t_earth, rtol=1e-8, atol=1e-8)
    plt.plot(sol_earth.y[0], sol_earth.y[1], 'b--', alpha=0.3, label='Earth Orbit')
    
    t_mars = np.linspace(0, 687.0 * 86400, 200)
    sol_mars = solve_ivp(two_body, [0, 687.0 * 86400], np.concatenate((r2, v2_m)), t_eval=t_mars, rtol=1e-8, atol=1e-8)
    plt.plot(sol_mars.y[0], sol_mars.y[1], 'r--', alpha=0.3, label='Mars Orbit')
    
    plt.plot(r1[0], r1[1], 'bo', label='Earth Departure')
    plt.plot(r2[0], r2[1], 'ro', label='Mars Arrival')
    
    # Calculate and plot the actual transfer orbit using 2-body integration
    state0 = np.concatenate((r1, v1_trans))
    t_eval = np.linspace(0, tof_actual_days * 86400, 200)
    sol = solve_ivp(two_body, [0, tof_actual_days * 86400], state0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    
    plt.plot(sol.y[0], sol.y[1], color='orange', lw=2, label='Transfer Orbit')
    
    plt.axis('equal')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Earth to Mars Realistic Transfer')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.savefig('assets/mars_hohmann_demo.png')
    print("\nPlot saved to assets/mars_hohmann_demo.png")

if __name__ == "__main__":
    mars_hohmann_demo()
