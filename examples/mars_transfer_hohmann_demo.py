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
    # Hohmann takes ~259 days, let's try 210 days for a slightly faster path
    tof_actual_days = 210
    et_dep = spice_manager.utc2et(utc_dep)
    et_arr = et_dep + tof_actual_days * 86400
    
    print(f"\n--- Realistic Lambert Solver (Dep: {utc_dep}, TOF: {tof_actual_days}d) ---")
    
    state_e = spice_manager.get_body_state('EARTH', 'SUN', et_dep, 'ECLIPJ2000')
    state_m = spice_manager.get_body_state('MARS', 'SUN', et_arr, 'ECLIPJ2000')
    
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
    # Simple top-down plot of orbits
    theta = np.linspace(0, 2*np.pi, 200)
    plt.figure(figsize=(8, 8))
    
    # Plot Sun
    plt.plot(0, 0, 'yo', markersize=10, label='Sun')
    
    # Plot Circular Orbits for reference
    plt.plot(r_earth*np.cos(theta), r_earth*np.sin(theta), 'b--', alpha=0.3, label='Earth Orbit')
    plt.plot(r_mars*np.cos(theta), r_mars*np.sin(theta), 'r--', alpha=0.3, label='Mars Orbit')
    
    # Plot Transfer
    # We can propagate the transfer orbit for visualization
    mu = mu_sun
    v1 = v1_trans
    h = np.cross(r1, v1)
    energy = 0.5 * np.linalg.norm(v1)**2 - mu / np.linalg.norm(r1)
    a = -mu / (2 * energy)
    # Simplify: just plot the points for the transfer
    # (In a real scenario we'd propagate N-Body or use Keplerian)
    # For now, let's just show r1 and r2
    plt.plot(r1[0], r1[1], 'bo', label='Earth Departure')
    plt.plot(r2[0], r2[1], 'ro', label='Mars Arrival')
    
    # Crude transfer arc (straight line for now, or just markers)
    plt.annotate('', xy=r2[:2], xytext=r1[:2], arrowprops=dict(arrowstyle="->", color='orange', lw=2, label='Transfer (approx)'))
    
    plt.axis('equal')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Earth to Mars Hohmann-ish Transfer')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.savefig('mars_hohmann_demo.png')
    print("\nPlot saved to mars_hohmann_demo.png")

if __name__ == "__main__":
    mars_hohmann_demo()
