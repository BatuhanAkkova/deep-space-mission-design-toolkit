"""
Earth Escape Trajectory Example
-------------------------------
This script demonstrates how to calculate Earth departure C3, V_inf asymptote, 
and hyperbola parameters for an interplanetary mission.

It uses SPICE for realistic Earth states and assumes a high-altitude departure.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics

def earth_escape_demo():
    # 1. Setup SPICE
    spice_manager.load_standard_kernels()
    
    # 2. Define Mission Epoch
    # Assume departure on 2028-09-22
    utc_departure = "2028-09-22T12:00:00"
    et_dep = spice_manager.utc2et(utc_departure)
    
    # 3. Get Earth State relative to Sun
    # In reality, the escape is relative to Earth, but the V_inf is the 
    # excess velocity relative to Earth's orbital velocity.
    earth_state = spice_manager.get_body_state('EARTH', 'SUN', et_dep, 'ECLIPJ2000')
    mu_earth = spice_manager.get_mu('EARTH')
    radii_earth = spice_manager.get_radii('EARTH')
    re = radii_earth[0] # Equatorial radius
    
    print(f"--- Earth Escape Analysis ({utc_departure}) ---")
    print(f"Earth Radius: {re:.2f} km")
    print(f"Earth GM: {mu_earth:.2f} km^3/s^2")
    
    # 4. Define Departure Hyperbola
    # Suppose we want to leave Earth with a specific V_inf vector.
    # For a Mars mission, we might need ~3 km/s excess velocity.
    v_inf_mag = 3.0 # km/s
    c3 = v_inf_mag**2
    
    print(f"Target V_inf: {v_inf_mag:.2f} km/s")
    print(f"Departure C3: {c3:.2f} km^2/s^2")
    
    # 5. Calculate Perigee Velocity required for escape
    # Energy: E = v^2/2 - mu/r = v_inf^2 / 2
    # So v_perigee = sqrt(v_inf^2 + 2*mu/rp)
    
    alt_p = 300.0 # km (Parking orbit altitude)
    rp = re + alt_p
    
    v_esc_local = np.sqrt(2 * mu_earth / rp)
    v_p = np.sqrt(v_inf_mag**2 + (2 * mu_earth / rp))
    dv_inj = v_p - np.sqrt(mu_earth / rp) # Delta-V from circular parking orbit
    
    print(f"Parking Orbit Altitude: {alt_p:.1f} km")
    print(f"Local Escape Velocity: {v_esc_local:.4f} km/s")
    print(f"Required Perigee Velocity: {v_p:.4f} km/s")
    print(f"Injection Delta-V from Circular Orbit: {dv_inj:.4f} km/s")
    
    # 6. Propagation Demonstration
    # Let's propagate the escape for 2 days to see the trajectory straighten out into the asymptote
    # We'll put the spacecraft at perigee on the X-axis, moving in Y.
    state0 = np.array([rp, 0, 0, 0, v_p, 0])
    
    dyn = NBodyDynamics(bodies=['EARTH'], central_body='EARTH')
    # Limit to Earth gravity for this local escape demo
    t_span = (0, 2 * 86400) # 2 days
    sol = dyn.propagate(state0, t_span, rtol=1e-10, atol=1e-12)
    
    # 7. Visualization
    t_days = sol.t / 86400
    r_mag = np.linalg.norm(sol.y[0:3], axis=0)
    v_mag = np.linalg.norm(sol.y[3:6], axis=0)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(sol.y[0], sol.y[1], 'b-', label='Hyperbolic Path')
    plt.plot(0, 0, 'go', label='Earth')
    # Draw Earth to scale-ish
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(re*np.cos(theta), re*np.sin(theta), 'g--', alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Earth Escape Trajectory (Inertial)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Velocity vs Distance
    plt.subplot(1, 2, 2)
    plt.plot(r_mag, v_mag, 'r-', label='Velocity')
    plt.axhline(v_inf_mag, color='k', linestyle='--', label='V_inf Asymptote')
    plt.xlabel('Distance from Earth Center [km]')
    plt.ylabel('Velocity [km/s]')
    plt.title('Velocity Decay toward V_infinity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('earth_escape_demo.png')
    print("\nPlot saved to earth_escape_demo.png")
    # plt.show()

if __name__ == "__main__":
    earth_escape_demo()
