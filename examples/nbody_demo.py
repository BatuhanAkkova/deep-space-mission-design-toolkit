import numpy as np
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
import matplotlib.pyplot as plt

def main():
    print("Initializing N-Body Demonstration...")
    
    # 1. Load Kernels
    # User is expected to put kernels in 'data/'
    # Required: de440.bsp (or similar), pck00010.tpc, naif0012.tls
    try:
        spice_manager.load_standard_kernels()
    except Exception as e:
        print(f"Error loading kernels: {e}")
        print("Please ensure you have .bsp, .tpc, and .tls files in the 'data' directory.")
        return

    if not spice_manager._kernels_loaded:
        print("No kernels loaded. Aborting demo.")
        return

    # 2. Define Scenario: Earth-Mars Transfer
    # Let's pick a date roughly around a launch window, e.g., 2020-07-30
    launch_date_utc = "2020-07-30T12:00:00"
    et_start = spice_manager.utc2et(launch_date_utc)
    
    # Simple guess for transfer: Earth velocity + some delta-V
    # Get Earth state
    earth_state = spice_manager.get_body_state("EARTH", "SUN", et_start, "ECLIPJ2000")
    r0 = earth_state[0:3]
    v0_earth = earth_state[3:6]
    
    # [FIX] Calculate required start velocity for escape
    # We want a hyperbolic excess velocity (V_inf) of ~3.0 km/s
    # Vis-Viva Equation: v^2 = GM * (2/r + V_inf^2/GM)
    
    mu_earth = spice_manager.get_mu("EARTH")
    r_parking = 7000.0 # km
    v_inf = 3.0 # km/s
    
    # Calculate magnitude of velocity at perigee
    v_mag_parking = np.sqrt(v_inf**2 + 2 * mu_earth / r_parking)
    # v_mag_parking should be approx 11 km/s
    
    # Direction: Velocity is tangential at perigee, in direction of Earth's motion around Sun
    # (Simplified approximation for maximum heliocentric energy gain)
    v_dir = v0_earth / np.linalg.norm(v0_earth)
    
    v0_sc_rel = v_dir * v_mag_parking
    v0_sc = v0_earth + v0_sc_rel
    
    start_offset = (r0 / np.linalg.norm(r0)) * r_parking 
    r0_sc = r0 + start_offset # Position is offset by radius

    initial_state = np.concatenate((r0_sc, v0_sc))
    
    # 3. Propagate
    print(f"Propagating directly from {launch_date_utc}...")
    duration_days = 200
    t_span = (et_start, et_start + duration_days * 86400.0)
    
    # Include perturbations from major bodies
    bodies = ['SUN', 'EARTH', 'MARS BARYCENTER', 'JUPITER BARYCENTER']
    nbody = NBodyDynamics(bodies, frame='ECLIPJ2000', central_body='SUN')
    
    sol = nbody.propagate(initial_state, t_span, rtol=1e-9, atol=1e-12)
    
    print(f"Propagation complete. Steps: {len(sol.t)}")
    
    # 4. Energy Conservation Check (Two-Body approximation)
    # E = v^2/2 - mu/r
    # Note: In n-body, energy is NOT strictly conserved relative to Sun due to perturbations
    # But it should be relatively stable.
    r_hist = sol.y[0:3, :].T
    v_hist = sol.y[3:6, :].T
    r_mags = np.linalg.norm(r_hist, axis=1)
    v_mags = np.linalg.norm(v_hist, axis=1)
    
    mu_sun = spice_manager.get_mu("SUN")
    energies = 0.5 * v_mags**2 - mu_sun / r_mags
    
    print(f"Initial Energy: {energies[0]:.6f} km^2/s^2")
    print(f"Final Energy:   {energies[-1]:.6f} km^2/s^2")
    print(f"Energy Drift:   {abs(energies[-1] - energies[0]):.6e} (Expected to be non-zero due to n-body)")

    # 5. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Trajectory
    ax.plot(r_hist[:,0], r_hist[:,1], r_hist[:,2], label='Spacecraft', color='lime')
    
    # Plot Planets (approximate positions for visualization)
    # Just plotting start and end for simplicity to avoid querying spice 1000 times in loop for demo
    for body in ['EARTH', 'MARS BARYCENTER']:
        color = 'blue' if body == 'EARTH' else 'red'
        # Start
        st_start = spice_manager.get_body_state(body, "SUN", et_start, "ECLIPJ2000")
        ax.scatter(st_start[0], st_start[1], st_start[2], color=color, marker='o', s=50, label=f'{body} Start')
        
        # End
        st_end = spice_manager.get_body_state(body, "SUN", t_span[1], "ECLIPJ2000")
        ax.scatter(st_end[0], st_end[1], st_end[2], color=color, marker='x', s=50, label=f'{body} End')

    # Sun
    ax.scatter(0, 0, 0, color='yellow', s=100, label='Sun')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f'N-Body Propagation ({duration_days} days)')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
