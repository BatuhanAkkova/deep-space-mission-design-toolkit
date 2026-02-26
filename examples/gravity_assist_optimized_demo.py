"""
Gravity Assist Optimization Example
-----------------------------------
This script demonstrates how to target a specific B-plane aim point at the Moon
starting from an Earth departure. It uses the FlybyCorrector and N-Body dynamics
to iteratively find the required Delta-V.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.flyby import FlybyCorrector, state_to_bplane
from src.trajectory.lambert import LambertSolver

def gravity_assist_optimization_demo():
    # 1. Setup SPICE
    spice_manager.load_standard_kernels()
    
    # 2. Define Mission Window
    # Launch from Earth towards the Moon
    utc_launch = "2025-06-15T10:00:00"
    et_launch = spice_manager.utc2et(utc_launch)
    
    # Estimate flight time to Moon (~3.5 days)
    tof_days = 3.5
    et_arrival = et_launch + tof_days * 86400
    
    print(f"--- Moon Gravity Assist Targeter ---")
    print(f"Launch Epoch: {utc_launch}")
    
    # 3. Initial Guess via Lambert (Two-body Earth-Moon)
    state_earth = spice_manager.get_body_state('EARTH', 'EARTH', et_launch, 'J2000') # 0
    state_moon = spice_manager.get_body_state('MOON', 'EARTH', et_arrival, 'J2000')
    
    r1 = np.array([6700.0, 0, 0]) # Start from LEO
    r2 = state_moon[0:3]
    mu_earth = spice_manager.get_mu('EARTH')
    
    v1_trans, _ = LambertSolver.solve(r1, r2, tof_days * 86400, mu_earth)
    state0 = np.concatenate((r1, v1_trans))
    
    # 4. Setup N-Body Model for Correction
    # We want to hit a specific point in the Moon's B-plane
    # target_br: distance from B-center in R direction
    # target_bt: distance from B-center in T direction
    # Aim for a 1000 km altitude polar flyby
    r_moon = 1737.4
    target_alt = 1000.0
    impact_param = r_moon + target_alt
    
    # Polar flyby: B vector along R or T depending on frame
    target_br = impact_param
    target_bt = 0.0
    
    print(f"Targeting Moon B-plane: BR={target_br:.1f} km, BT={target_bt:.1f} km")
    
    nbody = NBodyDynamics(bodies=['EARTH', 'MOON'], central_body='EARTH', frame='J2000')
    corrector = FlybyCorrector(nbody)
    
    # 5. Run Iterative Correction
    corrected_state0, success = corrector.target_b_plane(
        state0, et_launch, target_br, target_bt, 
        t_encounter=et_arrival, flyby_body='MOON', tol=0.1
    )
    
    if success:
        print("Targeting SUCCESSFUL!")
        # Verify result with final propagation
        sol = nbody.propagate(corrected_state0, (et_launch, et_arrival + 86400), rtol=1e-9, atol=1e-11)
        
        # Calculate final B-plane at closest approach
        # (This is a simplification, we find the min distance point first)
        rel_pos = sol.y[0:3].T - np.array([spice_manager.get_body_state('MOON', 'EARTH', t, 'J2000')[0:3] for t in sol.t])
        rel_dist = np.linalg.norm(rel_pos, axis=1)
        min_idx = np.argmin(rel_dist)
        t_ca = sol.t[min_idx]
        
        state_moon_ca = spice_manager.get_body_state('MOON', 'EARTH', t_ca, 'J2000')
        rf_rel = sol.y[0:3, min_idx] - state_moon_ca[0:3]
        vf_rel = sol.y[3:6, min_idx] - state_moon_ca[3:6]
        
        br_final, bt_final = state_to_bplane(rf_rel, vf_rel, spice_manager.get_mu('MOON'))
        print(f"Final B-plane: BR={br_final:.2f} km, BT={bt_final:.2f} km")
        print(f"Closest Approach Altitude: {rel_dist[min_idx] - r_moon:.2f} km")
        
        # 6. Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(sol.y[0], sol.y[1], 'b-', label='Interplanetary Path')
        # Plot Moon at CA
        plt.plot(state_moon_ca[0], state_moon_ca[1], 'ko', label='Moon at CA')
        plt.axis('equal')
        plt.xlabel('X [km]')
        plt.ylabel('Y [km]')
        plt.title('Earth-Moon Targeted Flyby (N-Body)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gravity_assist_optimized_demo.png')
        print("Plot saved to gravity_assist_optimized_demo.png")
    else:
        print("Targeting FAILED to converge.")

if __name__ == "__main__":
    gravity_assist_optimization_demo()
