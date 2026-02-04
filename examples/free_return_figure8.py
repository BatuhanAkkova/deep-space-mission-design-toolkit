
"""
Earth to Moon Free Return Trajectory (Figure 8) Example.

This script demonstrates how to solve for a free return trajectory using Single Shooting.
It uses N-Body dynamics (Earth, Moon, Sun) and scipy.optimize.minimize to target
specific conditions at the Moon (altitude) and Earth return (re-entry altitude).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

import os 
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.flyby import state_to_bplane

def get_hohmann_guess(r_park_mag, r_target_mag, mu_primary):
    """
    Calculates approximate TLI Delta-V magnitude for a Hohmann transfer.
    """
    # Semi-major axis of transfer orbit
    a_trans = (r_park_mag + r_target_mag) / 2.0
    
    # Vis-viva equation for velocity at perigee of transfer orbit
    v_trans_p = np.sqrt(mu_primary * (2.0 / r_park_mag - 1.0 / a_trans))
    
    # Velocity of parking orbit (circular)
    v_park = np.sqrt(mu_primary / r_park_mag)
    
    dv_mag = v_trans_p - v_park
    return dv_mag

def main():
    print("==========================================================")
    print("   Earth-Moon Free Return Trajectory Optimization (Fig-8) ")
    print("==========================================================")

    # 1. Setup Environment
    if not spice_manager._kernels_loaded:
        spice_manager.load_standard_kernels()
        
    start_date = "2025-01-15T12:00:00" 
    et0 = spice_manager.utc2et(start_date)
    
    mu_earth = spice_manager.get_mu("EARTH")
    mu_moon = spice_manager.get_mu("MOON")
    r_moon_radius = spice_manager.get_body_constant("MOON", "RADII", 3)[0]
    r_earth_radius = spice_manager.get_body_constant("EARTH", "RADII", 3)[0]
    
    # 2. Define Targets
    h_park = 185.0 # km
    r_park = r_earth_radius + h_park
    
    h_moon_target = 200.0 # km (closest approach)
    r_moon_target = r_moon_radius + h_moon_target
    
    h_reentry = 1000.0 # km (approx interface)
    r_reentry = r_earth_radius + h_reentry
    
    print(f"Start Epoch: {start_date}")
    print(f"Parking Radius: {r_park:.1f} km")
    print(f"Target Moon Periapsis: {r_moon_target:.1f} km (Alt: {h_moon_target} km)")
    print(f"Target Earth Reentry: {r_reentry:.1f} km")
    
    # 3. Initial Guess Calculation
    # Get Earth-Moon distance roughly
    state_moon_0 = spice_manager.get_body_state("MOON", "EARTH", et0, "ECLIPJ2000")
    dist_moon = np.linalg.norm(state_moon_0[0:3])
    
    print("Calculating Hohmann Guess...")
    dv_hohmann_mag = get_hohmann_guess(r_park, dist_moon, mu_earth)
    print(f"Hohmann Delta-V Estimate: {dv_hohmann_mag:.4f} km/s")
    
    # Construct Initial State (Parking Orbit)
    # Estimated time of flight ~ 4 days
    t_transfer_guess = 4.0 * 86400.0
    
    # Where will Moon be in 4 days
    state_moon_arr = spice_manager.get_body_state("MOON", "EARTH", et0 + t_transfer_guess, "ECLIPJ2000")
    r_moon_arr = state_moon_arr[0:3]
    
    # Injection Position (r0)
    r_moon_arr_hat = r_moon_arr / np.linalg.norm(r_moon_arr)
    r_inj_direction = -r_moon_arr_hat 
    r0 = r_inj_direction * r_park
    
    # Injection Velocity (v0)
    state_moon_init = spice_manager.get_body_state("MOON", "EARTH", et0, "ECLIPJ2000")
    h_vec = np.cross(state_moon_init[0:3], state_moon_init[3:6]) # Moon Angular Momentum
    h_hat = h_vec / np.linalg.norm(h_vec)
    
    # Velocity direction: Cross product of H and r0
    v_dir_hat = np.cross(h_hat, r0)
    v_dir_hat = v_dir_hat / np.linalg.norm(v_dir_hat)
    
    v_park_mag = np.sqrt(mu_earth / r_park)
    v_total_mag = v_park_mag + dv_hohmann_mag
    
    v0_guess = v_total_mag * v_dir_hat
    
    print(f"Initial Guess: r0={r0}, v0={v0_guess}")
    
    # 4. Dynamics
    bodies = ['EARTH', 'MOON', 'SUN']
    nbody = NBodyDynamics(bodies, frame='ECLIPJ2000', central_body='EARTH')
    
    # Define Events

    def moon_event(t, y):
        # Event will happen after 1 day
        if t < et0 + 86400:
            return 1.0 # Ignore
            
        state_moon = spice_manager.get_body_state('MOON', 'EARTH', t, 'ECLIPJ2000')
        r_m = state_moon[0:3]
        v_m = state_moon[3:6]
        
        r_sc = y[0:3]
        v_sc = y[3:6]
        
        # Relative
        rel_pos = r_sc - r_m
        rel_vel = v_sc - v_m
        
        return np.dot(rel_pos, rel_vel) # Zero at periapsis/apoapsis
    
    moon_event.direction = 0
    moon_event.terminal = False # Don't stop, we need to return to Earth
    
    def earth_event(t, y):
        if t < et0 + 4 * 86400: # Ignore the departure phase
             return 1.0
        
        r_sc = y[0:3]
        v_sc = y[3:6]
        return np.dot(r_sc, v_sc) # Zero at periapsis/apoapsis
        
    earth_event.direction = 0
    earth_event.terminal = True # Stop at return periapsis
    
    # 5. Objective Function
    def objective(u):
        """
        u: [vx, vy, vz] at TLI.
        """
        # Form initial state
        # r0 is fixed (defined above)
        v0 = u
        y0 = np.concatenate((r0, v0))
        
        t_span = (et0, et0 + 10 * 86400) # 10 days max
        
        sol = nbody.propagate(y0, t_span, events=[moon_event, earth_event], rtol=1e-6, atol=1e-9)
        
        # Analyze Events
        # sol.t_events[0] -> Moon Events
        # sol.t_events[1] -> Earth Events
        
        # 1. Did we reach closest approach to Moon?
        if len(sol.t_events[0]) == 0:
            # Use final state distance
            r_final = sol.y[0:3, -1]
            state_moon_f = spice_manager.get_body_state('MOON', 'EARTH', sol.t[-1], 'ECLIPJ2000')
            dist_moon = np.linalg.norm(r_final - state_moon_f[0:3])
            
            J1 = 1e6 + dist_moon # Huge penalty proportional to distance
            r_moon_achieved = dist_moon
        else:
            idx_ev = 0
            y_m = sol.y_events[0][idx_ev]
            t_m = sol.t_events[0][idx_ev]
            
            state_moon_m = spice_manager.get_body_state('MOON', 'EARTH', t_m, 'ECLIPJ2000')
            dist_moon = np.linalg.norm(y_m[0:3] - state_moon_m[0:3])
            
            # Compute B-plane parameters
            r_rel = y_m[0:3] - state_moon_m[0:3]
            v_rel = y_m[3:6] - state_moon_m[3:6]
            
            try:
                br, bt = state_to_bplane(r_rel, v_rel, mu_moon)
                # Target:
                term_br = (br - 0.0)**2
                term_bt = (bt - (r_moon_target))**2 
                
                J1 = term_br + term_bt
                r_moon_achieved = np.sqrt(br**2 + bt**2)
            except ValueError:
                J1 = 1e6 + dist_moon
                r_moon_achieved = dist_moon

        # 2. Did we return to Earth?
        if len(sol.t_events[1]) == 0:
            r_final = sol.y[0:3, -1]
            dist_earth = np.linalg.norm(r_final) # Central body
            J2 = 1e6 + dist_earth # Huge penalty proportional to distance
            r_earth_achieved = dist_earth
        else:
            y_e = sol.y_events[1][0]
            dist_earth = np.linalg.norm(y_e[0:3])
            
            J2 = (dist_earth - r_reentry)**2
            r_earth_achieved = dist_earth
            
        # Total cost
        return 0.5 * J1 + 0.5 * J2

    def perform_grid_search(u_nom):
        print("\n--- Performing Grid Search for Initial Guess ---")
        best_u = u_nom
        best_cost = objective(u_nom)
        print(f"Baseline Cost: {best_cost:.4e}")
        
        # Magnitude variation: +/- 20 m/s
        # Angle variation: +/- 5 degrees
        
        v_vec = u_nom
        v_mag = np.linalg.norm(v_vec)
        v_dir = v_vec / v_mag
        
        # Construct local frame for angle variations
        h_local = np.cross(r0, v_vec)
        h_local /= np.linalg.norm(h_local)
        
        # Basis
        # v_dir
        # h_local
        # perp = h x v
        perp = np.cross(h_local, v_dir)
        
        mags = np.linspace(v_mag * 0.99, v_mag * 1.01, 5) # +/- 1%
        angles = np.linspace(np.radians(-5), np.radians(5), 5) # +/- 5 deg
        
        total_checks = len(mags) * len(angles)
        print(f"Checking {total_checks} candidates...")
        
        count = 0
        for m in mags:
            for ang in angles:
                # Rotate v_dir by 'ang' around h_local
                # approx rotation in plane
                v_rot = v_dir * np.cos(ang) + perp * np.sin(ang)
                u_test = v_rot * m
                
                cost = objective(u_test)
                if cost < best_cost:
                    best_cost = cost
                    best_u = u_test
                    print(f"  Better candidate found! Cost: {cost:.4e}")
                count += 1
                
        print(f"Grid Search Complete. Best Cost: {best_cost:.4e}")
        return best_u

    # Wrapper to unpack for visualization/logging

    costs_log = []
    def callback(xk):
        cost = objective(xk)
        costs_log.append(cost)
        print(f"Iter: Cost={cost:.4e}, v={xk}")
        
    # Run Optimization
    # Run Optimization
    print("Starting Optimization (B-Plane Targeting)...")
    
    # Run Grid Search First
    best_u0 = perform_grid_search(v0_guess)
    u0 = best_u0
    
    res = minimize(objective, u0, method='Powell', callback=callback, options={'maxiter': 500, 'disp': True})
    
    print("\noptimization Complete!")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Optimal V: {res.x}")
    print(f"Final Cost: {res.fun}")
    
    # 6. Propagate Final Solution for Plotting
    v_opt = res.x
    y0_opt = np.concatenate((r0, v_opt))
    t_span = (et0, et0 + 10 * 86400)
    sol_final = nbody.propagate(y0_opt, t_span, events=[moon_event, earth_event], rtol=1e-9, atol=1e-12)

    # Check achieved values
    # Re-evaluate logic manually
    if len(sol_final.t_events[0]) > 0:
        t_m = sol_final.t_events[0][0]
        y_m = sol_final.y_events[0][0]
        s_m = spice_manager.get_body_state('MOON', 'EARTH', t_m, 'ECLIPJ2000')
        d_m = np.linalg.norm(y_m[0:3] - s_m[0:3])
        print(f"Achieved Moon Periapsis: {d_m:.2f} km (Target: {r_moon_target})")
    
    if len(sol_final.t_events[1]) > 0:
        y_e = sol_final.y_events[1][0]
        d_e = np.linalg.norm(y_e[0:3])
        print(f"Achieved Earth Periapsis: {d_e:.2f} km (Target: {r_reentry})")
    
    # 7. Visualization (Earth-Centered Inertial)
    r_hist = sol_final.y[0:3, :].T
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth
    av = np.linspace(0, 2*np.pi, 20)
    bv = np.linspace(0, np.pi, 20)
    x = r_earth_radius * np.outer(np.cos(av), np.sin(bv))
    y = r_earth_radius * np.outer(np.sin(av), np.sin(bv))
    z = r_earth_radius * np.outer(np.ones(np.size(av)), np.cos(bv))
    ax.plot_surface(x, y, z, color='b', alpha=0.3)
    
    # Plot Trajectory
    ax.plot(r_hist[:,0], r_hist[:,1], r_hist[:,2], label='Free Return Trajectory', color='k')
    
    # Plot Moon Path
    times = sol_final.t
    moon_pos = []
    for t in times:
        s = spice_manager.get_body_state("MOON", "EARTH", t, "ECLIPJ2000")
        moon_pos.append(s[0:3])
    moon_pos = np.array(moon_pos)
    ax.plot(moon_pos[:,0], moon_pos[:,1], moon_pos[:,2], label='Moon Path', color='gray', linestyle='--')
    
    # Plot Moon at Encounter
    if len(sol_final.t_events[0]) > 0:
        t_m = sol_final.t_events[0][0]
        s_m = spice_manager.get_body_state('MOON', 'EARTH', t_m, 'ECLIPJ2000')
        ax.scatter(s_m[0], s_m[1], s_m[2], color='gray', s=50, label='Moon Encounter')
        
    ax.set_title('Earth-Moon Free Return (Inertial)')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    
    # Rotating Frame Plot (Figure 8 validation)
    # Transform trajectory to Earth-Moon Rotating Frame
    # X axis: Earth -> Moon
    # Z axis: Moon Orbital Ang Mom
    # Y axis: Completes triad
    
    print("Generating Rotating Frame Plot...")
    rot_pos = []
    
    for i, t in enumerate(times):
        r_eci = r_hist[i]
        s_moon = spice_manager.get_body_state("MOON", "EARTH", t, "ECLIPJ2000")
        
        r_moon = s_moon[0:3]
        v_moon = s_moon[3:6]
        
        # Define Basis
        x_hat = r_moon / np.linalg.norm(r_moon)
        z_hat = np.cross(r_moon, v_moon)
        z_hat = z_hat / np.linalg.norm(z_hat)
        y_hat = np.cross(z_hat, x_hat)
        
        # Project
        x_rot = np.dot(r_eci, x_hat)
        y_rot = np.dot(r_eci, y_hat)
        z_rot = np.dot(r_eci, z_hat)
        
        rot_pos.append([x_rot, y_rot, z_rot])
        
    rot_pos = np.array(rot_pos)
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(rot_pos[:,0], rot_pos[:,1], label='Trajectory (Rotating)')
    
    # Earth
    circle1 = plt.Circle((0, 0), r_earth_radius, color='b', alpha=0.5, label='Earth')
    plt.gca().add_patch(circle1)
    
    # Moon
    avg_dist = np.mean(np.linalg.norm(moon_pos, axis=1))
    circle2 = plt.Circle((avg_dist, 0), r_moon_radius, color='gray', alpha=0.5, label='Moon')
    plt.gca().add_patch(circle2)
    
    plt.xlabel('X (Earth-Moon Line) [km]')
    plt.ylabel('Y [km]')
    plt.title('Free Return Trajectory in Earth-Moon Rotating Frame')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
