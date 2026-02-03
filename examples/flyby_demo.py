import numpy as np
import matplotlib.pyplot as plt

# Add src to path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trajectory.flyby import FlybyCalculator
from src.spice.manager import spice_manager


def main():
    print("Initializing SPICE...")
    #Ensure kernels are loaded
    if not spice_manager._kernels_loaded:
        spice_manager.load_standard_kernels()
        
    # Define Flyby Parameters
    body = 'EARTH'
    date = '2025-06-01'
    epoch = spice_manager.utc2et(date)
    
    # Check if Earth is available
    try:
        spice_manager.get_mu(body)
    except:
        print("Earth gravity not found. Using custom fallback if needed, but likely kernels missing.")
        return

    # Use Ecliptic Frame
    frame = 'ECLIPJ2000'
    calculator = FlybyCalculator(body, frame=frame)
    
    # Define Incoming Conditions
    # Assume we are coming from Mars to Earth Transfer? 
    # Let's just pick a generic V_inf vector in Ecliptic frame.
    # V_inf = 5 km/s, confined mostly to Ecliptic (XY), slight inclination.
    v_inf_in = np.array([5.0, 2.0, 0.5]) 
    
    periapsis_alt = 500.0 # km
    beta_angle = np.radians(45.0) # B-plane angle
    
    print("\n--- Analytic Calculation ---")
    analytic_res = calculator.analyze_flyby(v_inf_in, periapsis_alt, beta_angle)
    print(f"Incoming V_inf: {analytic_res['v_inf_in']} km/s")
    print(f"Mag: {np.linalg.norm(analytic_res['v_inf_in']):.3f} km/s")
    print(f"Periapsis Alt: {analytic_res['altitude']} km")
    print(f"Turn Angle: {analytic_res['delta_deg'].item():.3f} deg")
    print(f"Analytic Outgoing V_inf: {analytic_res['v_inf_out']} km/s")
    print(f"Mag: {np.linalg.norm(analytic_res['v_inf_out']):.3f} km/s")
    
    print("\n--- N-Body Propagation Verification ---")
    # Propagate for +/- 2 days around periapsis
    # dt_SOI should be large enough to be 'at infinity' effectively but small enough for stability.
    # Earth SOI ~ 900,000 km. v=5km/s. t ~ 180,000 s ~ 2 days.
    dt_soi = 2 * 86400 
    
    prop_res = calculator.propagate_flyby(v_inf_in, epoch, periapsis_alt, beta_angle, dt_SOI=dt_soi)
    
    measured_v_inf = prop_res['measured_v_inf_out']
    print(f"Measured Outgoing V_inf: {measured_v_inf} km/s")
    print(f"Mag: {np.linalg.norm(measured_v_inf):.3f} km/s")
    
    delta_err = abs(prop_res['analytic_delta'] - prop_res['measured_delta'])
    print(f"Turn Angle Error: {delta_err.item():.4f} degrees")
    
    v_diff = np.linalg.norm(analytic_res['v_inf_out'] - measured_v_inf)
    print(f"Vector Difference Norm: {v_diff:.4f} km/s")
    
    # Visualization
    plot_trajectory(prop_res, body, epoch, frame)

def plot_trajectory(results, body, epoch, frame='ECLIPJ2000'):
    sol_fwd = results['trajectory_fwd']
    sol_back = results['trajectory_back']
    
    # Get Planet Position over time (approximate centered for view)
    # Actually, we propagated in Heliocentric frame.
    # To see the flyby, we must plot in Planet-Centered Inertial frame.
    
    # Extract times
    times_fwd = sol_fwd.t
    times_back = sol_back.t
    
    # Convert states to Planet-Relative
    rel_pos_fwd = []
    for i, t in enumerate(times_fwd):
        state_planet = spice_manager.get_body_state(body, 'SUN', t, frame)
        r_planet = state_planet[0:3]
        r_sc = sol_fwd.y[0:3, i]
        rel_pos_fwd.append(r_sc - r_planet)
        
    rel_pos_back = []
    for i, t in enumerate(times_back):
        state_planet = spice_manager.get_body_state(body, 'SUN', t, frame)
        r_planet = state_planet[0:3]
        r_sc = sol_back.y[0:3, i]
        rel_pos_back.append(r_sc - r_planet) 
        
    rel_pos_fwd = np.array(rel_pos_fwd).T
    rel_pos_back = np.array(rel_pos_back).T
    
    # Debug prints
    radii = spice_manager.get_body_constant(body, 'RADII', 3)
    R = radii[0]
    print(f"\n--- Visualization Debug ---")
    print(f"Planet Radius: {R} km")
    dist_start = np.linalg.norm(rel_pos_fwd[:, 0])
    print(f"Trajectory Periapsis Distance: {dist_start} km")
    print(f"Gap (Alt): {dist_start - R} km")

    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Planet
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_wireframe(x, y, z, color="b", alpha=0.3)
    
    # Plot Trajectory (Near Encounter)
    # We plot all points, but we restrict axis limits
    ax.plot(rel_pos_back[0], rel_pos_back[1], rel_pos_back[2], 'g--', label='Inbound')
    ax.plot(rel_pos_fwd[0], rel_pos_fwd[1], rel_pos_fwd[2], 'r-', label='Outbound')
    
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title(f'{body} Flyby Trajectory (Planet-Centered)')
    ax.legend()
    
    # Zoom in to 3x Periapsis Radius
    limit = 3.0 * dist_start
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Try to set aspect ratio to equal if supported
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass
    
    plt.show()
    #plt.savefig('flyby_trajectory.png')
    #print("Trajectory verified and plotted to flyby_trajectory.png")

if __name__ == "__main__":
    main()
