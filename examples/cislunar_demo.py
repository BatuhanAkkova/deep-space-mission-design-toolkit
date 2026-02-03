
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics

def main():
    print("=============================================")
    print("   Cislunar Trajectory Demo (Earth-Moon)     ")
    print("=============================================")
    
    # 1. Load Kernels
    try:
        spice_manager.load_standard_kernels()
    except Exception as e:
        print(f"Error loading kernels: {e}")
        return

    # 2. Define Initial State: Trans-Lunar Injection (TLI)
    # Simplified TLI conditions
    # Epoch: Arbitrary recent date
    epoch_utc = "2023-01-01T12:00:00"
    et0 = spice_manager.utc2et(epoch_utc)
    
    # Frame: J2000 (Earth-Centered Inertial approx for this demo, usually GCRF)
    # We will simulate in J2000 relative to EARTH
    
    # TLI Burn Point (Perigee of transfer orbit)
    alt_p = 200.0 # km
    R_earth = spice_manager.get_radii("EARTH")[0]
    rp = R_earth + alt_p
    
    # Velocity for Transfer
    # Moon distance ~ 384,400 km
    # Vis-viva for transfer orbit: 
    # a_transfer approx (R_earth + R_moon) / 2 approx 200,000 km
    # v^2 = mu * (2/r - 1/a)
    
    mu_earth = spice_manager.get_mu("EARTH")
    
    # Let's say we want C3 ~ -2.0 km^2/s^2 (Captured but high apogee)
    # v = sqrt(C3 + 2*mu/rp)
    # C3 = -mu/a -> a = -mu/C3
    
    # For TLI, approximate V_p is around 10.9 km/s
    v_p_mag = 10.92 # km/s
    
    # Define State Vector in J2000
    # Position: On X-axis (arbitrary choice)
    r0 = np.array([rp, 0.0, 0.0])
    
    # Velocity: Purely tangential (Y-axis), prograde
    v0 = np.array([0.0, v_p_mag, 0.0])
    
    # Inclination adjustment
    inc_rad = np.radians(28.0)
    # Rotate velocity vector around X axis (radial) -> introduces Z component
    # v0 was [0, v, 0]. New v0 = [0, v*cos(i), v*sin(i)]
    v0 = np.array([0.0, v_p_mag * np.cos(inc_rad), v_p_mag * np.sin(inc_rad)])
    
    initial_state = np.concatenate((r0, v0))
    
    print(f"Initial State (Earth-Relative J2000):")
    print(f"R: {r0} km")
    print(f"V: {v0} km/s")
    
    # 3. Propagate
    # Bodies: Earth (Central), Moon, Sun
    bodies = ['EARTH', 'MOON', 'SUN']
    
    nbody = NBodyDynamics(bodies, frame='J2000', central_body='EARTH')
    
    # Duration: ~10 days
    days = 12.0
    t_span = (et0, et0 + days * 86400.0)
    
    print(f"Propagating for {days} days...")
    sol = nbody.propagate(initial_state, t_span, rtol=1e-9, atol=1e-12)
    
    if not sol.success:
        print("Propagation Failed!")
        return
        
    print(f"Simulated {len(sol.t)} steps.")
    
    # 4. Analysis
    r_sc = sol.y[0:3, :]
    
    # Get Moon Position over time
    r_moon = []
    for t in sol.t:
        st = spice_manager.get_body_state("MOON", "EARTH", t, "J2000")
        r_moon.append(st[0:3])
    r_moon = np.array(r_moon).T # (3, N)
    
    # Calculate distance to Moon
    dist_moon = np.linalg.norm(r_sc - r_moon, axis=0)
    min_dist = np.min(dist_moon)
    min_idx = np.argmin(dist_moon)
    time_min = (sol.t[min_idx] - et0) / 86400.0
    
    print(f"Closest Approach to Moon: {min_dist:.2f} km at T+{time_min:.2f} days")
    
    # 5. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trajectory
    ax.plot(r_sc[0], r_sc[1], r_sc[2], label='Spacecraft', color='lime')
    
    # Moon Trajectory
    ax.plot(r_moon[0], r_moon[1], r_moon[2], label='Moon Orbit', color='gray', linestyle='--')
    
    # Earth
    # Draw wireframe sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R_earth * np.cos(u)*np.sin(v)
    y = R_earth * np.sin(u)*np.sin(v)
    z = R_earth * np.cos(v)
    ax.plot_wireframe(x, y, z, color='blue', alpha=0.3)
    
    # Plot Moon at closest approach
    xm = r_moon[0, min_idx]
    ym = r_moon[1, min_idx]
    zm = r_moon[2, min_idx]
    
    xc = r_sc[0, min_idx]
    yc = r_sc[1, min_idx]
    zc = r_sc[2, min_idx]
    
    ax.scatter(xm, ym, zm, color='gray', s=50, label='Moon (Closest Approach)')
    ax.scatter(xc, yc, zc, color='magenta', s=20, label='SC (Closest Approach)')
    
    # Draw line
    ax.plot([xc, xm], [yc, ym], [zc, zm], color='magenta', linestyle=':')
    
    ax.set_title(f"Cislunar Trajectory\nClosest Moon Approach: {min_dist:.0f} km")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    
    output_file = "cislunar_demo.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
