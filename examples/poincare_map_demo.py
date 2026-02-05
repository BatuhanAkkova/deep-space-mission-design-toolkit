import numpy as np
import matplotlib.pyplot as plt
import os 
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.optimization.grid_search import grid_search_poincare

def get_hohmann_guess(r_park_mag, r_target_mag, mu_primary):
    a_trans = (r_park_mag + r_target_mag) / 2.0
    v_trans_p = np.sqrt(mu_primary * (2.0 / r_park_mag - 1.0 / a_trans))
    v_park = np.sqrt(mu_primary / r_park_mag)
    return v_trans_p - v_park

def main():
    print("==========================================================")
    print("       Poincare Map: Energy vs Phase (Free Return)        ")
    print("==========================================================")

    # 1. Setup Environment
    if not spice_manager._kernels_loaded:
        spice_manager.load_standard_kernels()
        
    # Constant Setup
    mu_earth = spice_manager.get_mu("EARTH")
    r_earth_radius = spice_manager.get_body_constant("EARTH", "RADII", 3)[0]
    
    h_park = 185.0 # km
    r_park = r_earth_radius + h_park
    v_park_mag = np.sqrt(mu_earth / r_park)
    
    # Nominal Epoch
    date_nom = "2025-01-15T12:00:00"
    et_nom = spice_manager.utc2et(date_nom)
    
    # Dynamics Setup
    bodies = ['EARTH', 'MOON', 'SUN']
    nbody = NBodyDynamics(bodies, frame='ECLIPJ2000', central_body='EARTH')

    # 2. Define Evaluation Function
    
    state_moon_nom = spice_manager.get_body_state("MOON", "EARTH", et_nom, "ECLIPJ2000")
    dist_moon_nom = np.linalg.norm(state_moon_nom[0:3])
    dv_hohmann_nom = get_hohmann_guess(r_park, dist_moon_nom, mu_earth)
    
    print(f"Nominal Delta-V Guess: {dv_hohmann_nom:.4f} km/s")
    
    def evaluate_trajectory(dv_mag, time_shift):
        """
        Returns closest approach distance to Moon.
        dv_mag: Delta-V magnitude (km/s)
        time_shift: Shift in seconds from et_nom
        """
        et0 = et_nom + time_shift
        
        # Target roughly 4 days out
        t_transfer = 4.0 * 86400
        state_moon_arr = spice_manager.get_body_state("MOON", "EARTH", et0 + t_transfer, "ECLIPJ2000")
        r_moon_arr = state_moon_arr[0:3]
        
        # Injection Position (Anti-Moon direction at arrival, standard Hohmann logic)
        r_moon_arr_hat = r_moon_arr / np.linalg.norm(r_moon_arr)
        r_inj_direction = -r_moon_arr_hat 
        r0 = r_inj_direction * r_park
        
        # Injection Velocity Direction
        # Perpendicular to r0 in the Moon's orbital plane (roughly)
        state_moon_t0 = spice_manager.get_body_state("MOON", "EARTH", et0, "ECLIPJ2000")
        h_vec = np.cross(state_moon_t0[0:3], state_moon_t0[3:6])
        h_hat = h_vec / np.linalg.norm(h_vec)
        
        v_dir_hat = np.cross(h_hat, r0)
        v_dir_hat = v_dir_hat / np.linalg.norm(v_dir_hat)
        
        # Total V
        v_total = v_park_mag + dv_mag
        v0 = v_total * v_dir_hat
        
        # Propagate
        y0 = np.concatenate((r0, v0))
        t_span = (et0, et0 + 5.0 * 86400) # 5 days
        
        sol = nbody.propagate(y0, t_span, rtol=1e-5, atol=1e-8)
        times = sol.t
        states = sol.y.T # (N, 6)
        
        # Vectorized Moon positions
        r_sc = states[:, 0:3]
        
        min_dist = 1e9
        
        for k in range(0, len(times), 5): # skip steps for speed
            t_k = times[k]
            s_m = spice_manager.get_body_state("MOON", "EARTH", t_k, "ECLIPJ2000")
            d = np.linalg.norm(r_sc[k] - s_m[0:3])
            if d < min_dist:
                min_dist = d
                
        return min_dist

    # 3. Setup Grid
    # Delta-V Range: +/- 50 m/s around nominal
    dv_vals = np.linspace(dv_hohmann_nom - 0.05, dv_hohmann_nom + 0.05, 20)
    
    # Time Range: +/- 2 Days
    dt_vals = np.linspace(-2.0 * 86400, 2.0 * 86400, 20)
    
    print("\nStarting Poincare Grid Search...")
    print(f"Delta-V Range: [{dv_vals[0]:.4f}, {dv_vals[-1]:.4f}] km/s")
    print(f"Time Shift Range: [{dt_vals[0]/3600:.1f}, {dt_vals[-1]/3600:.1f}] hours")
    
    results = grid_search_poincare(
        param1_vals=dv_vals, # X axis
        param2_vals=dt_vals, # Y axis
        eval_func=evaluate_trajectory,
        verbose=True
    )
    
    # 4. Visualization
    X = results['grid_p1'] # Delta V
    Y = results['grid_p2'] / 86400.0 # Time in Days
    Z = results['grid_z']
    
    # Log scale for distance visualization
    Z_log = np.log10(Z + 1.0) # avoid log(0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour
    cp = ax.contourf(X, Y, Z_log, levels=30, cmap='plasma_r') # Reversed plasma: Dark is close (low dist), Light is far
    cbar = fig.colorbar(cp)
    cbar.set_label('Log10(Distance to Moon [km])')
    
    ax.set_title('Poincare Map: Moon Approach Distance')
    ax.set_xlabel('TLI Delta-V [km/s]')
    ax.set_ylabel('Departure Time Shift [days]')
    
    # Mark Best
    best_dv = results['best_p1']
    best_dt = results['best_p2'] / 86400.0
    ax.scatter(best_dv, best_dt, color='lime', marker='*', s=200, label='Closest Approach')
    
    print(f"\nLowest Distance Found: {results['best_z']:.2f} km")
    print(f"At Delta-V: {best_dv:.5f} km/s")
    print(f"At Time Shift: {best_dt:.2f} days")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
