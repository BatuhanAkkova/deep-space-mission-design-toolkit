import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.lambert import LambertSolver
from scipy.integrate import solve_ivp

def kepler_step(r0, v0, mu, dt):
    """
    Analytic Two-Body Step relative to a central body.
    Used to safely step away from a planet's singularity before N-Body.
    """
    def two_body(t, y):
        # State = [rx, ry, rz, vx, vy, vz]
        r = y[0:3]
        v = y[3:6]
        r_norm = np.linalg.norm(r)
        acc = -mu * r / r_norm**3
        return np.concatenate((v, acc))
    
    # Precise integration for just one step
    sol = solve_ivp(two_body, (0, dt), np.concatenate((r0, v0)), rtol=1e-12, atol=1e-13)
    return sol.y[:, -1]


def differentiate_correction(nbody, initial_state_helio, t_span, r_target, et_arrival):
    """
    Performs differential correction (single shooting) to target a position.
    Adjusts standard initial velocity.
    """
    print("\nStarting Differential Correction...")
    
    current_state = initial_state_helio.copy()
    
    for i in range(10):
        # Propagate with STM
        sol = nbody.propagate(current_state, t_span, rtol=1e-9, atol=1e-12, stm=True)
        
        if not sol.success:
            print("Propagation failed during correction.")
            return initial_state_helio, False

        # Extract Final State and STM
        final_full = sol.y[:, -1]
        rf = final_full[0:3]
        # vf = final_full[3:6]
        phi_f = final_full[6:].reshape((6, 6))
        
        # Error
        r_err = rf - r_target
        miss = np.linalg.norm(r_err)
        print(f"Iteration {i}: Miss = {miss:.2f} km")
        
        if miss < 1000.0: # 1000 km tolerance (fine for heliocentric scale)
            print("Converged!")
            return current_state, True
            
        # Jacobian: dr_f / dv_0
        # This is the upper-right 3x3 block of Phi (if state is [r, v])
        # Phi = [[dxf/dx0, dxf/dv0], [dvf/dx0, dvf/dv0]]
        J = phi_f[0:3, 3:6]
        
        # Update: dv0 = -pinv(J) * r_err
        dv0 = -np.linalg.pinv(J) @ r_err
        
        # Dampening / Limits
        # If correction is huge, clamp it to avoid linearity violation
        dv_mag = np.linalg.norm(dv0)
        if dv_mag > 0.5: # max 500 m/s per step
            dv0 = dv0 * (0.5 / dv_mag)
        
        current_state[3:6] += dv0
        
    print("Max iterations reached.")
    return current_state, False

def main():
    print("=============================================")
    print("   N-Body Demo: Earth to Mars Transfer       ")
    print("=============================================")
    
    try:
        spice_manager.load_standard_kernels()
    except Exception as e:
        print(f"Error loading kernels: {e}")
        return

    # 2. Define Scenario: 2020 Earth-Mars Window
    launch_date_utc = "2020-07-30T12:00:00"
    arrival_date_utc = "2021-02-18T12:00:00"
    
    et_launch = spice_manager.utc2et(launch_date_utc)
    et_arrival = spice_manager.utc2et(arrival_date_utc)
    dt_transfer = et_arrival - et_launch
    
    print(f"Launch:  {launch_date_utc}")
    print(f"Arrival: {arrival_date_utc}")
    
    earth_state = spice_manager.get_body_state("EARTH", "SUN", et_launch, "ECLIPJ2000")
    mars_state  = spice_manager.get_body_state("MARS BARYCENTER", "SUN", et_arrival, "ECLIPJ2000")
    
    r1 = earth_state[0:3]
    r2 = mars_state[0:3]
    v1_planet = earth_state[3:6]
    
    # 4. Solve Lambert's Problem
    print("Solving Lambert Problem...")
    mu_sun = spice_manager.get_mu("SUN")
    try:
        v1_trans, v2_trans = LambertSolver.solve(r1, r2, dt_transfer, mu_sun, prograde=True)
    except Exception as e:
        print(f"Lambert Solver Failed: {e}")
        return

    # Compute Hyperbolic Excess
    v_inf_dep = v1_trans - v1_planet
    print(f"Departure V_inf: {np.linalg.norm(v_inf_dep):.3f} km/s")

    # 5. Seed N-Body Simulation with "Kepler Step"
    print("Stepping forward 3 days (SOI exit estimation) analytically...")
    dt_safe = 3.0 * 86400.0
    state_safe_rel_sun = kepler_step(r1, v1_trans, mu_sun, dt_safe)
    
    t_nbody_start = et_launch + dt_safe
    initial_state_nbody = state_safe_rel_sun # [r, v] heliocentric
    
    # Setup N-Body
    bodies = ['SUN', 'EARTH', 'MARS BARYCENTER', 'JUPITER BARYCENTER', 'VENUS']
    nbody = NBodyDynamics(bodies, frame='ECLIPJ2000', central_body='SUN')
    t_span = (t_nbody_start, et_arrival)

    # 6. Initial Propagation (Expect Miss)
    print("Propagating Initial Guess (Lambert Seed)...")
    sol_init = nbody.propagate(initial_state_nbody, t_span, rtol=1e-9, atol=1e-12)
    rf_init = sol_init.y[0:3, -1]
    miss_init = np.linalg.norm(rf_init - r2)
    print(f"Initial Miss Distance: {miss_init:.2f} km")
    
    # 7. Correction Loop
    corrected_state, success = differentiate_correction(nbody, initial_state_nbody, t_span, r2, et_arrival)
    
    # 8. Final Propagation
    print("Propagating Final Trajectory...")
    # Propagate purely for plotting (dense output)
    sol_final = nbody.propagate(corrected_state, t_span, rtol=1e-9, atol=1e-12)
    
    rf_final = sol_final.y[0:3, -1]
    miss_final = np.linalg.norm(rf_final - r2)
    print(f"Final Miss Distance: {miss_final:.2f} km")
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    r_hist = sol_final.y[0:3, :].T
    ax.plot(r_hist[:,0], r_hist[:,1], r_hist[:,2], label='Corrected Trajectory', color='lime')
    
    # Dotted line for initial guess
    r_hist_init = sol_init.y[0:3, :].T
    ax.plot(r_hist_init[:,0], r_hist_init[:,1], r_hist_init[:,2], label='Initial Guess', color='gray', linestyle='--')
    
    ax.scatter(r1[0], r1[1], r1[2], color='blue', s=60, label='Earth Dep')
    ax.scatter(r2[0], r2[1], r2[2], color='red', s=60, label='Mars Arr')
    ax.scatter(0, 0, 0, color='yellow', s=100, label='Sun')
    
    ax.set_title(f"Earth-Mars N-Body Transfer\nFinal Miss: {miss_final:.1f} km")
    ax.legend()
    
    output_file = 'earth_mars_nbody.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
