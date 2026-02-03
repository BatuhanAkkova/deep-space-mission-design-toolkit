import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.flyby import FlybyCorrector, state_to_bplane

def main():
    print("Testing Flyby Correction...")
    try:
        if not spice_manager._kernels_loaded:
             spice_manager.load_standard_kernels()
        mu_earth = spice_manager.get_mu('EARTH')
        print(f"Loaded Earth MU: {mu_earth}")
    except Exception as e:
        print(f"SPICE Error: {e}")
        return

    start_time = 0.0 # J2000
    
    # Initial State setup (Earth centered for clarity then shift)
    target_br = 10000.0
    target_bt = 5000.0
    
    state_earth = spice_manager.get_body_state('EARTH', 'SUN', start_time, 'ECLIPJ2000')
    r_e = state_earth[0:3]
    v_e = state_earth[3:6]
    
    # Relative state
    # Incoming asymptote approximate
    # B ~ [0, 10000, 5000]
    r_rel = np.array([-500000.0, 10000.0, 5000.0]) 
    v_rel = np.array([5.0, 0.0, 0.0]) # 5 km/s incoming
    
    state0 = np.concatenate((r_e + r_rel, v_e + v_rel))
    
    nbody = NBodyDynamics(bodies=['SUN', 'EARTH'], frame='ECLIPJ2000', central_body='SUN')
    corrector = FlybyCorrector(nbody)
    
    # Propagate with fixed time
    dt_flight = 150000.0
    
    print(f"Targeting B_R={target_br}, B_T={target_bt}...")
    
    try:
        corrected_state, success = corrector.target_b_plane(
            state0, start_time, 
            target_br, target_bt, 
            dt_max=dt_flight, 
            tol=1.0
        )
    except Exception as e:
        print(f"Correction Exception: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Success: {success}")
    
    if success:
        sol = nbody.propagate(corrected_state, (start_time, start_time + dt_flight))
        final = sol.y[:, -1]
        state_earth_f = spice_manager.get_body_state('EARTH', 'SUN', sol.t[-1], 'ECLIPJ2000')
        r_rel_f = final[0:3] - state_earth_f[0:3]
        v_rel_f = final[3:6] - state_earth_f[3:6]
        
        br, bt = state_to_bplane(r_rel_f, v_rel_f, mu_earth)
        print(f"Result B_R: {br:.2f}, B_T: {bt:.2f}")
        print(f"Errors: dR={br-target_br:.2f}, dT={bt-target_bt:.2f}")

if __name__ == "__main__":
    main()
