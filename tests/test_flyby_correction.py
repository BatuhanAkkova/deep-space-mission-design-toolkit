import pytest
import numpy as np
from src.trajectory.flyby import FlybyCorrector, state_to_bplane, compute_outgoing_v_inf
from src.dynamics.nbody import NBodyDynamics
from src.spice.manager import spice_manager

def test_flyby_correction_convergence():
    # Setup: Simple Earth Flyby scenario
    
    # Check if kernels loaded
    # Assuming standard kernels are available or mocked
    spice_manager.load_standard_kernels()
    mu_earth = spice_manager.get_mu('EARTH')
        
    start_time = 0.0 # J2000
    
    # 1. Define a "Truth" Flyby that hits a specific B-plane target
    # V_inf = [5, 0, 0]
    # Target B = [10000, 5000] (B_R, B_T)
    
    # Initial Guess: Pointing straight at Earth center (collision course)
    # Relative state
    # r = [-1000000, 0, 0]
    # v = [5, 0, 0]
    
    # Target: B_R = 7000 (just above surface), B_T = 0
    target_br = 7000.0
    target_bt = 0.0
    
    # Initial state (inertial)
    # Planet state at t=0
    state_earth = spice_manager.get_body_state('EARTH', 'SUN', start_time, 'ECLIPJ2000')
    r_e = state_earth[0:3]
    v_e = state_earth[3:6]
    
    # Spacecraft relative state (incoming on X axis)
    r_rel = np.array([-500000.0, 10000.0, 5000.0]) # Slightly offset
    v_rel = np.array([5.0, 0.0, 0.0])
    
    state0 = np.concatenate((r_e + r_rel, v_e + v_rel))
    
    # Setup NBody
    nbody = NBodyDynamics(bodies=['SUN', 'EARTH'], frame='ECLIPJ2000', central_body='SUN')
    
    corrector = FlybyCorrector(nbody)
    
    # Target
    # Propagate for enough time to pass Earth
    dt_flight = 150000.0 
    
    corrected_state, success = corrector.target_b_plane(
        state0, start_time, 
        target_br, target_bt, 
        dt_max=dt_flight, 
        tol=10.0 # 10 km tolerance
    )
    
    assert success
    
    # Verify Result
    # Propagate corrected state and check B-plane
    sol = nbody.propagate(corrected_state, (start_time, start_time + dt_flight))
    final = sol.y[:, -1]
    
    state_earth_f = spice_manager.get_body_state('EARTH', 'SUN', sol.t[-1], 'ECLIPJ2000')
    r_rel_f = final[0:3] - state_earth_f[0:3]
    v_rel_f = final[3:6] - state_earth_f[3:6]
    
    br, bt = state_to_bplane(r_rel_f, v_rel_f, mu_earth)
    
    assert np.abs(br - target_br) < 15.0 # Check against tolerance + numerical noise
    assert np.abs(bt - target_bt) < 15.0
