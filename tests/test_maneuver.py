import pytest
import numpy as np
from src.trajectory.maneuver import ImpulsiveManeuver, ManeuverError

def test_impulsive_maneuver_no_error():
    dv = np.array([1.0, 0.0, 0.0])
    man = ImpulsiveManeuver(epoch=0.0, delta_v=dv)
    
    state = np.array([1000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    new_state = man.apply_to_state(state, epoch=0.0)
    
    assert np.allclose(new_state[0:3], state[0:3]) # Position unchanged
    assert np.allclose(new_state[3:6], np.array([1.0, 1.0, 0.0])) # Velocity updated

def test_impulsive_maneuver_timing():
    dv = np.array([1.0, 0.0, 0.0])
    man = ImpulsiveManeuver(epoch=100.0, delta_v=dv)
    
    state = np.zeros(6)
    # Wrong time
    new_state = man.apply_to_state(state, epoch=0.0)
    assert np.allclose(new_state, state)
    
    # Correct time
    new_state = man.apply_to_state(state, epoch=100.0)
    assert new_state[3] == 1.0

def test_maneuver_error_magnitude():
    # 10% magnitude error, no pointing error
    err_model = ManeuverError(magnitude_sigma=0.1, pointing_sigma=0.0)
    dv = np.array([100.0, 0.0, 0.0])
    
    # Statistical test (check range)
    dvs = []
    for _ in range(100):
        dvs.append(np.linalg.norm(err_model.apply_error(dv)))
        
    dvs = np.array(dvs)
    # Mean should be close to 100
    # Std should be close to 10
    
    assert np.abs(np.mean(dvs) - 100.0) < 5.0 # Check bias
    assert np.abs(np.std(dvs) - 10.0) < 5.0 # Check spread

def test_maneuver_error_pointing():
    # 10 degree pointing error (large)
    sigma_rad = np.radians(10.0)
    err_model = ManeuverError(magnitude_sigma=0.0, pointing_sigma=sigma_rad)
    dv = np.array([100.0, 0.0, 0.0])
    
    dv_noisy = err_model.apply_error(dv)
    
    # Magnitude should be conserved
    assert np.isclose(np.linalg.norm(dv_noisy), 100.0)
    
    # Check angle
    # dot = cos(theta) * |a|*|b|
    cos_theta = np.dot(dv, dv_noisy) / (100.0 * 100.0)
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    
    # Should be non-zero (highly likely)
    assert angle > 0.0
