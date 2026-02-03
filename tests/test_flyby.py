import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.trajectory.flyby import compute_turn_angle, compute_aiming_radius, rotate_vector, compute_outgoing_v_inf

def test_compute_turn_angle():
    # Test case: Earth flyby
    # mu_earth = 398600.4418
    mu = 398600.4418
    v_inf = 5.0 # km/s
    rp = 6378.137 + 500 # 500 km altitude
    
    delta = compute_turn_angle(v_inf, mu, rp)
    
    # Validation calculation
    e = 1 + rp * v_inf**2 / mu
    expected_delta = 2 * np.arcsin(1/e)
    
    assert np.isclose(delta, expected_delta, atol=1e-8)
    
    # Check bounds
    assert delta > 0
    assert delta < np.pi

def test_compute_aiming_radius():
    mu = 398600.4418
    v_inf = 3.0
    rp = 7000.0
    
    b = compute_aiming_radius(v_inf, mu, rp)
    
    # From formula: b = rp * sqrt(1 + 2mu/(rp v^2))
    expected = rp * np.sqrt(1 + 2*mu/(rp * v_inf**2))
    
    assert np.isclose(b, expected, atol=1e-8)

def test_rotate_vector_simple():
    # Rotate X around Z by 90 degrees -> Y
    vec = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2
    
    rotated = rotate_vector(vec, angle, axis)
    
    expected = np.array([0.0, 1.0, 0.0])
    assert np.allclose(rotated, expected, atol=1e-8)

def test_compute_outgoing_v_inf_planar():
    # Planar case: V_inf along X, Beta=0
    
    v_inf_in = np.array([5.0, 0.0, 0.0])
    beta = 0.0
    rp = 7000.0
    mu = 398600.0
    
    v_out, B_vec, S_vec = compute_outgoing_v_inf(v_inf_in, beta, rp, mu)
    
    # Check magnitude conservation
    assert np.isclose(np.linalg.norm(v_out), np.linalg.norm(v_inf_in), atol=1e-8)
    
    # Check turn angle consistency
    delta = compute_turn_angle(5.0, mu, rp)
    calculated_delta = np.arccos(np.dot(v_inf_in, v_out) / (np.linalg.norm(v_inf_in)**2))
    
    assert np.isclose(delta, calculated_delta, atol=1e-8)

if __name__ == "__main__":
    test_compute_turn_angle()
    test_compute_aiming_radius()
    test_rotate_vector_simple()
    test_compute_outgoing_v_inf_planar()
    print("All tests passed.")
