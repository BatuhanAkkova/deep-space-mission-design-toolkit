import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is directly importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dynamics.nbody import NBodyDynamics

@pytest.fixture
def mock_spice():
    with patch('src.dynamics.nbody.spice_manager') as mock:
        # Mock MU for Sun and Earth
        def get_mu_side_effect(body):
            if body == 'SUN': return 1.327e11
            if body == 'EARTH': return 3.986e5
            return 0.0
        
        mock.get_mu.side_effect = get_mu_side_effect
        mock._kernels_loaded = True
        
        # Mock body state (Earth at 1 AU on X axis)
        # r = [1.5e8, 0, 0], v = [0, 30, 0]
        def get_body_state_side_effect(target, observer, et, frame):
            if target == 'EARTH' and observer == 'SUN':
                return np.array([1.5e8, 0, 0, 0, 30.0, 0])
            return np.zeros(6)
            
        mock.get_body_state.side_effect = get_body_state_side_effect
        yield mock

def test_nbody_init(mock_spice):
    nbody = NBodyDynamics(bodies=['SUN', 'EARTH'])
    assert 'SUN' in nbody.mus
    assert 'EARTH' in nbody.mus
    assert nbody.mus['SUN'] == 1.327e11

def test_equations_of_motion_two_body(mock_spice):
    """Test pure Keplerian motion (only Sun)."""
    nbody = NBodyDynamics(bodies=['SUN'], central_body='SUN')
    
    # Circular orbit at r = 1.5e8 km
    r = 1.5e8
    mu = 1.327e11
    v_circ = np.sqrt(mu/r) # ~29.7 km/s
    
    state = np.array([r, 0, 0, 0, v_circ, 0])
    
    # Derivative should be [vx, vy, vz, ax, ay, az]
    # ax = -mu/r^2 = -1.327e11 / (1.5e8)^2
    expected_acc = -mu / (r**2)
    
    ds = nbody.equations_of_motion(0, state)
    
    # Check velocity part matches input velocity
    assert np.allclose(ds[0:3], state[3:6])
    
    # Check acceleration
    # Acceleration should be purely int -x direction
    assert np.isclose(ds[3], expected_acc)
    assert np.isclose(ds[4], 0)
    assert np.isclose(ds[5], 0)

def test_perturbation_direction(mock_spice):
    """Test that a 3rd body actually adds acceleration."""
    # Only Sun + Earth
    nbody = NBodyDynamics(bodies=['SUN', 'EARTH'], central_body='SUN')
    
    # Spacecraft is at [2e8, 0, 0] (further out than Earth which is at [1.5e8, 0, 0])
    state = np.array([2.0e8, 0, 0, 0, 0, 0])
    
    # Calculate acceleration with Earth
    ds_perturbed = nbody.equations_of_motion(0, state)
    
    # Calculate acceleration without Earth
    nbody_unperturbed = NBodyDynamics(bodies=['SUN'], central_body='SUN')
    ds_unperturbed = nbody_unperturbed.equations_of_motion(0, state)
    
    assert ds_perturbed[3] < ds_unperturbed[3]
