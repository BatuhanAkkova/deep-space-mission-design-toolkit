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
        
        # Mock body state
        def get_body_state_side_effect(target, observer, et, frame):
            if target == 'EARTH' and observer == 'SUN':
                return np.array([1.5e8, 0, 0, 0, 30.0, 0])
            return np.zeros(6)
            
        mock.get_body_state.side_effect = get_body_state_side_effect
        yield mock

def test_stm_initialization(mock_spice):
    """Test that STM initializes to Identity and state is augmented."""
    nbody = NBodyDynamics(bodies=['SUN'])
    
    state_0 = np.array([1.5e8, 0, 0, 0, 30.0, 0])
    t_span = (0, 100)
    
    sol = nbody.propagate(state_0, t_span, stm=True, max_step=10)
    
    # Check output shape
    assert sol.y.shape[0] == 42
    
    # Check initial STM is Identity
    initial_stm = sol.y[6:, 0].reshape((6, 6))
    assert np.allclose(initial_stm, np.eye(6))

def test_stm_finite_difference(mock_spice):
    """
    Verify STM accuracy by comparing against finite difference perturbation.
    Delta X_final approx STM * Delta X_initial
    """
    nbody = NBodyDynamics(bodies=['SUN'], central_body='SUN')
    
    r = 1.5e8
    v = np.sqrt(1.327e11 / r)
    state_0 = np.array([r, 0, 0, 0, v, 0])
    
    t_end = 1000 # Short propagation
    t_span = (0, t_end)
    
    # 1. Propagate Nominal with STM
    sol_nom = nbody.propagate(state_0, t_span, stm=True, max_step=10, rtol=1e-12, atol=1e-12)
    xf_nom = sol_nom.y[:6, -1]
    stm_final = sol_nom.y[6:, -1].reshape((6, 6))
    
    # 2. Perturb Initial State (Position X by small amount)
    delta_x0 = 1.0 # 1 km perturbation
    perturbation = np.zeros(6)
    perturbation[0] = delta_x0
    
    state_perturbed = state_0 + perturbation
    
    # 3. Propagate Perturbed (no STM needed)
    sol_pert = nbody.propagate(state_perturbed, t_span, stm=False, max_step=10, rtol=1e-12, atol=1e-12)
    xf_pert = sol_pert.y[:, -1]
    
    # 4. Compare
    delta_xf_actual = xf_pert - xf_nom
    delta_xf_predicted = stm_final @ perturbation
    
    print(f"Actual Delta: {delta_xf_actual}")
    print(f"Pred Delta:   {delta_xf_predicted}")
    
    # Error should be very small (linear approximation holds for small pert/short time)
    # Relative error check
    rel_error = np.linalg.norm(delta_xf_actual - delta_xf_predicted) / np.linalg.norm(delta_xf_actual)
    assert rel_error < 1e-4

def test_stm_with_perturbation(mock_spice):
    """Test STM calculation with 3rd body perturbation (Earth)."""
    nbody = NBodyDynamics(bodies=['SUN', 'EARTH'], central_body='SUN')
    
    # Spacecraft near Earth
    # Earth at 1.5e8 km, SC at 1.5e8 + 10000 km
    state_0 = np.array([1.5e8 + 10000.0, 0, 0, 0, 30.0, 0])
    
    t_span = (0, 500)
    
    sol = nbody.propagate(state_0, t_span, stm=True, max_step=10, rtol=1e-12, atol=1e-12)
    stm_final = sol_nom = sol.y[6:, -1].reshape((6, 6))
    
    # STM should evolve (non-identity)
    assert not np.allclose(stm_final, np.eye(6))
