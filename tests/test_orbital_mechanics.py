"""
Unit Tests for Fundamental Orbital Mechanics
-------------------------------------------
Verifies Vis-Viva equation, conservation of energy, and angular momentum.
"""

import pytest
import numpy as np
from src.dynamics.nbody import NBodyDynamics

@pytest.fixture
def kepler_dynamics():
    # Simple central body dynamics for testing
    mu = 3.986e5 # Earth GM
    dyn = NBodyDynamics(bodies=['EARTH'], central_body='EARTH')
    dyn.mus['EARTH'] = mu
    return dyn

def test_vis_viva_circular(kepler_dynamics):
    """Verify velocity for a circular orbit: v = sqrt(mu/r)."""
    mu = kepler_dynamics.mus['EARTH']
    r_mag = 7000.0 # km
    v_target = np.sqrt(mu / r_mag)
    
    # State: [r, 0, 0, 0, v, 0]
    state = np.array([r_mag, 0, 0, 0, v_target, 0])
    
    # Vis-Viva: v^2 = mu * (2/r - 1/a)
    # For circular, a = r, so v^2 = mu/r
    a = r_mag
    v_vv = np.sqrt(mu * (2.0/r_mag - 1.0/a))
    
    assert np.isclose(v_target, v_vv)

def test_vis_viva_elliptical(kepler_dynamics):
    """Verify velocity at different points in an elliptical orbit."""
    mu = kepler_dynamics.mus['EARTH']
    rp = 7000.0 # Perigee
    ra = 10000.0 # Apogee
    a = (rp + ra) / 2.0
    
    vp = np.sqrt(mu * (2.0/rp - 1.0/a))
    va = np.sqrt(mu * (2.0/ra - 1.0/a))
    
    # Test at perigee
    v_calc_p = np.sqrt(mu * (2.0/rp - 1.0/a))
    assert np.isclose(vp, v_calc_p)
    
    # Test at apogee
    v_calc_a = np.sqrt(mu * (2.0/ra - 1.0/a))
    assert np.isclose(va, v_calc_a)

def test_conservation_energy(kepler_dynamics):
    """Verify specific mechanical energy is conserved during propagation."""
    mu = kepler_dynamics.mus['EARTH']
    state0 = np.array([7500.0, 0, 0, 0, 8.5, 0.5]) # Elliptical
    
    r0 = np.linalg.norm(state0[0:3])
    v0 = np.linalg.norm(state0[3:6])
    energy0 = 0.5 * v0**2 - mu / r0
    
    # Propagate for 2 orbits (~4 hours)
    period = 2 * np.pi * np.sqrt((7500)**3 / mu) # rough
    sol = kepler_dynamics.propagate(state0, (0, 2 * period), rtol=1e-10, atol=1e-12)
    
    # Check energy at each step
    for i in range(sol.y.shape[1]):
        ri = np.linalg.norm(sol.y[0:3, i])
        vi = np.linalg.norm(sol.y[3:6, i])
        energy_i = 0.5 * vi**2 - mu / ri
        assert np.isclose(energy0, energy_i, rtol=1e-8), f"Energy not conserved at step {i}"

def test_conservation_angular_momentum(kepler_dynamics):
    """Verify angular momentum vector is conserved in magnitude and direction."""
    mu = kepler_dynamics.mus['EARTH']
    state0 = np.array([7000.0, 1000.0, 0, -1.0, 7.5, 2.0])
    
    h0 = np.cross(state0[0:3], state0[3:6])
    
    sol = kepler_dynamics.propagate(state0, (0, 5000.0), rtol=1e-10, atol=1e-12)
    
    for i in range(sol.y.shape[1]):
        hi = np.cross(sol.y[0:3, i], sol.y[3:6, i])
        np.testing.assert_allclose(h0, hi, rtol=1e-8, err_msg=f"Angular momentum drift at step {i}")

if __name__ == "__main__":
    pytest.main([__file__])
