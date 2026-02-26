"""
Unit Tests for Trajectory Propagation Accuracy
-----------------------------------------------
Verifies that NBodyDynamics (with a single central body) matches analytic 
Keplerian propagation over a long duration.
"""

import pytest
import numpy as np
from src.dynamics.nbody import NBodyDynamics

def test_propagation_vs_analytic():
    """Compare N-Body propagation (Sun-only) with analytic 2-body orbit over 10 orbits."""
    mu = 1.32712440018e11 # Sun GM
    r_au = 1.495978707e8 # 1 AU
    v_circ = np.sqrt(mu / r_au)
    
    # State: Circular orbit at 1 AU
    state0 = np.array([r_au, 0, 0, 0, v_circ, 0])
    
    period = 2 * np.pi * np.sqrt(r_au**3 / mu)
    t_span = (0, 10 * period) # 10 full orbits (~10 years)
    
    dyn = NBodyDynamics(bodies=['SUN'], central_body='SUN')
    dyn.mus['SUN'] = mu
    
    sol = dyn.propagate(state0, t_span, rtol=1e-12, atol=1e-14)
    
    # Analytic position at 10 periods should be exactly state0[0:3]
    rf = sol.y[0:3, -1]
    vf = sol.y[3:6, -1]
    
    # Check position error (should be very small with high tolerance integrator)
    # 10 orbits is a long time, allow for some numerical drift
    # Drift usually shows up in phase (along-track)
    pos_err = np.linalg.norm(rf - state0[0:3])
    vel_err = np.linalg.norm(vf - state0[3:6])
    
    print(f"Error after 10 orbits: Pos={pos_err:.4e} km, Vel={vel_err:.4e} km/s")
    
    # Tolerance: 10 orbits at 1 AU is ~9.4 billion km traveled. 
    # 100 km error is 1e-8 relative error.
    assert pos_err < 100.0, f"Significant position drift after 10 orbits: {pos_err} km"

def test_high_eccentricity_stability():
    """Verify stability of the integrator for a high eccentricity (e=0.9) orbit."""
    mu = 3.986e5
    rp = 7000.0
    e = 0.9
    a = rp / (1 - e)
    vp = np.sqrt(mu * (2.0/rp - 1.0/a))
    
    state0 = np.array([rp, 0, 0, 0, vp, 0])
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    
    dyn = NBodyDynamics(bodies=['EARTH'], central_body='EARTH')
    dyn.mus['EARTH'] = mu
    
    # Propagate through one full orbit including perigee passage
    sol = dyn.propagate(state0, (0, period), rtol=1e-12, atol=1e-14)
    
    rf = sol.y[0:3, -1]
    err = np.linalg.norm(rf - state0[0:3])
    
    # For e=0.9, perigee is very fast, placing high demand on the integrator
    assert err < 1.0, f"High eccentricity drift: {err} km"

if __name__ == "__main__":
    pytest.main([__file__])
