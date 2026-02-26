"""
Unit Tests for SPICE Benchmarks
-------------------------------
Verifies SPICE ephemeris data against known reference values.
"""

import pytest
import numpy as np
from src.spice.manager import spice_manager

def test_spice_earth_state_ref():
    """Benchmark SPICE Earth state against J2000 epoch known values."""
    spice_manager.load_standard_kernels()
    
    # Epoch: 2000-01-01T12:00:00 UTC (J2000)
    et = spice_manager.utc2et("2000-01-01T12:00:00")
    assert et == 0.0 # By definition
    
    # Earth state relative to Sun (ECLIPJ2000)
    state = spice_manager.get_body_state('EARTH', 'SUN', et, 'ECLIPJ2000')
    
    # Reference values for Earth at J2000 (roughly 1 AU on X)
    # Note: These values can vary slightly between DE4xx versions
    # DE440 values are approximately:
    # r = [-2.6e7, 1.44e8, 9.6e3] km
    # v = [-29.8, -5.3, 0.001] km/s
    
    r_mag = np.linalg.norm(state[0:3])
    v_mag = np.linalg.norm(state[3:6])
    
    # Check if Earth is roughly at 1 AU (~1.49e8 km)
    assert 1.47e8 < r_mag < 1.53e8
    # Check if velocity is roughly ~29.8 km/s
    assert 29.0 < v_mag < 31.0

def test_spice_planetary_constants():
    """Verify GM and Radii for major bodies."""
    spice_manager.load_standard_kernels()
    
    # Earth
    mu_earth = spice_manager.get_mu('EARTH')
    # Expect ~398600.44
    assert np.isclose(mu_earth, 398600.44, rtol=1e-3)
    
    # Radius ~6378.1
    radii = spice_manager.get_radii('EARTH')
    assert np.isclose(radii[0], 6378.1, rtol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])
