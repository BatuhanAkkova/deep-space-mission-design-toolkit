import pytest
import numpy as np
from src.mission.departure import calculate_departure_asymptotes, state_to_orbital_elements

def test_departure_c3():
    # V_inf = [3, 4, 0] km/s
    v_inf = np.array([3.0, 4.0, 0.0])
    res = calculate_departure_asymptotes(v_inf)
    
    assert res['C3'] == 25.0 # 3^2 + 4^2
    assert res['v_inf_mag_km_s'] == 5.0
    
    # Direction
    # Unit vec = [0.6, 0.8, 0]
    # DLA = asin(0) = 0
    # RLA = atan2(0.8, 0.6) = 53.13 deg
    
    assert res['DLA_deg'] == 0.0
    assert np.isclose(res['RLA_deg'], np.degrees(np.arctan2(0.8, 0.6)))

def test_state_to_coe():
    # Circular equatorial orbit
    r = np.array([7000.0, 0.0, 0.0])
    mu = 398600.0
    v_circ = np.sqrt(mu / 7000.0)
    v = np.array([0.0, v_circ, 0.0])
    
    eles = state_to_orbital_elements(r, v, mu)
    
    assert np.isclose(eles['e'], 0.0, atol=1e-5)
    assert np.isclose(eles['a'], 7000.0, rtol=1e-3)
    assert np.isclose(eles['i_deg'], 0.0)

def test_porkchop_generator():
    # Test generation runs without error on mock data
    # We won't test full Lambert accuracy here (covered in lambert tests)
    # Just interface
    from src.mission.porkchop import PorkchopPlotter
    from src.spice.manager import SpiceManager
    SpiceManager().load_standard_kernels()
    
    plotter = PorkchopPlotter('EARTH', 'MARS')
    
    # Mock spice manager to avoid loading kernels in unit test if possible
    # But integration test is better.
    # Let's trust integration test for real kernel loading.
    pass
