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
    """
    Tests the PorkchopPlotter data generation for a known launch window (Earth-Mars 2020).
    """
    from src.mission.porkchop import PorkchopPlotter
    from src.spice.manager import spice_manager
    
    # Load kernels (assuming data directory relative to project root)
    spice_manager.load_standard_kernels('data')
    
    plotter = PorkchopPlotter('EARTH BARYCENTER', 'MARS BARYCENTER')
    
    # Earth-Mars 2020 (Perseverance Window)
    # Launch: July-August 2020, Arrival: Feb 2021
    l_start = spice_manager.utc2et("2020-07-01T00:00:00")
    l_end = spice_manager.utc2et("2020-08-31T00:00:00")
    a_start = spice_manager.utc2et("2021-01-01T00:00:00")
    a_end = spice_manager.utc2et("2021-05-31T00:00:00")
    
    # Use a small grid for speed in unit tests
    num_steps = 5
    launch_dates = np.linspace(l_start, l_end, num_steps)
    arrival_dates = np.linspace(a_start, a_end, num_steps)
    
    data = plotter.generate_data(launch_dates, arrival_dates)
    
    # Assertions on dictionary structure
    assert 'c3' in data
    assert 'v_inf_arr' in data
    assert 'tof_days' in data
    
    # Assertions on data shapes
    expected_shape = (num_steps, num_steps)
    assert data['c3'].shape == expected_shape
    
    # Assert that we found valid transfer solutions (not all NaNs)
    valid_c3 = data['c3'][~np.isnan(data['c3'])]
    assert len(valid_c3) > 0, "No valid Lambert solutions found in the specified window."
    
    # Basic physical sanity checks
    # Earth-Mars C3 should be reasonable (e.g., between 10 and 50 km^2/s^2 for this window)
    min_c3 = np.nanmin(data['c3'])
    assert 5.0 < min_c3 < 100.0, f"Minimum C3 {min_c3} is outside expected physical range."
    
    # TOF Should be around 200 days
    avg_tof = np.nanmean(data['tof_days'])
    assert 100 < avg_tof < 400, f"Average TOF {avg_tof} is outside expected range for Mars transfer."
