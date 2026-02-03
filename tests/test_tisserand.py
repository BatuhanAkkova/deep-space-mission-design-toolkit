
import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mission.tisserand import TisserandGraph

class TestTisserandGraph:
    def test_vinf_matching_orbit(self):
        """Test that V_inf is 0 when SC orbit matches Planet orbit."""
        tg = TisserandGraph(mu_sun=1.0)
        tg.add_body("TEST_BODY", a_km=1.0)
        
        # The class uses constants:
        AU_KM = 1.495978707e8
        DAY_SEC = 86400.0
        
        mu_sun = 1.327e11 # km^3/s^2 approx
        tg = TisserandGraph(mu_sun=mu_sun)
        
        R_pl = 1.5e8 # km approx 1 AU
        tg.add_body("PLANET", a_km=R_pl)
        # Circular velocity
        v_c = np.sqrt(mu_sun / R_pl)
        
        # Case 1: Matching orbit
        # a = R_pl, e = 0 -> rp = R_pl
        a_sc = R_pl
        P_sc_sec = 2 * np.pi * np.sqrt(a_sc**3 / mu_sun)
        P_sc_days = P_sc_sec / DAY_SEC
        rp_au = R_pl / AU_KM # close to 1
        
        v_inf = tg.compute_vinf(np.array([P_sc_days]), np.array([rp_au]), "PLANET")
        
        assert np.isclose(v_inf[0], 0.0, atol=1e-3)
        
    def test_vinf_hohmann(self):
        """
        Test V_inf for a Hohmann transfer-like orbit.
        Earth to Mars Hohmann.
        Departs Earth (at Earth distance) with V_inf > 0.
        """
        mu_sun = 1.327124e11
        tg = TisserandGraph(mu_sun=mu_sun)
        
        r_earth = 1.496e8
        r_mars = 2.279e8
        
        tg.add_body("EARTH", a_km=r_earth)
        
        # Transfer orbit
        a_trans = (r_earth + r_mars) / 2
        # Perihelion is r_earth
        rp_trans = r_earth
        
        P_trans_sec = 2 * np.pi * np.sqrt(a_trans**3 / mu_sun)
        P_trans_days = P_trans_sec / 86400.0
        rp_au = rp_trans / 1.495978707e8
        
        # Calculate expected v_inf manually
        # v_dep = sqrt(mu * (2/r_earth - 1/a_trans))
        # v_earth = sqrt(mu / r_earth)
        # v_inf = v_dep - v_earth (tangential departure)
        
        v_dep = np.sqrt(mu_sun * (2/r_earth - 1/a_trans))
        v_earth = np.sqrt(mu_sun / r_earth)
        expected_vinf = abs(v_dep - v_earth)
        
        computed_vinf = tg.compute_vinf(np.array([P_trans_days]), np.array([rp_au]), "EARTH")
        
        assert np.isclose(computed_vinf[0], expected_vinf, atol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])
