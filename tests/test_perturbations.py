import pytest
import numpy as np
from src.spice.manager import spice_manager
from src.dynamics.perturbations import J2Perturbation, SSRPerturbation, DragPerturbation
from src.dynamics.nbody import NBodyDynamics

class TestPerturbations:
    @classmethod
    def setup_class(cls):
        # Ensure kernels are loaded
        spice_manager.load_standard_kernels()
        
    def test_coord_transform(self):
        # Test Identity transform
        et = 0.0
        R = spice_manager.get_coord_transform("ECLIPJ2000", "ECLIPJ2000", et)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        
        # Test J2000 to ECLIPJ2000 (Non-Identity roughly)
        # Actually they are close but defined.
        # Just check it returns valid 3x3 matrix
        R2 = spice_manager.get_coord_transform("J2000", "ECLIPJ2000", et)
        assert R2.shape == (3, 3)
        assert np.abs(np.linalg.det(R2) - 1.0) < 1e-10 # Determinant 1

    def test_body_constant(self):
        # Check Earth Radius
        rad = spice_manager.get_body_constant('EARTH', 'RADII')
        assert len(rad) == 3
        assert rad[0] > 6000 # km

    def test_j2_acceleration(self):
        # J2 acceleration at equator should be non-zero
        j2_pert = J2Perturbation('EARTH', 'ECLIPJ2000', 'IAU_EARTH')
        if j2_pert.J2 == 0:
            pytest.skip("J2 constant not found for EARTH")
            
        t = 0.0
        # Position on Equator (approx)
        r_sc = np.array([7000.0, 0.0, 0.0]) 
        
        acc = j2_pert.compute_acceleration(t, r_sc)
        assert np.linalg.norm(acc) > 0
        
        # J2 acceleration is roughly -1.5 * J2 * (mu/r^2) * (Re/r)^2 ...
        # Just ensure it runs and produces reasonable output order of magnitude
        # J2 ~ 1e-3, mu ~ 4e5, r ~ 7e3 -> a_point ~ 4e5/49e6 ~ 0.008
        # a_j2 ~ 1e-3 * 0.008 ~ 1e-5 km/s^2
        assert 1e-7 < np.linalg.norm(acc) < 1e-3

    def test_srp_acceleration(self):
        # Solar Radiation Pressure
        area = 10.0 # m^2
        mass = 1000.0 # kg
        cr = 1.8
        srp = SSRPerturbation(area, mass, cr, 'SUN')
        
        t = 0.0
        r_sc = np.array([1.5e8, 0.0, 0.0]) # 1 AU
        
        acc = srp.compute_acceleration(t, r_sc)
        
        # Should be small, radial
        assert np.linalg.norm(acc) > 0
        
        # Direction should be roughly away from Sun (along +X if r_sc is +X)
        # r_sc is relative to Sun in this case since central_body='SUN'
        acc_dir = acc / np.linalg.norm(acc)
        # S/C is at +X. Sun is at 0. Force is +X.
        np.testing.assert_allclose(acc_dir, [1, 0, 0], atol=1e-1) 

    def test_nbody_integration_with_j2(self):
        # Test full integration wrapper
        dynamics = NBodyDynamics(['EARTH'], 'ECLIPJ2000', 'EARTH', 
                                 perturbations=[J2Perturbation('EARTH')])
        
        t_span = (0, 1000)
        state0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]) # LEO-ish
        
        sol = dynamics.propagate(state0, t_span)
        assert sol.status == 0
        assert sol.y.shape[0] == 6

    def test_stm_integration_with_j2_no_partials(self):
        # Currently J2 partials are zero, so STM should just propagate but not reflect J2 variations perfectly
        # But code should run.
        dynamics = NBodyDynamics(['EARTH'], 'ECLIPJ2000', 'EARTH', 
                                 perturbations=[J2Perturbation('EARTH', 'ECLIPJ2000', 'IAU_EARTH')])
        
        state0 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        sol = dynamics.propagate(state0, (0, 100), stm=True)
        assert sol.y.shape[0] == 42
