import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.optimization.initial_guess import InitialGuesser

def test_initial_guess_projectile():
    """
    Tests grid search and propagation on a simple projectile problem.
    Target: Hit x=10, y=0 at t=2.
    Gravity: g=1.0 (downwards)
    Start: x=0, y=0.
    
    Analytical solution:
    x(t) = vx * t
    y(t) = vy * t - 0.5 * g * t^2
    
    At t=2:
    10 = vx * 2  => vx = 5
    0 = vy * 2 - 0.5 * 1.0 * 4 => 0 = 2*vy - 2 => vy = 1
    
    So target velocity is (5, 1, 0) magnitude sqrt(26) approx 5.099.
    """
    
    # Dynamics: [x, y, z, vx, vy, vz]
    def dynamics(t, state):
        x, y, z, vx, vy, vz = state
        g = 1.0
        ax = 0
        ay = -g
        az = 0
        return np.array([vx, vy, vz, ax, ay, az])
        
    def objective(v_vec):
        # Propagate manually for objective to keep it decoupled from the class being tested if we wanted,
        # but here we trust the physics is simple.
        # Actually grid_search expects objective_func(v_vec) -> cost.
        # We can use analytical end state for speed in the objective or short propagation.
        # Let's use analytical for the objective function to test the Grid Search part pure.
        
        t = 2.0
        g = 1.0
        # Use offset r0
        r0 = np.array([0.0, 100.0, 0.0])
        
        # v_vec is (vx, vy, vz)
        r_final_x = r0[0] + v_vec[0]*t
        r_final_y = r0[1] + v_vec[1]*t - 0.5*g*t**2
        
        # Target also offset by 100 in Y
        target = np.array([10.0, 100.0, 0.0])
        diff = np.array([r_final_x, r_final_y, 0.0]) - target
        
        return np.linalg.norm(diff)

    # Offset r0 to avoid singularity in grid search plane definition
    r0 = np.array([0.0, 100.0, 0.0])
    
    # Nominal guess: say (4, 2, 0)
    v0_nom = np.array([4.0, 2.0, 0.0])  
    
    # Search around nominal
    # vx needs to go 4 -> 5 (+25%)
    # vy needs to go 2 -> 1 (-50%)
    
    # Let's give a wide search range
    res = InitialGuesser.find_guess_and_resample(
        objective,
        dynamics,
        r0,
        v0_nom,
        t_span=(0, 2.0),
        num_points=20,
        mag_diff_pct=0.5, # Search +/- 50% magnitude
        angle_diff_rad=0.5, # Search +/- 0.5 rad (~28 deg)
        grid_points=50
    )
    
    best_v = res['best_v']
    print(f"Result Velocity: {best_v}")
    
    # Check values
    assert np.isclose(best_v[0], 5.0, atol=0.2) # Allow some grid coarseness error
    assert np.isclose(best_v[1], 1.0, atol=0.2)
    
    # Check trajectory shape
    t_eval = res['t_eval']
    state = res['state_eval']
    
    assert len(t_eval) == 20
    assert state.shape == (20, 6)
    
    # Check final state of trajectory
    xf = state[-1]
    # x should be ~10, y should be ~100
    assert np.isclose(xf[0], 10.0, atol=0.2) 
    assert np.isclose(xf[1], 100.0, atol=0.2)

if __name__ == "__main__":
    test_initial_guess_projectile()
