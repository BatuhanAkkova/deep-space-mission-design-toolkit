import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from optimization.single_shooting import SingleShootingSolver
from optimization.multiple_shooting import MultipleShootingSolver

def double_integrator_dynamics(t, y, p=None):
    # y = [x1, x2, lam1, lam2]
    x1, x2, lam1, lam2 = y
    
    # u = -lam2
    u = -lam2
    
    dx1 = x2
    dx2 = u
    dlam1 = 0.0
    dlam2 = -lam1
    
    return np.array([dx1, dx2, dlam1, dlam2])

def bc_func_rest_to_rest(y0, yf, p=None):
    # x(0) = 0, v(0) = 0
    # x(1) = 1, v(1) = 0
    
    res = [
        y0[0] - 0.0,
        y0[1] - 0.0,
        yf[0] - 1.0,
        yf[1] - 0.0
    ]
    return np.array(res)

def test_single_shooting():
    solver = SingleShootingSolver(double_integrator_dynamics)
    
    t0 = 0.0
    tf = 1.0
    
    # Guess: x roughly 0.5, v roughly 0, costates usually order 1-10
    y0_guess = np.array([0.0, 0.0, 1.0, 1.0]) 
    
    res = solver.solve(
        t_guess=np.array([t0, tf]),
        y_guess=y0_guess,
        bc_func=bc_func_rest_to_rest,
        verbose=1
    )
    
    assert res.success, f"Single shooting failed: {res.message}"
    
    # Check final state
    xf = res.y[:, -1]
    assert np.isclose(xf[0], 1.0, atol=1e-3)
    assert np.isclose(xf[1], 0.0, atol=1e-3)
    
    print("\nSingle Shooting Solution (Rest-to-Rest):")
    print("Initial Costates:", res.y[2:, 0])

def test_multiple_shooting():
    solver = MultipleShootingSolver(double_integrator_dynamics)
    
    t0 = 0.0
    tf = 1.0
    num_nodes = 5
    t_guess = np.linspace(t0, tf, num_nodes)
    
    # Guess trajectory
    # Linear interpolation for x1 from 0 to 1
    # x2 constant 0 (bad guess but should work)
    # Costates constant 1
    y_guess = np.zeros((4, num_nodes))
    y_guess[0, :] = np.linspace(0, 1, num_nodes)
    y_guess[2, :] = -10.0 # Heuristic
    y_guess[3, :] = 10.0
    
    res = solver.solve(
        t_guess=t_guess,
        y_guess=y_guess,
        bc_func=bc_func_rest_to_rest,
        verbose=1
    )
    
    assert res.success, f"Multiple shooting failed: {res.message}"
    
    xf = res.y[:, -1]
    assert np.isclose(xf[0], 1.0, atol=1e-3)
    assert np.isclose(xf[1], 0.0, atol=1e-3)

    print("\nMultiple Shooting Solution:")
    print("Initial Costates:", res.y[2:, 0])

if __name__ == "__main__":
    test_single_shooting()
    test_multiple_shooting()
