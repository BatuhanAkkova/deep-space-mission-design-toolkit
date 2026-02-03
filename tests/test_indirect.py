import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import sympy as sp
from src.optimization.ocp import OptimalControlProblem
from src.optimization.indirect import IndirectSolver

class TestIndirectSolver:
    
    def test_double_integrator_fixed_final_state(self):
        """
        Test simple double integrator with fixed final state.
        min J = integral(0.5 * u^2) dt
        s.t. x_dot = v, v_dot = u
        x(0)=0, v(0)=0
        x(1)=1, v(1)=0
        """
        ocp = OptimalControlProblem()
        ocp.set_time_variable('t')
        
        x, v = ocp.define_states(['x', 'v'])
        u_list = ocp.define_controls(['u'])
        u = u_list[0]
        
        ocp.set_dynamics([v, u])
        
        # Cost: 0.5 * u^2
        ocp.set_running_cost(0.5 * u**2)
        
        solver = IndirectSolver(ocp)
        solver.derive_conditions()
        solver.lambdify_system()
        
        # BCs
        x0 = np.array([0.0, 0.0])
        xf = np.array([1.0, 0.0])
        
        bc_func = solver.create_bc_function(x0, xf, free_time=False)
        
        # Initial guess
        t_guess = np.linspace(0, 1, 11)
        y_guess = np.zeros((4, 11))
        # Guess for states: linear interp
        y_guess[0, :] = t_guess # x goes 0 to 1
        y_guess[1, :] = 0 # v
        
        res = solver.solve(t_guess, y_guess, bc_func=bc_func, verbose=0)
        
        assert res.success
        
        # Check final state
        files_x = res.y[0, -1]
        files_v = res.y[1, -1]
        
        assert np.isclose(files_x, 1.0, atol=1e-3)
        assert np.isclose(files_v, 0.0, atol=1e-3)
        
    def test_double_integrator_partial_state_constraint(self):
        """
        Test double integrator with free final velocity.
        x(1)=1, v(1)=FREE (implies lambda_v(1) = 0)
        Analytical result: x(t) = 1.5t^2 - 0.5t^3
        v(1) should be 1.5
        """
        ocp = OptimalControlProblem()
        ocp.set_time_variable('t')
        
        x, v = ocp.define_states(['x', 'v'])
        u_list = ocp.define_controls(['u'])
        u = u_list[0]
        
        ocp.set_dynamics([v, u])
        ocp.set_running_cost(0.5 * u**2)
        
        solver = IndirectSolver(ocp)
        solver.derive_conditions()
        solver.lambdify_system()
        
        x0 = np.array([0.0, 0.0])
        # Partial constraint: x(1)=1, v(1)=NaN (free)
        xf = np.array([1.0, np.nan])
        
        bc_func = solver.create_bc_function(x0, xf, free_time=False)
        
        t_guess = np.linspace(0, 1, 11)
        y_guess = np.zeros((4, 11))
        
        res = solver.solve(t_guess, y_guess, bc_func=bc_func, verbose=0)
        
        assert res.success
        
        final_x = res.y[0, -1]
        final_v = res.y[1, -1]
        final_lam_v = res.y[3, -1] # 4th component is lambda_v
        
        # Check constraints
        assert np.isclose(final_x, 1.0, atol=1e-3)
        
        # Check transversality condition: lambda_v(1) ~ 0
        assert np.isclose(final_lam_v, 0.0, atol=1e-3)
        
        # Check analytical prediction for v(1)
        # v(1) = 1.5
        assert np.isclose(final_v, 1.5, atol=1e-3)

if __name__ == "__main__":
    # Allow running directly
    t = TestIndirectSolver()
    t.test_double_integrator_fixed_final_state()
    t.test_double_integrator_partial_state_constraint()
    print("All tests passed!")
