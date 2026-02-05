import numpy as np
from scipy.integrate import solve_ivp
from src.optimization.grid_search import grid_search_velocity_plane

class InitialGuesser:
    """
    Refined initial guess generator using grid search and propagation.
    """
    
    @staticmethod
    def propagate_and_resample(dynamics_func, x0, t_span, num_points=100, integrator_params=None):
        """
        Propagates a trajectory and resamples it to a fixed grid.
        
        Args:
            dynamics_func (callable): Function dynamics(t, y) returning dy/dt.
            x0 (np.array): Initial state vector.
            t_span (tuple): (t0, tf).
            num_points (int): Number of points for the output grid.
            integrator_params (dict): Optional parameters for solve_ivp (e.g. 'rtol', 'atol').
            
        Returns:
            np.array: t_eval (time grid).
            np.array: state_eval (interpolated state history, shape (num_points, state_dim)).
        """
        if integrator_params is None:
            # Default tight tolerances for decent precision
            integrator_params = {'rtol': 1e-9, 'atol': 1e-9}
            
        t0, tf = t_span
        t_eval = np.linspace(t0, tf, num_points)
        
        sol = solve_ivp(dynamics_func, t_span, x0, t_eval=t_eval, **integrator_params)
        
        if not sol.success:
            print(f"Warning: Propagation failed: {sol.message}")
            
        return sol.t, sol.y.T

    @staticmethod
    def find_guess_and_resample(objective_func, dynamics_func, r0, v0_nom, t_span, num_points=100, 
                                mag_diff_pct=0.1, angle_diff_rad=0.2, grid_points=10):
        """
        Uses grid search to find the best initial velocity, then generates a full trajectory guess.
        
        Args:
            objective_func (callable): Function(v_vec) -> cost. Used by grid search.
            dynamics_func (callable): Function used for propagation.
            r0 (np.array): Initial position.
            v0_nom (np.array): Nominal initial velocity guess.
            t_span (tuple): (t0, tf).
            num_points (int): Number of points for the resampled trajectory.
            mag_diff_pct (float): Grid search velocity magnitude range (fraction).
            angle_diff_rad (float): Grid search angle range (radians).
            grid_points (int): Resolution of the grid search (N x N).
            
        Returns:
            dict: Dictionary containing:
                - 't_eval': Time history.
                - 'state_eval': State history.
                - 'best_v': Optimized initial velocity.
                - 'best_cost': Cost of the best solution.
                - 'grid_results': Full output from grid_search_velocity_plane.
        """
        
        # 1. Run Grid Search
        print("Running grid search for initial guess...")
        grid_res = grid_search_velocity_plane(
            objective_func, 
            r0, 
            v0_nom, 
            mag_diff_pct=mag_diff_pct, 
            angle_diff_rad=angle_diff_rad, 
            num_points=grid_points
        )
        
        best_v = grid_res['best_v']
        best_cost = grid_res['best_cost']
        
        print(f"Grid search found best v: {best_v} with cost {best_cost:.4e}")
        
        # 2. Propagate and Resample
        x0 = np.concatenate((r0, best_v))
        t_eval, state_eval = InitialGuesser.propagate_and_resample(
            dynamics_func, x0, t_span, num_points=num_points
        )
        
        return {
            't_eval': t_eval,
            'state_eval': state_eval,
            'best_v': best_v,
            'best_cost': best_cost,
            'grid_results': grid_res
        }
