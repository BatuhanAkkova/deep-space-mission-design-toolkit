import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from typing import Callable, List, Optional, Union

class MultipleShootingSolver:
    """
    Solves Two-Point Boundary Value Problems (TPBVP) using the Multiple Shooting method.
    This method breaks the trajectory into multiple segments and ensures continuity 
    while satisfying boundary conditions.
    """

    def __init__(self, dynamics: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray]):
        """
        Args:
            dynamics: Callable f(t, y, p) -> dy/dt.
        """
        self.dynamics = dynamics

    def solve(self, 
              t_guess: np.ndarray, 
              y_guess: np.ndarray, 
              bc_func: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
              verbose: int = 2, 
              tol: float = 1e-6, 
              max_nodes: Optional[int] = None, 
              free_time: bool = False, 
              events: Optional[List[Callable]] = None):
        """
        Solves the TPBVP using Multiple Shooting.

        Args:
            t_guess: Initial time mesh (nodes).
            y_guess: Initial guess for states at each node (num_vars, num_nodes).
            bc_func: Boundary condition function bc(y_0, y_f, p).
            verbose: Verbosity level.
            tol: Integration tolerance.
            max_nodes: Max iterations for root finder.
            free_time: If True, optimizes for unknown tf (assumed p[0]).
            events: List of event functions for each segment (optional).
        
        Returns:
            Result object.
        """
        
        # 1. Setup
        num_nodes = len(t_guess)
        num_segments = num_nodes - 1
        
        # Check y_guess shape
        if y_guess.shape[1] != num_nodes:
            raise ValueError(f"y_guess shape {y_guess.shape} must match number of nodes {num_nodes}.")
            
        num_vars = y_guess.shape[0]
        
        # P handling
        p_guess = []
        if free_time:
            tf_guess = t_guess[-1]
            p_guess = [tf_guess]
            t_grid = t_guess / tf_guess
        else:
            t_grid = t_guess
            
        num_params = len(p_guess)

        # Flatten Z = [y_0, y_1, ..., y_N-1, p]
        Z_nodes = y_guess.T.flatten()
        Z0 = np.concatenate((Z_nodes, p_guess))

        # 2. Shooting Function
        def shooting_function(Z):
            # Extract params
            if num_params > 0:
                p_current = Z[-num_params:]
                Y_flat = Z[:-num_params]
            else:
                p_current = []
                Y_flat = Z
            
            # Reshape Y back to (n_nodes, n_vars)
            Y_nodes = Y_flat.reshape((num_nodes, num_vars))
            
            residuals = []
            
            # Loop over segments
            for i in range(num_segments):
                t_start_fraction = t_grid[i]
                t_end_fraction = t_grid[i+1]
                
                # Determine integration times
                if free_time:
                    tf_val = p_current[0]
                    t_start = t_start_fraction * tf_val
                    t_end = t_end_fraction * tf_val
                    
                    if tf_val < 1e-6:
                        pass
                else:
                    t_start = t_grid[i]
                    t_end = t_grid[i+1]
                
                y_start = Y_nodes[i]
                y_target = Y_nodes[i+1]
                
                # Event handling
                event_i = events[i] if events and i < len(events) else None
                ivp_events = [event_i] if event_i else None
                if ivp_events and not hasattr(event_i, 'terminal'):
                     event_i.terminal = True
                
                # Integrate
                try:
                    # solve_ivp expects fun(t, y)
                    fun = lambda t, y: self.dynamics(t, y, p_current)
                    
                    sol = solve_ivp(fun, (t_start, t_end), y_start, 
                                    t_eval=[t_end] if not event_i else None,
                                    events=ivp_events, rtol=tol, atol=tol, method='DOP853')
                    
                    if not sol.success:
                        # Large defect
                        return np.ones_like(Z) * 1e3
                    
                    # Get end state
                    if event_i and sol.t_events and len(sol.t_events[0]) > 0:
                        y_end_integrated = sol.y_events[0][0]
                    else:
                        y_end_integrated = sol.y[:, -1]
                    
                    # Defect: y_integrated - y_guess_next
                    defect = y_end_integrated - y_target
                    residuals.append(defect)
                    
                except Exception:
                     return np.ones_like(Z) * 1e3

            residuals = np.array(residuals).flatten()
            
            # Boundary Conditions
            y_initial = Y_nodes[0]
            y_final = Y_nodes[-1]
            
            bc_res = bc_func(y_initial, y_final, p_current)
            
            total_residuals = np.concatenate((residuals, bc_res))
            return total_residuals

        # 3. Solve
        if verbose > 0:
            print(f"Starting Multiple Shooting (Free Time={free_time}, Nodes={num_nodes})...")
            
        sol = root(shooting_function, Z0, method='lm', options={'maxiter': 100 if max_nodes is None else max_nodes})
        
        if verbose > 0:
            print(f"Multiple Shooting {'Converged' if sol.success else 'Failed'}: {sol.message}")
            if hasattr(sol, 'fun'):
                print(f"Final residual norm: {np.linalg.norm(sol.fun)}")
                
        # 4. Result
        if num_params > 0:
            p_sol = sol.x[-num_params:]
            Y_sol_flat = sol.x[:-num_params]
        else:
            p_sol = None
            Y_sol_flat = sol.x
            
        Y_sol = Y_sol_flat.reshape((num_nodes, num_vars)).T # (n_vars, n_nodes)
        
        # Reconstruct Time
        if free_time and p_sol is not None:
            t_sol = t_grid * p_sol[0]
        else:
            t_sol = t_guess
            
        return Result(t_sol, Y_sol, p_sol, sol.success, sol.message, sol.fun)

class Result:
    """
    Data class for storing solver results.
    """
    def __init__(self, t, y, p, success, message, fun=None):
        self.x = t
        self.y = y
        self.p = p
        self.success = success
        self.message = message
        self.fun = fun
