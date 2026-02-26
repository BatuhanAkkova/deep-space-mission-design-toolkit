import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Callable, Optional, Union, List

class SingleShootingSolver:
    """
    Solves Two-Point Boundary Value Problems (TPBVP) using the Single Shooting method.
    This method interprets the BVP as a Root-Finding problem for the unknown 
    initial states or parameters.
    """

    def __init__(self, dynamics: Callable[[float, np.ndarray, Optional[np.ndarray]], np.ndarray]):
        """
        Args:
            dynamics: Callable f(t, y, p) -> dy/dt.
                      t: scalar time
                      y: state vector
                      p: parameter vector (optional)
        """
        self.dynamics = dynamics

    def solve(self, 
              t_guess: np.ndarray, 
              y_guess: np.ndarray, 
              bc_func: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
              verbose: int = 2,
              tol: float = 1e-6,
              max_nodes: Optional[int] = None, # Used as max_nfev
              free_time: bool = False,
              bounds: tuple = (-np.inf, np.inf)):
        """
        Solves the TPBVP using Single Shooting.

        Args:
            t_guess: Initial time mesh. Used for t0 and tf.
            y_guess: Initial guess for the state. If 2D (n_vars, n_nodes), y_guess[:, 0] is used.
                     If 1D (n_vars), it is used as y0.
            bc_func: Boundary condition function bc(y0, yf, p) -> residuals.
            verbose: Verbosity level (0, 1, 2).
            tol: Integration tolerance (rtol, atol).
            max_nodes: Maximum number of function evaluations (optimizer iterations).
            free_time: If True, optimizes for final time tf. 
                       Assumes p[0] is tf (or total duration).
                       Integrates on [0, 1] internally.
            bounds: Bounds for optimization variables [y0, p].
                    Consistent with scipy.optimize.least_squares.
        
        Returns:
            Result object with fields: t, y, p, success, message, fun.
        """
        
        # 1. Parse Inputs
        t0 = t_guess[0]
        tf_guess = t_guess[-1]

        # Handle y_guess
        if y_guess.ndim == 2:
            y0_guess = y_guess[:, 0]
        else:
            y0_guess = y_guess
            
        num_vars = len(y0_guess)

        # Handle Parameters (p)
        
        p_guess = []
        if free_time:
            p_guess = [tf_guess]
        
        Z0 = np.concatenate((y0_guess, p_guess))
        num_params = len(p_guess)

        # 2. Define Shooting Function
        def shooting_function(Z):
            # Extract y0 and p
            if num_params > 0:
                current_y0 = Z[:-num_params]
                current_p = Z[-num_params:]
            else:
                current_y0 = Z
                current_p = []
                
            # Setup Integration
            if free_time:
                # Scaled dynamics: dt/dtau = tf
                # t in [0, 1]
                tf_val = current_p[0]
                if tf_val <= 1e-6:
                     # Penalize invalid time
                     return np.ones(num_vars + num_params) * 1e3
                
                t_span = (0.0, 1.0)
                t_eval = [1.0]
                
                def scaled_dynamics(t, y):
                    return self.dynamics(t, y, current_p) * tf_val
                
                fun = scaled_dynamics
            else:
                t_span = (t0, tf_guess)
                t_eval = [tf_guess]
                fun = lambda t, y: self.dynamics(t, y, current_p)

            # Integrate
            try:
                sol = solve_ivp(fun, t_span, current_y0, t_eval=t_eval, rtol=tol, atol=tol, method='DOP853')
                
                if not sol.success:
                    return np.ones(num_vars + num_params) * 1e6
                
                y_final = sol.y[:, -1]
                
                # Evaluate BCs
                # bc_func(y0, yf, p)
                residuals = bc_func(current_y0, y_final, current_p)
                return np.array(residuals).flatten()
            except Exception:
                return np.ones(num_vars + num_params) * 1e6

        # 3. Optimize
        if verbose > 0:
            print(f"Starting Single Shooting (Free Time={free_time})...")
            
        max_nfev = 100 if max_nodes is None else max_nodes
        
        res = least_squares(shooting_function, Z0, bounds=bounds, verbose=verbose, max_nfev=max_nfev)
        
        # 4. Reconstruct Solution
        if num_params > 0:
            p_sol = res.x[-num_params:]
            y0_sol = res.x[:-num_params]
        else:
            p_sol = []
            y0_sol = res.x

        if free_time:
            t_span = (0.0, 1.0)
            tf_sol = p_sol[0]
            # Grid for output
            steps = len(t_guess) if len(t_guess) > 1 else 100
            t_eval = np.linspace(0, 1, steps)
            
            fun = lambda t, y: self.dynamics(t, y, p_sol) * tf_sol
        else:
            t_span = (t0, tf_guess)
            t_eval = t_guess if len(t_guess) > 1 else np.linspace(t0, tf_guess, 100)
            fun = lambda t, y: self.dynamics(t, y, p_sol)

        final_sol = solve_ivp(fun, t_span, y0_sol, t_eval=t_eval, rtol=tol, atol=tol, method='DOP853')
        
        Y_sol = final_sol.y
        t_sol = final_sol.t
        
        if free_time:
            t_sol = t_sol * p_sol[0] + t0 # Rescale time
            
        return Result(t_sol, Y_sol, p_sol if len(p_sol) > 0 else None, res.success, res.message, res.fun)

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
