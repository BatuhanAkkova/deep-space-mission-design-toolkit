import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from typing import Callable, List, Union
from .indirect import IndirectSolver

class MultipleShootingSolver(IndirectSolver):
    """
    Solves the Indirect OCP using Multiple Shooting.
    Inherits from IndirectSolver to reuse symbolic derivation of PMP conditions.
    """
    def __init__(self, ocp):
        super().__init__(ocp)

    def solve(self, t_guess, y_guess, bc_func: Callable, verbose=2, tol=1e-6, max_nodes=None, free_time=False, events=None):
        """
        Solves the TPBVP using Multiple Shooting.

        Args:
            t_guess: Initial mesh.
            y_guess: Initial guess.
            bc_func: Boundary condition function bc(y_0, y_f, p). 
            free_time: If True, optimizes for unknown parameters (assumed [tf]).
            events: Optional. List of callables, one per segment (or None).
                    If events[i] is provided, segment i integration stops at the event.
                    The "end time" of this segment becomes variable.
                    Note: If using events, the time grid structure changes.
        """
        if self._numeric_dynamics is None:
            raise RuntimeError("System not lambdified. Call derive_conditions() and lambdify_system() first.")

        # 1. Setup
        num_nodes = len(t_guess)
        num_segments = num_nodes - 1
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

        # Flatten Z = [Y_0, ..., Y_N-1, p]
        Z_nodes = y_guess.T.flatten() 
        Z0 = np.concatenate((Z_nodes, p_guess))

        # 2. Integration Function
        ode_fun = self.get_ode_fun(free_time=free_time)

        def shooting_function(Z):
            # Extract params
            if num_params > 0:
                p_current = Z[-num_params:]
                Y_flat = Z[:-num_params]
            else:
                p_current = []
                Y_flat = Z
            
            # Reshape Y
            Y_nodes = Y_flat.reshape((num_nodes, num_vars))
            
            residuals = []
            
            # 2.1 Integrate each segment
            for i in range(num_segments):
                t_start = t_grid[i]
                t_end = t_grid[i+1]
                
                # If leg ends on event, t_end is just a max horizon, integration stops earlier
                event_i = events[i] if events and i < len(events) else None
                
                y_start = Y_nodes[i]
                y_next_guess = Y_nodes[i+1]
                
                # Propagate
                def step_fun(t, y):
                    return ode_fun(t, y, p_current)
                
                # Setup events for solve_ivp
                ivp_events = [event_i] if event_i else None
                if ivp_events:
                     # Helper to wrap event if needed (e.g. to ensure terminal=True)
                     # But we assume user passes proper event function
                     if not hasattr(event_i, 'terminal'):
                         event_i.terminal = True
                     
                sol = solve_ivp(step_fun, (t_start, t_end), y_start, t_eval=[t_end] if not event_i else None, 
                                events=ivp_events, rtol=tol, atol=tol)
                
                if not sol.success:
                    return np.ones_like(Z) * 1e6
                
                # Get final state
                if event_i and sol.t_events and len(sol.t_events[0]) > 0:
                    # Stopped at event
                    y_end_integrated = sol.y_events[0][0]
                    # t_event = sol.t_events[0][0]
                    # Note: y_next_guess corresponds to state AT THE EVENT?
                    # Yes. Standard MS matches state at node i+1 with integrated state from i.
                    pass
                else:
                    # Reached t_end
                    y_end_integrated = sol.y[:, -1]
                
                # Defect
                defect = y_end_integrated - y_next_guess
                residuals.append(defect)
            
            residuals = np.array(residuals).flatten() 
            
            # 2.2 Boundary Conditions
            y_initial = Y_nodes[0]
            y_final = Y_nodes[-1]
            
            bc_res = bc_func(y_initial, y_final, p_current)
            
            total_residuals = np.concatenate((residuals, bc_res))
            
            return total_residuals

        # 3. Solve
        if verbose > 0:
            print(f"Starting Multiple Shooting (Free Time={free_time}, Events={events is not None})...")
            
        sol = root(shooting_function, Z0, method='lm', options={'maxiter': 100 if max_nodes is None else max_nodes}) 
        
        if verbose > 0:
            print(f"Multiple Shooting {'Converged' if sol.success else 'Failed'}: {sol.message}")
            if hasattr(sol, 'fun'):
                print(f"Final residual norm: {np.linalg.norm(sol.fun)}")

        # 4. Format Output
        if num_params > 0:
            p_sol = sol.x[-num_params:]
            Y_sol_flat = sol.x[:-num_params]
        else:
            p_sol = None
            Y_sol_flat = sol.x
            
        Y_sol = Y_sol_flat.reshape((num_nodes, num_vars)).T 
        
        # Denormalize time if needed
        if free_time and p_sol is not None:
             t_final_res = t_grid * p_sol[0]
        else:
             t_final_res = t_guess
        
        return Result(t_final_res, Y_sol, p_sol, sol.success, sol.message)

class Result:
    def __init__(self, t, y, p, success, message):
        self.x = t
        self.y = y
        self.p = p
        self.success = success
        self.message = message
