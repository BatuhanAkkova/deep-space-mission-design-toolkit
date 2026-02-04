import numpy as np
import sympy as sp
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple, Callable
from .ocp import OptimalControlProblem

class DirectCollocation:
    """
    Solves an Optimal Control Problem using Direct Collocation (Hermite-Simpson).
    """
    def __init__(self, ocp: OptimalControlProblem, num_nodes: int = 50, method: str = "hermite-simpson"):
        self.ocp = ocp
        self.num_nodes = num_nodes
        self.method = method
        self.N_nodes = num_nodes
        self.N_segments = num_nodes - 1
        
        # Determine problem dimensions
        self.n_x = len(ocp.states)
        self.n_u = len(ocp.controls)
        self.n_p = len(ocp.parameters)
        
        # Internal storage for lambdified functions
        self.func_dynamics = None
        self.func_L = None
        self.func_phi = None
        self.func_constraints: Dict[str, List] = {}
        
        # Prepare functions
        self._lambdify_problem()

    def _lambdify_problem(self):
        """Converts symbolic expressions to fast numpy callbacks."""
        # Standard args: (t, x, u, p)
        
        args = [self.ocp.t] + self.ocp.states + self.ocp.controls + self.ocp.parameters
        
        # Dynamics f(t, x, u, p)
        self.func_dynamics = sp.lambdify(args, self.ocp.dynamics, modules=['numpy', 'math'])
        
        # Running cost L(t, x, u, p)
        self.func_L = sp.lambdify(args, self.ocp.L, modules=['numpy', 'math'])
        
        # Terminal cost phi(t, x, p)
        # Usually phi is phi(x(tf), tf, p)
        args_terminal = [self.ocp.t] + self.ocp.states + self.ocp.parameters
        self.func_phi = sp.lambdify(args_terminal, self.ocp.phi, modules=['numpy', 'math'])
        
        # Boundary constraints
        self.func_constraints = {
            'initial': [],
            'final': [],
            'path': []
        }
        
        for kind, exprs in self.ocp.constraints.items():
            for expr in exprs:
                # Use full args for generality
                f = sp.lambdify(args, expr, modules=['numpy', 'math'])
                self.func_constraints[kind].append(f)

    def _unpack_decision_vector(self, Z):
        """
        Z = [p, x0, u0, x1, u1, ..., xN, uN]
        """
        idx = 0
        
        # Parameters
        p = Z[idx : idx + self.n_p]
        idx += self.n_p
        
        # Trajectories
        traj_data = Z[idx:]
        traj_matrix = traj_data.reshape((self.N_nodes, self.n_x + self.n_u))
        
        X = traj_matrix[:, :self.n_x] # Shape (N, n_x)
        U = traj_matrix[:, self.n_x:] # Shape (N, n_u)
        
        return p, X, U

    def solve(self, t0, tf_guess, initial_guess_x=None, initial_guess_u=None, initial_guess_p=None, bounds=None):
        """
        Solves the NLP.
        
        Args:
            t0: Initial time (fixed).
            tf_guess: Final time guess.
            initial_guess_x: Array (N, n_x) or None
            initial_guess_u: Array (N, n_u) or None
            initial_guess_p: Array (n_p) or None
        """
        # 1. Setup Time Grid
        tf = tf_guess # Fixed for now
        t_grid = np.linspace(t0, tf, self.N_nodes)
        dt = t_grid[1] - t_grid[0] # Constant step size for now
        
        # 2. Construct Initial Guess Z0
        if initial_guess_p is None:
            initial_guess_p = np.zeros(self.n_p)
            
        if initial_guess_x is None:
            initial_guess_x = np.zeros((self.N_nodes, self.n_x))
            
        if initial_guess_u is None:
            initial_guess_u = np.zeros((self.N_nodes, self.n_u))
            
        # Interleave
        traj_data = np.hstack((initial_guess_x, initial_guess_u)) # (N, n_x + n_u)
        Z0 = np.concatenate((initial_guess_p, traj_data.flatten()))
        
        # 3. Define Objective Function
        def objective(Z):
            p, X, U = self._unpack_decision_vector(Z)
            
            # Mayer Term (Terminal Cost)
            cost = self.func_phi(t_grid[-1], *X[-1], *p) 
            
            # Lagrange Term (Running Cost)
            running_cost = 0.0
            for k in range(self.N_segments):
                x_k = X[k]
                u_k = U[k]
                t_k = t_grid[k]
                
                x_k1 = X[k+1]
                u_k1 = U[k+1]
                t_k1 = t_grid[k+1]
                
                val_k = self.func_L(t_k, *x_k, *u_k, *p)
                val_k1 = self.func_L(t_k1, *x_k1, *u_k1, *p)
                
                if self.method == "hermite-simpson":
                    # Simpson Quadrature
                    # Need x_mid, u_mid to evaluate L_mid
                    
                    # Re-evaluate dynamics for interpolation
                    f_k = np.array(self.func_dynamics(t_k, *x_k, *u_k, *p))
                    f_k1 = np.array(self.func_dynamics(t_k1, *x_k1, *u_k1, *p))
                    
                    x_mid = 0.5 * (x_k + x_k1) + (dt / 8.0) * (f_k - f_k1)
                    u_mid = 0.5 * (u_k + u_k1)
                    t_mid = t_k + 0.5 * dt
                    
                    val_mid = self.func_L(t_mid, *x_mid, *u_mid, *p)
                    
                    running_cost += (dt / 6.0) * (val_k + 4*val_mid + val_k1)
                else:
                    # Trapezoidal
                    running_cost += 0.5 * (val_k + val_k1) * dt
            
            return cost + running_cost

        # 4. Define Constraints
        constraints = []
        
        # 4.1 Defect Constraints (Dynamics)
        def defect_constraints(Z):
            p, X, U = self._unpack_decision_vector(Z)
            defects = []
            
            for k in range(self.N_segments):
                x_k = X[k]
                u_k = U[k]
                t_k = t_grid[k]
                
                x_k1 = X[k+1]
                u_k1 = U[k+1]
                t_k1 = t_grid[k+1]
                
                # Eval dynamics
                f_k = np.array(self.func_dynamics(t_k, *x_k, *u_k, *p))
                f_k1 = np.array(self.func_dynamics(t_k1, *x_k1, *u_k1, *p))
                
                if self.method == "hermite-simpson":
                    # Hermite-Simpson Collocation
                    # x_mid_interp = 0.5 * (x_k + x_k1) + (dt / 8.0) * (f_k - f_k1)
                    # u_mid = 0.5 * (u_k + u_k1) (FOH)
                    # t_mid = t_k + 0.5 * dt
                    
                    # f_mid = f(t_mid, x_mid_interp, u_mid, p)
                    # defect = x_k1 - x_k - (dt / 6.0) * (f_k + 4 * f_mid + f_k1)
                    
                    x_mid = 0.5 * (x_k + x_k1) + (dt / 8.0) * (f_k - f_k1)
                    u_mid = 0.5 * (u_k + u_k1)
                    t_mid = t_k + 0.5 * dt
                    
                    f_mid = np.array(self.func_dynamics(t_mid, *x_mid, *u_mid, *p))
                    
                    defect = x_k1 - x_k - (dt / 6.0) * (f_k + 4*f_mid + f_k1)
                    
                else:
                    # Trapezoidal defaulting
                    defect = x_k1 - x_k - 0.5 * dt * (f_k + f_k1)
                
                defects.extend(defect)
            
            return np.array(defects)

        constraints.append({'type': 'eq', 'fun': defect_constraints})
        
        # 4.2 Boundary Constraints
        def boundary_constraints(Z):
            p, X, U = self._unpack_decision_vector(Z)
            vals = []
            
            # Initial Constraints
            if 'initial' in self.func_constraints:
                for f in self.func_constraints['initial']:
                    # Eval at k=0
                    vals.append(f(t_grid[0], *X[0], *U[0], *p))
                    
            # Final Constraints
            if 'final' in self.func_constraints:
                for f in self.func_constraints['final']:
                    # Eval at k=N
                    vals.append(f(t_grid[-1], *X[-1], *U[-1], *p))
            
            return np.array(vals)
        
        if self.func_constraints.get('initial') or self.func_constraints.get('final'):
            constraints.append({'type': 'eq', 'fun': boundary_constraints})
            
        # 4.3 Path Constraints (Equality for now)
        
        # 5. Run Optimization
        res = minimize(
            objective, 
            Z0, 
            method='SLSQP', 
            constraints=constraints, 
            bounds=bounds, 
            options={'maxiter': 500, 'disp': True, 'ftol': 1e-6}
        )
        
        # 6. Extract Result
        p_opt, X_opt, U_opt = self._unpack_decision_vector(res.x)
        
        return {
            'success': res.success,
            'message': res.message,
            't': t_grid,
            'x': X_opt,
            'u': U_opt,
            'p': p_opt,
            'cost': res.fun,
            'raw_res': res # Return raw for inspection
        }
