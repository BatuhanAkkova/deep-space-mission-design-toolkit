import sympy as sp
import numpy as np
from scipy.integrate import solve_bvp
from typing import List, Callable, Optional, Dict
from .ocp import OptimalControlProblem

class IndirectSolver:
    """
    Solves an Optimal Control Problem using the Indirect Method.
    Transforms OCP -> TPBVP via Pontryagin's Maximum Principle.
    """
    def __init__(self, ocp: OptimalControlProblem):
        self.ocp = ocp
        self.costates = []
        self.H = None
        self.control_eqs = {} # Maps control symbol to expression (if explicit)
        self.state_eqs = []
        self.costate_eqs = []
        
        # Numeric functions
        self.f_sys = None # Function for dy/dt = [x_dot, lambda_dot]
        self.f_bc = None  # Function for boundary conditions
        
        self.parameter_values = {} # Stores current value of parameters

    def derive_conditions(self):
        """
        Symbolically derives the necessary conditions of optimality.
        """
        # 1. Define Costates (Lagrange Multipliers)
        # Note: Costates are functions of time, same as states
        self.costates = [sp.Function(f'lambda_{s.name}')(self.ocp.t) for s in self.ocp.states]
        
        # 2. Form Hamiltonian
        self.H = self.ocp.get_hamiltonian(self.costates)
        
        # 3. Stationarity Condition: dH/du = 0
        # If possible, solve for u* explicitly in terms of x and lambda
        # For now, we assume we can solve for u explicitly or u is unconstrained in simple cases.
        # TODO: Handle numerical root finding for u if explicit solution fails.
        for u in self.ocp.controls:
            stationarity = sp.diff(self.H, u)
            sol = sp.solve(stationarity, u)
            if sol:
                # Take the first solution (careful with multiple roots!)
                self.control_eqs[u] = sol[0]
            else:
                 print(f"Warning: Could not solve explicitly for control {u}. Numerical optimization required.")

        # 4. State Dynamics: x_dot = dH/dlambda (recovers original dynamics)
        # We substitute u* into these
        
        # 5. Costate Dynamics: lambda_dot = -dH/dx
        # Euler-Lagrange Equations
        self.state_eqs = []
        self.costate_eqs = []
        
        substitutions = {u: expr for u, expr in self.control_eqs.items()}
        
        # Process State Equations (x_dot)
        for f_expr in self.ocp.dynamics:
            # Substitute control u* into dynamics
            f_sub = f_expr.subs(substitutions)
            self.state_eqs.append(f_sub)
            
        # Process Costate Equations (lambda_dot)
        for x, lam in zip(self.ocp.states, self.costates):
            # lambda_dot = - partial H / partial x
            # Note: We differentiate H BEFORE substituting u (partial derivative rule)
            # OR we can use the total derivative if we consider u(x, lambda).
            # PMP says partial H / partial x holding u const.
            dh_dx = sp.diff(self.H, x)
            codyn = -dh_dx
            # Now substitute u*
            codyn_sub = codyn.subs(substitutions)
            self.costate_eqs.append(codyn_sub)
            
    def lambdify_system(self):
        """
        Converts symbolic equations to numerical functions for solve_bvp.
        """
        # State vector Y = [x1, ..., xn, lam1, ..., lamn]
        t_sym = self.ocp.t
        all_vars = self.ocp.states + self.costates
        
        # Dynamics vector [x_dot..., lam_dot...]
        dynamics_exprs = self.state_eqs + self.costate_eqs
        
        # Since solve_bvp passes y of shape (n, N), we need vectorized functions?
        # sp.lambdify usually handles numpy arrays if 'numpy' module is passed.
        
        # However, sympy Function objects like x(t) need to be treated as symbols for lambdify
        # We replace f(t) with dummy symbols for lambdification
        
        dummy_vars = [sp.Symbol(v.name) for v in all_vars]
        subs_map = dict(zip(all_vars, dummy_vars))
        
        dynamics_numeric = [expr.subs(subs_map) for expr in dynamics_exprs]
        
        # Create the function f(t, y, p)
        # Note: solve_bvp expects f(t, y) -> shape (n, m)
        # We include params in the lambdified function signature
        params = self.ocp.parameters
        self._numeric_dynamics = sp.lambdify((t_sym, dummy_vars, params), dynamics_numeric, modules='numpy')

        # Boundary Conditions
        # BCs usually come from:
        # 1. Initial State: x(t0) - x0 = 0
        # 2. Terminal State: x(tf) - xf = 0 (if fixed)
        # 3. Transversality: lambda(tf) - dPhi/dx(tf) = 0 (if free state)
        # 4. Transversality: H(tf) + dPhi/dt(tf) = 0 (if free time)
        
        # For this version, let's assume specific BoundaryConstraints are passed numerically or
        # we implement specific generic transversality.
        # To keep it flexible, we might let the user define the BC function for scipy?
        # Or we automate it. Let's try to automate standard fixed/free cases.
        pass

    def get_ode_fun(self):
        """Returns the function f(t, y) formatted for scipy.integrate.solve_bvp."""
        def fun(t, y, p=None): # p is parameters
            # y shape (n_states + n_costates, n_points)
            # unpack y
            args = [y[i] for i in range(len(y))]
            
            # Get parameter values in order
            param_vals = [self.parameter_values.get(p.name, 0.0) for p in self.ocp.parameters]
            
            # call compiled dynamics
            # Note: t might be an array or scalar. lambdify handles broadcasting usually.
            res_list = self._numeric_dynamics(t, args, param_vals)
            
            # Ensure all elements are arrays of the correct shape (broadcast scalars)
            # t is (n_nodes,)
            
            final_res = []
            shape_target = t.shape if isinstance(t, np.ndarray) else ()
            
            for item in res_list:
                if np.ndim(item) == 0: # scalar or 0-d array
                    if shape_target: # if t is an array, broadcast
                        final_res.append(np.full(shape_target, item))
                    else: # t is scalar
                        final_res.append(item)
                else:
                    final_res.append(item)
            
            return np.array(final_res)
        return fun

    def solve(self, t_guess, y_guess, bc_func: Callable, verbose=2, tol=1e-3, max_nodes=1000):
        """
        Solves the BVP.
        
        Args:
            t_guess: Initial mesh (array).
            y_guess: Initial guess for y [x..., lam...] at mesh points.
            bc_func: Boundary condition function bc(y_a, y_b) -> residuals.
        """
        if self._numeric_dynamics is None:
            raise RuntimeError("System not lambdified. Call derive_conditions() and lambdify_system() first.")
            
        fun = self.get_ode_fun()
        
        res = solve_bvp(fun, bc_func, t_guess, y_guess, verbose=verbose, tol=tol, max_nodes=max_nodes)
        
        return res

    def solve_with_homotopy(self, t_guess, y_guess, bc_func: Callable, param_name: str, param_values: List[float], verbose=2):
        """
        Solves the problem using parameter continuation (homotopy).
        
        Args:
            param_name: The name of the symbol in the OCP parameters to vary.
            param_values: A sequence of values for the parameter.
            Other args: Same as solve().
            
        Returns:
            The final result object.
        """
        current_t = t_guess
        current_y = y_guess
        
        # We need a way to update the numerical function with the new parameter value.
        # Currently, numeric_dynamics depends on (t, y). 
        # If the standard lambdify was used, it expects specific arguments.
        # If OCP has parameters, we need to handle them in lambdify.
        
        # TODO: This requires OCP to support parameters and lambdify to include them.
        # For now, we will assume the user manually updates the OCP or we re-lambdify? 
        # Re-lambdifying is slow.
        # Better: pass parameters as arguments to the symbolic function.
        
        print(f"Homotopy: Starting continuation on {param_name} over {len(param_values)} steps.")
        
        last_res = None
        
        for i, val in enumerate(param_values):
            print(f"Step {i+1}/{len(param_values)}: {param_name} = {val}")
            
            # Update the parameter value in the solver's context
            self.set_parameter_value(param_name, val)
            
            try:
                 res = self.solve(current_t, current_y, bc_func, verbose=0) # reduced verbosity
                 if not res.success:
                     print(f"Warning: Convergence failed at {param_name}={val}. Retrying with higher verbosity/tolerance could be needed.")
                     # Might want to break or continue?
                 
                 # Update guess for next step
                 current_t = res.x
                 current_y = res.y
                 last_res = res
            except Exception as e:
                print(f"Error during homotopy step: {e}")
                break
                
        return last_res

    def set_parameter_value(self, name: str, value: float):
        """
        Sets a value for a symbolic parameter.
        Note: This is a placeholder. Real implementation needs `lambdify` to accept args.
        For this prototype, we'll store it in a dict and `get_ode_fun` should use it.
        """
        if not hasattr(self, 'parameter_values'):
            self.parameter_values = {}
        self.parameter_values[name] = value

