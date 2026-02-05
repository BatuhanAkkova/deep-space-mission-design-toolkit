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
        self._numeric_dynamics = None
        self._numeric_H = None
        
        self.parameter_values = {} # Stores current value of parameters

    def derive_conditions(self):
        """
        Symbolically derives the necessary conditions of optimality.
        """
        # 1. Define Costates (Lagrange Multipliers)
        # Check if already defined (manual override support)
        if not self.costates:
            self.costates = [sp.Function(f'lambda_{s.func.name}')(self.ocp.t) for s in self.ocp.states]
        
        # 2. Form Hamiltonian
        self.H = self.ocp.get_hamiltonian(self.costates)
        
        if self.ocp.objective_type == 'min_fuel_singular':
             # Special handling for singular arcs if needed, or expected user to provide smoothed L
             pass

        # 3. Stationarity Condition: dH/du = 0
        for u in self.ocp.controls:
            # Check if user manually provided the control law
            if u in self.control_eqs:
                continue

            stationarity = sp.diff(self.H, u)
            
            # Try to solve for u explicitly
            try:
                # Direct solve without simplify
                sol = sp.solve(stationarity, u)
            except Exception:
                sol = []
            
            if sol:
                # If multiple solutions, heuristic: take the real one or the first one
                self.control_eqs[u] = sol[0] 
            else:
                # If explicit solution fails, check if the user provided a control law manually
                print(f"Warning: Could not solve explicitly for control {u}. Stationarity: {stationarity}")
                pass


        # 4. State Dynamics: x_dot = dH/dlambda
        
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
            dh_dx = sp.diff(self.H, x)
            codyn = -dh_dx
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
        
        # Replace f(t) with dummy symbols for lambdification
        def get_name(v):
            return v.func.name if hasattr(v, 'func') else v.name

        dummy_vars = [sp.Symbol(get_name(v)) for v in all_vars]
        subs_map = dict(zip(all_vars, dummy_vars))
        
        dynamics_numeric = [expr.subs(subs_map) for expr in dynamics_exprs]
        
        # Create the function f(t, y, p)
        params = self.ocp.parameters
        self._numeric_dynamics = sp.lambdify((t_sym, dummy_vars, params), dynamics_numeric, modules='numpy')

        # Also compile Hamiltonian for Transversality Conditions
        substitutions = {u: expr for u, expr in self.control_eqs.items()}
        H_optimal = self.H.subs(substitutions)
        H_expr = H_optimal.subs(subs_map)
        
        self._numeric_H = sp.lambdify((t_sym, dummy_vars, params), H_expr, modules='numpy')

    def get_ode_fun(self, free_time=False):
        """
        Returns the function f(t, y, p) formatted for scipy.integrate.solve_bvp.
        
        Args:
            free_time: If True, assumes p[0] is t_f and scales dynamics by t_f.
                       The independent variable t is assumed to be normalized [0, 1].
        """
        def fun(t, y, p=None): # p is parameters
            # y shape (n_states + n_costates, n_points)
            # unpack y
            args = [y[i] for i in range(len(y))]
            
            # Get parameter values
            # If free_time is True, we assume the first element of p is tf.
            tf = 1.0
            if free_time:
                if p is None or len(p) == 0:
                    raise ValueError("free_time=True but no parameters passed to ODE function.")
                tf = p[0]
            
            # Get user-defined constant parameters
            param_vals = [self.parameter_values.get(param.name, 0.0) for param in self.ocp.parameters]
            
            # call compiled dynamics
            res_list = self._numeric_dynamics(t, args, param_vals)
            
            # Ensure all elements are arrays of the correct shape (broadcast scalars)
            final_res = []
            shape_target = t.shape if isinstance(t, np.ndarray) else ()
            
            for item in res_list:
                val = item
                if np.ndim(val) == 0: 
                    if shape_target: 
                        val = np.full(shape_target, val)
                
                # Scale by tf if free time (dx/dtau = dx/dt * dt/dtau = f * tf)
                if free_time:
                    val = val * tf
                    
                final_res.append(val)
            
            return np.array(final_res)
        return fun

    def create_bc_function(self, x0, xf=None, free_time=False, custom_transversality=None):
        """
        Creates a boundary condition function for solve_bvp.
        
        Args:
            x0: Initial state (numpy array).
            xf: Target final state (numpy array, Optional). 
                If None, assumes free final state for all components.
                If xf contains NaNs, those specific components are treated as free.
            free_time: Boolean. If True, adds transversality condition for Hamiltonian.
        
        Returns:
             Function bc(ya, yb, p) -> residuals
        """
        if self._numeric_H is None:
             raise RuntimeError("System not lambdified. Call lambdify_system() first.")

        n_x = len(self.ocp.states)
        n_lam = len(self.costates)
        
        def bc(ya, yb, p=None):
            # ya, yb are [x..., lam...] at 0 and tf
            
            res = []
            
            # 1. Initial State Condition: x(0) - x0 = 0
            res.extend(ya[:n_x] - x0)
            
            # 2. Final Conditions
            if xf is not None:
                for i in range(n_x):
                    if np.isnan(xf[i]):
                        # Free final state for this component -> Costate is zero at tf
                        # lambda_i(tf) = 0
                        res.append(yb[n_x + i])
                    else:
                        # Fixed final state
                        # x_i(tf) - xf_i = 0
                        res.append(yb[i] - xf[i])
            else:
                # Free final state -> Costates are zero
                res.extend(yb[n_x:]) 
            
            # 3. Free Time Transversality
            if free_time:
                tf = p[0]
                
                # Get User Params
                param_vals = [self.parameter_values.get(param.name, 0.0) for param in self.ocp.parameters]
                
                # yb is the state at t=End (normalized t=1). Actual t=tf if autonomous.
                # If non-autonomous, pass tf as time.
                h_val = self._numeric_H(tf, yb, param_vals)
                
                target_H = 0.0
                # Assuming minimal time or Hamiltonian = 0 at tf for autonomous free-time
                res.append(h_val - target_H)
                
            return np.array(res)
            
        return bc

    def solve(self, t_guess, y_guess, bc_func: Callable = None, verbose=2, tol=1e-3, max_nodes=1000, free_time=False):
        """
        Solves the BVP.
        """
        if self._numeric_dynamics is None:
            raise RuntimeError("System not lambdified. Call derive_conditions() and lambdify_system() first.")
            
        fun = self.get_ode_fun(free_time=free_time)
        
        if bc_func is None:
            raise ValueError("Please provide bc_func. Automated generation requires calling create_bc_function first.")
            
        # P for parameters
        p = None
        if free_time:
            tf_guess = t_guess[-1]
            p = [tf_guess]
            
            # Normalize t_guess to [0, 1]
            t_guess_norm = t_guess / tf_guess
            t_eval = t_guess_norm
        else:
            t_eval = t_guess

        res = solve_bvp(fun, bc_func, t_eval, y_guess, p=p, verbose=verbose, tol=tol, max_nodes=max_nodes)

        # Denormalize if free time
        if free_time and res.success:
            tf_found = res.p[0]
            res.x = res.x * tf_found
            
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
        """
        if not hasattr(self, 'parameter_values'):
            self.parameter_values = {}
        self.parameter_values[name] = value

