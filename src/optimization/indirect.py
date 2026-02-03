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
        self.costates = [sp.Function(f'lambda_{s.func.name}')(self.ocp.t) for s in self.ocp.states]
        
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
        
        # Helper to get name
        def get_name(v):
            return v.func.name if hasattr(v, 'func') else v.name

        dummy_vars = [sp.Symbol(get_name(v)) for v in all_vars]
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
            # If free_time, p[0] is tf. Other parameters come after?
            # For simplicity, let's assume if free_time=True, p is [tf, user_params...]
            # But currently user_params are stored in self.parameter_values dict.
            # Let's keep it simple: p passed from solve_bvp contains the UNKNOWN parameters.
            # Known parameters are in self.parameter_values.
            
            # If free_time is True, we assume the first element of p is tf.
            tf = 1.0
            if free_time:
                if p is None or len(p) == 0:
                    raise ValueError("free_time=True but no parameters passed to ODE function.")
                tf = p[0]
            
            # Get user-defined constant parameters
            # Note: This logic assumes p ONLY contains the unknown parameters optimized by solve_bvp.
            param_vals = [self.parameter_values.get(param.name, 0.0) for param in self.ocp.parameters]
            
            # call compiled dynamics
            # Note: t might be an array or scalar. lambdify handles broadcasting usually.
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
            xf: Target final state (numpy array, Optional). If None, assumes free final state for components.
            free_time: Boolean. If True, adds transversality condition for Hamiltonian.
        
        Returns:
             Function bc(ya, yb, p) -> residuals
        """
        n_x = len(self.ocp.states)
        n_lam = len(self.costates)
        
        # We need the numeric Hamiltonian function for transversality
        # Symbolically substitute control u* into H
        substitutions = {u: expr for u, expr in self.control_eqs.items()}
        H_optimal = self.H.subs(substitutions)
        
        # Lambdify H(t, x, lam, params)
        all_vars = self.ocp.states + self.costates
        
        # Helper to get name (redefined locally or make method?)
        def get_name(v):
            return v.func.name if hasattr(v, 'func') else v.name
            
        dummy_vars = [sp.Symbol(get_name(v)) for v in all_vars]
        subs_map = dict(zip(all_vars, dummy_vars))
        H_expr = H_optimal.subs(subs_map)
        
        t_sym = self.ocp.t
        params = self.ocp.parameters
        
        numeric_H = sp.lambdify((t_sym, dummy_vars, params), H_expr, modules='numpy')

        def bc(ya, yb, p=None):
            # ya, yb are [x..., lam...] at 0 and tf (or 1)
            
            res = []
            
            # 1. Initial State Condition: x(0) - x0 = 0
            res.extend(ya[:n_x] - x0)
            
            # 2. Final Conditions
            # If xf is provided, enforce x(tf) = xf
            # If xf is None (or partial), enforce lambda(tf) = ... (Transversality)
            # For now, simplistic: either process fully fixed or fully free per component?
            # Or handle provided xf.
            
            if xf is not None:
                # Assuming fully fixed state for now
                # TODO: Support partial state constraints via NaNs or separate mask
                res.extend(yb[:n_x] - xf)
            else:
                # Free final state -> Costates are zero (if no terminal cost phi)
                # lambda(tf) = dphi/dx
                # Assuming phi=0 for now or handled manually.
                # Simplest Transversality: lambda(tf) = 0
                res.extend(yb[n_x:]) 
            
            # 3. Free Time Transversality
            # H(tf) + dphi/dt = 0
            # For Min Time: L=1, phi=0 -> H(tf) = -1 ?
            # Wait, Standard result: H(tf) = -1 for min time (L=1).
            # If L=0 and phi=tf, then H(tf) = -1.
            
            if free_time:
                # p[0] is tf
                tf = p[0]
                
                # Calculate H at final point
                # yb is the state/costate at normalized time 1.
                # But numeric_H expects 't'. If autonomous, t doesn't matter.
                # If non-autonomous, t should be tf.
                
                # Get User Params
                param_vals = [self.parameter_values.get(param.name, 0.0) for param in self.ocp.parameters]
                
                h_val = numeric_H(tf, yb, param_vals)
                
                # Condition depends on objective
                # If Time Optimal (L=1), H(tf) should be 0? No.
                # Min J = int(1) dt. H = 1 + lambda*f.
                # Transversality: H(tf) = 0 (if final time free and fixed end state).
                # WAIT. Pontryagin: H(tf) = - dPhi/dt.
                # If Phi=0, H(tf) = 0. 
                # BUT for Min Time, Hamiltonian is often defined with p0 (abnormal multiplier).
                # Common form: H + 1 = 0 or H = 0 depending on definition.
                
                # Let's use the property:
                # If min time, H(tf) = -1? 
                # Let's assume the user sets up the problem such that H(tf) should be 0 (general case for free time autonomous).
                # OR user can specific target H value.
                
                # For Generic Free Time with fixed endpoint: H(tf) = 0 is common for autonomous systems.
                # Let's enforce H(tf) = 0 by default for free time.
                
                target_H = 0.0
                if self.ocp.objective_type == 'min_time':
                    # If user didn't put '1' in L, but put 'tf' in Phi?
                    # If L=1, then H = 1 + ...
                    # H(tf) = 0 => 1 + lambda*f = 0.
                    pass
                
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
        
        # If bc_func is None, we need x0 and xf from somewhere? 
        # For now, require bc_func or upgrade this method.
        if bc_func is None:
            raise ValueError("Please provide bc_func. Automated generation requires calling create_bc_function first.")
            
        # P for parameters
        p = None
        if free_time:
            # We need an initial guess for tf.
            # Assuming t_guess is normalized [0,1], we can try to guess tf from original t_guess?
            # Or user must provide it.
            # Let's assume t_guess is NOT normalized and we normalize it here?
            # Complexity: mixing raw and normalized.
            # Let's assume input t_guess is real time.
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

