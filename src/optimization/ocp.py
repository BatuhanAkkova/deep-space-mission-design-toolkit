import sympy as sp
from typing import List, Callable, Union, Optional

class OptimalControlProblem:
    """
    Defines an Optimal Control Problem (OCP) symbolically.
    """
    def __init__(self):
        self.t = sp.Symbol('t')  # Independent variable (time)
        self.states = []
        self.controls = []
        self.parameters = [] # Static parameters
        self.dynamics = []  # List of symbolic expressions for x_dot
        self.L = 0  # Running cost (Lagrangian)
        self.phi = 0  # Terminal cost (Mayer term)
        self.constraints = {} # Boundary constraints
        self.objective_type = 'general' # 'min_time', 'min_fuel', 'general'

    def set_time_variable(self, name: str = 't'):
        """Sets the symbol for the independent variable."""
        self.t = sp.Symbol(name)

    def define_states(self, names: List[str]):
        """Defines state variables as functions of time."""
        self.states = [sp.Function(name)(self.t) for name in names]
        return self.states

    def define_controls(self, names: List[str]):
        """Defines control variables as functions of time."""
        self.controls = [sp.Function(name)(self.t) for name in names]
        return self.controls
    
    def define_parameters(self, names: List[str]):
        """Defines static parameters."""
        params = [sp.Symbol(name) for name in names]
        self.parameters.extend(params)
        return params

    def set_dynamics(self, equations: List[sp.Expr]):
        """
        Sets the system dynamics: x_dot = f(x, u, t).
        The list should correspond to the order of states defined.
        """
        if len(equations) != len(self.states):
            raise ValueError("Number of dynamic equations must match number of states.")
        self.dynamics = equations

    def set_running_cost(self, cost_expression: sp.Expr):
        """Sets the running cost L(x, u, t)."""
        self.L = cost_expression

    def set_terminal_cost(self, cost_expression: sp.Expr):
        """Sets the terminal cost phi(x(tf), tf)."""
        self.phi = cost_expression
    
    def add_constraint(self, kind: str, expression: sp.Expr):
        """
        Adds a boundary constraint.
        kind: 'initial', 'final', or 'path'
        expression: Symbolic expression that must equal 0.
        """
        if kind not in self.constraints:
            self.constraints[kind] = []
        self.constraints[kind].append(expression)

    def add_path_constraint(self, g: sp.Expr, penalty_weight: Union[sp.Symbol, float], method='quadratic_penalty'):
        """
        Adds an inequality path constraint g(x, u, t) <= 0 via penalty function.
        
        Args:
            g: The constraint expression (supposed to be <= 0).
            penalty_weight: The weight factor for the penalty (e.g., 100 or a symbol).
            method: 'quadratic_penalty' is currently supported.
        """
        if method == 'quadratic_penalty':
            penalty = 0.5 * penalty_weight * sp.Max(0, g)**2
            self.L += penalty
        else:
            raise NotImplementedError(f"Constraint method {method} not supported.")


    def get_hamiltonian(self, costates: List[sp.Symbol]):
        """Constructs the Hamiltonian: H = L + lambda^T * f"""
        if len(costates) != len(self.dynamics):
            raise ValueError("Number of costates must match number of states.")
        
        H = self.L
        for lam, f in zip(costates, self.dynamics):
            H += lam * f
        return H

    def set_smoothed_fuel_cost(self, control_symbols: List[sp.Symbol], weight: float = 1.0, epsilon: float = 0.1):
        """
        Sets a smoothed L1 cost: L = weight * sqrt(u^2 + epsilon^2).
        Approximates |u| as epsilon -> 0.
        """
        cost = 0
        for u in control_symbols:
            cost += weight * sp.sqrt(u**2 + epsilon**2)
        
        self.L = cost
        self.objective_type = 'min_fuel'
