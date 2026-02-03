import numpy as np
from typing import Callable, List, Dict
from .indirect import IndirectSolver
from .multiple_shooting import MultipleShootingSolver

def solve_minimum_fuel_with_homotopy(solver, 
                                     t_guess, 
                                     y_guess, 
                                     bc_func, 
                                     control_symbols, 
                                     weight=1.0, 
                                     epsilon_schedule=[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6],
                                     verbose=2):
    """
    Solves a minimum fuel problem by strictly reducing epsilon in the smoothed cost L = weight * sqrt(u^2 + eps^2).
    
    Args:
        solver: An instance of IndirectSolver or MultipleShootingSolver.
        t_guess: Initial mesh.
        y_guess: Initial guess.
        bc_func: Boundary condition function.
        control_symbols: List of control variables (SymPy symbols).
        weight: Weight factor for the cost.
        epsilon_schedule: List of epsilon values to solve for sequentially.
        
    Returns:
        The final result object.
    """
    
    current_t = t_guess
    current_y = y_guess
    last_res = None
    
    print(f"=== Starting Smoothing Homotopy (Steps: {len(epsilon_schedule)}) ===")
    
    for i, eps in enumerate(epsilon_schedule):
        print(f"\n[Homotopy Step {i+1}] Epsilon = {eps}")
        
        # 1. Update OCP Cost
        solver.ocp.set_smoothed_fuel_cost(control_symbols, weight=weight, epsilon=eps)
        
        if verbose > 1:
            print("  Deriving conditions...")
        solver.derive_conditions()
        solver.lambdify_system()
        
        # 2. Solve
        # Use previous solution as guess
        try:
             res = solver.solve(current_t, current_y, bc_func, verbose=0, tol=1e-4)
             
             if res.success:
                 print(f"  Converged. Max residual? (Check solver output)")
                 last_res = res
                 current_t = res.x
                 current_y = res.y
             else:
                 print(f"  Failed. Keeping previous guess (or breaking).")
                 print(res.message)
                 break
                 
        except Exception as e:
            print(f"  Error: {e}")
            break
            
    print("\n=== Homotopy Complete ===")
    return last_res
