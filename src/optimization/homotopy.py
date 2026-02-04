import numpy as np
from typing import Callable, List, Dict
from .indirect import IndirectSolver

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
        solver: An instance of IndirectSolver.

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
                 max_res = np.max(res.rms_residuals) if hasattr(res, 'rms_residuals') and len(res.rms_residuals) > 0 else 'N/A'
                 print(f"  Converged. Max residual: {max_res}")
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

def analyze_switching_function(t, switching_func_vals, tol=1e-3):
    """
    Analyzes the switching function S(t) to detect switch points (zeros).
    
    Args:
        t: Time grid.
        switching_func_vals: Values of S(t).
        
    Returns:
        List of switch times (approximate).
    """
    switch_times = []
    
    # Simple sign change detection
    signs = np.sign(switching_func_vals)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    
    for idx in sign_changes:
        # Linear interpolation to find zero
        t1, t2 = t[idx], t[idx+1]
        s1, s2 = switching_func_vals[idx], switching_func_vals[idx+1]
        
        if s1 == s2: continue # Should not happen if sign changed
        
        # t_zero = t1 + (0 - s1) * (t2 - t1) / (s2 - s1)
        t_zero = t1 - s1 * (t2 - t1) / (s2 - s1)
        switch_times.append(t_zero)
        
    return switch_times

def plot_switching_function(t, switching_func_vals, title="Switching Function"):
    """
    Helper to visualize switching function.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, switching_func_vals)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("S(t)")
        plt.grid(True)
        plt.show()
    except ImportError:
        pass
