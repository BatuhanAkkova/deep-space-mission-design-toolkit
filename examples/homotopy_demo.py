
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.optimization.ocp import OptimalControlProblem
from src.optimization.indirect import IndirectSolver
from src.optimization.homotopy import solve_minimum_fuel_with_homotopy

def run_homotopy_demo():
    print("=== Smoothing Homotopy Demo (Double Integrator) ===")
    
    # Problem: Min Fuel to move from x=0,v=0 to x=1,v=0 in fixed time tf=2.
    # Dynamics: x_dot = v, v_dot = u.
    # Control: unconstrained u? Wait, min fuel usually implies bounded u.
    # But classical PMP for Fuel min (|u|) leads to bang-bang-off.
    # If u is unbounded, min |u| leads to impulses.
    # Let's assume bounded u in [-1, 1].
    # But our Indirect solver stationarity dH/du=0 assumes u is determined by stationary point.
    # For H = |u| + v*lambda_x + u*lambda_v, H is linear in u.
    # dH/du = sign(u) + lambda_v.
    # This doesn't give u=f(lambda). Minimization of H gives u = -sign(lambda_v) * u_max.
    # Our simple IndirectSolver assumes we can solve dH/du=0.
    
    # Smoothing: L = sqrt(u^2 + eps^2).
    # H = sqrt(u^2 + eps^2) + lambda_x * v + lambda_v * u.
    # dH/du = u / sqrt(u^2 + eps^2) + lambda_v = 0.
    # => u / sqrt(...) = -lambda_v.
    # => u^2 = lambda_v^2 * (u^2 + eps^2).
    # => u^2 (1 - lambda_v^2) = lambda_v^2 eps^2.
    # => u = +/- eps * lambda_v / sqrt(1 - lambda_v^2).
    # This provides a smooth mapping from lambda to u!
    
    # 1. Define OCP
    ocp = OptimalControlProblem()
    x, v = ocp.define_states(['x', 'v'])
    
    # We manually define the costate symbols to construct the control law
    # Note: IndirectSolver generates costates named lambda_x, lambda_v
    # We need to match those names for the substitution to work, OR we define them here and use them?
    # IndirectSolver creates its OWN costate symbols.
    # To inject the control law, we can't use OCP.control_eqs directly because OCP doesn't store them.
    # We need to set the DYNAMICS to be the closed loop dynamics.
    
    # But IndirectSolver derives costate dynamics from H.
    # H = L + lam*f.
    # If we substitute u(lam) into L and f, H becomes function of x, lam.
    # Then IndirectSolver calculates partial H / partial x.
    # This works.
    
    # We need symbols for lambdas that match what IndirectSolver WILL create.
    lam_x = sp.Function('lambda_x')(ocp.t)
    lam_v = sp.Function('lambda_v')(ocp.t)
    
    # Parameters
    eps = sp.Symbol('eps')
    weight = 1.0
    
    # Derived Control Law: u = - eps * lam_v / sqrt(w^2 - lam_v^2)
    # Protection against singularity
    denom_sq = weight**2 - lam_v**2
    # Smooth max or just assumes solver handles it?
    # Let's use sp.sqrt(denom_sq) but we need to ensure it doesn't crash during evaluation.
    # In homotopy, eps starts large, so lam_v should be small?
    # Actually if eps is large, cost is dominated by eps?
    # Check: u near 0.
    
    u_expr = - eps * lam_v / sp.sqrt(denom_sq)
    
    # Closed Loop Dynamics
    ocp.set_dynamics([v, u_expr])
    
    # We must also set the Cost Function with substituted u to get correct H
    # L = weight * sqrt(u^2 + eps^2)
    # This might be messy symbolically. 
    # H = weight*sqrt(u^2+eps^2) + lam_v*u + lam_x*v
    # Just set H directly? IndirectSolver usually calculates H from L and f.
    # If we don't set L, H is wrong.
    
    L_expr = weight * sp.sqrt(u_expr**2 + eps**2)
    ocp.set_running_cost(L_expr)
    
    # We also need to tell IndirectSolver to use our pre-defined costates?
    # IndirectSolver creates costates. We need to alias them.
    # The solver uses `sp.Function(f'lambda_{s.name}')`.
    # Our manual `lam_v` must match.
    
    # To make this robust, let's subclass IndirectSolver or patch it? 
    # Or just rely on string matching if we used same names?
    # SymPy functions with same name and argument are identical.
    
    solver = IndirectSolver(ocp)
    
    # Inject parameter value for eps (will be updated by loop)
    # We used 'eps' symbol.
    ocp.parameters.append(eps)
    solver.set_parameter_value('eps', 1.0)
    
    # Condition derivation
    # derive_conditions will compute H = L + lam*f.
    # It will try to solve dH/du ONLY IF `ocp.controls` is not empty.
    # We did NOT define controls in OCP (variable u).
    # So it skips stationarity.
    # It derives state_eqs = dynamics (already function of lam).
    # It derives costate_eqs = -dH/dx.
    # This is exactly what we want!
    
    solver.derive_conditions()
    solver.lambdify_system()
    
    # 3. BCs
    x0 = np.array([0, 0])
    xf = np.array([1, 0])
    
    # Fixed time tf=2
    t_guess = np.linspace(0, 2, 21)
    
    # Initial Guess
    y_guess = np.zeros((4, 21)) # x, v, lam_x, lam_v
    y_guess[0, :] = np.linspace(0, 1, 21)
    y_guess[1, :] = 0.5 
    y_guess[3, :] = 0.5 # Start with small lambda to avoid singularity (w=1)
    
    bc_helper = solver.create_bc_function(x0, xf, free_time=False)
    
    # 4. Run Homotopy
    # Custom homotopy loop because we managed parameters manually
    # solve_minimum_fuel_with_homotopy assumes it sets 'smoothed_fuel_cost' and calls set_parameter.
    # But we set up OCP manually.
    # We can just iterate here.
    
    eps_schedule = [1.0, 0.5, 0.2, 0.1]
    
    print(f"=== Manual Smoothing Homotopy (Steps: {len(eps_schedule)}) ===")
    
    for val in eps_schedule:
        print(f"Epsilon = {val}")
        solver.set_parameter_value('eps', val)
        
        try:
             res = solver.solve(t_guess, y_guess, bc_helper, verbose=0, tol=1e-4)
             if res.success:
                 print("  Converged.")
                 t_guess = res.x
                 y_guess = res.y
                 
                 # Check max lambda_v
                 max_lam = np.max(np.abs(res.y[3]))
                 print(f"  Max Lambda_v: {max_lam:.4f}")
                 if max_lam >= 0.99:
                     print("  Warning: Approaching singularity.")
             else:
                 print("  Failed.")
                 break
        except Exception as e:
            print(f"  Error: {e}")
            break
            
    # Plot final result
    t_sol = t_guess
    x_sol = y_guess[0]
    v_sol = y_guess[1]
    lam_v_sol = y_guess[3]
    
    # Recompute u
    u_sol = []
    curr_eps = eps_schedule[-1]
    for lv in lam_v_sol:
        denom = 1 - lv**2
        if denom < 1e-6: denom = 1e-6
        val = - curr_eps * lv / np.sqrt(denom)
        u_sol.append(val)
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_sol, x_sol, label='x')
    plt.plot(t_sol, v_sol, label='v')
    plt.legend() 
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(t_sol, u_sol, label='u')
    plt.plot(t_sol, lam_v_sol, '--', label='lambda_v')
    plt.legend()
    plt.grid()
    plt.savefig("homotopy_demo.png")
    print("Saved homotopy_demo.png")

if __name__ == "__main__":
    run_homotopy_demo()
