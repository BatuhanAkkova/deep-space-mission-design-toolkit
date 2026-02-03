
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.optimization.ocp import OptimalControlProblem
from src.optimization.indirect import IndirectSolver
from src.optimization.multiple_shooting import MultipleShootingSolver

def run_brachistochrone_comparison():
    print("=== Brachistochrone Problem Comparison ===")
    
    # 1. Define OCP
    ocp = OptimalControlProblem()
    ocp.objective_type = 'min_time'
    
    # Variables
    x, y, v = ocp.define_states(['x', 'y', 'v'])
    u = ocp.define_controls(['u'])[0] # theta angle
    g_sym = ocp.define_parameters(['g'])[0]
    
    # Dynamics (y positive down)
    # v_dot = g sin(u)
    # x_dot = v cos(u)
    # y_dot = v sin(u)
    ocp.set_dynamics([
        v * sp.cos(u),
        v * sp.sin(u),
        g_sym * sp.sin(u)
    ])
    
    # Cost
    ocp.set_running_cost(1) # Min time
    
    # 2. Setup Solver
    solver_indirect = IndirectSolver(ocp)
    solver_indirect.derive_conditions()
    solver_indirect.lambdify_system()
    solver_indirect.set_parameter_value('g', 9.81)
    
    # 3. Define Boundary Values
    x0 = np.array([0, 0, 0.1]) # v=0.1 to avoid singularity/stagnation at start
    xf = np.array([5, 2, 0]) # Target x=5, y=2. v free? No usually v is result. Let's fix x, y. v is free.
    
    # Wait, simple Brachistochrone: min time to reach point (x,y). v is free at end.
    
    # Create BC function
    # xf provided is [5, 2, ?]. Solver expects exact match if array provided.
    # Our simple create_bc_function assumes full state if provided.
    # We need partial state support. 
    # Let's override the BC function for this problem or use a specific full target.
    # For Brachistochrone, let's just fix x and y. v is free -> lambda_v(tf) = 0.
    
    # Custom BC for comparison
    def bc_func(ya, yb, p):
        # ya, yb: [x, y, v, lam_x, lam_y, lam_v]
        # p[0] = tf (if free time)
        
        # Initial: x=0, y=0, v=0.1
        res = [
            ya[0] - 0,
            ya[1] - 0,
            ya[2] - 0.1
        ]
        
        # Final: x=5, y=2
        res.append(yb[0] - 5)
        res.append(yb[1] - 2)
        
        # Transversality: v free -> lam_v(tf) = 0
        res.append(yb[5])
        
        # Free Time condition: H(tf) = 0
        # Instead of generic H, let's code specific H for check? 
        # Or blindly trust p[0] is tf and solver handles size.
        # Wait, usually solve_bvp needs N_bc = N_states + N_parameters?
        # N_states = 6. N_p = 1. Total 7 BCs needed?
        # We have 3 initial + 2 final state + 1 trans state = 6.
        # Plus 1 for Hamiltonian = 7. Correct.
        
        if len(p) > 0:
            tf = p[0]
            # H(tf) approx 0?
            # Let's use the indirect solver helper if possible or just hardcode H
            # H = 1 + lam_x v cos u + lam_y v sin u + lam_v g sin u
            # At tf:
            v_f = yb[2]
            lam_x_f = yb[3]
            lam_y_f = yb[4]
            lam_v_f = yb[5] # Should be 0
            
            # We need u(tf). 
            # tan u = (lam_y v + lam_v g) / (lam_x v)
            # If lam_v=0, tan u = lam_y / lam_x
            # sin u = lam_y / sqrt(lam_x^2 + lam_y^2)
            # cos u = lam_x / sqrt(...)
            
            # H = 1 + v * (lam_x cos + lam_y sin)
            # H = 1 + v * sqrt(lam_x^2 + lam_y^2)
            # This should be 0 ???
            # Wait, 1 + ... = 0 implies negative costates?
            # Yes, costates often negative.
            
            # Simplified Transversality for Free Time: H(tf) = 0
            # We can use solver.H to evaluate it numerically if we want, or just add the condition.
            # Let's try to trust the solver creates generic BCs correctly?
            pass

        # For this custom function we must return 7 residuals.
        # We need H(tf).
        # To avoid re-implementing H calculation, let's use the object.
        return np.array(res + [0]) # Placeholder for H logic if not calling helper
        
    # Let's strictly use the provided create_bc_function first to see if it works!
    # But create_bc_function expects full xf if provided.
    # We can pass xf=None and let it assume all free, then we override residuals? No.
    
    # HACK: We will manually define BC using the internal H function of solver
    # This ensures we match the symbolic H.
    
    
    # 4. Solvers
    t_guess = np.linspace(0, 2, 21) # 2 seconds guess
    y_guess = np.zeros((6, 21))
    y_guess[0, :] = np.linspace(0, 5, 21) # x
    y_guess[1, :] = np.linspace(0, 2, 21) # y
    y_guess[2, :] = np.linspace(0.1, 5, 21) # v
    y_guess[3, :] = 0.1 # lam_x guess
    y_guess[4, :] = 0.1 # lam_y
    y_guess[5, :] = 0.1 
    
    # --- Indirect Solver (Solve BVP) ---
    print("\nRunning Indirect Solver (solve_bvp)...")
    
    # We need the numeric H for the BC
    # solver_indirect._numeric_dynamics is available.
    # We can create the BC function using the helper with xf=None, 
    # but manually patch the state constraints.
    
    # Using the helper
    bc_helper = solver_indirect.create_bc_function(x0, xf=None, free_time=True)
    
    def hybrid_bc(ya, yb, p):
        # Call standard free-state BC
        # This enforces x(0)=x0, lambda(tf)=0, H(tf)=0
        res = list(bc_helper(ya, yb, p))
        
        # res has: 
        # 0-2: ya - x0 (OK)
        # 3-5: yb_lam (lam_x, lam_y, lam_v) = 0 (We want to CHANGE this)
        # 6: H(tf) = 0 (OK)
        
        # We want x(tf)=5, y(tf)=2.
        # So we replace res[3] (lam_x) with yb[0] - 5
        # And res[4] (lam_y) with yb[1] - 2
        
        res[3] = yb[0] - 5
        res[4] = yb[1] - 2
        # res[5] is lam_v=0 (Correct)
        
        return np.array(res)

    res_bvp = solver_indirect.solve(t_guess, y_guess, hybrid_bc, verbose=2, free_time=True)
    
    if res_bvp.success:
        print(f"BVP Success! Tf = {res_bvp.p[0]:.4f}")
    else:
        print("BVP Failed.")

    # --- Multiple Shooting Solver ---
    print("\nRunning Multiple Shooting Solver...")
    solver_ms = MultipleShootingSolver(ocp)
    solver_ms.derive_conditions()
    solver_ms.lambdify_system()
    solver_ms.set_parameter_value('g', 9.81)
    
    # MS needs same BC.
    # Note: MS uses solve_ivp which propagates segments.
    # The BC function for MS in my implementation must return residuals.
    # My MS implementation takes t_guess and y_guess.
    
    res_ms = solver_ms.solve(t_guess, y_guess, hybrid_bc, verbose=2, tol=1e-4, free_time=True) 
    
    if res_ms.success:
        tf_ms = res_ms.p[0] if res_ms.p is not None else 0.0
        print(f"MS Success! Tf = {tf_ms:.4f}")
    else:
        print("MS Failed.")

    # Plot
    plt.figure()
    if res_bvp.success:
        t_sol = res_bvp.x
        x_sol = res_bvp.y[0]
        y_sol = res_bvp.y[1]
        plt.plot(x_sol, -y_sol, 'b-', label=f'BVP (Tf={res_bvp.p[0]:.2f})')
        
    if res_ms.success:
        # MS result x might be normalized or not depending on implementation return.
        # My updated implementation returns denormalized t (t_final_res).
        t_sol_ms = res_ms.x
        x_sol_ms = res_ms.y[0]
        y_sol_ms = res_ms.y[1]
        plt.plot(x_sol_ms, -y_sol_ms, 'r--', label=f'MS (Tf={res_ms.p[0]:.2f})')
        
    plt.scatter([0], [0], label='Start')
    plt.scatter([5], [-2], label='End')
    plt.legend()
    plt.title("Brachistochrone Comparison")
    plt.grid(True)
    plt.savefig("brachistochrone_test.png")
    print("Saved plot to brachistochrone_test.png")

if __name__ == "__main__":
    run_brachistochrone_comparison()
