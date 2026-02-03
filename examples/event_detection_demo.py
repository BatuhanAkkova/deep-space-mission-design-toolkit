
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src.optimization.ocp import OptimalControlProblem
from src.optimization.multiple_shooting import MultipleShootingSolver

def run_event_demo():
    print("=== Event Detection Demo (Multiple Shooting) ===")
    
    # Problem: Drop a particle from height H. Stop when it hits ground (y=0).
    # This tests if the solver correctly handles variable leg duration due to events.
    
    # 1. Define OCP
    ocp = OptimalControlProblem()
    y, v = ocp.define_states(['y', 'v'])
    # No control needed for simple drop, but let's add dummy
    u = ocp.define_controls(['u'])[0]
    
    # Dynamics: y_dot = v, v_dot = -g
    g = 9.81
    ocp.set_dynamics([v, -g + 0*u])
    ocp.set_running_cost(0) # minimal
    
    # 2. Solver
    solver = MultipleShootingSolver(ocp)
    solver.derive_conditions()
    solver.lambdify_system()
    
    # 3. Setup
    # Height 10m. Free fall time = sqrt(2*10/9.81) approx 1.42s.
    t_guess = np.linspace(0, 2.0, 3) # 2 segments. Guess longer than impact.
    
    # Initial Guess
    y_guess = np.zeros((4, 3)) 
    y_guess[0, :] = [10, 5, 0] # y
    y_guess[1, :] = [0, -5, -10] # v
    
    # BC Function (Standard)
    def bc_func(ya, yb, p):
        # ya: start, yb: end
        # Start: y=10, v=0
        # End: y=0 (Impact)
        # Note: Event detection stops integration at y=0.
        # But the solver constraints (defects) ensure continuity.
        # The LAST node should satisfy y=0 if the event triggered.
        # However, Multiple Shooting enforces Y_end_integrated = Y_next_guess.
        # If integration stops at event, Y_end_integrated is state at event.
        # So Y_next_guess MUST match state at event.
        # And the LAST node Y_N is Y_next_guess of last segment.
        # So Y_N will be at the event.
        
        return np.array([
            ya[0] - 10,
            ya[1] - 0,
            ya[2] - 0, # lam_y = 0
            ya[3] - 0, # lam_v = 0
            yb[0] - 0  # Enforce strictly y=0 at end (redundant with event but ok)
        ])

    # Define Event
    # Event: y = 0.
    # We need a callable with signature event(t, y, p) or event(t, y)?
    # solve_ivp events expects (t, y) usually.
    # Our solver wraps it.
    
    def impact_event(t, y, p=None):
        return y[0] # y position
    impact_event.terminal = True
    impact_event.direction = -1 # Cross from + to -
    
    # Pass event to the LAST segment only? Or all?
    # If we pass to all, it might trigger early if guesses are bad.
    # Let's pass to all segments.
    events = [None, impact_event] # First segment normal, second segment stops at impact?
    # Actually, if we want to find impact, the last segment should definitely have it.
    # If intermediate segments hit it, that's also fine (but implies multiphase?)
    # Let's try passing to last segment only.
    
    # Note: solve method takes list of events corresponding to segments.
    # We have 2 segments (3 nodes).
    seg_events = [None, impact_event]
    
    print("Running solver with event detection...")
    res = solver.solve(t_guess, y_guess, bc_func, verbose=2, events=seg_events)
    
    if res.success:
        print("Success!")
        t_sol = res.x
        y_sol = res.y
        
        print(f"Final Time: {t_sol[-1]:.4f}")
        print(f"Final State: y={y_sol[0, -1]:.4f}, v={y_sol[1, -1]:.4f}")
        
        true_tf = np.sqrt(2*10/9.81)
        print(f"True Impact Time: {true_tf:.4f}")
        
        # Plot
        plt.figure()
        plt.plot(t_sol, y_sol[0], 'o-', label='y')
        plt.axhline(0, color='k', linestyle='--')
        plt.legend()
        plt.title(f"Event Detection (Impact at t={t_sol[-1]:.2f})")
        plt.savefig("event_demo.png")
    else:
        print("Failed.")

if __name__ == "__main__":
    run_event_demo()
