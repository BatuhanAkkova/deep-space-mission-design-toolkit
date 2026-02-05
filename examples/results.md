**nbody_demo.py:**
Launch:  2020-07-30T12:00:00
Arrival: 2021-02-18T12:00:00
Solving Lambert Problem...
Departure V_inf: 3.816 km/s
Stepping forward 3 days (SOI exit estimation) analytically...
Propagating Initial Guess (Lambert Seed)...
Initial Miss Distance: 4349610.16 km

Starting Differential Correction...
Iteration 0: Miss = 4349610.17 km
Iteration 1: Miss = 22769.74 km
Iteration 2: Miss = 5266.02 km
Iteration 3: Miss = 1011.70 km
Iteration 4: Miss = 766.85 km
Converged!
Propagating Final Trajectory...
Final Miss Distance: 766.84 km
- image: nbody.png

**flyby_demo.py:**
--- Analytic Calculation ---
Incoming V_inf: [5.  2.  0.5] km/s
Mag: 5.408 km/s
Periapsis Alt: 500.0 km
Turn Angle: 83.299 deg
Analytic Outgoing V_inf: [-1.15316509  3.62945442  3.84021756] km/s
Mag: 5.408 km/s

--- N-Body Propagation Verification ---
Measured Outgoing V_inf: [-1.16140829  3.67985551  3.89038373] km/s
Mag: 5.480 km/s
Turn Angle Error: 0.0773 degrees
Vector Difference Norm: 0.0716 km/s

--- Visualization Debug ---
Planet Radius: 6378.1366 km
Trajectory Periapsis Distance: 6878.136599999752 km
Gap (Alt): 499.9999999997526 km
- image: flyby.png

**flyby_correction_demo.py:**
Refining initial guess for MOON encounter at T+302400.0s...
Lambert Solution Found.

--- Case 1: Vertical Offset (Polar) ---
Target: B_R = 3737.40 km, B_T = 0.00 km
Achieved: B_R = 3737.40 km, B_T = -0.00 km
Error:    dR  = 0.00 km, dT  = -0.00 km
Approx Periapsis Altitude: -232.93 km

--- Case 2: Horizontal Offset (Equatorial) ---
Target: B_R = 0.00 km, B_T = 5000.00 km
Achieved: B_R = 0.24 km, B_T = 4999.54 km
Error:    dR  = 0.24 km, dT  = -0.46 km
Approx Periapsis Altitude: 749.50 km

--- Case 3: Altitude ~100.0 km, Theta=45 deg (Approx) ---
Target: B_R = 3970.35 km, B_T = 3970.35 km
Achieved: B_R = 3970.35 km, B_T = 3970.35 km
Error:    dR  = -0.00 km, dT  = 0.00 km
Approx Periapsis Altitude: 1231.24 km
- image: flyby_correct.png

**direct_optimization_demo.py:**
Setting up Optimal Control Problem (Direct Method)...
Targeting Transfer time: 4.50 TU
Solving Nonlinear Programming Problem...
Optimization terminated successfully    (Exit mode 0)
            Current function value: 0.010509441894284478
            Iterations: 30
            Function evaluations: 9030
            Gradient evaluations: 30
Optimization Successful!
Cost: 0.010509441894284478
- image: direct_coll.png

**indirect_optimization_demo.py:**
Setting up Optimal Control Problem...
Targeting Transfer time: 4.50 TU
Deriving Necessary Conditions (Symbolic)...
Compiling Numerical Functions...
Generating Boundary Condition Function...
Solving BVP...
   Iteration    Max residual  Max BC residual  Total nodes    Nodes added
       1          3.71e-06       4.19e-22          50              0       
Solved in 1 iterations, number of nodes 50.
Maximum relative residual: 3.71e-06
Maximum boundary residual: 4.19e-22
Optimization Successful!
- image: indirect.png

**homotopy_demo.py:**
========================================
   Smoothing Homotopy Demo
   Double Integrator Rest-to-Rest (Min Fuel)
========================================

Starting p-Norm Homotopy Loop...

Optimization Step: p = 2.0
  Converged.

Optimization Step: p = 1.8
  Converged.

Optimization Step: p = 1.6
  Converged.

Optimization Step: p = 1.4
  Converged.

Optimization Step: p = 1.2
  Converged.

Homotopy Successful!
Final Control Equation: -Abs(lambda_v(t))**5.0*sign(lambda_v(t))
- image: homotopy.png

**free_return_figure8.py:**
--- SOLVER STARTED (Continuous Mode) ---
[Phase 1] Finding Basin of Attraction...
   Scan: Theta=180, dV=3.0800 -> Miss=218878 km
   Scan: Theta=180, dV=3.1050 -> Miss=216465 km
   Scan: Theta=180, dV=3.1300 -> Miss=217974 km
   Scan: Theta=180, dV=3.1550 -> Miss=221672 km
   Scan: Theta=180, dV=3.1800 -> Miss=222240 km
   Scan: Theta=190, dV=3.0800 -> Miss=190539 km
   Scan: Theta=190, dV=3.1050 -> Miss=180737 km
   Scan: Theta=190, dV=3.1300 -> Miss=175330 km
   Scan: Theta=190, dV=3.1550 -> Miss=174249 km
   Scan: Theta=190, dV=3.1800 -> Miss=177156 km
   Scan: Theta=200, dV=3.0800 -> Miss=168940 km
   Scan: Theta=200, dV=3.1050 -> Miss=146405 km
   Scan: Theta=200, dV=3.1300 -> Miss=138788 km
   Scan: Theta=200, dV=3.1550 -> Miss=130066 km
   Scan: Theta=200, dV=3.1800 -> Miss=126389 km
   Scan: Theta=210, dV=3.0800 -> Miss=157115 km
   Scan: Theta=210, dV=3.1050 -> Miss=119964 km
   Scan: Theta=210, dV=3.1300 -> Miss=93223 km
   Scan: Theta=210, dV=3.1550 -> Miss=81329 km
   Scan: Theta=210, dV=3.1800 -> Miss=79597 km
   Scan: Theta=220, dV=3.0800 -> Miss=166076 km
   Scan: Theta=220, dV=3.1050 -> Miss=107709 km
   Scan: Theta=220, dV=3.1300 -> Miss=61757 km
   Scan: Theta=220, dV=3.1550 -> Miss=33846 km
   Scan: Theta=220, dV=3.1800 -> Miss=23572 km
   -> Basin Found: Theta=220.00, dV=3.1800 (Dist=23572 km)

[Phase 2] Gradient Descent...
   Converged: True
   Final State: Theta=225.0058 deg, V=3.186179 km/s
   Moon Altitude: 262.86 km
   Earth Return Altitude: 49.93 km
- image: free_return.png

**porkchop_plot_demo.py:**
- image: porkchop.png

**tisserand_demo.py:**
- image: tisserand.png