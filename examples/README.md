# 🚀 Astrodynamics Examples

This directory contains a suite of demonstration scripts showcasing the core capabilities of the **Deep Space Mission Design Toolkit**. These examples range from fundamental N-body propagation to advanced indirect optimization and mission-specific trajectory design.

---

## 📂 Gallery of Results

### 🔭 Trajectory Propagation & Targeting
| Mission Analysis | Visualization |
| :--- | :--- |
| **N-Body Propagation & Correction**<br>Demonstrates Lambert problem solving followed by differential correction in a full N-body environment (Earth-Mars transfer). | ![N-Body Trajectory](../assets/nbody.png) |
| **B-Plane Flyby Targeting**<br>Shows how to precisely target specific B-plane coordinates (e.g., polar vs. equatorial flybys) using iterative correction. | ![Flyby Correction](../assets/flyby_correrct.png) |
| **Free Return Trajectory (Figure-8)**<br>A complex Earth-Moon-Earth free return trajectory optimized for specific perilune and reentry altitudes. | ![Free Return](../assets/free_return.png) |

### 🛠️ Advanced Optimization
| Method | Visualization |
| :--- | :--- |
| **Indirect Optimization**<br>Solving the Optimal Control Problem (OCP) using Pontryagin's Minimum Principle and BVP solvers. | ![Indirect Optimization](../assets/indirect.png) |
| **Homotopy (Smoothing)**<br>Demonstrates the transition from a Minimum Energy problem to a Minimum Fuel problem using smoothing homotopy. | ![Homotopy](../assets/homotopy.png) |

### 📊 Mission Planning Tools
| Tool | Visualization |
| :--- | :--- |
| **Porkchop Plotter**<br>Global Scan of launch windows and arrival dates to identify optimal C3 and TOF opportunities. | ![Porkchop Plot](../assets/porkchop.png) |
| **Tisserand Analysis**<br>Visualizing gravity assist opportunities and energy changes between planetary encounters. | ![Tisserand](../assets/tisserand.png) |

---

## 📝 Detailed Script Descriptions

### 1. `nbody_demo.py`
*   **Purpose:** Validates the N-body propagator and differential corrector.
*   **Workflow:** Solves a Lambert problem for an Earth-Mars transfer, then uses the solution as a seed for a high-fidelity N-body propagation. It iteratively corrects the initial velocity to hit the target Mars position.
*   **Key Result:** Achieves sub-1000km miss distance in a high-fidelity environment.

### 2. `flyby_demo.py` & `flyby_correction_demo.py`
*   **Purpose:** Analyzes gravity assist dynamics.
*   **Features:** 
    *   `flyby_demo.py` compares analytic patched-conic flybys with N-body integrated flybys.
    *   `flyby_correction_demo.py` implements B-plane targeting for specific flyby geometries (Polar, Equatorial, and custom offsets).

### 3. `indirect_optimization_demo.py`
*   **Purpose:** Optimal trajectory design using first-order necessary conditions.
*   **Features:** Uses symbolic math to derive Euler-Lagrange equations and solves the resulting Boundary Value Problem (BVP).

### 4. `homotopy_demo.py`
*   **Purpose:** Robustly solving hard optimization problems.
*   **Features:** Implements a p-norm homotopy loop to find minimum-fuel solutions by starting from a "well-behaved" minimum-energy problem.

### 5. `free_return_demo.py`
*   **Purpose:** Designing iconic Lunar free-return trajectories.
*   **Workflow:** Uses a two-phase approach: a grid search to find the basin of attraction, followed by gradient-based optimization to meet precise altitude constraints at the Moon and Earth.

### 6. `porkchop_plot_demo.py`
*   **Purpose:** Preliminary mission design.
*   **Features:** Scans departure and arrival dates for a Mars mission, plotting C3 energy contours and time-of-flight lines.

### 7. `tisserand_demo.py`
*   **Purpose:** Multi-flyby mission design.
*   **Features:** Uses Tisserand graphs to analyze how flybys change the spacecraft's orbital semi-major axis and eccentricity.

### 8. `poincare_map_demo.py`
*   **Purpose:** Chaos and stability analysis.
*   **Features:** Generates Poincaré sections in the Circular Restricted Three-Body Problem (CR3BP) to visualize stable and unstable regions.

---

## 🏃 How to Run

Ensure you have the toolkit installed in your environment, then run any script from the root directory:

```bash
# Example: Running the N-body demo
python examples/nbody_demo.py
```

Results (plots) will typically be displayed or saved depending on the script configuration.
