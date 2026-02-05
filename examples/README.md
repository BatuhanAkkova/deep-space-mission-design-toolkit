# 🚀 Astrodynamics Examples

This directory contains demonstration scripts showcasing the core capabilities of the **Deep Space Mission Design Toolkit**. These examples range from fundamental N-body propagation to advanced mission-specific trajectory design.

---

## 📂 Gallery of Results

### 🔭 Targeting & Mission Design
| Example | Summary | Visualization |
| :--- | :--- | :--- |
| `nbody_demo.py` | **N-Body Propagation & Correction**<br>Solves a Lambert problem for Earth-Mars, then uses differential correction to hit the target in a full N-body environment. | ![N-Body](../assets/nbody.png) |
| `flyby_correction_demo.py` | **B-Plane Targeting**<br>Precisely targets specific B-plane coordinates for polar and equatorial flybys at the Moon. | ![Flyby](../assets/flyby_correrct.png) |
| `free_return_demo.py` | **Free Return Trajectory**<br>Earth-Moon-Earth "Figure-8" trajectory optimized for precise perilune and reentry altitudes. | ![Free Return](../assets/free_return.png) |

### 📊 Mission Planning & Analysis
| Example | Summary | Visualization |
| :--- | :--- | :--- |
| `porkchop_plot_demo.py` | **Porkchop Plotter**<br>Global Scan of launch windows (e.g., Earth-Mars 2005) identifying optimal C3 and TOF opportunities. | ![Porkchop](../assets/porkchop.png) |
| `tisserand_demo.py` | **Tisserand Analysis**<br>Analyzes gravity assist sequences and orbital resonances (e.g., Earth -> Mars -> Jupiter). | ![Tisserand](../assets/tisserand.png) |
| `poincare_map_demo.py` | **Poincare Basin Map**<br>Varies TLI energy and phase to map the ease of Lunar capture/approach. | ![Poincare](../assets/poincare.png) |

### 🛠️ Optimal Control & Math
| Example | Summary | Visualization |
| :--- | :--- | :--- |
| `indirect_optimization_demo.py`| **Indirect Solver**<br>Solving the BVP derived from Pontryagin's Minimum Principle for an Earth-Mars transfer. | ![Indirect](../assets/indirect.png) |
| `homotopy_demo.py` | **Smoothing Homotopy**<br>Demonstrates the robust transition from Minimum-Energy to Minimum-Fuel solutions. | ![Homotopy](../assets/homotopy.png) |

---

## 📝 Detailed Script Descriptions

### 1. `nbody_demo.py`
- **Workflow**: Solves Lambert -> Seeds N-Body -> Corrects initial $V_0$.
- **Demonstrates**: `NBodyDynamics`, `LambertSolver`, and Shooting-based correction.

### 2. `flyby_demo.py` & `flyby_correction_demo.py`
- **Focus**: Gravity Assist Dynamics.
- **Key Tool**: `FlybyCorrector`.
- **Logic**: Iteratively adjusts the incoming state to hit specific $B \cdot R$ and $B \cdot T$ coordinates.

### 3. `free_return_demo.py`
- **Mission**: Earth-Moon-Earth Free Return.
- **Strategy**: Two-phase approach using a grid-search for the basin of attraction followed by a Nelder-Mead optimization for precise targeting.

### 4. `porkchop_plot_demo.py`
- **Mission**: Mars Reconnaissance Orbiter (MRO) 2005 Window.
- **Shows**: Interactive search for the lowest C3 launch energy.

### 5. `poincare_map_demo.py`
- **Concept**: State-space mapping.
- **Logic**: Sweeps Delta-V and Departure Date to visualize the sensitivity of Moon encounters.

### 6. `homotopy_demo.py`
- **Method**: Parameter continuation (Smoothing).
- **Application**: Solving a hard Bang-Bang control problem by starting from a smooth quadratic cost.

---

## 🏃 How to Run

Generic execution from the repository root:

```bash
python examples/nbody_demo.py
```

*Note: Most scripts will generate a plot (.png) in the root directory or display it interactively.*
