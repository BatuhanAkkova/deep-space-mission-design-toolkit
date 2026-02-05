# 🚀 Astrodynamics Examples

This directory contains demonstration scripts showcasing the core capabilities of the **Deep Space Mission Design Toolkit**. These examples range from fundamental N-body propagation to advanced mission-specific trajectory design.

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
