# 🌌 Deep Space Mission Design Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A powerful, high-fidelity Python toolkit designed for the analysis and optimization of interplanetary trajectories. This repository provides a suite of tools for N-body propagation, differential correction, optimal control solving (indirect/direct methods), and preliminary mission planning.

---

## 🚀 Key Features

-   **High-Fidelity Dynamics**: Full N-body propagation using SPICE ephemeris data.
-   **Mission Design Tools**: Porkchop plot generation, Tisserand analysis, and Lambert solvers.
-   **Advanced Optimization**: 
    -   **Indirect Methods**: Solving TPBVPs via Pontryagin's Minimum Principle with symbolic derivation.
    -   **Homotopy/Continuation**: Robustly solving hard optimization problems (e.g., Min-Energy to Min-Fuel).
    -   **Direct Methods**: Initial guess generation via grid search and single/multiple shooting.
-   **Targeting & Correction**: B-plane targeting for precise gravity assists and free-return trajectories.
-   **SPICE Integration**: Seamless handling of planetary constants and kernels via `SpiceManager`.

---

## 🛠️ Installation & Setup

### 1. Clone & Install
```bash
git clone https://github.com/BatuhanAkkova/deep-space-mission-design-toolkit.git
cd deep-space-mission-design-toolkit
pip install -e .
```

### 2. SPICE Kernels
This toolkit requires SPICE kernels for planetary ephemerides. Place your `.bsp`, `.tpc`, and `.tls` files in the `data/` directory. 
Recommended kernels:
- `de440.bsp` (Planetary Ephemeris)
- `naif0012.tls` (Leapseconds)
- `pck00010.tpc` (Planetary Constants)

---

## 📏 Mission Design Conventions

All calculations within this toolkit follow strict astrodynamics standards:
-   **Time System**: Barycentric Dynamical Time (TDB) / Ephemeris Time (ET).
-   **Coordinate Frames**: `ECLIPJ2000` (Ecliptic) for interplanetary, `J2000` (Equatorial) for near-Earth.
-   **Units**: SI-based with **Kilometers (km)** and **km/s** as standard for SPICE compatibility.
-   **Validation**: All patched-conic results are validated against full N-body numerical integration.

*For more details, see [mission_design_conventions.md](mission_design_conventions.md).*

---

## 🔭 Examples

The [examples/](examples/) directory contains a suite of demonstration scripts. Below is a gallery of key results:

### Trajectory Propagation & Targeting
| Mission Analysis | Visualization |
| :--- | :--- |
| **N-Body Propagation & Correction**<br>Demonstrates Lambert problem solving followed by differential correction in a full N-body environment. | ![N-Body Trajectory](assets/nbody.png) |
| **B-Plane Flyby Targeting**<br>Precisely target specific B-plane coordinates (e.g., polar vs. equatorial flybys). | ![Flyby Correction](assets/flyby_correrct.png) |
| **Free Return Trajectory (Figure-8)**<br>Earth-Moon-Earth free return optimized for specific perilune and reentry altitudes. | ![Free Return](assets/free_return.png) |

### Advanced Optimization
| Method | Visualization |
| :--- | :--- |
| **Indirect Optimization**<br>Solving Optimal Control Problems (OCP) using Pontryagin's Minimum Principle. | ![Indirect Optimization](assets/indirect.png) |
| **Homotopy (Smoothing)**<br>Transitioning from Minimum Energy to Minimum Fuel problems. | ![Homotopy](assets/homotopy.png) |

### Mission Planning Tools
| Tool | Visualization |
| :--- | :--- |
| **Porkchop Plotter**<br>Global scan of launch windows identifying optimal C3 and TOF opportunities. | ![Porkchop Plot](assets/porkchop.png) |
| **Tisserand Analysis**<br>Visualizing gravity assist opportunities and energy changes. | ![Tisserand](assets/tisserand.png) |

*For detailed script descriptions, see the [Examples README](examples/README.md).*

---

## 🗺️ Roadmap

-   [ ] **Navigation & GNC**: Orbit Determination (EKF, Batch Least Squares), Covariance Analysis.
-   [ ] **Multiple Flybys**: MGA (Multiple Gravity Assist) solver.
-   [ ] **Low Thrust**: Continuous thrust trajectory optimization.
-   [ ] **Invariant Manifolds**: Transfers between Lagrange points (L1/L2).
-   [ ] **Generalized Free Return**: Robust grid-search based free-return solver.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or it's MIT by default for this toolkit).
