# Mission Design Conventions & Standards

This document establishes the mandatory conventions for all astrodynamics analysis, software development, and mission design activities within this workspace.

## 1. Mission Phases
All mission profiles must be decomposed into the following distinct phases for analysis:

1.  **Launch & Departure (L+D)**
    *   From launch vehicle separation to Earth Sphere of Influence (SOI) exit.
    *   Key Metrics: C3 (Characteristic Energy), DLA (Declination of Launch Asymptote), RLA (Right Ascension of Launch Asymptote).
2.  **Interplanetary Cruise (IC)**
    *   Ballistic arcs or low-thrust segments between planetary encounters.
    *   Dynamics: N-body heliocentric.
3.  **Gravity Assist / Flyby (GA)**
    *   Hyperbolic passage through a body's SOI.
    *   Key Metrics: B-plane parameters ($B \cdot R$, $B \cdot T$), Turn Angle ($\delta$), Pericenter Radius ($r_p$).
4.  **Arrival & Insertion (ARR)**
    *   Approach and capture maneuver at the target body.
    *   Key Metrics: $V_{\infty}$ (Arrival), Capture $\Delta V$, Initial Orbit Elements.

## 2. Naming Conventions

### 2.1 Coordinate Frames
Explicitly declare frames in all code and documentation.
*   **Inertial (Heliocentric/Geocentric)**: `ECLIPJ2000` (Ecliptic Plane, Equinox of J2000) is preferred for interplanetary. `J2000` (Earth Equatorial) for near-Earth operations.
*   **Body-Fixed**: `IAU_BODYNAME` (e.g., `IAU_EARTH`, `IAU_MARS`) as defined by SPICE PCK.
*   **Local Orbital**: `VNB` (Velocity-Normal-Binormal) or `LVLH` (Local Vertical Local Horizontal).

### 2.2 Time Systems
*   **Internal Calculation**: `TDB` (Barycentric Dynamical Time) expressed as **Ephemeris Time (ET)** seconds past J2000.
    *   Variable Name: `et` or `et_seconds`.
*   **Input/Output**: `UTC` (Coordinated Universal Time) in ISO 8601 format (e.g., `2026-05-20T12:00:00`).

### 2.3 Units
While SI is preferred for physics, **km** is the standard for astrodynamics compatibility with SPICE.
*   **Distance**: Kilometers (**km**)
*   **Velocity**: Kilometers per second (**km/s**)
*   **Mass**: Kilograms (**kg**)
*   **Angle**: Radians (**rad**) (Degrees permissible for I/O only).
*   **Time**: Seconds (**s**)

*Note: All physical constants ($GM$, radii) must be loaded from SPICE kernels to ensure unit consistency.*

### 2.4 Variable Naming
*   Position Vector: `r_NAME` or `pos_NAME` (e.g., `r_earth`, `pos_sc`)
*   Velocity Vector: `v_NAME` or `vel_NAME`
*   State Vector (6x1): `state` or `rv`
*   Mu ($GM$): `mu` or `gm`

## 3. Mandatory Assumptions & Rules

### 3.1 Dynamics Models
*   **Primary Source**: All planetary ephemerides and constants must be retrieved via `spiceypy` using NASA's DE4xx kernels (e.g., `de440.bsp`).
*   **Integrator**: Variable-step integrators (e.g., RK45, DOP853, LSODA) must be used.
    *   **Tolerance**: Relative error tolerance $\le 10^{-9}$, Absolute error tolerance $\le 10^{-12}$.
*   **N-Body Perturbations**:
    *   Minimum set: Sun + Earth + Moon + Jupiter + Saturn + Target Body.
    *   Solar Radiation Pressure (SRP) and Relativistic corrections must be explicitly "OPT-IN" if used.

### 3.2 State Transition Matrices (STM)
*   For GNC and optimization, STMs must be computed consistent with the dynamics (variational equations preferred over finite differencing).

### 3.3 Trajectory Validation
*   All patched-conic approximations must be validated against full N-body numerical propagation.
*   Continuity: Position error $< 1$ meter, Velocity error $< 1$ mm/s at patch points.
