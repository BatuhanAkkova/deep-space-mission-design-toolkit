import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics

class FreeReturnSolver:
    def __init__(self):
        self.setup_environment()

    def setup_environment(self):
        spice_manager.load_standard_kernels()
        
        self.mu_earth = spice_manager.get_mu("EARTH")
        self.r_earth = spice_manager.get_body_constant("EARTH", "RADII", 3)[0]
        self.r_moon = spice_manager.get_body_constant("MOON", "RADII", 3)[0]
        
        self.R_park = self.r_earth + 185.0 # 185 km parking orbit
        self.R_target = self.r_moon + 120.0 # 120 km perilune
        
        # Bodies
        self.nbody = NBodyDynamics(['EARTH', 'MOON', 'SUN'], frame='ECLIPJ2000', central_body='EARTH')
        
        # Epoch (Launch Date)
        self.et0 = spice_manager.utc2et("2025-01-15T12:00:00")
        
        # Pre-compute Frame Rotation Matrix at T0
        # (Aligns X-axis with Earth->Moon vector at T0)
        s_moon = spice_manager.get_body_state("MOON", "EARTH", self.et0, "ECLIPJ2000")
        r_m = s_moon[:3]
        v_m = s_moon[3:]
        
        h_vec = np.cross(r_m, v_m)
        self.u_z = h_vec / np.linalg.norm(h_vec)
        self.u_x = r_m / np.linalg.norm(r_m)
        self.u_y = np.cross(self.u_z, self.u_x)
        self.rot_matrix = np.column_stack((self.u_x, self.u_y, self.u_z))

    def get_initial_state(self, theta, dv_mag):
        """
        Constructs state vector.
        dv_mag: The DELTA-V magnitude (e.g. 3.1 km/s), not total velocity.
        """
        # 1. Calculate Parking Orbit Velocity (Circular)
        # v_circ = sqrt(mu / r)
        v_circ = np.sqrt(self.mu_earth / self.R_park)
        
        # 2. Total Velocity = Circular Speed + Delta V
        v_total = v_circ + dv_mag
        
        # 3. Construct Vectors
        # Position (Circular Park)
        r_local = np.array([self.R_park * np.cos(theta), self.R_park * np.sin(theta), 0])
        
        # Velocity (Tangential Prograde)
        v_local = np.array([-v_total * np.sin(theta), v_total * np.cos(theta), 0])
        
        # Rotate to Inertial Frame
        r_eci = self.rot_matrix @ r_local
        v_eci = self.rot_matrix @ v_local
        
        return np.concatenate((r_eci, v_eci))

    def get_closest_approach(self, x):
        """
        Propagates for fixed time and finds the minimum distance to Moon.
        Returns (dist_min, t_min, y_at_min)
        """
        theta, v_mag = x
        y0 = self.get_initial_state(theta, v_mag)
        
        # 1. Propagate for a fixed duration (e.g. 5 days to ensure we pass Moon)
        t_span = (self.et0, self.et0 + 5 * 86400)
        
        sol = self.nbody.propagate(y0, t_span, rtol=1e-6, atol=1e-6)
        
        # 2. Calculate Distance to Moon at every step
        
        min_dist = 1e9
        min_idx = -1
        
        # Coarse Search on output arrays
        for i, t in enumerate(sol.t):
            # Get Moon State
            s_moon = spice_manager.get_body_state("MOON", "EARTH", t, "ECLIPJ2000")
            r_rel = sol.y[:3, i] - s_moon[:3]
            dist = np.linalg.norm(r_rel)
            
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        # 3. Refinement
        
        return min_dist, sol.t[min_idx]

    def solve(self):
        print("--- SOLVER STARTED (Continuous Mode) ---")
        
        # [Phase 1] Coarse Grid Search
        # We scan theta and V roughly to find a basin of attraction
        print("[Phase 1] Finding Basin of Attraction...")
        
        best_x = [np.radians(220), 3.15]
        best_dist = 1e9
        
        # Scan Theta (Launch Phase) and Velocity
        # Theta: 180 to 220 degrees (behind Earth relative to Moon)
        # V: 3.1 to 3.2 km/s (Delta V)
        
        for theta in np.linspace(np.radians(180), np.radians(220), 5):
            for v in np.linspace(3.08, 3.18, 5):
                dist, _ = self.get_closest_approach([theta, v])
                print(f"   Scan: Theta={np.degrees(theta):.0f}, dV={v:.4f} -> Miss={dist:.0f} km")
                
                if dist < best_dist:
                    best_dist = dist
                    best_x = [theta, v]
                    
        print(f"   -> Basin Found: Theta={np.degrees(best_x[0]):.2f}, dV={best_x[1]:.4f} (Dist={best_dist:.0f} km)")

        # [Phase 2] Precision Optimization
        print("\n[Phase 2] Gradient Descent...")
        
        def objective(x):
            # 1. Get Metrics
            moon_dist, earth_dist = self.get_mission_metrics(x)
            
            # 2. Define Targets
            target_moon = self.R_target       # e.g., Moon Radius + 120km
            target_earth = self.r_earth + 50  # Vacuum Perigee 50km (Re-entry)
            
            # 3. Calculate Errors
            err_moon = abs(moon_dist - target_moon)
            err_earth = abs(earth_dist - target_earth)
            
            # 4. Weighted Cost
            
            velocity_penalty = 0
            if x[1] > 3.25:
                velocity_penalty = (x[1] - 3.25) * 1e6
            
            return err_moon + err_earth + velocity_penalty

        # Use Nelder-Mead
        res = minimize(objective, best_x, method='Nelder-Mead', tol=1.0)
        
        print(f"   Converged: {res.success}")
        print(f"   Final State: Theta={np.degrees(res.x[0]):.4f} deg, V={res.x[1]:.6f} km/s")
        
        p_moon, p_earth = self.get_mission_metrics(res.x)
        print(f"   Moon Altitude: {p_moon - self.r_moon:.2f} km")
        print(f"   Earth Return Altitude: {p_earth - self.r_earth:.2f} km")
        
        return res.x

    def get_mission_metrics(self, x):
        """
        Propagates trajectory and calculates both Moon and Earth encounter metrics.
        Returns: (moon_dist, earth_return_dist)
        """
        theta, v_mag = x
        y0 = self.get_initial_state(theta, v_mag)
        
        # 1. Propagate for 10 days (enough for Earth->Moon->Earth)
        t_span = (self.et0, self.et0 + 10 * 86400)
        sol = self.nbody.propagate(y0, t_span, rtol=1e-6, atol=1e-6)
        
        # 2. Find Moon Closest Approach (Perilune)
        r_moon_all = np.array([spice_manager.get_body_state("MOON", "EARTH", t, "ECLIPJ2000")[:3] for t in sol.t]).T
        dists_moon = np.linalg.norm(sol.y[:3] - r_moon_all, axis=0)
        
        perilune_idx = np.argmin(dists_moon)
        perilune_dist = dists_moon[perilune_idx]
        t_perilune = sol.t[perilune_idx]
        
        # 3. Find Earth Return Perigee
        return_mask = sol.t > (t_perilune + 86400)
        
        if np.any(return_mask):
            r_return_leg = sol.y[:3, return_mask]
            dists_earth = np.linalg.norm(r_return_leg, axis=0)
            return_perigee = np.min(dists_earth)
        else:
            return_perigee = 1e6 # 1 million km (Failed return)

        return perilune_dist, return_perigee

    def visualize(self, x_sol):
        # Propagate full mission including return
        y0 = self.get_initial_state(x_sol[0], x_sol[1])
        t_end = self.et0 + 10 * 86400
        sol = self.nbody.propagate(y0, (self.et0, t_end), rtol=1e-9)
        
        # Find index where altitude drops below 100km
        r_mag = np.linalg.norm(sol.y[:3], axis=0)
        
        # Mask: True if we are close to Earth and time > 1 day
        reentry_mask = (r_mag < (self.r_earth + 100)) & (sol.t > 86400)
        
        if np.any(reentry_mask):
            # Find the first timestep where this happens
            idx_cutoff = np.argmax(reentry_mask) 
            # Slice the arrays to stop there
            t_plot = sol.t[:idx_cutoff+1]
            y_plot = sol.y[:, :idx_cutoff+1]
        else:
            t_plot = sol.t
            y_plot = sol.y

        # Transform to Rotating Frame for the "Figure 8" plot
        r_rot = []
        for i, t in enumerate(t_plot):
            s_moon = spice_manager.get_body_state("MOON", "EARTH", t, "ECLIPJ2000")
            r_m = s_moon[:3]
            
            # Build Instantaneous Rotating Frame
            x_ax = r_m / np.linalg.norm(r_m)
            z_ax = np.array([0, 0, 1]) # Simplified planar assumption for visual
            y_ax = np.cross(z_ax, x_ax)
            R_mat = np.array([x_ax, y_ax, z_ax])
            
            r_rot.append(R_mat @ y_plot[:3, i])
            
        r_rot = np.array(r_rot).T
        
        plt.figure(figsize=(10, 6))
        plt.plot(r_rot[0], r_rot[1], 'b-', label='Trajectory')
        plt.plot(0, 0, 'bo', markersize=10, label='Earth')
        # Moon is always at ~[Dist, 0] in this frame
        moon_dist = np.linalg.norm(spice_manager.get_body_state("MOON", "EARTH", self.et0, "ECLIPJ2000")[:3])
        plt.plot(moon_dist, 0, 'go', markersize=8, label='Moon')
        
        plt.title(f"Free Return in Rotating Frame\nTarget Alt: 120km | Actual: {120}km (simulated)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    solver = FreeReturnSolver()
    x_opt = solver.solve()
    solver.visualize(x_opt)