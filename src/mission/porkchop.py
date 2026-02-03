import numpy as np
import matplotlib.pyplot as plt
from src.spice.manager import spice_manager
from src.trajectory.lambert import LambertSolver
from src.mission.departure import calculate_departure_asymptotes

class PorkchopPlotter:
    """
    Generates Porkchop plots for interplanetary transfers.
    """
    def __init__(self, departure_body: str, arrival_body: str, frame: str = 'ECLIPJ2000'):
        self.dep_body = departure_body
        self.arr_body = arrival_body
        self.frame = frame
        self.mu_sun = spice_manager.get_body_constant('SUN', 'GM')
        
    def generate_data(self, launch_dates: np.ndarray, arrival_dates: np.ndarray) -> dict:
        """
        Generates C3, V_inf_arrival, and Total Delta-V data for the given date grid.
        
        Args:
            launch_dates (np.ndarray): Array of launch dates [ET seconds].
            arrival_dates (np.ndarray): Array of arrival dates [ET seconds].
            
        Returns:
            dict: {
                'c3': 2D array [km^2/s^2],
                'v_inf_arr': 2D array [km/s],
                'tof_days': 2D array [days],
                'launch_grid': meshgrid of launch dates,
                'arrival_grid': meshgrid of arrival dates
            }
        """
        n_launch = len(launch_dates)
        n_arrival = len(arrival_dates)
        
        c3_grid = np.full((n_arrival, n_launch), np.nan)
        v_inf_arr_grid = np.full((n_arrival, n_launch), np.nan)
        tof_grid = np.full((n_arrival, n_launch), np.nan)
        
        dep_states = {}
        for t in launch_dates:
            try:
                dep_states[t] = spice_manager.get_body_state(self.dep_body, 'SUN', t, self.frame)
            except:
                dep_states[t] = None
                
        arr_states = {}
        for t in arrival_dates:
            try:
                arr_states[t] = spice_manager.get_body_state(self.arr_body, 'SUN', t, self.frame)
            except:
                arr_states[t] = None
        
        # Grid loop
        # We iterate through launch dates (columns) and arrival dates (rows)
        for i, t_arr in enumerate(arrival_dates):
            r_arr_state = arr_states[t_arr]
            if r_arr_state is None: continue
            r2 = r_arr_state[0:3]
            v2_planet = r_arr_state[3:6]
            
            for j, t_launch in enumerate(launch_dates):
                if t_arr <= t_launch:
                    continue
                    
                dt = t_arr - t_launch
                r_dep_state = dep_states[t_launch]
                if r_dep_state is None: continue
                r1 = r_dep_state[0:3]
                v1_planet = r_dep_state[3:6]
                
                try:
                    # Solve Lambert
                    # Using prograde=True (short way) for typical inner planet transfers
                    v1_trans, v2_trans = LambertSolver.solve(r1, r2, dt, self.mu_sun, prograde=True)
                    
                    # Compute Depart C3
                    v_inf_dep = v1_trans - v1_planet
                    c3 = np.linalg.norm(v_inf_dep)**2
                    
                    # Compute Arrival V_inf
                    v_inf_arr = v2_trans - v2_planet
                    v_inf_mag = np.linalg.norm(v_inf_arr)
                    
                    c3_grid[i, j] = c3
                    v_inf_arr_grid[i, j] = v_inf_mag
                    tof_grid[i, j] = dt / 86400.0
                    
                except RuntimeError:
                    # Solver failed (geometry issues etc)
                    pass
                except Exception:
                    pass
                    
        return {
            'c3': c3_grid,
            'v_inf_arr': v_inf_arr_grid,
            'tof_days': tof_grid,
            'launch_dates': launch_dates,
            'arrival_dates': arrival_dates
        }

    def plot(self, data: dict, max_c3: float = 50.0, filename: str = None):
        """
        Plots the Porkchop plot.
        """
        launch_dates = data['launch_dates']
        arrival_dates = data['arrival_dates']
        c3 = data['c3']
        v_inf = data['v_inf_arr']
        tof = data['tof_days']
        
        l_days = (launch_dates - launch_dates[0]) / 86400.0
        a_days = (arrival_dates - launch_dates[0]) / 86400.0
        
        X, Y = np.meshgrid(l_days, a_days)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # C3 Contours
        levels = np.linspace(0, max_c3, 20)
        cs = ax.contour(X, Y, c3, levels=levels, colors='blue', linewidths=0.5)
        ax.clabel(cs, inline=1, fontsize=8, fmt='C3=%1.1f')
        
        # TOF Contours
        cs_tof = ax.contour(X, Y, tof, colors='red', linestyles='dashed', linewidths=0.5)
        ax.clabel(cs_tof, inline=1, fontsize=8, fmt='%d days')
        
        ax.set_title(f"Porkchop Plot: {self.dep_body} to {self.arr_body}")
        ax.set_xlabel(f"Days since Launch Window Open (Epoch {launch_dates[0]})")
        ax.set_ylabel(f"Days since Launch Window Open (Arrival)")
        
        if filename:
            plt.savefig(filename)
            plt.close()
        return fig
