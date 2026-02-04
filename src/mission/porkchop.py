import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from src.spice.manager import spice_manager
from src.trajectory.lambert import LambertSolver

class PorkchopPlotter:
    """
    Generates Porkchop plots for interplanetary transfers.
    Calculates departure C3, arrival V_inf, and Time of Flight (TOF) for a grid of launch and arrival dates.
    """
    def __init__(self, departure_body: str, arrival_body: str, frame: str = 'ECLIPJ2000'):
        """
        Args:
            departure_body (str): Check SPICE name (e.g., 'EARTH BARYCENTER').
            arrival_body (str): Check SPICE name (e.g., 'MARS BARYCENTER').
            frame (str): Reference frame (default 'ECLIPJ2000').
        """
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
                'launch_dates': input array,
                'arrival_dates': input array
            }
        """
        n_launch = len(launch_dates)
        n_arrival = len(arrival_dates)
        
        c3_grid = np.full((n_arrival, n_launch), np.nan)
        v_inf_arr_grid = np.full((n_arrival, n_launch), np.nan)
        tof_grid = np.full((n_arrival, n_launch), np.nan)
        
        print(f"Generating Porkchop data for {n_launch}x{n_arrival} grid...")
        
        # Pre-fetch states to avoid repeated SPICE calls
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
        
        # Loop through grid
        for i, t_arr in enumerate(arrival_dates):
            r_arr_state = arr_states[t_arr]
            if r_arr_state is None: continue
            
            r2 = r_arr_state[0:3]
            v2_planet = r_arr_state[3:6]
            
            for j, t_launch in enumerate(launch_dates):
                if t_arr <= t_launch:
                    continue # Arrival must be after launch
                    
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

    def plot(self, data: dict, max_c3: float = 50.0, max_vinf: float = 10.0, filename: str = None):
        """
        Plots the Porkchop plot.
        
        Args:
            data (dict): Result from generate_data.
            max_c3 (float): Max C3 to plot contours for [km^2/s^2].
            max_vinf (float): Max V_inf at arrival to plot contours for [km/s].
            filename (str): If provided, save to file.
        """
        launch_dates = data['launch_dates']
        arrival_dates = data['arrival_dates']
        c3 = data['c3']
        v_inf = data['v_inf_arr']
        tof = data['tof_days']
        
        # Convert ET to Python Datetime objects for plotting
        l_dates_dt = [spice_manager.et2datetime(et) for et in launch_dates]
        a_dates_dt = [spice_manager.et2datetime(et) for et in arrival_dates]
        
        # Format dates for matplotlib
        l_num = mdates.date2num(l_dates_dt)
        a_num = mdates.date2num(a_dates_dt)
        
        X, Y = np.meshgrid(l_num, a_num)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. C3 Contours (Departure Energy) - Blue Solid Lines
        levels_c3 = np.linspace(np.nanmin(c3), max_c3, 20)
        cs = ax.contour(X, Y, c3, levels=levels_c3, colors='blue', linewidths=1.2)
        ax.clabel(cs, inline=1, fontsize=9, fmt=r'%1.1f')
        
        # 2. TOF Contours (Time of Flight) - Red Dashed/Solid Lines
        # The user requested "red lines". I'll use solid or dashed red. 
        # Typically TOF is dashed to distinguish from energy, but red color is the key.
        tof_min = np.nanmin(tof)
        tof_max = np.nanmax(tof)
        levels_tof = np.linspace(tof_min, tof_max, 15)
        cs_tof = ax.contour(X, Y, tof, levels=levels_tof, colors='red', linestyles='-', linewidths=1.0, alpha=0.6)
        ax.clabel(cs_tof, inline=1, fontsize=8, fmt='%d d')
        
        # 3. Mark Optimal Point (Lowest C3)
        min_idx = np.unravel_index(np.nanargmin(c3), c3.shape)
        opt_launch = X[min_idx]
        opt_arrival = Y[min_idx]
        opt_c3 = c3[min_idx]
        
        ax.plot(opt_launch, opt_arrival, 'k*', markersize=12, label=f'Optimal: C3={opt_c3:.2f}')
        ax.annotate(f'C3={opt_c3:.2f}', (opt_launch, opt_arrival), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

        # Formatting
        ax.set_title(f"Porkchop Plot: {self.dep_body} to {self.arr_body}", fontsize=14, pad=20)
        ax.set_xlabel("Departure Date (UTC)", fontsize=12)
        ax.set_ylabel("Arrival Date (UTC)", fontsize=12)
        
        # Date formatting on axes
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10)]
        ax.legend(custom_lines, [r'Departure $C_3$ ($km^2/s^2$)', 'Time of Flight (days)', 'Optimal Departure'])
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
            plt.close()
        return fig

