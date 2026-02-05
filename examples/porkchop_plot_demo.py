import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mission.porkchop import PorkchopPlotter
from src.spice.manager import spice_manager

def main():
    print("====================================")
    print("   Porkchop Plot Demo (Earth-Mars)  ")
    print("====================================")
    
    # 1. Load kernels
    if not spice_manager._kernels_loaded:
        try:
            spice_manager.load_standard_kernels()
        except Exception as e:
            print(f"Error loading kernels: {e}")
            print("Ensure 'data/' folder exists with SPICE kernels (de432s.bsp, etc).")
            return

    # 2. Define Mission Window
    # The 2005 Earth-Mars transfer (famous example)
    # Launch: ~August 2005
    # Arrival: ~July 2006 (Mars Reconnaissance Orbiter timing)
    
    launch_start_str = "2005-04-01T12:00:00"
    launch_end_str   = "2005-11-01T12:00:00"
    
    arr_start_str    = "2005-12-01T12:00:00"
    arr_end_str      = "2006-12-01T12:00:00"

    # Convert to ET using SPICE
    t_l_start = spice_manager.utc2et(launch_start_str)
    t_l_end   = spice_manager.utc2et(launch_end_str)
    
    t_a_start = spice_manager.utc2et(arr_start_str)
    t_a_end   = spice_manager.utc2et(arr_end_str)

    # Grid Resolution
    n_pts_launch = 50 
    n_pts_arr = 50
    
    launch_dates = np.linspace(t_l_start, t_l_end, n_pts_launch)
    arrival_dates = np.linspace(t_a_start, t_a_end, n_pts_arr)
    
    # 3. Compute Data
    plotter = PorkchopPlotter('EARTH BARYCENTER', 'MARS BARYCENTER')
    data = plotter.generate_data(launch_dates, arrival_dates)
    
    # 4. Plot
    filename = "porkchop.png"
    # Max C3 = 50, Max V_inf = 10 (km/s)
    plotter.plot(data, max_c3=30.0, max_vinf=7.0, filename=filename)

if __name__ == "__main__":
    main()
