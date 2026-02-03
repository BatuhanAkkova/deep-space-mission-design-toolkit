import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mission.porkchop import PorkchopPlotter
from src.spice.manager import spice_manager
import datetime

def date_to_et(date_str):
    # Rough approximation or use spice
    # Assume input is YYYY-MM-DD
    # We can use simple arithmetic if SPICE not immediately handy for strings,
    # but let's use a fixed offset.
    # Actually, we rely on SPICE to load kernels.
    # If kernels loaded, use str2et (not exposed in manager yet?).
    # Manager uses 'get_body_state' by time. 
    # Python datetime to et:
    # J2000 is 2000-01-01 12:00:00
    dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    delta = dt_obj - datetime.datetime(2000, 1, 1, 12, 0, 0)
    return delta.total_seconds()

def main():
    print("Generating Porkchop Plot for Earth-Mars 2005...")
    
    # Load Kernels
    if not spice_manager._kernels_loaded:
        spice_manager.load_standard_kernels()
        
    plotter = PorkchopPlotter('EARTH BARYCENTER', 'MARS BARYCENTER')
    
    # 2005 Opportunity
    # Launch: mid 2005
    # Arrival: 2006
    
    launch_start = date_to_et("2005-01-01")
    launch_end = date_to_et("2005-12-01")
    
    arr_start = date_to_et("2005-06-01")
    arr_end = date_to_et("2006-12-01")
    
    n_pts = 30 # Coarse grid for demo
    
    launch_dates = np.linspace(launch_start, launch_end, n_pts)
    arrival_dates = np.linspace(arr_start, arr_end, n_pts)
    
    print("Calculating Lambert solutions...")
    data = plotter.generate_data(launch_dates, arrival_dates)
    
    filename = "earth_mars_2005_porkchop.png"
    print(f"Plotting to {filename}...")
    plotter.plot(data, max_c3=50.0, filename=filename)
    print("Done.")

if __name__ == "__main__":
    main()
