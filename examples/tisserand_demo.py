
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mission.tisserand import TisserandGraph
from src.spice.manager import spice_manager

def main():
    print("Generating Tisserand Graph...")
    
    # Initialize Manager (Optional, TisserandGraph handles constants if SPICE fails, but we want SPICE if possible)
    try:
        spice_manager.load_standard_kernels('data') # path relative to run location usually
    except:
        print("Warning: Could not load SPICE kernels. Using default constants.")

    # Create Graph Instance
    tg = TisserandGraph()
    
    # Add Bodies
    # Using explicit values for clarity and robustness in this demo
    AU = 1.495978707e8 # km
    
    # Earth
    tg.add_body("EARTH", a_km=1.000*AU)
    # Venus
    tg.add_body("VENUS", a_km=0.723*AU)
    # Mars
    tg.add_body("MARS", a_km=1.524*AU)
    # Jupiter
    tg.add_body("JUPITER", a_km=5.203*AU)
    
    # Define Contours
    # V_inf in km/s
    contours = {
        'EARTH': [3.0, 5.0, 7.0, 9.0],
        'VENUS': [3.0, 5.0, 7.0],
        'MARS': [3.0, 5.0, 7.0],
        'JUPITER': [5.0, 7.0, 9.0, 12.0]
    }
    
    # Define Plot Range
    # Period: 0.5 Years to 12 Years (to reach Jupiter) -> 180 days to 4500 days
    # Rp: 0.5 AU to 6.0 AU
    
    p_range = (100, 5000) # Days
    rp_range = (0.4, 6.0) # AU
    
    output_file = 'tisserand_plot_demo.png'
    
    # Plot
    tg.plot_graph(p_range, rp_range, 
                  v_inf_contours=contours, 
                  filename=output_file)
                  
    print(f"Done. Check {output_file}")

if __name__ == "__main__":
    main()
