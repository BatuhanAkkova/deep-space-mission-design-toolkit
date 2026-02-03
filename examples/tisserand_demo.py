
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
    
    # Initialize Manager
    try:
        spice_manager.load_standard_kernels('data') 
    except:
        print("Warning: Could not load SPICE kernels. Using default constants.")

    # Create Graph Instance
    tg = TisserandGraph()
    
    # Add Bodies
    AU = 1.495978707e8 # km
    
    # Earth
    tg.add_body("EARTH", a_km=1.000*AU)
    # Venus
    tg.add_body("VENUS", a_km=0.723*AU)
    # Mars
    tg.add_body("MARS", a_km=1.524*AU)
    # Jupiter
    tg.add_body("JUPITER", a_km=5.203*AU)
    
    # Define V_inf Contours (km/s)
    contours = {
        'EARTH': [3.0, 5.0, 7.0, 9.0],
        'VENUS': [3.0, 4.0, 6.0],
        'MARS': [3.0, 5.0, 7.0],
        'JUPITER': [6.0, 7.0, 9.0]
    }
    
    # Define Resonances
    # (m:n) -> m spacecraft orbits = n planet orbits
    # T_sc / T_pl = n / m
    resonances = {
        'EARTH': [(1, 1), (2, 1), (3, 2), (2, 3)],  # 1:1, 2:1 (2 year), 3:2, 2:3
        'JUPITER': [(2, 5), (1, 2)]
    }
    
    # Define Plot Range
    # Period: 100 days to 13 years (~4800 days)
    p_range = (100, 4800) # Days
    rp_range = (0.5, 5.5) # AU
    
    output_file = 'tisserand_plot_demo.png'
    
    # Plot
    tg.plot_graph(p_range, rp_range, 
                  v_inf_contours=contours, 
                  resonance_lines=resonances,
                  filename=output_file)
                  
    print(f"Done. Check {output_file}")

if __name__ == "__main__":
    main()
