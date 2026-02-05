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
    # Mars
    tg.add_body("MARS", a_km=1.524*AU)
    # Jupiter
    tg.add_body("JUPITER", a_km=5.203*AU)
    
    # Define V_inf Contours (km/s)
    contours = {
        'EARTH': [3.0, 5.0, 7.0, 9.0],
        'MARS': [3.0, 4.0, 5.0, 6.0, 7.0],
        'JUPITER': [3.0, 4.0, 5.0,6.0, 7.0]
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
    fig, ax = tg.plot_graph(p_range, rp_range, 
                  v_inf_contours=contours, 
                  resonance_lines=resonances,
                  filename=output_file,
                  show=False)
                  
    # Define a Mock Flyby Sequence (Earth -> Mars -> Jupiter)
    # This represents a path through the Tisserand Graph
    # 1. Earth Departure (Rp=1.0 AU) -> Transfer Orbit 1 (P=1.4 yr, Rp=1.0 AU)
    # 2. Mars Flyby (Rp=1.52 AU) -> Transfer Orbit 2 (P=6.2 yr, Rp=1.52 AU)
    # 3. Jupiter Arrival
    
    sequence = [
        {'P': 365.25 * 1.0, 'Rp': 1.0, 'name': 'Earth Dep'}, # Earth
        {'P': 518.0,        'Rp': 1.0, 'name': 'Trans 1'}, # E-M Transfer
        {'P': 687.0,        'Rp': 1.524, 'name': 'Mars Flyby'}, # Mars Match
        {'P': 2253.0,       'Rp': 1.524, 'name': 'Trans 2'}, # M-J Transfer
        {'P': 4333.0,       'Rp': 5.203, 'name': 'Jupiter Arr'} # Jupiter
    ]
    
    tg.overlay_sequence(ax, sequence, label="E-M-J Sequence")
    
    # Save again with overlay
    fig.savefig(output_file, dpi=300)
    print(f"Done. Check {output_file}")

if __name__ == "__main__":
    main()
