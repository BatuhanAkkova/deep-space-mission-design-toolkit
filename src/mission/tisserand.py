import numpy as np
import matplotlib.pyplot as plt
from src.spice.manager import spice_manager

class TisserandGraph:
    """
    Class to generate Tisserand Graphs for gravity assist sequence planning.
    Plots Orbital Period (or Energy) vs Perihelion Radius.
    Draws contours of constant V_inf relative to flyby bodies.
    Supports drawing orbital resonance lines (e.g., 2:3 Earth Resonance).
    """
    def __init__(self, mu_sun: float = None):
        """
        Args:
            mu_sun (float): Gravitational parameter of the central body (Sun). 
                            If None, attempts to get from SPICE.
        """
        if mu_sun is None:
            try:
                # Try to get Sun GM from SPICE if kernels are loaded
                self.mu = spice_manager.get_body_constant('SUN', 'GM')[0]
            except:
                # Fallback to standard value if SPICE not loaded (km^3/s^2)
                self.mu = 132712440018.9 
        else:
            self.mu = mu_sun
            
        self.bodies = {} # Dictionary to store body parameters

    def add_body(self, name: str, a_km: float, radius_km: float = 0):
        """
        Add a body to the graph definition.
        
        Args:
            name (str): Name of the body (e.g. 'EARTH')
            a_km (float): Semi-major axis of the body [km]. Assumed circular orbit.
            radius_km (float): Physical radius of the body [km] (optional, for masking).
        """
        period = 2 * np.pi * np.sqrt(a_km**3 / self.mu)
        v_orb = np.sqrt(self.mu / a_km)
        
        self.bodies[name] = {
            'a': a_km,
            'P': period,
            'V_orb': v_orb,
            'R_mean': a_km, # Assuming circular orbit, R = a
            'Radius': radius_km
        }

    def add_body_from_spice(self, name: str):
        """
        Add a body using approximate mean parameters from SPICE (J2000).
        Logic: Get state at J2000, compute 'a'. This is an approximation.
        """
        try:
            state = spice_manager.get_body_state(name, 'SUN', 0.0, 'ECLIPJ2000')
            r = state[0:3]
            v = state[3:6]
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            
            # Vis-viva equation: v^2 = mu(2/r - 1/a) -> 1/a = 2/r - v^2/mu
            inv_a = 2.0/r_mag - v_mag**2/self.mu
            a = 1.0/inv_a
            
            # Radii
            try:
                radii = spice_manager.get_radii(name)
                radius = radii[0] # Equatorial radius
            except:
                radius = 0
            
            self.add_body(name, a, radius)
            print(f"Added {name} from SPICE: a={a:.2f} km")
            
        except Exception as e:
            print(f"Failed to add {name} from SPICE: {e}")

    def compute_vinf(self, period_days: np.ndarray, rp_au: np.ndarray, body_name: str):
        """
        Compute V_inf magnitude field for a grid of (Period, Rp).
        
        Args:
            period_days (np.ndarray): Meshgrid of orbital periods [days].
            rp_au (np.ndarray): Meshgrid of perihelion radii [AU].
            body_name (str): Name of the flyby body to compute V_inf wrt.
            
        Returns:
            np.ndarray: V_inf values [km/s]. NaN where invalid (no intersection).
        """
        if body_name not in self.bodies:
            raise ValueError(f"Body {body_name} not found. Add it first.")
            
        body = self.bodies[body_name]
        R_pl = body['R_mean'] # km
        V_pl = body['V_orb'] # km/s
        
        # Conversions
        AU_KM = 1.495978707e8
        DAY_SEC = 86400.0
        
        P_sec = period_days * DAY_SEC
        rp_km = rp_au * AU_KM
        
        # Compute spacecraft semi-major axis 'a' from Period
        # P = 2*pi * sqrt(a^3/mu) -> a = (P * sqrt(mu) / 2pi)^(2/3) or mu P^2 / 4pi^2 = a^3
        a_sc = (self.mu * P_sec**2 / (4 * np.pi**2))**(1.0/3.0)
        
        # Intersection Check
        # 1. perihelion must be <= R_pl (SC must go inside or touch planet orbit)
        # 2. aphelion must be >= R_pl (SC must go outside or touch planet orbit)
        # ra = 2*a - rp
        ra_km = 2 * a_sc - rp_km
        
        # Add small tolerance for floating point comparisons (1e-6 km)
        tol = 1e-6
        valid_mask = (rp_km <= R_pl + tol) & (ra_km >= R_pl - tol)
        
        # Tisserand relation for V_inf (assuming i=0, tangent V_pl)
        # v_inf^2 = V_pl^2 * (3 - R_pl/a - 2*sqrt(a/R_pl * (1-e^2)))
        # e = 1 - rp/a
        # 1 - e^2 = 1 - (1 - rp/a)^2 = rp/a * (2 - rp/a)
        # Term = (a/R_pl) * (rp/a) * (2 - rp/a) = (rp/R_pl) * (2 - rp/a)
        # v_inf^2 = V_pl^2 * (3 - R_pl/a - 2*sqrt( (rp/R_pl) * (2 - rp_km/a_sc) ))
        
        # Initialize output
        v_inf_sq = np.full_like(period_days, np.nan)
        
        # Calculation on valid indices only to avoid domain errors
        term_1 = R_pl / a_sc
        
        # Inner term for sqrt: (rp/R_pl) * (2 - rp/a)
        # 2 - rp/a = 2 - rp_km / a_sc
        # rp <= a for valid elliptic orbit
        valid_mask = valid_mask & (rp_km <= a_sc + tol)
        
        inner_term = (rp_km / R_pl) * (2.0 - rp_km / a_sc)
        inner_term = np.maximum(inner_term, 0.0) 
        term_2 = 2.0 * np.sqrt(inner_term)
        
        val = 3.0 - term_1 - term_2
        
        v_inf_sq = V_pl**2 * val
        
        # apply mask
        v_inf = np.sqrt(np.maximum(v_inf_sq, 0.0), where=(valid_mask & (v_inf_sq > -1.0)), out=np.full_like(v_inf_sq, np.nan))
        
        return v_inf

    def plot_graph(self, period_range: tuple, rp_range: tuple, 
                  v_inf_contours: dict = None,
                  resonance_lines: dict = None
                  ):
        """
        Plot the Tisserand Graph.
        
        Args:
            period_range (tuple): (min, max) Period in Days.
            rp_range (tuple): (min, max) Perihelion Radius in AU.
            v_inf_contours (dict): definition of contours. {BodyName: [level1, level2, ...]}
            resonance_lines (dict): definition of resonances. {BodyName: [(num, den), ...]}
                                    e.g. {'EARTH': [(1,1), (2,3)]} means Earth 1:1 and 2:3.
        """
        # Create grid
        p_min, p_max = period_range
        rp_min, rp_max = rp_range
        
        P_grid, Rp_grid = np.meshgrid(np.linspace(p_min, p_max, 300), np.linspace(rp_min, rp_max, 300))
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        c_idx = 0
        
        # 1. Plot V_inf contours
        if v_inf_contours:
            for body_name, levels in v_inf_contours.items():
                if body_name not in self.bodies:
                    print(f"Skipping {body_name}, not in bodies.")
                    continue
                    
                v_inf_field = self.compute_vinf(P_grid, Rp_grid, body_name)
                
                # Plot contours
                color = colors[c_idx % len(colors)]
                c_idx += 1
                
                cs = ax.contour(P_grid, Rp_grid, v_inf_field, levels=levels, colors=color, linewidths=1.5)
                ax.clabel(cs, inline=1, fontsize=9, fmt=f'{body_name} %1.1f')
                
                # Mark the body itself on the graph
                b_p = self.bodies[body_name]['P'] / 86400.0 # days
                b_rp = self.bodies[body_name]['a'] / 1.495978707e8 # AU
                
                ax.plot(b_p, b_rp, 'o', color=color, markersize=8, label=f'{body_name} Orbit', markeredgecolor='black')
        
        # 2. Plot Resonance Lines
        # A resonance m:n means m spacecraft periods = n planet periods.
        # T_sc = (n/m) * T_planet
        # This is a horizontal line on the Period plot.
        if resonance_lines:
            for body_name, ratios in resonance_lines.items():
                if body_name not in self.bodies:
                    continue
                
                body = self.bodies[body_name]
                T_pl = body['P'] / 86400.0 # days
                
                for (m, n) in ratios:
                    # Spacecraft Period
                    T_sc = (float(n) / float(m)) * T_pl
                    
                    if p_min <= T_sc <= p_max:
                        ax.hlines(y=T_sc, xmin=rp_min, xmax=rp_max, colors='gray', linestyles=':', alpha=0.5) 
                        ax.vlines(x=T_sc, ymin=rp_min, ymax=rp_max, colors='gray', linestyles=':', alpha=0.5)
                        
                        # Label
                        label_y = rp_max - (rp_max - rp_min) * 0.05
                        ax.text(T_sc, label_y, f"{body_name} {m}:{n}", rotation=90, verticalalignment='top', fontsize=8, color='gray')


        ax.set_xlabel('Orbital Period [Days]', fontsize=12)
        ax.set_ylabel('Perihelion Radius [AU]', fontsize=12)
        ax.set_title('Tisserand Graph (Energy vs Shape)', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')
        
        # Fix Limits
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(rp_min, rp_max)
        
        plt.tight_layout()
        plt.show()
            
        return fig, ax

    def overlay_sequence(self, ax, sequence: list, label: str = "Trajectory"):
        """
        Overlay a trajectory sequence on the graph.
        
        Args:
            ax: Matplotlib axis.
            sequence: List of dicts with {'P': day, 'Rp': AU, 'name': str}
            label: Label for the legend.
        """
        # Unzip
        ps = [item['P'] for item in sequence]
        rps = [item['Rp'] for item in sequence]
        
        # Plot path
        ax.plot(ps, rps, 'k--', linewidth=2, label=label, marker='x', markersize=8)
        
        # Label points
        for item in sequence:
            txt = item.get('name', '')
            if txt:
                ax.text(item['P'], item['Rp']+0.1, txt, fontsize=9, fontweight='bold', ha='center')
