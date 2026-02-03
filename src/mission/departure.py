import numpy as np
from src.spice.manager import spice_manager

def calculate_departure_asymptotes(v_inf_vec: np.ndarray, departure_body: str = 'EARTH') -> dict:
    """
    Calculates departure asymptote parameters (C3, DLA, RLA) from a V-infinity vector.
    
    Args:
        v_inf_vec (np.ndarray): Hyperbolic excess velocity vector [vx, vy, vz] in Body-Centered Inertial Frame (usually EME2000 for Earth).
        departure_body (str): Name of the departure body (default: 'EARTH'). 
                              (Context: Used for reference frame checks if needed, currently assumes input is in the correct equatorial frame).
                              
    Returns:
        dict: {
            'C3': characteristic energy [km^2/s^2],
            'DLA_deg': Declination of Launch Asymptote [deg],
            'RLA_deg': Right Ascension of Launch Asymptote [deg],
            'v_inf_mag_km_s': Magnitude of V_inf [km/s]
        }
    """
    # 1. Calculate C3 and V_inf Magnitude
    v_inf_mag = np.linalg.norm(v_inf_vec)
    c3 = v_inf_mag**2
    
    # 2. Calculate DLA and RLA
    # These are essentially the declination and right ascension of the V_inf vector
    # in the equatorial frame of the departure body.
    # Assuming v_inf_vec is provided in the Equatorial Inertial Frame (e.g. J2000 for Earth).
    
    # Unit vector
    if v_inf_mag == 0:
        u = np.zeros(3)
    else:
        u = v_inf_vec / v_inf_mag
        
    # Declination (delta): angle from equatorial plane (-90 to +90)
    # sin(delta) = z / 1
    dla_rad = np.arcsin(u[2])
    
    # Right Ascension (alpha): angle in equatorial plane from X axis (0 to 360)
    # tan(alpha) = y / x
    rla_rad = np.arctan2(u[1], u[0])
    
    # Wrap RLA to [0, 360]
    if rla_rad < 0:
        rla_rad += 2 * np.pi
        
    return {
        'C3': c3,
        'DLA_deg': np.degrees(dla_rad),
        'RLA_deg': np.degrees(rla_rad),
        'v_inf_mag_km_s': v_inf_mag
    }

def state_to_orbital_elements(r: np.ndarray, v: np.ndarray, mu: float) -> dict:
    """
    Converts Cartesian state (r, v) to Keplerian Orbital Elements.
    
    Args:
        r (np.ndarray): Position vector [km].
        v (np.ndarray): Velocity vector [km/s].
        mu (float): Gravitational parameter [km^3/s^2].
        
    Returns:
        dict: Keplerian elements (a, e, i, raan, arg_p, nu).
    """
    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)
    
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # Specific Energy
    E = (v_mag**2) / 2 - mu / r_mag
    
    # Semi-major axis
    if abs(E) < 1e-9:
        a = np.inf # Parabolic
    else:
        a = -mu / (2 * E)
        
    # Eccentricity
    e_vec = (1/mu) * ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)
    
    # Inclination
    i_rad = np.arccos(h_vec[2] / h_mag)
    
    # Node Vector (line of nodes)
    # n = k x h = [-hy, hx, 0]
    n_vec = np.array([-h_vec[1], h_vec[0], 0])
    n_mag = np.linalg.norm(n_vec)
    
    # RAAN (Right Ascension of Ascending Node)
    if n_mag == 0:
        raan_rad = 0.0 # Define as 0 for equatorial orbits
    else:
        raan_rad = np.arccos(n_vec[0] / n_mag)
        if n_vec[1] < 0:
            raan_rad = 2 * np.pi - raan_rad
            
    # Argument of Periapsis (omega)
    if n_mag == 0:
        # Equatorial: angle between I and e?
        arg_p_rad = 0.0 # Ambiguous
    else:
        if e < 1e-9:
             arg_p_rad = 0.0
        else:
             arg_p_rad = np.arccos(np.dot(n_vec, e_vec) / (n_mag * e))
             if e_vec[2] < 0:
                 arg_p_rad = 2 * np.pi - arg_p_rad
                 
    # True Anomaly (nu)
    if e < 1e-9:
        # Circular: angle between n and r? 
        # Actually usually argument of latitude u = omega + nu
        pass 
        
    if e == 0:
        xp = np.dot(r, n_vec)/n_mag if n_mag > 0 else r[0] # simplification
        # This part is tricky for special cases.
        # Let's simple check angle between e and r.
        nu_rad = 0.0
    else:
        nu_rad = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            nu_rad = 2 * np.pi - nu_rad
            
    return {
        'a': a,
        'e': e,
        'i_deg': np.degrees(i_rad),
        'raan_deg': np.degrees(raan_rad),
        'arg_p_deg': np.degrees(arg_p_rad),
        'nu_deg': np.degrees(nu_rad)
    }
