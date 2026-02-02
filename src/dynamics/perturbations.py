from abc import ABC, abstractmethod
import numpy as np
from src.spice.manager import spice_manager
import warnings

class Perturbation(ABC):
    """
    Abstract base class for all perturbations.
    J2 CANNOT BE FETCHED FROM SPICE. SOLVE IT LATER.
    """
    def __init__(self):
        pass

    @abstractmethod
    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        """
        Compute the perturbative acceleration.

        Args:
            t (float): Ephemeris Time.
            r_sc (np.ndarray): Spacecraft position vector relative to central body (Inertial Frame).
            state (np.ndarray): Full state vector (optional, required for drag).

        Returns:
            np.ndarray: Acceleration vector (km/s^2).
        """
        pass

    def compute_partial_derivatives(self, t: float, r_sc: np.ndarray) -> np.ndarray:
        """
        Compute the partial derivative of acceleration w.r.t position (3x3 Jacobian).
        Default is zero matrix (conservative approximation or not supported).

        Args:
             t (float): Ephemeris Time.
             r_sc (np.ndarray): Spacecraft position vector.
        
        Returns:
            np.ndarray: 3x3 Matrix da/dr.
        """
        return np.zeros((3, 3))

class J2Perturbation(Perturbation):
    """
    J2 Zonal Harmonic Perturbation.
    """
    def __init__(self, central_body: str, frame: str = 'ECLIPJ2000', fixed_frame: str = 'IAU_EARTH'):
        """
        Args:
            central_body (str): Name of the central body (e.g., 'EARTH').
            frame (str): The inertial integration frame (e.g., 'ECLIPJ2000').
            fixed_frame (str): The body-fixed frame (e.g., 'IAU_EARTH').
        """
        super().__init__()
        self.central_body = central_body
        self.frame = frame
        self.fixed_frame = fixed_frame
        
        try:
            self.mu = spice_manager.get_mu(central_body)
            # RADII usually has 3 elements (a, b, c)
            self.R_body = spice_manager.get_body_constant(central_body, 'RADII')[0] 
            self.J2 = spice_manager.get_body_constant(central_body, 'J2')
        except Exception as e:
            warnings.warn(f"Failed to initialize J2 for {central_body}: {e}")
            self.mu = 0
            self.J2 = 0

    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        if self.J2 == 0: return np.zeros(3)

        # 1. Get Rotation Matrix (Inertial -> Fixed)
        # R * r_inertial = r_fixed
        R = spice_manager.get_coord_transform(self.frame, self.fixed_frame, t)
        
        r_fixed = R @ r_sc
        x, y, z = r_fixed
        r_sq = x*x + y*y + z*z
        r = np.sqrt(r_sq)
        
        if r == 0: return np.zeros(3)

        # 2. Compute Acceleration in Fixed Frame
        # a_x = - (3/2) J2 (mu/r^2) (Re/r)^2 (x/r) (1 - 5(z/r)^2)
        # a_z = - (3/2) J2 (mu/r^2) (Re/r)^2 (z/r) (3 - 5(z/r)^2)
        
        factor = -(3.0/2.0) * self.J2 * (self.mu / r_sq) * (self.R_body / r)**2
        z_r_sq = (z/r)**2
        
        ax = factor * (x/r) * (1 - 5 * z_r_sq)
        ay = factor * (y/r) * (1 - 5 * z_r_sq)
        az = factor * (z/r) * (3 - 5 * z_r_sq)
        
        a_fixed = np.array([ax, ay, az])
        
        # 3. Rotate back to Inertial
        # a_inertial = R^T * a_fixed
        return R.T @ a_fixed

class SSRPerturbation(Perturbation):
    """
    Solar Radiation Pressure (Cannonball Model).
    """
    def __init__(self, area: float, mass: float, CR: float, central_body: str = 'SUN'):
        super().__init__()
        self.area = area
        self.mass = mass
        self.CR = CR
        self.central_body = central_body
        self.P0 = 4.56e-6 * 1e-3 # N/m^2 -> km/s^2 conversion factor adjustment? 
        # P0 is ~4.56e-6 Pa (N/m^2) at 1 AU.
        # Force = P0 * (1AU/d)^2 * CR * A
        # Accel = Force / Mass
        # Units:
        # P0 [kg/(m s^2)]
        # Area [m^2]
        # Mass [kg]
        # Accel [m/s^2]
        # We work in km/s^2 usually.
        # Let's keep input Area in m^2, Mass in kg.
        # P0 = 4.56e-6 N/m^2.
        # Accel (m/s^2) = (4.56e-6) * (AU/d)^2 * CR * A / m
        # Accel (km/s^2) = Accel (m/s^2) / 1000
        
        self.AU_km = 1.495978707e8

    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        # Vector from Sun to Spacecraft
        if self.central_body == 'SUN':
            r_sun_sc = r_sc
        else:
            # Need Sun position relative to Central Body
            # r_sc_rel_sun = r_sc_rel_cb + r_cb_rel_sun
            r_cb_sun = spice_manager.get_body_state('SUN', self.central_body, t)[0:3]
            r_sun_sc = r_sc - r_cb_sun # Wait. 
            # r_sc is S/C relative to Central Body.
            # We need vector S/C relative to Sun = (S/C - Sun)
            # = (S/C - CB) + (CB - Sun)
            # = r_sc + r_cb_sun NOT minus.
            # Using spice: get_body_state('SUN', central_body) gives Sun pos wrt CB using generic function?
            # Actually get_body_state(target='SUN', observer=central_body) returns Sun vector relative to CB.
            # So r_sun_sc = r_sc - r_sun_wrt_cb.
            r_sun_wrt_cb = spice_manager.get_body_state('SUN', self.central_body, t)[0:3]
            r_sun_sc = r_sc - r_sun_wrt_cb
            
        dist_km = np.linalg.norm(r_sun_sc)
        if dist_km == 0: return np.zeros(3)
        
        # P_SRP = P0 * (AU / dist)^2
        # a = - P_SRP * CR * (A/m) * unit_vec_sc_sun ?? 
        # Force acts AWAY from Sun.
        # Direction is r_sun_sc / |r_sun_sc|
        
        P_1AU = 4.56e-6 # N/m^2
        dist_AU = dist_km / self.AU_km
        P_dist = P_1AU / (dist_AU**2)
        
        force_mag_N = P_dist * self.CR * self.area
        accel_mag_m_s2 = force_mag_N / self.mass
        accel_mag_km_s2 = accel_mag_m_s2 / 1000.0
        
        return accel_mag_km_s2 * (r_sun_sc / dist_km)


class DragPerturbation(Perturbation):
    """
    Atmospheric Drag (Exponential Model).
    Currently implemented for Earth and Mars only as defaults.
    """
    def __init__(self, cd: float, area: float, mass: float, central_body: str = 'EARTH'):
        super().__init__()
        self.cd = cd
        self.area = area # m^2
        self.mass = mass # kg
        self.central_body = central_body
        
        # Simple Parameters
        if central_body == 'EARTH':
            self.rho0 = 1.225 # kg/m^3
            self.h0 = 0.0 # km
            self.H = 7.2 # km scale height (approx)
            self.R_body = 6378.14
            self.rot_rate = 7.292115e-5 # rad/s
        elif central_body == 'MARS':
            self.rho0 = 0.020 # kg/m^3
            self.h0 = 0.0
            self.H = 11.1 # km
            self.R_body = 3396.2
            self.rot_rate = 7.088e-5
        else:
            self.rho0 = 0
            self.H = 1
            self.R_body = 1
            warnings.warn(f"Drag not configured for {central_body}, ignoring.")

    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        if self.rho0 == 0: return np.zeros(3)
        if state is None or len(state) < 6: return np.zeros(3)
        
        r_vec = state[0:3]
        v_vec = state[3:6] # Inertial velocity
        
        r_mag = np.linalg.norm(r_vec)
        h = r_mag - self.R_body
        
        if h < 0: 
            # Subsurface?
            return np.zeros(3)
        if h > 1000: # Cutoff
            return np.zeros(3)

        # Atmosphere Rotation (assume rigid body rotation with planet)
        # v_atm = w x r
        w_vec = np.array([0, 0, self.rot_rate]) # Assuming Z-axis rotation approximation
        v_atm = np.cross(w_vec, r_vec)
        
        # Relative velocity
        v_rel = v_vec - v_atm
        v_rel_mag = np.linalg.norm(v_rel)
        
        # Density
        rho = self.rho0 * np.exp(-(h - self.h0) / self.H)
        
        # Drag Force: F = -0.5 * rho * v^2 * Cd * A * unit_v
        # Accel = F / m
        
        # Units:
        # rho: kg/m^3
        # v: km/s -> Need m/s?
        # Let's convert everything to coherent units.
        
        rho_kg_km3 = rho * 1e9 # kg/m^3 * (1e9 m^3 / 1 km^3) = kg/km^3
        # That's huge.
        
        # Let's convert to SI (m, kg, s) then back to km/s^2
        v_rel_m = v_rel * 1000.0
        v_rel_mag_m = v_rel_mag * 1000.0
        
        F_drag_N = 0.5 * rho * (v_rel_mag_m**2) * self.cd * self.area
        a_drag_m_s2 = F_drag_N / self.mass
        
        a_drag_vec_m_s2 = -a_drag_m_s2 * (v_rel_m / v_rel_mag_m)
        
        return a_drag_vec_m_s2 / 1000.0
