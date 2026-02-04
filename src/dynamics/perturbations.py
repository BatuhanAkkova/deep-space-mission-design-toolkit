from abc import ABC, abstractmethod
import numpy as np
from src.spice.manager import spice_manager
import warnings

class Perturbation(ABC):
    """
    Abstract base class for all perturbations.
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

class J2Gravity(Perturbation):
    """
    J2 Zonal Harmonic Perturbation.
    """
    def __init__(self, central_body: str, frame: str = 'ECLIPJ2000', fixed_frame: str = 'IAU_EARTH'):
        """
        Args:
            central_body (str): Name of the central body (Earth, Sun, and Mars only).
            frame (str): The inertial integration frame (e.g., 'ECLIPJ2000').
            fixed_frame (str): The body-fixed frame (e.g., 'IAU_EARTH').
        """
        super().__init__()
        self.central_body = central_body
        self.frame = frame
        self.fixed_frame = fixed_frame
        
        self.mu = spice_manager.get_body_constant(central_body, 'GM', 1)
        self.R_body = spice_manager.get_body_constant(central_body, 'RADII', 3)[0] 

        if central_body == 'EARTH':
            self.J2 = spice_manager.get_body_constant(central_body, 'J2E', 1)
        elif central_body == 'MARS':
            self.J2 = spice_manager.get_body_constant(central_body, 'J2M', 1)
        elif central_body == 'SUN':
            self.J2 = spice_manager.get_body_constant(central_body, 'J2SUN', 1)
        else:
            raise ValueError(f"J2 not configured for {central_body}")

    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        if self.J2 == 0: return np.zeros(3)

        # 1. Get Rotation Matrix (Inertial -> Fixed)
        R = spice_manager.get_coord_transform(self.frame, self.fixed_frame, t)
        
        r_fixed = R @ r_sc
        x, y, z = r_fixed
        r_sq = x*x + y*y + z*z
        r = np.sqrt(r_sq)
        
        if r == 0: return np.zeros(3)

        # 2. Compute Acceleration in Fixed Frame
        factor = -(3.0/2.0) * self.J2 * (self.mu / r_sq) * (self.R_body / r)**2
        z_r_sq = (z/r)**2
        
        ax = factor * (x/r) * (1 - 5 * z_r_sq)
        ay = factor * (y/r) * (1 - 5 * z_r_sq)
        az = factor * (z/r) * (3 - 5 * z_r_sq)
        
        a_fixed = np.array([ax, ay, az])
        
        # 3. Rotate back to Inertial
        return (R.T @ a_fixed).flatten()

class SRP(Perturbation):
    """
    Solar Radiation Pressure (Cannonball Model).
    """
    def __init__(self, area: float, mass: float, CR: float, central_body: str = 'SUN'):
        super().__init__()
        self.area = area
        self.mass = mass
        self.CR = CR
        self.central_body = central_body
        self.P0 = 4.56e-6 # N/m^2
        self.AU = 1.495978707e8 # km

    def compute_acceleration(self, t: float, r_sc: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        if self.central_body == 'SUN':
            r_sun_sc = r_sc
        else:
            r_sun_wrt_cb = spice_manager.get_body_state('SUN', self.central_body, t)[0:3]
            r_sun_sc = r_sc - r_sun_wrt_cb
        
        dist = np.linalg.norm(r_sun_sc)
        if dist == 0: return np.zeros(3)
        
        # P_SRP = P0 * (AU / dist)^2
        dist_AU = self.AU / dist
        P_dist = self.P0 * (dist_AU**2)
        
        force_mag = P_dist * self.CR * self.area # N
        accel_mag = force_mag / self.mass # m/s^2
        accel_mag /= 1000.0 # km/s^2
        
        return (accel_mag * (r_sun_sc / dist)).flatten()

class Drag(Perturbation):
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
        
        if h < 0: return np.zeros(3)
        if h > 1000: return np.zeros(3)

        # Atmosphere Rotation
        # v_atm = w x r
        w_vec = np.array([0, 0, self.rot_rate]) # Assuming Z-axis rotation approximation
        v_atm = np.cross(w_vec, r_vec)
        
        # Relative velocity
        v_rel = v_vec - v_atm # km/s
        v_rel_mag = np.linalg.norm(v_rel) # m/s
        
        # Density
        rho = self.rho0 * np.exp(-(h - self.h0) / self.H)

        v_rel *= 1000.0 # m/s
        v_rel_mag *= 1000.0 # m/s
        
        F_drag = 0.5 * rho * (v_rel_mag**2) * self.cd * self.area # N
        a_drag = F_drag / self.mass # m/s^2
        
        a_drag_vec = -a_drag * (v_rel / v_rel_mag) # m/s^2
        
        return (a_drag_vec / 1000.0).flatten() # km/s^2
