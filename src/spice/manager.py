import spiceypy as spice
import os
import glob
import numpy as np

class SpiceManager:
    """
    A singleton-like class to manage SPICE kernels and providing high-level interfaces
    for astrodynamics calculations.
    """
    _instance = None
    _kernels_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpiceManager, cls).__new__(cls)
        return cls._instance

    def load_standard_kernels(self, base_dir: str = 'data'):
        """
        Loads standard SPICE kernels (.bsp, .tpc, .tls, .ker) from a specified directory.
        Uses glob to find files.
        """
        # Unload previous to prevent duplication warnings if called multiple times
        if self._kernels_loaded:
            spice.kclear()
            
        print(f"Loading SPICE kernels from {os.path.abspath(base_dir)}...")
        
        patterns = ['*.bsp', '*.tpc', '*.tls', '*.tf', '*.ker']
        count = 0
        for pattern in patterns:
            search_path = os.path.join(base_dir, pattern)
            kernels = glob.glob(search_path)
            for kernel in kernels:
                try:
                    spice.furnsh(kernel)
                    print(f"Loaded: {os.path.basename(kernel)}")
                    count += 1
                except Exception as e:
                    print(f"Failed to load {kernel}: {e}")
        
        if count == 0:
            print("[WARNING] No SPICE kernels were loaded! Please ensure .bsp, .tpc, .tls, .tf, .ker files are in the data directory.")
        else:
            self._kernels_loaded = True

    def get_body_state(self, target: str, observer: str, et: float, frame: str = 'ECLIPJ2000'):
        """
        Get the state vector (position, velocity) of a target body relative to an observer.
        
        Args:
            target (str): Name of target body (e.g., 'MARS', 'EARTH')
            observer (str): Name of observing body (e.g., 'SUN')
            et (float): Ephemeris Time (seconds past J2000)
            frame (str): Reference frame (default: 'ECLIPJ2000')
            
        Returns:
            numpy.ndarray: 6-element state vector [x, y, z, vx, vy, vz] in km and km/s
        """
        try:
            state, _ = spice.spkezr(target, et, frame, 'NONE', observer)
            return state
        except Exception as e:
            raise RuntimeError(f"SPICE Error getting state for {target} wrt {observer}: {e}")

    def utc2et(self, utc_str: str) -> float:
        """Converts UTC string (ISO 8601) to Ephemeris Time."""
        return spice.str2et(utc_str)

    def et2utc(self, et: float, format_str: str = "ISOC", precision: int = 3) -> str:
        """Converts Ephemeris Time to UTC string."""
        return spice.et2utc(et, format_str, precision)

    def get_coord_transform(self, from_frame: str, to_frame: str, et: float) -> np.ndarray:
        """
        Returns the 3x3 rotation matrix from one frame to another at a given epoch.
        """
        try:
            mat = spice.pxform(from_frame, to_frame, et)
            return np.array(mat)
        except Exception as e:
            raise RuntimeError(f"SPICE Error getting rotation from {from_frame} to {to_frame}: {e}")

    def get_state_transform(self, from_frame: str, to_frame: str, et: float) -> np.ndarray:
        """
        Returns the 6x6 state transformation matrix from one frame to another at a given epoch.
        """
        try:
            mat = spice.sxform(from_frame, to_frame, et)
            return np.array(mat)
        except Exception as e:
            raise RuntimeError(f"SPICE Error getting state transform from {from_frame} to {to_frame}: {e}")

    def get_body_constant(self, body: str, constant_name: str, max_val: int = 1):
        """
        Wraps spice.bodvrd to retrieve body constants like radii or pole orientation.
        
        Args:
            body (str): Body name (e.g., 'EARTH')
            constant_name (str): Constant name (e.g., 'RADII', 'J2')
            
        Returns:
            Numpy array.
        """
        try:
            dim, values = spice.bodvrd(body, constant_name, max_val)
            return values
        except Exception as e:
            raise RuntimeError(f"Could not find constant {constant_name} for {body}: {e}")
        
    def get_mu(self, body: str) -> float:
        """
        Get the gravitational parameter (GM) for a body.
        """
        return self.get_body_constant(body, "GM")

# Global accessibility
spice_manager = SpiceManager()
