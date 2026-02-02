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
        Loads standard SPICE kernels (.bsp, .tpc, .tls) from a specified directory.
        Uses glob to find files.
        """
        # Unload previous to prevent duplication warnings if called multiple times
        if self._kernels_loaded:
            spice.kclear()
            
        print(f"Loading SPICE kernels from {os.path.abspath(base_dir)}...")
        
        patterns = ['*.bsp', '*.tpc', '*.tls', '*.tf']
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
            print("[WARNING] No SPICE kernels were loaded! Please ensure .bsp, .tpc, and .tls files are in the data directory.")
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

    def get_mu(self, body: str) -> float:
        """
        Get the gravitational parameter (GM) for a body.
        
        Args:
            body (str): Body name (e.g., 'SUN', 'EARTH')
            
        Returns:
            float: Gravitational parameter (km^3/s^2)
        """
        try:
            # Check if we need to look up by ID or name
            # GM is usually stored as 'BODYnnn_GM' where nnn is the ID
            try:
                # Try getting the ID
                body_id = spice.bodn2c(body)
                key = f"BODY{body_id}_GM"
                n, mu = spice.gdpool(key, 0, 1)
                if n > 0:
                    return mu[0]
            except:
                pass

            # Fallback for common names if standard lookup fails or simpler kernel structure
            # Sometimes just 'GM_BODY' logic applies or directly looking up properties
            # This part can be tricky with different kernel conventions.
            # Let's try the high level bodvrd
            
            # bodvrd returns n values. GM is usually 1 value.
            dim, values = spice.bodvrd(body, "GM", 1)
            return values[0]
            
        except Exception as e:
            raise RuntimeError(f"Could not determine GM for body '{body}'. Check pck kernel. Error: {e}")

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
            return spice.pxform(from_frame, to_frame, et)
        except Exception as e:
            raise RuntimeError(f"SPICE Error getting rotation from {from_frame} to {to_frame}: {e}")

    def get_state_transform(self, from_frame: str, to_frame: str, et: float) -> np.ndarray:
        """
        Returns the 6x6 state transformation matrix from one frame to another at a given epoch.
        """
        try:
            return spice.sxform(from_frame, to_frame, et)
        except Exception as e:
            raise RuntimeError(f"SPICE Error getting state transform from {from_frame} to {to_frame}: {e}")

    def get_body_constant(self, body: str, constant_name: str):
        """
        Wraps spice.bodvrd to retrieve body constants like radii or pole orientation.
        
        Args:
            body (str): Body name (e.g., 'EARTH')
            constant_name (str): Constant name (e.g., 'RADII', 'J2')
            
        Returns:
            Numpy array.
        """
        try:
            dim, values = spice.bodvrd(body, constant_name, 3)
            return values

        except Exception as e:
            # Check if it might be in the pool directly (sometimes J2 is BODYnnn_J2)
            try:
                # Try generic pool lookup if bodvrd fails, though bodvrd is preferred for body variables
                # Construct fallback keys? No, let's just expose gdpool if needed or fail.
                # Usually BODVRD handles 'RADII', 'NUTATION', etc.
                # For J2, it might not be a standard keyword for all bodies in all kernels.
                raise RuntimeError(f"Could not find constant {constant_name} for {body}: {e}")
            except:
                raise RuntimeError(f"Could not find constant {constant_name} for {body}: {e}")

# Global accessibility
spice_manager = SpiceManager()
