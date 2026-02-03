import numpy as np

class ManeuverError:
    """
    Represents execution errors for a maneuver.
    """
    def __init__(self, magnitude_sigma: float = 0.0, pointing_sigma: float = 0.0):
        """
        Args:
            magnitude_sigma (float): Standard deviation of magnitude error (relative, e.g. 0.01 for 1%).
                                     Actually, usually defined as 1-sigma relative error or absolute?
                                     Let's assume RELATIVE error (percentage/100).
            pointing_sigma (float): Standard deviation of pointing error [radians].
        """
        self.magnitude_sigma = magnitude_sigma
        self.pointing_sigma = pointing_sigma

    def apply_error(self, delta_v: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian error to the delta_v vector.
        """
        if self.magnitude_sigma == 0.0 and self.pointing_sigma == 0.0:
            return delta_v

        mag = np.linalg.norm(delta_v)
        if mag == 0:
            return delta_v

        # Magnitude Error
        mag_error = np.random.normal(0, self.magnitude_sigma * mag)
        new_mag = mag + mag_error
        
        # Pointing Error
        # Generate a random vector orthogonal to delta_v
        # Rotate delta_v by pointing_sigma (approx)
        # Actually random angle around the vector? No, deviation angle.
        
        if self.pointing_sigma > 0:
            # Random deviation angle
            deviation = np.abs(np.random.normal(0, self.pointing_sigma))
            # Random roll angle around the vector
            roll = np.random.uniform(0, 2*np.pi)
            
            # Construct a coordinate system aligned with delta_v
            u = delta_v / mag
            
            # Find arbitrary orthogonal vector v
            if np.abs(u[0]) < 0.9:
                v = np.cross(u, np.array([1, 0, 0]))
            else:
                v = np.cross(u, np.array([0, 1, 0]))
            v = v / np.linalg.norm(v)
            
            w = np.cross(u, v)
            
            # Perturbed direction in local frame (u, v, w)
            # u is forward.
            # rotate by deviation
            u_prime = np.cos(deviation) * u + np.sin(deviation) * (np.cos(roll) * v + np.sin(roll) * w)
            
            result = u_prime * new_mag
        else:
            result = (delta_v / mag) * new_mag
            
        return result

class ImpulsiveManeuver:
    """
    Represents an impulsive delta-v maneuver.
    """
    def __init__(self, epoch: float, delta_v: np.ndarray, frame: str = 'J2000', error_model: ManeuverError = None):
        """
        Args:
            epoch (float): Time of maneuver execution [ET].
            delta_v (np.ndarray): Delta-V vector [km/s] (3,).
            frame (str): Reference frame of the delta-v vector.
            error_model (ManeuverError): Error model to apply during execution.
        """
        self.epoch = epoch
        self.delta_v_nominal = delta_v
        self.frame = frame
        self.error_model = error_model

    def get_delta_v(self, apply_errors: bool = False) -> np.ndarray:
        """
        Returns the delta-v vector, optionally with errors applied.
        """
        if apply_errors and self.error_model:
            return self.error_model.apply_error(self.delta_v_nominal)
        return self.delta_v_nominal

    def apply_to_state(self, state: np.ndarray, epoch: float, apply_errors: bool = False) -> np.ndarray:
        """
        Applies the maneuver to a state vector if the time matches.
        
        Args:
            state (np.ndarray): State vector [rx, ry, rz, vx, vy, vz].
            epoch (float): Current time.
            
        Returns:
            np.ndarray: Updated state vector.
        """
        # Note: In a continuous propagation, this needs event detection.
        # This function acts as an instantaneous jump.
        if abs(epoch - self.epoch) < 1e-6: # Tolerance check
            dv = self.get_delta_v(apply_errors)
            new_state = state.copy()
            new_state[3:6] += dv
            return new_state
        return state
