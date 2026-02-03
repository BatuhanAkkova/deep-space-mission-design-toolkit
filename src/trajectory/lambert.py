import numpy as np
from scipy.optimize import newton

class LambertSolver:
    """
    A robust Lambert solver using Universal Variables.
    This implementation solves the boundary value problem: finding the velocity vectors
    at two points (r1, r2) given the time of flight (dt).
    """

    @staticmethod
    def solve(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float, prograde: bool = True, max_iter: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves Lambert's problem for the transfer between position vectors r1 and r2
        with time of flight dt.

        Args:
            r1 (np.ndarray): Initial position vector [km].
            r2 (np.ndarray): Final position vector [km].
            dt (float): Time of flight [seconds].
            mu (float): Gravitational parameter [km^3/s^2].
            prograde (bool): If True, solve for prograde orbit (inclination < 90).
                             If False, retrograde.
            max_iter (int): Maximum iterations for the solver.
            tol (float): Tolerance for convergence.

        Returns:
            tuple[np.ndarray, np.ndarray]: (v1, v2) - Velocity vectors at r1 and r2 [km/s].
        
        Raises:
            RuntimeError: If the solver fails to converge.
        """
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        
        cross_12 = np.cross(r1, r2)
        r1_dot_r2 = np.dot(r1, r2)
        
        # Determine the change in true anomaly, dnu
        cos_dnu = r1_dot_r2 / (r1_mag * r2_mag)
        
        # Numerical stability clamp
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
        
        dnu = np.arccos(cos_dnu)

        # Standard logic:
        if prograde:
            # Short way
            if cross_12[2] < 0:
                dnu = 2*np.pi - dnu # Ensure prograde/consistent? 
                pass
        else:
            # Long way
            dnu = 2*np.pi - dnu

        # "A" constant
        A = np.sin(dnu) * np.sqrt((r1_mag * r2_mag) / (1 - np.cos(dnu)))
        
        if abs(A) < 1e-12:
            raise RuntimeError("Limit case A=0 (180 degree transfer) not handled.")

        def tof_equation(z):
            # Universal variable time of flight equation
            c_z = LambertSolver.stumpC(z)
            s_z = LambertSolver.stumpS(z)
            
            # Use small epsilon to avoid divide by zero
            y = r1_mag + r2_mag + A * (z * s_z - 1.0) / np.sqrt(max(c_z, 1e-12))
            
            if y <= 0:
                # Return a large value or nan to steer solver away from unphysical regions
                return np.nan
            
            x = np.sqrt(y / c_z)
            t_flight = (x**3 * s_z + A * np.sqrt(y)) / np.sqrt(mu)
            return t_flight - dt

        try:
            z_sol = newton(tof_equation, 0.0, tol=tol, maxiter=max_iter)
        except (RuntimeError, ValueError):
            # Try a slightly different guess if it failed to converge
            try:
                z_sol = newton(tof_equation, 1.0, tol=tol, maxiter=max_iter)
            except:
                raise RuntimeError("Lambert solver failed to converge.")

        z = z_sol
        C = LambertSolver.stumpC(z)
        S = LambertSolver.stumpS(z)
        y = r1_mag + r2_mag + A * (z * S - 1.0) / np.sqrt(C)
        
        f = 1 - (y / r1_mag)
        g = A * np.sqrt(y / mu)
        g_dot = 1 - (y / r2_mag)
        
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
        
        return v1, v2

    @staticmethod
    def stumpS(z):
        if z > 1e-6:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z))**3
        elif z < -1e-6:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z))**3
        else:
            return 1.0/6.0 - z/120.0

    @staticmethod
    def stumpC(z):
        if z > 1e-6:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < -1e-6:
            return (np.cosh(np.sqrt(-z)) - 1) / (-z)
        else:
            return 0.5 - z/24.0
