import numpy as np
import math
from scipy.integrate import solve_ivp

class ShapeBasedGuesser:
    """
    Provides shape-based methods for initial trajectory guessing.
    """
    @staticmethod
    def exponential_sinusoid(r0, r_f, t0, tf, n_points=100, k2=1.0, phi=0.0, rotations=0):
        """
        Generates an initial guess using the Exponential Sinusoid shape.
        r = k0 * exp(k1 * sin(k2 * theta + phi))
        
        Args:
            r0: Initial radius (scalar).
            r_f: Final radius (scalar).
            t0: Initial time.
            tf: Final time.
            n_points: Number of points.
            k2: Winding parameter (approx number of revs * 2pi).
            phi: Phase angle.
        
        Returns:
            t: Time grid.
            x: State guess (cartesian [x, y, vx, vy]).
        """
        # 1. Solve for k0, k1
        
        theta_0 = 0.0
        if rotations == 0:
             theta_f = np.pi # Hohmann-like
        else:
             theta_f = 2 * np.pi * rotations

        # k0 * exp(k1 * sin(k2*theta0 + phi)) = r0
        # k0 * exp(k1 * sin(k2*thetaf + phi)) = rf
        
        # Log:
        # ln(k0) + k1 * sin(phi) = ln(r0)
        # ln(k0) + k1 * sin(k2*thetaf + phi) = ln(rf)
        
        s0 = math.sin(phi)
        sf = math.sin(k2 * theta_f + phi)
        
        # If possible to solve
        if abs(s0 - sf) > 1e-6:
            k1 = (np.log(r_f) - np.log(r0)) / (sf - s0)
            ln_k0 = np.log(r0) - k1 * s0
            k0 = np.exp(ln_k0)
        else:
            # Fallback to linear spiral
            k1 = 0
            k0 = (r0 + r_f) / 2.0
            
        theta = np.linspace(theta_0, theta_f, n_points)
        r = k0 * np.exp(k1 * np.sin(k2 * theta + phi))
        
        t = np.linspace(t0, tf, n_points)
        
        # Cartesian conversion (Planar)
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        
        # Velocities
        # v ~ sqrt(mu/r)
        mu = 1.0 # Canonical
        v = np.sqrt(mu / r)
        VX = -v * np.sin(theta)
        VY = v * np.cos(theta)
        
        state = np.vstack((X, Y, np.zeros_like(X), VX, VY, np.zeros_like(X))).T # 3D state
        
        return t, state

class QLawGuesser:
    """
    Q-Law Feedback Control Guesser.
    """
    def __init__(self, mu=1.0, max_thrust=0.1, isp=2000, g0=9.81):
        self.mu = mu
        self.T_max = max_thrust
        self.Isp = isp
        self.g0 = g0
        self.c = isp * g0
        
    def q_law_control(self, oe, oe_target, weight=[1,1,1,1,1]):
        """
        Computes Q-law control acceleration.
        oe: [a, e, i, RAAN, argp]
        """
        # Placeholder for full Q-law logic.
        # Returning a simplified "thrust in velocity direction" (tangential)
        # modified by error in semi-major axis.
        
        # Real Q-law requires sum of (Wi * dQ/dOE) ...
        # For the purpose of initial guess for low-thrust, we often use
        # simple Lyapunov: u = - sign( P^T * (OE - OE_target) )
        
        # Returning [ur, ut, uh] (Radial, Tangential, Normal)
        
        da = oe_target[0] - oe[0]
        di = oe_target[2] - oe[2]
        
        # If semi-major axis needs changing -> Tangential
        # If inclination needs changing -> Normal
        
        ur = 0.0
        ut = 1.0 if da > 0 else -1.0
        uh = 1.0 if di > 0 else -1.0
        
        # Normalize
        u = np.array([ur, ut, uh])
        norm = np.linalg.norm(u)
        if norm > 1e-6:
            u = u / norm
            
        return u

    def generate_guess(self, r0, v0, oe_target, t_max):
        """
        Propagates under Q-Law control.
        """
        y0 = np.concatenate((r0, v0, [1.0])) # State + Mass
        
        def dynamics(t, y):
            r = y[:3]
            v = y[3:6]
            m = y[6]
            
            r_norm = np.linalg.norm(r)
            v_norm = np.linalg.norm(v)
            
            # Keplerian
            acc_grav = -self.mu * r / r_norm**3
            
            # Control (convert Cartesian to OE first)
            # For speed, we skip OE conversion in this placeholder and just allow
            # Tangential thrust
            
            # Tangential direction
            t_vec = v / v_norm
            
            acc_thrust = (self.T_max / m) * t_vec
            
            # Mass flow
            dm = -self.T_max / self.c
            
            dydt = np.concatenate((v, acc_grav + acc_thrust, [dm]))
            return dydt
            
        sol = solve_ivp(dynamics, (0, t_max), y0, rtol=1e-3, atol=1e-3)
        
        return sol.t, sol.y.T, np.zeros((len(sol.t), 3)) # Returns t, x, u=0 (placeholder)
