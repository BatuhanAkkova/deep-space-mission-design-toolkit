import numpy as np
from scipy.integrate import solve_ivp
from src.spice.manager import spice_manager
from src.dynamics.perturbations import Perturbation
import warnings

class NBodyDynamics:
    """
    Propagates spacecraft trajectory under the influence of N point-mass bodies and additional perturbations.
    """
    def __init__(self, bodies: list[str], frame: str = 'ECLIPJ2000', central_body: str = 'SUN', perturbations: list[Perturbation] = []):
        """
        Initialize the N-Body dynamics model.

        Args:
            bodies (list[str]): List of SPICE body names exerting gravity (e.g., ['SUN', 'EARTH', 'JUPITER']).
            frame (str): Reference frame for the state vector (default: 'ECLIPJ2000').
            central_body (str): The body at the center of integration (usually 'SUN' or a planet for local operations).
            perturbations (list[Perturbation]): List of additional perturbations (J2, SRP, Aerodynamics).
        """
        self.bodies = bodies
        self.frame = frame
        self.central_body = central_body
        self.perturbations = perturbations
        self.mus = {}
        
        if not spice_manager._kernels_loaded:
            # Try loading
            spice_manager.load_standard_kernels()

        for body in bodies:
            try:
                self.mus[body] = spice_manager.get_mu(body)
            except Exception as e:
                warnings.warn(f"Could not load GM for {body}, it will be ignored in dynamics. Error: {str(e)}")

    def compute_gravity_gradient(self, r_sc: np.ndarray, t: float) -> np.ndarray:
        """
        Computes the gravitational gradient matrix (3x3 Jacobian of acceleration wrt position).
        G = da/dr

        Args:
            r_sc (np.ndarray): Spacecraft position [x, y, z] relative to central body.
            t (float): Ephemeris Time.

        Returns:
            np.ndarray: 3x3 Gravity Gradient Matrix
        """
        G = np.zeros((3, 3))

        for body in self.bodies:
            mu = self.mus.get(body, 0.0)
            if mu == 0.0:
                continue
            
            if body == self.central_body:
                # Central Body Gradient
                # G = -mu/r^3 * (I - 3*r*r^T/r^2)
                r_mag = np.linalg.norm(r_sc)
                if r_mag == 0: continue # Should not happen

                r3 = r_mag**3
                r5 = r_mag**5
                
                I = np.eye(3)
                rrT = np.outer(r_sc, r_sc)
                
                G_body = -(mu / r3) * I + (3 * mu / r5) * rrT
                G += G_body
            else:
                # 3rd Body Gradient
                # a_3rd = -mu * ( (r - r_b)/|r - r_b|^3 )  (Indirect term is constant wrt r)
                # G = -mu/d^3 * (I - 3*d*d^T/d^2)
                try:
                    state_body = spice_manager.get_body_state(body, self.central_body, t, self.frame)
                    r_body = state_body[0:3]
                    
                    d_vec = r_sc - r_body
                    d_mag = np.linalg.norm(d_vec)
                    if d_mag == 0: continue

                    d3 = d_mag**3
                    d5 = d_mag**5
                    
                    I = np.eye(3)
                    ddT = np.outer(d_vec, d_vec)
                    
                    G_body = -(mu / d3) * I + (3 * mu / d5) * ddT
                    G += G_body
                    
                except Exception:
                    pass
        
        # Add perturbation gradients
        for pert in self.perturbations:
            G += pert.compute_partial_derivatives(t, r_sc)
            
        return G

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Computes derivative of state.
        
        If len(state) == 6: returns [v, a] (6 elements)
        If len(state) == 42: returns [v, a, dPhi/dt] (42 elements)
                             where Phi is 6x6 STM flattened.
        """
        # Unpack State
        r_sc = state[0:3]
        v_sc = state[3:6]
        
        # 1. Compute Point Mass Acceleration
        a_total = np.zeros(3)
        
        for body in self.bodies:
            mu = self.mus.get(body, 0.0)
            if mu == 0.0: continue

            if body == self.central_body:
                r_mag = np.linalg.norm(r_sc)
                a_total += -mu * r_sc / (r_mag**3)
            else:
                try:
                    state_body = spice_manager.get_body_state(body, self.central_body, t, self.frame)
                    r_body = state_body[0:3]
                    
                    r_body_to_sc = r_sc - r_body
                    dist = np.linalg.norm(r_body_to_sc)
                    
                    term1 = r_body_to_sc / (dist**3)
                    
                    # Indirect term
                    r_body_mag = np.linalg.norm(r_body)
                    term2 = r_body / (r_body_mag**3)
                    
                    a_total += -mu * (term1 + term2)
                except Exception:
                    pass
        
        # 2. Add Perturbations
        for pert in self.perturbations:
            a_total += pert.compute_acceleration(t, r_sc, state)

        # Base derivative
        d_state_base = np.concatenate((v_sc, a_total))
        
        if len(state) == 6:
            return d_state_base
            
        elif len(state) == 42:
            # STM Propagation detected
            # State vector has 6 (orb) + 36 (STM) elements
            
            # Extract STM Phi (6x6)
            Phi = state[6:].reshape((6, 6))
            
            # Compute Jacobian Matrix A(t)
            # A = [[0, I], [G, 0]]
            G = self.compute_gravity_gradient(r_sc, t)
            
            A = np.zeros((6, 6))
            A[0:3, 3:6] = np.eye(3) # Top right is Identity
            A[3:6, 0:3] = G # Bottom left is Gravity Gradient
            
            # dPhi/dt = A * Phi
            dPhi = A @ Phi
            
            # Flatten and concatenate
            return np.concatenate((d_state_base, dPhi.flatten()))
        
        else:
            raise ValueError(f"State vector length {len(state)} not supported. Expected 6 or 42.")

    def propagate(self, initial_state: np.ndarray, t_span: tuple[float, float], 
                  max_step: float = np.inf, rtol: float = 1e-9, atol: float = 1e-12, stm: bool = False):
        """
        Propagate the state from t_span[0] to t_span[1].
        
        Args:
            initial_state (np.ndarray): Initial [x, y, z, vx, vy, vz].
            t_span (tuple): (t_start, t_end) in ET seconds.
            max_step (float): Maximum step size for integrator.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            stm (bool): If True, propagate and return 6x6 STM appended to state.
            
        Returns:
            scipy.integrate.OdeResult: Integration result. 
                                     y will be (6, N) or (42, N).
        """
        y0 = initial_state
        
        if stm:
            if len(initial_state) != 6:
                raise ValueError("Initial state must be length 6 to propagate STM.")
            
            # Append Identity matrix flattened
            phi0 = np.eye(6).flatten()
            y0 = np.concatenate((initial_state, phi0))
            
        # By default, we want dense output to interpolate later if needed
        sol = solve_ivp(
            fun=self.equations_of_motion,
            t_span=t_span,
            y0=y0,
            method='DOP853',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True
        )
        return sol
