import numpy as np
from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.maneuver import ImpulsiveManeuver

def compute_turn_angle(v_inf_mag: float, mu: float, rp: float) -> float:
    """
    Computes the turn angle (delta) for a hyperbolic flyby.
    
    Args:
        v_inf_mag (float): Hyperbolic excess velocity magnitude [km/s].
        mu (float): Gravitational parameter of the flyby body [km^3/s^2].
        rp (float): Periapsis radius [km].
        
    Returns:
        float: Turn angle in radians.
    """
    # delta = 2 * arcsin(1 / (1 + rp * v_inf^2 / mu))
    # eccentricity e = 1 + rp * v_inf^2 / mu
    e = 1.0 + (rp * v_inf_mag**2) / mu
    delta = 2.0 * np.arcsin(1.0 / e)
    return delta

def compute_aiming_radius(v_inf_mag: float, mu: float, rp: float) -> float:
    """
    Computes the aiming radius (b), also known as the impact parameter.
    
    Args:
        v_inf_mag (float): Hyperbolic excess velocity magnitude [km/s].
        mu (float): Gravitational parameter [km^3/s^2].
        rp (float): Periapsis radius [km].
        
    Returns:
        float: Aiming radius [km].
        
    Note:
        b = mu/v_inf^2 * sqrt(e^2 - 1)
        or b = rp * sqrt(1 + 2*mu/(rp*v_inf^2))
    """
    term = 1.0 + (2.0 * mu) / (rp * v_inf_mag**2)
    b = rp * np.sqrt(term)
    return b

def rotate_vector(vec: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Rotates a vector by a given angle around a specified axis using Rodrigues' rotation formula.
    
    Args:
        vec (np.ndarray): Vector to rotate.
        angle (float): Rotation angle [radians].
        axis (np.ndarray): Axis of rotation (should be normalized).
        
    Returns:
        np.ndarray: Rotated vector.
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Rodrigues formula: v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1 - cos(a))
    cross_prod = np.cross(axis, vec)
    dot_prod = np.dot(axis, vec)
    
    return vec * cos_a + cross_prod * sin_a + axis * dot_prod * (1 - cos_a)

def compute_outgoing_v_inf(v_inf_in: np.ndarray, beta: float, rp: float, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the outgoing V-infinity vector based on B-plane targeting.
    
    Args:
        v_inf_in (np.ndarray): Incoming V-infinity vector [vx, vy, vz] in body-centered frame.
        beta (float): B-plane angle (B_theta) [radians]. 
        rp (float): Desired periapsis radius [km].
        mu (float): Gravitational parameter [km^3/s^2].
        
    Returns:
        tuple: (v_inf_out, B_vector, S_vector)
            v_inf_out (np.ndarray): Outgoing V-infinity vector.
            B_vector (np.ndarray): B-vector (aiming point) in body-centered frame.
            S_vector (np.ndarray): S-vector (direction of incoming asymptote).
    """
    v_inf_mag = np.linalg.norm(v_inf_in)
    S = v_inf_in / v_inf_mag
    
    # Define T and R axes for B-plane
    pole = np.array([0.0, 0.0, 1.0])
    
    if np.abs(np.dot(S, pole)) > 0.999:
        # S is nearly parallel to Pole, choose arbitrary T
        T_temp = np.array([1.0, 0.0, 0.0])
        T = np.cross(S, T_temp)
    else:
        T = np.cross(S, pole)
        
    T = T / np.linalg.norm(T)
    R = np.cross(S, T)
    
    # B vector definition
    
    b = compute_aiming_radius(v_inf_mag, mu, rp)
    delta = compute_turn_angle(v_inf_mag, mu, rp)
    
    B_vec = b * (np.cos(beta) * T + np.sin(beta) * R)
    
    h_vec = np.cross(B_vec, S)
    h_hat = h_vec / np.linalg.norm(h_vec)
    
    # We rotate S by angle delta around h_hat.
    v_inf_out = rotate_vector(v_inf_in, delta, h_hat)
    
    # We rotate S by angle delta around h_hat.
    v_inf_out = rotate_vector(v_inf_in, delta, h_hat)
    
    return v_inf_out, B_vec, S

def state_to_bplane(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> tuple[float, float]:
    """
    Converts a planet-relative state (r, v) to B-plane parameters (B_R, B_T).
    
    Args:
        r_rel (np.ndarray): Position vector relative to planet [km].
        v_rel (np.ndarray): Velocity vector relative to planet [km/s].
        mu (float): Gravitational parameter [km^3/s^2].
        
    Returns:
        tuple[float, float]: (B_R, B_T) in km.
    """
    # 1. Calculate Hyperbolic properties
    r_mag = np.linalg.norm(r_rel)
    v_mag = np.linalg.norm(v_rel)
    
    # Specific Energy
    E = (v_mag**2)/2 - mu/r_mag
    if E <= 0:
        raise ValueError("State is not hyperbolic (E <= 0)")
        
    v_inf_mag = np.sqrt(2*E)
    
    # Angular Momentum
    h_vec = np.cross(r_rel, v_rel)
    h_mag = np.linalg.norm(h_vec)
    
    # Eccentricity Vector
    e_vec = (1/mu) * ((v_mag**2 - mu/r_mag)*r_rel - np.dot(r_rel, v_rel)*v_rel)
    e = np.linalg.norm(e_vec)
    
    # Incoming Asymptote (S vector)
    
    theta_inf = np.arccos(-1/e)
    # Incoming asymptote is at nu = -theta_inf
    
    # Velocity direction at -theta_inf:
    sin_nu = np.sin(-theta_inf)
    cos_nu = np.cos(-theta_inf)
    
    v_inf_x = -sin_nu
    v_inf_y = e + cos_nu
    # in Perifocal frame (x along e, y perpendicular)
    
    s_perifocal = np.array([v_inf_x, v_inf_y, 0])
    s_perifocal /= np.linalg.norm(s_perifocal)
    
    # Transform back to Inertial
    # Basis vectors of Perifocal Frame:
    p_hat = e_vec / e
    w_hat = h_vec / h_mag
    q_hat = np.cross(w_hat, p_hat)
    
    S = s_perifocal[0] * p_hat + s_perifocal[1] * q_hat
    
    # Construct B-plane axes
    pole = np.array([0.0, 0.0, 1.0])
    T = np.cross(S, pole)
    if np.linalg.norm(T) < 1e-4:
        T = np.cross(S, np.array([1.0, 0.0, 0.0]))
    T = T / np.linalg.norm(T)
    R = np.cross(S, T)
    
    # B-vector
    B_vec = np.cross(S, h_vec) / v_inf_mag
    
    br = np.dot(B_vec, R)
    bt = np.dot(B_vec, T)
    
    return br, bt

class FlybyCalculator:
    """
    Calculates flyby parameters and verifies with N-Body propagation.
    """
    def __init__(self, body: str, frame: str = 'J2000'):
        self.body = body
        self.frame = frame
        self.mu = spice_manager.get_mu(body)
        self.radii = spice_manager.get_body_constant(body, 'RADII', 3)
        self.planet_radius = self.radii[0] # Approx
        
    def analyze_flyby(self, v_inf_in: np.ndarray, periapsis_alt: float, beta_angle: float) -> dict:
        """
        Analytic flyby calculation.
        
        Args:
            v_inf_in (np.ndarray): Incoming V_inf vector [km/s] (3,).
            periapsis_alt (float): Periapsis altitude [km].
            beta_angle (float): B-plane angle [rad].
            
        Returns:
            dict: Result dictionary containing outgoing state and geometry.
        """
        rp = self.planet_radius + periapsis_alt
        v_inf_mag = np.linalg.norm(v_inf_in)
        
        v_inf_out, B_vec, S_vec = compute_outgoing_v_inf(v_inf_in, beta_angle, rp, self.mu)
        
        delta = compute_turn_angle(v_inf_mag, self.mu, rp)
        b_mag = compute_aiming_radius(v_inf_mag, self.mu, rp)
        
        return {
            "v_inf_in": v_inf_in,
            "v_inf_out": v_inf_out,
            "delta_deg": np.degrees(delta),
            "rp": rp,
            "altitude": periapsis_alt,
            "b_mag": b_mag,
            "B_vector": B_vec
        }

    def propagate_flyby(self, v_inf_in: np.ndarray, epoch_et: float, periapsis_alt: float, beta_angle: float, dt_SOI: float = 3*86400) -> dict:
        """
        Numerically propagates the flyby using N-Body dynamics to verify analytic results.
        
        Returns:
            dict: Propagation results.
        
        """
        # 1. Analytic setup
        rp = self.planet_radius + periapsis_alt
        mu = self.mu
        v_inf_mag = np.linalg.norm(v_inf_in)
        
        v_inf_out, B_vec, S_vec = compute_outgoing_v_inf(v_inf_in, beta_angle, rp, mu)
        delta = compute_turn_angle(v_inf_mag, mu, rp)
        
        # Calculate Angular Momentum unit vector (axis of rotation)
        h_vec_temp = np.cross(B_vec, v_inf_in)
        h_hat = h_vec_temp / np.linalg.norm(h_vec_temp)
        
        # Velocity at periapsis: Rotate V_inf_in by delta/2
        v_p_hat = rotate_vector(v_inf_in / v_inf_mag, delta / 2.0, h_hat)
        
        # Magnitude at periapsis
        vp = np.sqrt(v_inf_mag**2 + 2 * mu / rp)
        v_p_vec_rel = v_p_hat * vp
        
        # Position at periapsis
        r_p_hat = np.cross(v_p_hat, h_hat)
        r_p_vec_rel = r_p_hat * rp
        
        # 2. Get Planet State at Epoch
        # Assume epoch_et is in TDB/ET
        state_planet = spice_manager.get_body_state(self.body, 'SUN', epoch_et, self.frame)
        r_planet = state_planet[0:3]
        v_planet = state_planet[3:6]
        
        # 3. Form initial full state (Heliocentric)
        r0 = r_planet + r_p_vec_rel
        v0 = v_planet + v_p_vec_rel
        state0 = np.concatenate((r0, v0))
        
        # 4. Propagate
        # Create NBody model
        bodies = ['SUN', self.body] # simplified
        nbody = NBodyDynamics(bodies=bodies, frame=self.frame, central_body='SUN')
        
        # Backwards (-dt) and Forwards (+dt)
        # Times relative to epoch
        t_span_back = (epoch_et, epoch_et - dt_SOI)
        t_span_fwd = (epoch_et, epoch_et + dt_SOI)
        
        sol_back = nbody.propagate(state0, t_span_back, rtol=1e-9, atol=1e-12)
        sol_fwd = nbody.propagate(state0, t_span_fwd, rtol=1e-9, atol=1e-12)
        
        # Extract endpoints
        final_state = sol_fwd.y[:, -1]
        initial_state = sol_back.y[:, -1]
        
        t_final = sol_fwd.t[-1]
        state_p_final = spice_manager.get_body_state(self.body, 'SUN', t_final, self.frame)
        v_p_final = state_p_final[3:6]
        
        v_inf_out_meas = final_state[3:6] - v_p_final
        
        return {
            "analytic_v_inf_out": v_inf_out,
            "measured_v_inf_out": v_inf_out_meas,
            "analytic_delta": np.degrees(delta),
            "measured_delta": np.degrees(np.arccos(np.dot(v_inf_in, v_inf_out_meas)/(np.linalg.norm(v_inf_in)*np.linalg.norm(v_inf_out_meas)))),
            "trajectory_fwd": sol_fwd,
            "trajectory_back": sol_back
        }

class FlybyCorrector:
    """
    Tools for correcting flyby trajectories to target specific parameters.
    """
    def __init__(self, nbody_model: NBodyDynamics):
        self.nbody = nbody_model
        
    def target_b_plane(self, state0: np.ndarray, epoch: float, 
                       target_br: float, target_bt: float, 
                       t_encounter: float = None, dt_max: float = 30*86400,
                       tol: float = 1.0, flyby_body: str = None) -> tuple[np.ndarray, bool]:
        """
        Differential correction to achieve target B-plane parameters.
        Adjusts initial velocity.
        
        Args:
            state0 (np.ndarray): Initial state [r, v] (6,).
            epoch (float): Initial epoch.
            target_br (float): Target B_R [km].
            target_bt (float): Target B_T [km].
            t_encounter (float): Estimated time of encounter (optional). 
            dt_max (float): Propagation duration [s].
            tol (float): Tolerance in B-plane distance [km].
            flyby_body (str): Name of the flyby body (optional). If None, tries to use index 1.
            
        Returns:
            tuple: (corrected_state0, success)
        """
        # Targeting Loop (Newton-Raphson)
        max_iter = 10
        current_state = state0.copy()
        
        # Determine flyby body
        if flyby_body:
            target_body = flyby_body
        elif len(self.nbody.bodies) >= 2:
            target_body = self.nbody.bodies[1]
        else:
            raise ValueError("NBody model must have at least 2 bodies or flyby_body must be specified.")
        
        mu_flyby = self.nbody.mus.get(target_body)
        if mu_flyby is None:
             mu_flyby = spice_manager.get_mu(target_body)

        central_body = self.nbody.central_body
        
        # Integration span
        t_end = epoch + dt_max
        if t_encounter:
            t_end = t_encounter
            
        for i in range(max_iter):
            # 1. Propagate with STM
            
            # Prepare state with STM
            phi0 = np.eye(6).flatten()
            y0 = np.concatenate((current_state, phi0))
            
            # Propagate
            sol = self.nbody.propagate(current_state, (epoch, t_end), stm=True, rtol=1e-8, atol=1e-10)
            
            final_full_state = sol.y[:, -1]
            rf = final_full_state[0:3]
            vf = final_full_state[3:6]
            phi_f = final_full_state[6:].reshape((6, 6))
            t_f = sol.t[-1]
            
            # Get Relative State to Flyby Body
            state_body = spice_manager.get_body_state(target_body, central_body, t_f, self.nbody.frame)
            r_body = state_body[0:3]
            v_body = state_body[3:6]
            
            r_rel = rf - r_body
            v_rel = vf - v_body
            
            # 2. Compute Current B-plane
            try:
                br, bt = state_to_bplane(r_rel, v_rel, mu_flyby)
            except ValueError:
                print("Trajectory not hyperbolic.")
                return current_state, False
                
            # Error
            b_err = np.array([br - target_br, bt - target_bt])
            if np.linalg.norm(b_err) < tol:
                return current_state, True
                
            # 3. Compute Jacobian d(B)/d(V0)
            
            feature_Jacobian = np.zeros((2, 6))
            eps_fd = 1e-4 # km or km/s
            
            # Base B
            b_nom = np.array([br, bt])
            
            # Perturb each component of relative state
            for k in range(3): # Position
                r_p = r_rel.copy(); r_p[k] += eps_fd
                bru, btu = state_to_bplane(r_p, v_rel, mu_flyby)
                feature_Jacobian[:, k] = (np.array([bru, btu]) - b_nom) / eps_fd
                
            for k in range(3): # Velocity
                v_p = v_rel.copy(); v_p[k] += eps_fd * 1e-3 # smaller step for V
                bru, btu = state_to_bplane(r_rel, v_p, mu_flyby)
                feature_Jacobian[:, 3+k] = (np.array([bru, btu]) - b_nom) / (eps_fd * 1e-3)
            
            # J_tot = J_feat(2x6) * Phi(6x6) * [0; I](6x3)
            # Submatrix of Phi corresponding to d(Sf)/d(V0) is Phi[:, 3:6]
            phi_v = phi_f[:, 3:6]
            
            J = feature_Jacobian @ phi_v
            
            # 4. Update V0
            # delta_v = - pinv(J) * error
            # Check condition
            cond = np.linalg.cond(J)
            if cond > 1e12:
                print(f"Singular Jacobian. Cond={cond}")
                return current_state, False
                
            jt_inv = np.linalg.pinv(J)
            delta_v0 = -jt_inv @ b_err
            
            # Debug shapes if error expected
            # print(f"DEBUG: J={J.shape}, J_inv={jt_inv.shape}, b_err={b_err.shape}, delta_v0={delta_v0.shape}")
            
            # Limit step size
            dv_mag = np.linalg.norm(delta_v0)
            if dv_mag > 0.5: # Limit to 500 m/s per step to avoid non-linearity explosions
                delta_v0 = delta_v0 * (0.5 / dv_mag)
                
            current_state[3:6] += delta_v0.flatten()
            
        return current_state, False
            
        return current_state, False
