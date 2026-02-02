import numpy as np
from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics

class GravityAssistParamError(Exception):
    pass

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
                      Measured clockwise from T axis in B-plane? 
                      Standard definition: angle between B vector and T vector.
        rp (float): Desired periapsis radius [km].
        mu (float): Gravitational parameter [km^3/s^2].
        
    Returns:
        tuple: (v_inf_out, B_vector, S_vector)
            v_inf_out (np.ndarray): Outgoing V-infinity vector.
            B_vector (np.ndarray): B-vector (aiming point) in body-centered frame.
            S_vector (np.ndarray): S-vector (direction of incoming asymptote).
    """
    v_inf_mag = np.linalg.norm(v_inf_in)
    S = v_inf_in / v_inf_mag # S vector acts as the Z-axis of the B-plane frame
    
    # Define T and R axes for B-plane
    # T is typically in the ecliptic plane, perpendicular to S. 
    # Or defined by the intersection of the trajectory plane and reference plane.
    # Common convention: Let N be a reference pole (e.g., North Pole of Ecliptic: [0, 0, 1])
    # T = (S x N) / |S x N|
    # R = S x T
    
    # Assume Ecliptic J2000 Frame, so Pole is roughly +Z.
    # Check if S is parallel to Z (e.g. polar approach)
    pole = np.array([0.0, 0.0, 1.0])
    
    if np.abs(np.dot(S, pole)) > 0.999:
        # S is nearly parallel to Pole, choose arbitrary T
        T_temp = np.array([1.0, 0.0, 0.0])
        T = np.cross(S, T_temp)
    else:
        T = np.cross(S, pole)
        
    T = T / np.linalg.norm(T)
    R = np.cross(S, T) # Completes the triad S, T, R (right-handed? typically R, S, T order varies. B-plane is R-T)
    
    # B vector definition
    # B vector magnitude is b (aiming radius)
    # B = b * (cos(beta)*T + sin(beta)*R)  <-- Check convention? 
    # Usually BdotR vs BdotT. 
    # Let's assume beta is angle measured from T axis towards R axis.
    
    b = compute_aiming_radius(v_inf_mag, mu, rp)
    delta = compute_turn_angle(v_inf_mag, mu, rp)
    
    # The B vector points to the pierce point.
    # The trajectory bends *towards* the planet.
    # The B vector is in the plane perpendicular to incoming asymptote.
    B_vec = b * (np.cos(beta) * T + np.sin(beta) * R)
    
    # The turning plane is defined by S and B.
    # The rotation axis is perpendicular to the turning plane.
    # Rotation axis H_hat (direction of angular momentum) = (S x B) / |S x B| ??
    # Wait, B vector is "where we aim". The planet pulls us *in*.
    # So the turn happens in the plane containing S and B?
    # Yes. The hyperbola lies in the plane defined by S and the center of the planet.
    # Since B points from the center of the planet (projected) to the asymptote...
    # Actually B points from the center of the B-plane (planet center projection) to the asymptote pierce point.
    # The spacecraft is at B (in the far field). The planet is at Origin.
    # So the spacecraft passes "above" the planet in the direction of B.
    # The gravity pulls it "down" towards the origin.
    # So the velocity vector rotates *away* from B? No, towards the planet.
    # Which means V_out will be bent towards -B? 
    # Let's visualize: S is initial velocity direction.
    # We fly by at distance b in direction B (relative to planet center).
    # Gravity pulls us towards center.
    # So the velocity vector rotates in the plane formed by S and B.
    # The rotation axis is perpendicular to orbital plane.
    # h = r x v. Initial r is effectively B (at infinity). Initial v is S.
    # h_hat direction = (B x S) / |B x S|
    
    # Check cross product: S is into the page? No S is velocity.
    # B is perpendicular to S.
    # Orbital angular momentum h = r x v.
    # At simplified infinity: r ~ B, v ~ S * v_inf
    # h ~ B x S.
    
    h_vec = np.cross(B_vec, S)
    h_hat = h_vec / np.linalg.norm(h_vec)
    
    # We rotate S by angle delta around h_hat.
    v_inf_out = rotate_vector(v_inf_in, delta, h_hat)
    
    return v_inf_out, B_vec, S


class FlybyCalculator:
    """
    Calculates flyby parameters and verifies with N-Body propagation.
    """
    def __init__(self, body: str, frame: str = 'J2000'):
        self.body = body
        self.frame = frame
        self.mu = spice_manager.get_mu(body)
        self.radii = spice_manager.get_radii(body)
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
        
        Strategy:
        1. Calculate analytic B-vector and incoming asymptote.
        2. Back-propagate from 'infinity' (SOI edge) to periapsis? 
           Or simpler: Start at -dt_SOI with state derived from V_inf and B-plane.
        
        State at t = -dt:
           r(-dt) approx R_planet(-dt) + B_vec - V_inf_in * dt
           v(-dt) approx V_planet(-dt) + V_inf_in
           
           *Correction*: This linear approximation is poor for large dt due to gravity curving the path.
           Better approach:
           - Use hyperbolic orbit equations to find state (r, v) at true anomaly corresponding to -dt_SOI?
           - Or just start at periapsis and propagate backwards and forwards?
           
           Let's start at PERIAPSIS. It is the most constrained point.
           
        1. Determine Periapsis State (r_p, v_p) from geometry.
           - r_p magnitude is rp.
           - r_p direction? 
             - The eccentricity vector e points towards periapsis.
             - The turning happens in the plane.
             - The turn angle is delta.
             - The periapsis is halfway through the turn.
             - So r_p direction is rotated -delta/2 from V_inf_out? 
             - Or rotated +delta/2 from V_inf_in? No.
             
             Let's use the orbital frame (h_hat is normal).
             Check geometry:
             Incoming asymptote is at angle -theta_inf = -(pi - arccos(1/e)) from periapsis.
             So periapsis is at angle +theta_inf from incoming asymptote?
             Wait, theta_inf is the angle of asymptote limit.
             True anomaly nu goes from -theta_inf to +theta_inf.
             So incoming is at nu = -theta_inf.
             
             We have V_inf_in and V_inf_out.
             The periapsis vector bisects angle between V_inf_in and V_inf_out? 
             Actually, the turning angle is delta.
             The incoming vector is deflected by delta.
             The periapsis vector must be along the axis of symmetry.
             So angle(V_inf_in, r_p) = 90 + delta/2 ??
             
             Let's construct it:
             V_turn_plane_normal = h_hat (calculated in compute_outgoing)
             
             Let V_in_hat = V_inf_in / |V_inf_in|
             Let V_out_hat = V_inf_out / |V_inf_out|
             
             r_p_direction should be V_out_hat - V_in_hat (vector difference points towards focus? No)
             The change in velocity DeltaV = V_out - V_in points towards the center of attraction? (For circular, yes).
             For flyby, the force forms the turn.
             The force is always along -r.
             So sum of forces ...
             
             Correct geometry:
             The periapsis vector bisects the angle between the incoming and outgoing asymptotes?
             NO. The asymptotes make angle delta.
             The periapsis is the point of closest approach.
             The symmetry axis is the line of apsides.
             The asymptotes are symmetric wrt line of apsides.
             So yes, the line of apsides bisects the angle between (-V_inf_in) and (V_inf_out).
             (Note -V_inf_in is the direction the s/c came FROM, extended).
             
             So r_p_hat direction is bisector of (-V_inf_in) and (V_inf_out)? No.
             The SC bends AROUND the planet.
             V_in --------->
                              \
                               \  (Planet)
                               /
             V_out <----------/   
             
             Let's use the specific turn plane.
             Axis of symmetry (eccentricity vector) points to periapsis.
             Angle between e_vec and V_inf_in (velocity vector) at infinity is ...
               cos(phi) = -1/e
             
             Easier construction:
             We already have h_hat (normal).
             We have V_inf_in.
             We can rotate V_inf_in by (pi/2 + delta/2)??
             
             Let's rotate V_inf_in by an angle `psi` around `h_hat` to get `r_p_hat`.
             What is `psi`?
             At periapsis, velocity is purely perpendicular to radius. V_p is perpendicular to r_p.
             V_p magnitude = sqrt(mu/rp * (1+e)) ? Vis-viva: v^2 = mu(2/r - 1/a). a < 0 for hyp.
             v_inf^2 = -mu/a  => a = -mu/v_inf^2.
             v_p^2 = mu(2/rp + v_inf^2/mu) = 2mu/rp + v_inf^2. Correct.
             
             Direction of V_p:
             It is rotated by delta/2 from V_inf_in?
             Let's look at the turn. Total turn is delta.
             At periapsis, we have turned delta/2.
             So V_p direction is Rotate(V_inf_in, delta/2, h_hat).
             
             Direction of r_p:
             Since V_p is tangent at periapsis, and orbit is around center,
             r_p is perpendicular to V_p.
             And r_p points from Center to SC.
             Gravity pulls IN (-r).
             So V turns "inwards".
             r_p should be Rotate(V_p, -90deg, h_hat).
             
        2. Set state at t=t_periapsis + epoch?
           Let t_periapsis = epoch.
           r_vec_p = r_p_hat * rp
           v_vec_p = v_p_hat * vp
           
        3. Convert state to inertial frame.
           The calculated vectors are relative to the Planet Center.
           Need to add Planet State at `epoch` to get Heliocentric State.
           
        4. Propagate Forward and Backward.
        
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
        # (Since we are turning 'delta' total, half is at periapsis)
        # Verify rotation direction:
        # We rotate TOWARDS the planet. 
        # B_vec points 'above' the planet. Gravity pulls 'down'.
        # So we rotate around h_hat? 
        # h = r x v ~ B x V_in.
        # Rotate V_in by delta/2 around h_hat?
        # Yes, standard orbit mechanics.
        
        v_p_hat = rotate_vector(v_inf_in / v_inf_mag, delta / 2.0, h_hat)
        
        # Magnitude at periapsis
        vp = np.sqrt(v_inf_mag**2 + 2 * mu / rp)
        v_p_vec_rel = v_p_hat * vp
        
        # Position at periapsis
        # Perpendicular to velocity, in the plane.
        # r x v = h.
        # r must be perpendicular to v.
        # And r points from planet to S/C.
        # r = V x h_hat ??
        # Let's check: (V x h) direction. 
        # h = r x v. 
        # v x (r x v) = v x h
        # v x (r x v) = r(v.v) - v(r.v) = r*v^2 (since r.v=0 at periapsis)
        # So r direction is (v x h).
        # We have h_hat.
        # r_direction = cross(v_p_hat, h_hat) ??
        # Let's check: h = r x v.
        # If r = v x h => h = (v x h) x v = -v x (v x h) = - ( v(v.h) - h(v.v) ) = h(v.v).
        # So yes, r is parallel to v x h.
        
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
        # Include Sun and Planet at minimum.
        # Ideally include major other bodies?
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
        
        # Real V_inf checks
        # V_inf_out_meas = (V_sc_final - V_planet_final)
        # We need Planet state at final time
        t_final = sol_fwd.t[-1]
        state_p_final = spice_manager.get_body_state(self.body, 'SUN', t_final, self.frame)
        v_p_final = state_p_final[3:6]
        
        v_inf_out_meas = final_state[3:6] - v_p_final
        
        # Comparison with analytic
        # Rotate back to roughly compare? 
        # Just return magnitude and angle error
        
        return {
            "analytic_v_inf_out": v_inf_out,
            "measured_v_inf_out": v_inf_out_meas,
            "analytic_delta": np.degrees(delta),
            "measured_delta": np.degrees(np.arccos(np.dot(v_inf_in, v_inf_out_meas)/(np.linalg.norm(v_inf_in)*np.linalg.norm(v_inf_out_meas)))),
            "trajectory_fwd": sol_fwd,
            "trajectory_back": sol_back
        }
