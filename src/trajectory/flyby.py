import numpy as np
from src.spice.manager import spice_manager
from src.dynamics.nbody import NBodyDynamics
from src.trajectory.maneuver import ImpulsiveManeuver


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
    # S = - (e_vec + e_vec / e) ? No.
    # For a hyperbola, the asymptotes are at angle beta such that cos(beta) = -1/e
    # The incoming asymptote is "along" the direction -S (S usually points away).
    # Let's use the standard definition where S is the INCOMING velocity direction (V_inf_in / |V_inf_in|).
    # Wait, usually S is defined along V_inf_incoming.
    # From geometric relations:
    # S = (1/e) * (e_vec + (sqrt(e^2-1)/h_mag) * (h_vec x e_vec) ) ??
    # Careful with direction (incoming vs outgoing).
    # The incoming asymptote points TOWARDS the planet if we follow time.
    # But S is usually the direction OF the velocity, so S points towards planet?
    # No, velocity vector at infinity (incoming) is V_inf.
    # So S = V_inf / |V_inf|.
    
    # Formula for S in terms of e and h constants (assuming we are on the incoming branch):
    # This is ambiguous if we don't know if we are pre-periapsis or post-periapsis.
    # Assuming pre-periapsis (approaching):
    # We can determine true anomaly nu.
    # cos(nu) = dot(e, r) / (e*r).
    # If dot(r, v) < 0, we are inbound.
    
    term1 = e_vec / e
    term2 = np.cross(h_vec, e_vec) / (e * h_mag)
    sqrt_e2_1 = np.sqrt(e**2 - 1)
    
    # Incoming asymptote direction (unit vector)
    # S = (e_vec + sqrt(e^2-1) * (h x e_vec)/h ) / e  <-- Check sign
    # Actually, let's use the definition:
    # cos(delta/2) = 1/e.
    # S is rotated from e_vec by -(pi - arccos(1/e))?
    
    # Let's rely on the fact that S is the limit of v/|v| as t -> -inf.
    # But we don't want to propagate.
    # Analytic formula: 
    # S = ( e_vec + sqrt(e^2-1) * cross(h_hat, e_vec) ) / e ? (Plus or minus)
    # For INCOMING (Hyperbolic Mean Anomaly M -> -inf),
    # The velocity direction approaches asymptote.
    
    # Let's try:
    # S = (e_vec + sqrt(e^2-1) * np.cross(h_vec/h_mag, e_vec)) / e  (This might be one asymptote)
    # The other is with minus sign.
    # The incoming asymptote has velocity dot(r, v) < 0 (always?).
    # Actually, S is parallel to V_inf.
    
    # Let's assume standard "S" definition as Velocity Direction.
    # If we use strict formula:
    beta = np.arccos(1/e) # Half turn angle
    # The angle between e_vec and asymptote is (pi - beta).
    # So we rotate e_vec by +(pi-beta) or -(pi-beta).
    # In the orbital plane (defined by h).
    # Rotation axis h.
    
    # To determine sign?
    # We effectively want the vector that we are "coming from".
    # Actually, we want S to be the V_inf vector direction.
    
    # Let's use simpler approach:
    # V_inf vector can be computed from state!
    # v_inf_vec = (v - v_c) - ... no.
    # Use Lagrange coefficients or just geometric limit?
    
    # F and G series limit? No.
    # Used Vallado/Bate Mueller White formulas.
    # S = (e + sqrt(e^2-1) * R_cross) / e ??
    
    # Let's try a robust way:
    # Calculate the turning angle from current position? No.
    
    # Use e_vec and h_vec.
    # In perifocal frame (P along e, Q along h x e, W along h):
    # Incoming asymptote angle is -(pi - arccos(1/e)) ?
    # v_inf direction in perifocal:
    # theta_inf = arccos(-1/e). (This is true anomaly of asymptote).
    # The INCOMING asymptote is at nu = -theta_inf.
    # So S direction is direction of velocity at nu = -theta_inf.
    # Velocity direction psi wrt local horizontal?
    # Flight path angle gamma.
    
    # Velocity vector in Perifocal at nu = -theta_inf:
    # v = mu/h * [-sin(nu), e+cos(nu), 0]
    # nu = -theta_inf.
    # sin(-theta_inf) = -sin(theta_inf).
    # cos(-theta_inf) = -1/e.
    # v ~ [sin(theta_inf), e - 1/e, 0] ?
    # v ~ [sqrt(1 - 1/e^2), e - 1/e, 0] = [sqrt(e^2-1)/e, (e^2-1)/e, 0].
    # Unit vector:
    # Normalize.
    # S_perifocal = ...
    
    # Let's just implement the vector rotation.
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
    # B = (r x v) x S / v_inf ?? No.
    # "b = h / v_inf" is the magnitude.
    # Direction?
    # B points perpendicular to S.
    # And it lies in the orbital plane.
    # It is the "offset" vector.
    # B = (h x S) / v_inf_mag is checking units:
    # (L^2/T * 1) / (L/T) = L. Correct units.
    # Direction: h is perp to plane. S is in plane. h x S is in plane, perp to S.
    # Is it the correct direction?
    # B vector points to point of closest approach?
    # If S is velocity, h x S points...
    # Top view (h out of page). S up. h x S is Left.
    # Planet at center. S is incoming (up).
    # If we pass on the Right, L is 'down' (into page).
    # S (up) x L (down) = Right. Correct.
    # So B = (S x h) / v_inf ??
    # Wait, h x S vs S x h.
    # h = r x v.
    # If r=(b, 0, 0), v=(0, v, 0). h = (0, 0, bv).
    # S = (0, 1, 0).
    # h x S = (-bv, 0, 0) x (0, 1, 0) = (0, 0, -bv). Wrong.
    # (0, 0, bv) x (0, 1, 0) = (-bv, 0, 0) = -b * i.
    # We want +b * i.
    # So S x h.
    # (0, 1, 0) x (0, 0, bv) = (bv, 0, 0). Correct.
    
    B_vec = np.cross(S, h_vec) / v_inf_mag
    
    # Project onto R, T
    # B_R = dot(B, R)
    # B_T = dot(B, T)
    
    # Check definition of B_theta.
    # Usually B_R is component along R, B_T along T.
    
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

class FlybyCorrector:
    """
    Tools for correcting flyby trajectories to target specific parameters.
    """
    def __init__(self, nbody_model: NBodyDynamics):
        self.nbody = nbody_model
        
    def target_b_plane(self, state0: np.ndarray, epoch: float, 
                       target_br: float, target_bt: float, 
                       t_encounter: float = None, dt_max: float = 30*86400,
                       tol: float = 1.0) -> tuple[np.ndarray, bool]:
        """
        Differential correction to achieve target B-plane parameters.
        Adjusts INITIAL VELOCITY.
        
        Args:
            state0 (np.ndarray): Initial state [r, v] (6,).
            epoch (float): Initial epoch.
            target_br (float): Target B_R [km].
            target_bt (float): Target B_T [km].
            t_encounter (float): Estimated time of encounter (optional). 
                                 If None, propagates for dt_max or until periapsis?
                                 Better to specify a fixed time horizon or target plane crossing?
                                 Let's assume we propagate to a fixed time where we are "close enough" 
                                 to evaluate B-plane stability (asymptotes).
                                 Or just propagate to SOI exit?
                                 Actually, B-plane is defined by asymptotes, so any point in SOI works 
                                 if we use the hyperbolic analytic map.
            dt_max (float): Propagation duration [s].
            tol (float): Tolerance in B-plane distance [km].
            
        Returns:
            tuple: (corrected_state0, success)
        """
        # Targeting Loop (Newton-Raphson)
        max_iter = 10
        current_state = state0.copy()
        
        # Central Body MU (for B-plane calc)
        # Assuming nbody.bodies[1] is the flyby body?
        # We need the flyby body name.
        # Let's extract from NBody or ask user?
        # Assume 2nd body in NBody list is the target.
        if len(self.nbody.bodies) < 2:
            raise ValueError("NBody model must have at least 2 bodies (Central + Flyby Target).")
        
        flyby_body = self.nbody.bodies[1] 
        mu_flyby = self.nbody.mus[flyby_body]
        central_body = self.nbody.central_body
        
        # Integration span
        t_end = epoch + dt_max
        if t_encounter:
            t_end = t_encounter
            
        for i in range(max_iter):
            # 1. Propagate with STM
            # We need d(State_f)/d(State_0)
            
            # Prepare state with STM
            phi0 = np.eye(6).flatten()
            y0 = np.concatenate((current_state, phi0))
            
            # Propagate
            # We need to detect periapsis or just go to t_end?
            # If t_end is far enough, the state is hyperbolic.
            sol = self.nbody.propagate(current_state, (epoch, t_end), stm=True, rtol=1e-8, atol=1e-10)
            
            final_full_state = sol.y[:, -1]
            rf = final_full_state[0:3]
            vf = final_full_state[3:6]
            phi_f = final_full_state[6:].reshape((6, 6))
            t_f = sol.t[-1]
            
            # Get Relative State to Flyby Body
            state_body = spice_manager.get_body_state(flyby_body, central_body, t_f, self.nbody.frame)
            r_body = state_body[0:3]
            v_body = state_body[3:6]
            
            r_rel = rf - r_body
            v_rel = vf - v_body
            
            # 2. Compute Current B-plane
            try:
                br, bt = state_to_bplane(r_rel, v_rel, mu_flyby)
            except ValueError:
                # Not hyperbolic?
                print("Trajectory not hyperbolic.")
                return current_state, False
                
            # Error
            b_err = np.array([br - target_br, bt - target_bt])
            if np.linalg.norm(b_err) < tol:
                return current_state, True
                
            # 3. Compute Jacobian d(B)/d(V0)
            # J = d(B)/d(Sf_rel) * d(Sf_rel)/d(Sf_inertial) * d(Sf_inertial)/d(S0_inertial) * d(S0)/d(V0)
            
            # d(Sf_rel)/d(Sf_inertial) = Identity (since Planet state is fixed at t_f)
            # d(Sf_inertial)/d(S0_inertial) = Phi (6x6)
            # d(S0)/d(V0) = [0; I_3x3] (6x3)
            
            # We need d(B)/d(Sf_rel) (2x6)
            # Use Finite Difference for this part (Analytic B-plane partials are distinctively painful)
            
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
            
            # Total Jacobian d(B)/d(V0) (2x3)
            # J_tot = J_feat(2x6) * Phi(6x6) * [0; I](6x3)
            
            # Submatrix of Phi corresponding to d(Sf)/d(V0) is Phi[:, 3:6]
            phi_v = phi_f[:, 3:6] # (6x3)
            
            J = feature_Jacobian @ phi_v # (2x6) * (6x3) = (2x3)
            
            # 4. Update V0
            # delta_v = - pinv(J) * error
            # Check condition
            if np.linalg.cond(J) > 1e12:
                print("Singular Jacobian.")
                return current_state, False
                
            delta_v0 = -np.linalg.pinv(J) @ b_err
            
            # Limit step size
            dv_mag = np.linalg.norm(delta_v0)
            if dv_mag > 0.5: # Limit to 500 m/s per step to avoid non-linearity explosions
                delta_v0 = delta_v0 * (0.5 / dv_mag)
                
            current_state[3:6] += delta_v0
            
        return current_state, False
