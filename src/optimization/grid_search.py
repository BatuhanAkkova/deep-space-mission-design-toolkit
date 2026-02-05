import numpy as np

def grid_search_velocity_plane(objective_func, r0, v0_nom, 
                             mag_diff_pct=0.01, angle_diff_rad=0.08, 
                             num_points=10, verbose=False):
    """
    Performs a 2D grid search on velocity magnitude and in-plane flight path angle.
    
    This function explores a grid of possible initial velocities by varying the 
    magnitude and the direction within the orbital plane defined by (r0, v0_nom).
    
    Args:
        objective_func (callable): Function that takes velocity vector (np.array 3,) 
                                   and returns a scalar cost.
        r0 (np.array): Position vector (3,) relative to central body. 
                       Used to define the orbital plane (h = r x v).
        v0_nom (np.array): Nominal initial velocity vector (3,).
        mag_diff_pct (float): Search range for magnitude as a fraction of nominal.
                              e.g. 0.01 means +/- 1% range.
        angle_diff_rad (float): Search range for in-plane angle in radians.
                                e.g. 0.08 is approx +/- 4.5 degrees.
        num_points (int): Number of points along each axis (N x N grid).
        verbose (bool): If True, prints progress updates.
        
    Returns:
        dict: A dictionary containing the results:
            - 'best_v': (np.array 3,) The velocity vector corresponding to the lowest cost.
            - 'best_cost': (float) The minimum cost found.
            - 'grid_mag': (np.ndarray NxN) Meshgrid of velocity magnitudes tested.
            - 'grid_ang': (np.ndarray NxN) Meshgrid of angle deviations tested (radians).
            - 'grid_cost': (np.ndarray NxN) Meshgrid of computed costs.
            - 'nom_v': (np.array 3,) The input nominal velocity.
    """
    
    # Vectors must be floats
    r0 = np.array(r0, dtype=float)
    v_vec = np.array(v0_nom, dtype=float)
    
    v_mag_nom = np.linalg.norm(v_vec)
    v_dir = v_vec / v_mag_nom
    
    # Construct Local Basis Frame
    # h = r x v defines the orbital plane normal
    h_local = np.cross(r0, v_vec)
    h_norm = np.linalg.norm(h_local)
    if h_norm < 1e-9:
        # Singularity: Rectilinear trajectory (r parallel to v)
        # Define arbitrary normal
        if verbose:
             print("Warning: Rectilinear trajectory detected. Using arbitrary normal.")
        perp_arbitrary = np.array([0, 0, 1]) if np.abs(v_dir[2]) < 0.9 else np.array([0, 1, 0])
        h_local = np.cross(v_dir, perp_arbitrary)
        h_local /= np.linalg.norm(h_local)
    else:
        h_local /= h_norm
        
    # 'perp' vector in the plane, perpendicular to v_dir
    # This allows rotation of v_dir within the plane
    perp = np.cross(h_local, v_dir)
    
    # Grid Definition
    # Magnitude: +/- mag_diff_pct
    mags = np.linspace(v_mag_nom * (1.0 - mag_diff_pct), 
                       v_mag_nom * (1.0 + mag_diff_pct), 
                       num_points)
                       
    # Angle: +/- angle_diff_rad
    angles = np.linspace(-angle_diff_rad, angle_diff_rad, num_points)
    
    X, Y = np.meshgrid(mags, angles) # X=Mags, Y=Angles
    Z = np.zeros_like(X)
    
    best_cost = np.inf
    best_u = v0_nom.copy()
    
    total_checks = num_points * num_points
    if verbose:
        print(f"Grid Search: Checking {total_checks} points...")
        
    # Iterate
    for i in range(num_points): # Angle index (rows)
        for j in range(num_points): # Mag index (cols)
            m = mags[j]
            ang = angles[i]
            
            # Rotate v_dir by 'ang' around h_local
            # Rodriques' rotation formula simplified for planar:
            # v_new = v_dir * cos(ang) + (h x v_dir) * sin(ang)
            # h x v_dir is exactly our 'perp' vector
            v_rot = v_dir * np.cos(ang) + perp * np.sin(ang)
            
            u_test = v_rot * m
            
            cost = objective_func(u_test)
            Z[i, j] = cost
            
            if cost < best_cost:
                best_cost = cost
                best_u = u_test.copy()
                if verbose:
                     print(f"  New Best: Cost={cost:.4e} | Mag={m:.4f}, Ang={np.degrees(ang):.2f} deg")

    if verbose:
        print(f"Grid Search Complete. Min Cost: {best_cost:.4e}")
        
    return {
        'best_v': best_u,
        'best_cost': best_cost,
        'grid_mag': X,
        'grid_ang': Y, # Radians
        'grid_cost': Z,
        'nom_v': v0_nom
    }

def grid_search_poincare(param1_vals, param2_vals, eval_func, verbose=False):
    """
    Performs a generic 2D grid search over two parameters.
    Returns a Poincare map structure.
    
    Args:
        param1_vals (np.array): 1D array of values for the first parameter (X-axis).
        param2_vals (np.array): 1D array of values for the second parameter (Y-axis).
        eval_func (callable): Function that takes (p1, p2) and returns a scalar cost/metric.
        verbose (bool): If True, prints progress.
        
    Returns:
        dict:
            - 'grid_p1': Meshgrid of Param 1
            - 'grid_p2': Meshgrid of Param 2
            - 'grid_z': Meshgrid of evaluation results (The Map)
            - 'best_p1': Param 1 value for minimum Z
            - 'best_p2': Param 2 value for minimum Z
            - 'best_z': Minimum Z value
    """
    X, Y = np.meshgrid(param1_vals, param2_vals)
    Z = np.zeros_like(X)
    
    rows, cols = X.shape
    total = rows * cols
    
    if verbose:
        print(f"Poincare Grid Search: Evaluating {total} points...")
        
    best_z = np.inf
    best_p1 = None
    best_p2 = None
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            p1 = X[i, j]
            p2 = Y[i, j]
            
            val = eval_func(p1, p2)
            Z[i, j] = val
            
            if val < best_z:
                best_z = val
                best_p1 = p1
                best_p2 = p2
                if verbose:
                     print(f"  New Best: Z={val:.4e} at ({p1:.4f}, {p2:.4f})")
            
            count += 1
            if verbose and count % 10 == 0:
                print(f"  Progress: {count}/{total}", end='\r')
                
    if verbose:
        print(f"\nPoincare Search Complete. Min Z: {best_z:.4e}")
        
    return {
        'grid_p1': X,
        'grid_p2': Y,
        'grid_z': Z,
        'best_p1': best_p1,
        'best_p2': best_p2,
        'best_z': best_z
    }
