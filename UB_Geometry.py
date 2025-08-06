import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
Please do NOT read or write any file, or show any images in your final submission! 
'''

# =============================================================================
# TASK 1: ROTATION MATRICES
# =============================================================================

def _create_rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Create rotation matrix around Z-axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def _create_rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Create rotation matrix around X-axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Compute rotation matrix from xyz coordinate system to XYZ coordinate system.
    
    Transformation sequence: 
    1. Rotate around z-axis by alpha → x'y'z'
    2. Rotate around x'-axis by beta → x''y''z''  
    3. Rotate around z''-axis by gamma → XYZ
    
    Args:
        alpha: Rotation angle around z-axis (degrees)
        beta: Rotation angle around x'-axis (degrees) 
        gamma: Rotation angle around z''-axis (degrees)
    Returns:
        3x3 rotation matrix from xyz to XYZ
    '''
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    R_z_alpha = _create_rotation_matrix_z(alpha_rad)
    R_x_beta = _create_rotation_matrix_x(beta_rad)
    R_z_gamma = _create_rotation_matrix_z(gamma_rad)
    
    rot_xyz2XYZ = R_z_gamma @ R_x_beta @ R_z_alpha
    
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Compute rotation matrix from XYZ coordinate system to xyz coordinate system.
    This is the inverse of findRot_xyz2XYZ().
    
    Args:
        alpha: Rotation angle around z-axis (degrees)
        beta: Rotation angle around x'-axis (degrees)
        gamma: Rotation angle around z''-axis (degrees) 
    Returns:
        3x3 rotation matrix from XYZ to xyz (transpose of xyz2XYZ)
    '''
    rot_xyz2XYZ = findRot_xyz2XYZ(alpha, beta, gamma)
    return rot_xyz2XYZ.T






# =============================================================================
# TASK 2: CAMERA CALIBRATION  
# =============================================================================

def _solve_homogeneous_system(A: np.ndarray) -> np.ndarray:
    """
    Solve homogeneous system Ax = 0 using SVD.
    Returns the solution vector (last column of V).
    """
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1]

def _build_projection_matrix_system(img_coord: np.ndarray, world_coord: np.ndarray) -> np.ndarray:
    """
    Build the system matrix A for solving projection matrix via Ax = 0.
    Each point contributes 2 equations to the system.
    """
    n_points = len(img_coord)
    A = np.zeros((2 * n_points, 12))
    
    for i in range(n_points):
        X, Y, Z = world_coord[i]
        x, y = img_coord[i]
        
        # First equation: x constraint
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
        # Second equation: y constraint  
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]
    
    return A

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Detect 32 checkerboard corner coordinates in the image.
    
    Args: 
        image: Input BGR image (MxNx3)
    Returns:
        Array of shape (32,2) containing pixel coordinates of corners
        Coordinate system: top-left = (0,0), bottom-right = (N,M)
    '''
    corners_size = (9, 4)  # 9x4 grid of corners
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(grayscale_image, corners_size)
    if not ret:
        raise RuntimeError("Could not detect chessboard corners")
    
    # Remove 4 corners to get 32 points (originally detects 36)
    indices_to_remove = [4, 13, 22, 31]  # Specific corner indices to exclude
    corners = np.delete(corners, indices_to_remove, axis=0)
    
    return corners.reshape(32, 2)


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    Define 3D world coordinates for the 32 checkerboard corners.
    The coordinates correspond to the same ordering as img_coord.
    
    Args: 
        img_coord: Image coordinates (not used, but kept for interface consistency)
    Returns:
        Array of shape (32,3) with world coordinates in millimeters
        Grid spacing: 10mm, arranged in 4 layers (z=10,20,30,40mm)
    '''
    # Define world coordinates in millimeters
    # Grid pattern: 4 rows x 8 cols x 4 layers (z-heights)
    world_coord = np.array([
        # Layer 1 (z=10mm)
        [40,0,10], [30,0,10], [20,0,10], [10,0,10], [0,10,10], [0,20,10], [0,30,10], [0,40,10],
        # Layer 2 (z=20mm) 
        [40,0,20], [30,0,20], [20,0,20], [10,0,20], [0,10,20], [0,20,20], [0,30,20], [0,40,20],
        # Layer 3 (z=30mm)
        [40,0,30], [30,0,30], [20,0,30], [10,0,30], [0,10,30], [0,20,30], [0,30,30], [0,40,30],
        # Layer 4 (z=40mm)
        [40,0,40], [30,0,40], [20,0,40], [10,0,40], [0,10,40], [0,20,40], [0,30,40], [0,40,40]
    ], dtype=float)
    
    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Calculate camera intrinsic parameters from corresponding 2D-3D point pairs.
    
    Uses Direct Linear Transform (DLT) method to solve for projection matrix M,
    then extracts intrinsic parameters using the relationships:
    - cx = m1 · m3, cy = m2 · m3
    - fx = sqrt(m1·m1 - cx²), fy = sqrt(m2·m2 - cy²)
    
    Args: 
        img_coord: Image coordinates (32x2)
        world_coord: World coordinates (32x3)
    Returns:
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates in pixels
    '''
    # Build and solve the DLT system Ax = 0
    A = _build_projection_matrix_system(img_coord, world_coord)
    solution = _solve_homogeneous_system(A)
    
    # Reshape solution to 3x4 projection matrix M
    M = solution.reshape(3, 4)
    
    # Extract the first 3 columns (rotation part) 
    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    
    # Calculate intrinsic parameters using derived formulas
    cx = float(m1 @ m3)
    cy = float(m2 @ m3) 
    fx = float(np.sqrt(m1 @ m1 - cx**2))
    fy = float(np.sqrt(m2 @ m2 - cy**2))

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate camera extrinsic parameters (rotation R and translation T).
    
    Method:
    1. Solve for projection matrix M using DLT
    2. Extract intrinsic parameters K from M  
    3. Compute extrinsic matrix [R|T] = K⁻¹ * M
    
    Args: 
        img_coord: Image coordinates (32x2)
        world_coord: World coordinates (32x3)
    Returns:
        R: Rotation matrix (3x3) from world to camera coordinates
        T: Translation vector (3,) from world to camera coordinates
    '''
    # Solve for projection matrix using same method as intrinsic calculation
    A = _build_projection_matrix_system(img_coord, world_coord)
    solution = _solve_homogeneous_system(A)
    M = solution.reshape(3, 4)
    
    # Extract intrinsic parameters to build K matrix
    m1, m2, m3 = M[0, :3], M[1, :3], M[2, :3]
    cx = m1 @ m3
    cy = m2 @ m3
    fx = np.sqrt(m1 @ m1 - cx**2)
    fy = np.sqrt(m2 @ m2 - cy**2)
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy], 
                  [0, 0, 1]])
    
    # Extract extrinsic parameters: [R|T] = K⁻¹ * M
    extrinsic_matrix = np.linalg.inv(K) @ M
    
    R = extrinsic_matrix[:, :3] 
    T = extrinsic_matrix[:, 3]
    
    return R, T

# =============================================================================
# END OF IMPLEMENTATION
# =============================================================================







#---------------------------------------------------------------------------------------------------------------------