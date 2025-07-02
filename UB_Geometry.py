import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
Please do NOT read or write any file, or show any images in your final submission! 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    rot_z_alpha = np.array([[np.cos(alpha_r), -np.sin(alpha_r), 0],
                            [np.sin(alpha_r), np.cos(alpha_r), 0],
                            [0, 0, 1]])
    
    rot_x_beta = np.array([[1, 0, 0],
                           [0, np.cos(beta_r), -np.sin(beta_r)],
                           [0, np.sin(beta_r), np.cos(beta_r)]])
    
    rot_z_gamma = np.array([[np.cos(gamma_r), -np.sin(gamma_r), 0],
                            [np.sin(gamma_r), np.cos(gamma_r), 0],
                            [0, 0, 1]])
    
    rot_xyz2XYZ = rot_z_alpha @ rot_xyz2XYZ
    rot_xyz2XYZ = rot_x_beta @ rot_xyz2XYZ
    rot_xyz2XYZ = rot_z_gamma @ rot_xyz2XYZ

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    rot_z_alpha = np.array([[np.cos(alpha_r), -np.sin(alpha_r), 0],
                            [np.sin(alpha_r), np.cos(alpha_r), 0],
                            [0, 0, 1]])
    
    rot_x_beta = np.array([[1, 0, 0],
                           [0, np.cos(beta_r), -np.sin(beta_r)],
                           [0, np.sin(beta_r), np.cos(beta_r)]])
    
    rot_z_gamma = np.array([[np.cos(gamma_r), -np.sin(gamma_r), 0],
                            [np.sin(gamma_r), np.cos(gamma_r), 0],
                            [0, 0, 1]])
    
    rot_XYZ2xyz = rot_z_gamma.T @ rot_XYZ2xyz
    rot_XYZ2xyz = rot_x_beta.T @ rot_XYZ2xyz
    rot_XYZ2xyz = rot_z_alpha.T @ rot_XYZ2xyz

    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    img_coord = np.zeros([36, 2], dtype=float)
    cornersSize = (9, 4)
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(grayscaleImage, cornersSize)
    indicesToRemove = [4, 13, 22, 31]
    corners = np.delete(corners, indicesToRemove, axis=0)
    img_coord = corners.reshape(32, 2)

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord[:] = [
    [40,0,10], [30,0,10], [20,0,10], [10,0,10], [0,10,10], [0,20,10], [0,30,10], [0,40,10],
    [40,0,20], [30,0,20], [20,0,20], [10,0,20], [0,10,20], [0,20,20], [0,30,20], [0,40,20],
    [40,0,30], [30,0,30], [20,0,30], [10,0,30], [0,10,30], [0,20,30], [0,30,30], [0,40,30],
    [40,0,40], [30,0,40], [20,0,40], [10,0,40], [0,10,40], [0,20,40], [0,30,40], [0,40,40]
    ]

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    # Your implementation

    A = []
    for i in range(32):
        X, Y, Z = world_coord[i]
        x, y = img_coord[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    last_row = Vt[-1]
    M = last_row.reshape(3, 4)
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]

    cx = m1 @ m3
    cy = m2 @ m3
    fx = np.sqrt(m1 @ m1 - cx**2)
    fy = np.sqrt(m2 @ m2 - cy**2)

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation

    A = []
    for i in range(32):
        X, Y, Z = world_coord[i]
        x, y = img_coord[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    last_row = Vt[-1]
    M = last_row.reshape(3, 4)
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]
    ox = m1 @ m3
    oy = m2 @ m3
    fx = np.sqrt(m1 @ m1 - ox**2)
    fy = np.sqrt(m2 @ m2 - oy**2)

    intrinsic_matrix = np.array([[fx, 0, ox],
                                [0, fy, oy],
                                [0, 0, 1]])
    extrinsic_matrix = np.linalg.inv(intrinsic_matrix) @ M
    
    R = extrinsic_matrix[:, :3]
    T = extrinsic_matrix[:, 3]
    
    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2







#---------------------------------------------------------------------------------------------------------------------