#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UB_Geometry.py

Library module defining exactly the functions called by task1.py and task2.py.
Do NOT add any I/O or top-level code here.
"""

import numpy as np
import cv2
from typing import Tuple

# -------------------------------------------------------------------
# Task 1: Euler rotation matrices
# -------------------------------------------------------------------

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Args:
        alpha, beta, gamma: rotation angles in degrees.
    Returns:
        3×3 matrix R = Rz(alpha) · Rx(beta) · Rz(gamma).
    """
    a = np.deg2rad(alpha)
    b = np.deg2rad(beta)
    g = np.deg2rad(gamma)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)

    Rz1 = np.array([[ ca, -sa, 0],
                    [ sa,  ca, 0],
                    [  0,   0, 1]], dtype=float)
    Rx1 = np.array([[1,   0,    0],
                    [0,  cb,  -sb],
                    [0,  sb,   cb]], dtype=float)
    Rz2 = np.array([[ cg, -sg, 0],
                    [ sg,  cg, 0],
                    [  0,   0, 1]], dtype=float)
    return Rz1 @ Rx1 @ Rz2

def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Args:
        alpha, beta, gamma: same angles as above.
    Returns:
        R^{-1} = (Rz(alpha)·Rx(beta)·Rz(gamma))^T.
    """
    R = findRot_xyz2XYZ(alpha, beta, gamma)
    return R.T

# -------------------------------------------------------------------
# Task 2: Single-image checkerboard calibration via Homography
# -------------------------------------------------------------------

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    """
    Detect the 8×4 inner corners in 'image'.
    Args:
        image: H×W×3 BGR image array.
    Returns:
        (32×2) float32 array of (u,v) pixel coords, row-major.
    Raises:
        RuntimeError if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (8, 4)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        raise RuntimeError("Checkerboard corners not found")
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return corners.reshape(-1, 2).astype(np.float32)

def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    """
    Given img_coord of shape (32,2), produce the matching
    world coords for an 8×4 grid on Z=0, 10 mm squares.
    Returns:
        (32×3) float32 array of (X,Y,0).
    """
    nx, ny, s = 8, 4, 10.0
    pts3d = np.zeros((nx*ny, 3), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            idx = j*nx + i
            pts3d[idx, 0] = i * s
            pts3d[idx, 1] = j * s
    return pts3d

def find_intrinsic(img_coord: np.ndarray,
                   world_coord: np.ndarray
                  ) -> Tuple[float, float, float, float]:
    """
    Homography-based intrinsic estimation:
      Solve H from 2D–3D (Z=0) points, then
      cx,cy = mean(u),mean(v),
      fx = avg of focal estimates from H columns.
    Returns:
        fx, fy, cx, cy
    """
    N = img_coord.shape[0]
    A = []
    for (u,v), (X,Y,_) in zip(img_coord, world_coord):
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    A = np.array(A, dtype=float)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    H = h.reshape(3,3)

    cx = float(np.mean(img_coord[:,0]))
    cy = float(np.mean(img_coord[:,1]))

    h1 = H[:,0]
    h2 = H[:,1]
    def comp_f(hc):
        dx = hc[0] - cx*hc[2]
        dy = hc[1] - cy*hc[2]
        hz = hc[2]
        val = (dx*dx + dy*dy) / max(1e-8, (1 - hz*hz))
        if val <= 0:
            raise RuntimeError("Negative focal square")
        return float(np.sqrt(val))

    f1 = comp_f(h1)
    f2 = comp_f(h2)
    f = (f1 + f2) / 2.0
    return f, f, cx, cy

def find_extrinsic(img_coord: np.ndarray,
                   world_coord: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose homography H into R,t given K from find_intrinsic().
    Returns:
      R (3×3), t (3,)
    """
    N = img_coord.shape[0]
    A = []
    for (u,v), (X,Y,_) in zip(img_coord, world_coord):
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    A = np.array(A, dtype=float)
    _, _, VT = np.linalg.svd(A)
    H = VT[-1].reshape(3,3)

    fx, fy, cx, cy = find_intrinsic(img_coord, world_coord)
    K = np.array([[fx, 0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=float)
    K_inv = np.linalg.inv(K)

    h1 = H[:,0]; h2 = H[:,1]; h3 = H[:,2]
    lam = 1.0 / np.linalg.norm(K_inv.dot(h1))
    r1 = lam * (K_inv.dot(h1))
    r2 = lam * (K_inv.dot(h2))
    r3 = np.cross(r1, r2)
    t  = lam * (K_inv.dot(h3))

    R = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(R)
    R = U.dot(Vt)

    return R, t
