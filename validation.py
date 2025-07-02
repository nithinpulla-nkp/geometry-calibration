import numpy as np
import cv2
from typing import List, Tuple
import matplotlib.pyplot as plt
import json

def validate_projection(img_coord, world_coord, fx, fy, cx, cy, R, T):
    K = np.array([[fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])
    P = K @ np.hstack((R, T.reshape(-1, 1)))
    world_homog = np.hstack([world_coord, np.ones((world_coord.shape[0], 1))])
    projected = (P @ world_homog.T).T
    projected /= projected[:, 2, np.newaxis]
    projected_2d = projected[:, :2]
    error = np.linalg.norm(img_coord - projected_2d, axis=1)
    print(f"\n Mean Reprojection Error: {np.mean(error):.2f} pixels")
    return projected_2d

def show_projection(img, img_coord, projected_2d):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.scatter(img_coord[:, 0], img_coord[:, 1], c='lime', label='Original')
    plt.scatter(projected_2d[:, 0], projected_2d[:, 1], c='red', marker='+', label='Projected')
    plt.legend()
    plt.title('Original vs. Reprojected Points')
    plt.show()


with open("result_task2.json","r") as f:
    data = json.load(f)

img_coord   = np.array(data["img_coord"],   dtype=np.float32)  # (32,2)
world_coord = np.array(data["world_coord"], dtype=np.float32)  # (32,3)
fx, fy, cx, cy = (data["fx"], data["fy"],
                  data["cx"], data["cy"])
R = np.array(data["R"], dtype=np.float64)  # (3,3)
T = np.array(data["T"], dtype=np.float64)  # (3,)

# ————— run validation —————

img = cv2.imread("checkboard.png")
projected_2d = validate_projection(img_coord, world_coord, fx, fy, cx, cy, R, T)
show_projection(img, img_coord, projected_2d)