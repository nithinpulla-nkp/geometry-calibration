'''
Camera Calibration: 
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''
import sys
import cv2
import json
import numpy as np
from UB_Geometry import find_corner_img_coord, find_corner_world_coord, find_intrinsic, find_extrinsic
from helper import show_image, check_output_format


def save_result(
    img_coord: np.ndarray, 
    world_coord: np.ndarray,
    fx, fy, cx, cy: float,
    R, T: np.ndarray,
    save_path='result_task2.json',
):
    result = {}
    result['img_coord'] = img_coord.tolist()
    result['world_coord'] = world_coord.tolist()
    result['fx'] = fx
    result['fy'] = fy
    result['cx'] = cx
    result['cy'] = cy
    result['R'] = R.tolist()
    result['T'] = T.tolist()
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    if cv2.__version__ != '4.5.4':
        print("Please use OpenCV 4.5.4")
        sys.exit(1)
    img = cv2.imread('checkboard.png')
    img_coord = find_corner_img_coord(img)
    world_coord = find_corner_world_coord(img_coord)
    fx, fy, cx, cy = find_intrinsic(img_coord, world_coord)
    R, T = find_extrinsic(img_coord, world_coord)

    if not (
        check_output_format(img_coord, 'img_coord', np.ndarray, dimensions=(32,2)) and
        check_output_format(world_coord, 'world_coord', np.ndarray, dimensions=(32,3)) and
        check_output_format(fx, 'fx', float) and
        check_output_format(fy, 'fy', float) and
        check_output_format(cx, 'cx', float) and
        check_output_format(cy, 'cy', float) and
        check_output_format(R, 'R', np.ndarray, dimensions=(3,3)) and
        check_output_format(T, 'T', np.ndarray, dimensions=(3,))
    ):
        sys.exit(2)

    save_result(img_coord, world_coord, fx, fy, cx, cy, R, T)
    print('img_coord:')
    print(img_coord)
    print('world_coord:')
    print(world_coord)
    print('fx, fy, cx, cy')
    print(fx, fy, cx, cy)
    print('R')
    print(R)
    print('T')
    print(T)