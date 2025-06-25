'''
Helper and zip functions.
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''


import zipfile
import os
import argparse
import cv2
import numpy as np


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def check_output_format(output, output_name, tp, dimensions=(0,)):
    if not isinstance(output, tp):
        print('Wrong output type in %s! Should be %s, but you get %s.' % (output_name, tp, type(output)))
        return False
    else:
        if tp == np.ndarray:
            if not output.shape == dimensions:
                print('Wrong output dimensionsin %s! Should be %s, but you get %s.' % (output_name, dimensions, output.shape))
                return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project Geometry submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args

def check_submission(py_file, check_list=['cv2.imshow(', 'cv2.imwrite(', 'cv2.imread(', 'open(']):
    res = True
    with open(py_file, 'r') as f:
        lines = f.readlines()
    for nline, line in enumerate(lines):
        for string in check_list:
            if line.find(string) != -1:
                print('You submitted code (in line %d) cannot have %s (Even if it is commented). Please remove that and zip again.' % (nline + 1, string[:-1]))
                res = False
    return res

def files2zip(files: list, zip_file_name: str):
    res = True
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                if name == 'UB_Geometry.py':
                    if not check_submission(file):
                        print('Zipping error!')
                        res = False
                zf.write(file, arcname=name)
            else:
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)
                res = False
    return res

if __name__ == "__main__":
    args = parse_args()
    file_list = ['UB_Geometry.py', 'result_task1.json', 'result_task2.json']
    res = files2zip(file_list, 'submission_' + args.ubit + '.zip')
    if not res:
        print('Zipping failed.')
    else:
        print('Zipping succeed.')
