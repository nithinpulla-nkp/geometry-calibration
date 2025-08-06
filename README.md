# 3D Geometry and Camera Calibration
**CSE 473/573 Computer Vision Assignment 1 - Summer 2025**

This project implements fundamental 3D geometry transformations and camera calibration algorithms using computer vision techniques.

## Overview

### Task 1: 3D Rotation Matrices
- Implements Euler angle rotations (Z-X-Z sequence)
- Computes rotation matrices between coordinate systems xyz ↔ XYZ
- Uses modular helper functions for cleaner code organization

### Task 2: Camera Calibration  
- Detects checkerboard corners in calibration images
- Calculates intrinsic camera parameters (fx, fy, cx, cy)
- Computes extrinsic parameters (rotation R and translation T)
- Uses Direct Linear Transform (DLT) method with SVD

## Requirements
- Python 3.x
- OpenCV 4.5.4 (exactly this version)
- NumPy
- See `requirements.txt` for complete dependencies

## Usage

Execute all commands from the project root directory.

### Task 1: Rotation Matrix Testing
```bash
python task1.py --alpha 45 --beta 30 --gamma 50
```
- Tests 3D rotation transformations with specified Euler angles
- Outputs rotation matrices to `result_task1.json`

### Task 2: Camera Calibration
```bash
python task2.py
```
- Processes checkerboard calibration image (`checkboard.png`)
- Extracts camera intrinsic and extrinsic parameters
- Outputs results to `result_task2.json`

### Validation (Optional)
```bash
python validation.py
```
- Validates Task 2 results by reprojecting 3D points
- Shows visual comparison of original vs reprojected corners
- Computes reprojection error metrics

## Submission

Pack your submission (runs tests automatically):
```bash
sh pack_submission.sh <YourUBITName>
```

**Expected output:** `submission_<YourUBITName>.zip` containing:
- `UB_Geometry.py` (your implementation)
- `result_task1.json` (Task 1 results)  
- `result_task2.json` (Task 2 results)

Submit only the zip file.

## Code Structure

- `UB_Geometry.py`: Main implementation (reorganized for clarity)
- `task1.py`: Task 1 runner and validation
- `task2.py`: Task 2 runner and validation  
- `helper.py`: Utility functions for validation and submission
- `validation.py`: Visual validation of camera calibration results