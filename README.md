# 3D Geometry and Camera Calibration

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


## Code Structure

- `UB_Geometry.py`: Main implementation (reorganized for clarity)
- `task1.py`: Task 1 runner and validation
- `task2.py`: Task 2 runner and validation  
- `helper.py`: Utility functions for validation and submission
- `validation.py`: Visual validation of camera calibration results

## Disclaimer

**This project is developed solely for educational purposes as part of CSE 473/573 Computer Vision coursework.**

- **Academic Use Only**: This implementation is intended for learning fundamental computer vision concepts including 3D transformations, camera calibration, and geometric algorithms
- **Educational Value**: If you're interested in understanding classical computer vision techniques, camera geometry, and calibration methods, you can learn through this structured implementation
- **No Responsibility for Misuse**: I am not responsible for any misuse of this code, algorithms, or computational methods outside of educational contexts
- **Portfolio & Academic Demonstration**: This project is designed for academic coursework, portfolio demonstration, and educational reference purposes only
- **Proper Attribution**: Please respect academic integrity guidelines and provide appropriate citations when referencing this work

This implementation demonstrates core computer vision principles and should be used as a learning resource for understanding geometric transformations and camera calibration fundamentals.