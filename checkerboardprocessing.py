import os
import cv2
import glob
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# hi
# Extract and Process Frames
def extract_and_process_frames(video_file, checkerboard_size, output_folder):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Failed to open video file: {video_file}")
        return
    
    camera_id = os.path.basename(video_file).split('_')[-1].split('.')[0]  # Extracting 'x' from 'checkerboard_static_x.mp4'
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_filename = f"{output_folder}/camera_{camera_id}_frame_{frame_count}.png"
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # Rename file based on success or failure
        if ret:
            cv2.imwrite(frame_filename.replace(".png", "_success.png"), frame)
        else:
            cv2.imwrite(frame_filename.replace(".png", "_fail.png"), frame)
        
        frame_count += 1
    
    cap.release()

# Parameters
checkerboard_size = (7, 4)  # (rows, cols) - Number of inner corners per a chessboard row and column.
output_folder = "/output_folder" # Your path

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each video file
video_files = glob.glob("/checkerboard_static_*.mp4") # Your path

threads = []
for video_file in video_files:
    thread = threading.Thread(target=extract_and_process_frames, args=(video_file, checkerboard_size, output_folder))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("Frame extraction and processing complete.")

# Initialize variables for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((4 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:4].T.reshape(-1, 2) * 100  # 100mm squares

objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

# Initialize video writer
video_path = '/calibrationtest2.mp4' # Your path
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_shape = None  # To store the shape of the last valid frame

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('calibration_output.mp4', fourcc, 30.0, (frame_width, frame_height))

# Read the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_shape = frame.shape  # Store the shape of the last valid frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 4), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw corners and write the frame
        cv2.drawChessboardCorners(frame, (7, 4), corners2, ret)
        out.write(frame)

# Camera calibration
if len(objpoints) > 0 and len(imgpoints) > 0 and frame_shape is not None:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape[::-1][0:2], None, None)
    print("Calibration successful")

    # Calculate re-projection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print("Re-projection error: ", total_error / len(objpoints))

    # Save to CSV with descriptors
    df_mtx = pd.DataFrame({'Element': ['Row1Col1', 'Row1Col2', 'Row1Col3', 'Row2Col1', 'Row2Col2', 'Row2Col3', 'Row3Col1', 'Row3Col2', 'Row3Col3'],
                           'Camera Matrix': mtx.flatten()})
    df_dist = pd.DataFrame({'Element': ['k1', 'k2', 'p1', 'p2', 'k3'],
                            'Distortion Coefficients': dist.flatten()})
    df_mtx.to_csv('camera_matrix.csv', index=False)
    df_dist.to_csv('distortion_coefficients.csv', index=False)

    # Save to XML with descriptors
    fs = cv2.FileStorage('calibration_parameters.xml', cv2.FileStorage_WRITE)
    fs.writeComment('Camera Matrix and Distortion Coefficients for Camera Calibration')
    fs.write("CameraMatrix", mtx)
    fs.writeComment('Camera Matrix (Intrinsic Parameters)')
    fs.write("DistortionCoefficients", dist)
    fs.writeComment('Distortion Coefficients (k1, k2, p1, p2, k3)')
    fs.release()
else:
    print("No valid frames for calibration")

# Release the video objects
out.release()
cap.release()