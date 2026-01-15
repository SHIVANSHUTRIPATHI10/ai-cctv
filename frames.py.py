import cv2
import os

# Path to video
video_path = "avenue.mp4"

# Folder to save frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Read video
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Save frame as image
    frame_path = f"{output_folder}/frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)

    frame_count += 1

cap.release()
