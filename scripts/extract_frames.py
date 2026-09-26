import cv2
import os
from pathlib import Path

# Automatically resolves the project root (vision-target-tracker/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video is located inside the 'assets' folder
video_path = str(PROJECT_ROOT / "assets" / "desk_video2.mov")

# Output directory: places extracted frames inside data/raw inside the project
output_dir = str(PROJECT_ROOT / "data" / "raw")
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at: {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

# Sample 4 frames per second
interval = max(int(fps / 4), 1)

frame_idx = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        out_path = os.path.join(output_dir, f"frame_{saved_count:03d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_count += 1

    frame_idx += 1

cap.release()
print(f"Extracted {saved_count} frames to {output_dir}")