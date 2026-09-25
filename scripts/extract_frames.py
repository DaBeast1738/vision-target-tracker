import cv2
import os

video_path = "desk_video.mov" 
output_dir = "C:\Users\mThamilchelvan\Downloads\images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Sample 2 to 3 frames per second of footage
interval = max(int(fps / 2), 1)

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