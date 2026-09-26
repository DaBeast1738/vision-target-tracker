import os
from pathlib import Path
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
  raise ValueError(f"Missing ROBOFLOW_API_KEY. Checked path: {ENV_PATH}")

# Configure video paths
VIDEO_INPUT_PATH = str(PROJECT_ROOT / "assets" / "trial_video.mov")
OUTPUT_VIDEO_PATH = str(PROJECT_ROOT / "assets" / "output_tracked.mp4")

if not os.path.exists(VIDEO_INPUT_PATH):
  raise FileNotFoundError(f"Input video not found at: {VIDEO_INPUT_PATH}")

print(f"Loading input video from: {VIDEO_INPUT_PATH}")

# Initialize Standard HTTP Client 
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
)

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

print(f"Processing {total_frames} frames via HTTP workflow endpoint...")

frame_idx = 0
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  try:
    # Query workflow over HTTPS
    result = client.run_workflow(
        workspace_name="menakant",
        workflow_id=(
            "detect-simple-desk-objects-vdetect-simple-desk-objects-2-yolo11n-t1-logic"
        ),
        images={"image": frame},
    )

    # Parse predictions
    predictions = []
    if isinstance(result, list) and len(result) > 0:
      output_obj = result[0]
      if "predictions" in output_obj:
        preds = output_obj["predictions"]
        predictions = (
            preds.get("predictions", []) if isinstance(preds, dict) else preds
        )
      elif "output" in output_obj:
        predictions = output_obj["output"].get("predictions", [])

    for pred in predictions:
      cx = int(pred.get("x", 0))
      cy = int(pred.get("y", 0))
      w = int(pred.get("width", 0))
      h = int(pred.get("height", 0))
      label = pred.get("class", pred.get("class_name", "target"))
      conf = float(pred.get("confidence", 0.0))

      # Calculate bounding box coordinates
      x1 = max(0, int(cx - w / 2))
      y1 = max(0, int(cy - h / 2))
      x2 = min(width, int(cx + w / 2))
      y2 = min(height, int(cy + h / 2))

      # Draw tracking visual overlays
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
      cv2.putText(
          frame,
          f"{label} {conf:.2f} ({cx}, {cy})",
          (x1, max(20, y1 - 8)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

  except Exception as e:
    print(f"Frame {frame_idx} inference warning: {e}")

  out.write(frame)
  frame_idx += 1
  if frame_idx % 10 == 0 or frame_idx == total_frames:
    print(f"Processed frame {frame_idx}/{total_frames}")

cap.release()
out.release()
print(f"\nFinished, Annotated video saved to: {OUTPUT_VIDEO_PATH}")