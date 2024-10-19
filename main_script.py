import os
import cv2
import numpy as np
import torch  # Import torch to check for GPU availability
from ultralytics import YOLO
from sort import Sort

# Set environment variable to avoid OpenMP error (optional workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLOv8 model and send it to the appropriate device (CPU/GPU)
model = YOLO("./models/best.pt").to(device)

# Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# Open the video file
video_path = "sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create a VideoWriter to save the output
out = cv2.VideoWriter(
    "output_with_tracking.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, device=device)

    detections = []  # To store valid detections for the tracker
    for result in results:
        if result.masks is not None:
            for box, mask in zip(result.boxes.xyxy, result.masks.data):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = result.boxes.conf[0]

                if conf > 0.5:
                    detections.append([x1, y1, x2, y2])
        else:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = result.boxes.conf[0]

                if conf > 0.5:
                    detections.append([x1, y1, x2, y2])

    # Convert detections to numpy array and pass to SORT tracker
    tracked_objects = tracker.update(np.array(detections))

    # Draw bounding boxes and tracking IDs on the frame
    for x1, y1, x2, y2, obj_id in tracked_objects:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Write the frame with tracking to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Pothole Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as 'output_with_tracking.mp4'")
