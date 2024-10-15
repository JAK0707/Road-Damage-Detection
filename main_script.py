import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 model
from sort import Sort  # Import the SORT tracker

# Load your custom YOLOv8 model (adjust the path if necessary)
model = YOLO('best.pt')  # Example: Replace with your custom model

# Initialize SORT tracker
tracker = Sort()

# Open the video file
cap = cv2.VideoCapture('sample_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLOv8 on the current frame
    results = model(frame)

    # Prepare list of detections (format: [x1, y1, x2, y2, confidence])
    detections = []
    for result in results:
        for box, mask in zip(result.boxes.xyxy, result.masks.data):
            x1, y1, x2, y2 = map(int, box[:4])  # Extract only coordinates
            conf = result.boxes.conf[0]  # Confidence score

            if conf > 0.5:  # Confidence threshold
            # Append only the coordinates, without confidence
                detections.append([x1, y1, x2, y2])


    # Update SORT tracker with new detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw tracked bounding boxes and IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the video frame with annotations
    cv2.imshow('YOLOv8 + SORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit if 'q' is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
