import streamlit as st
import cv2
import numpy as np
import torch

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Streamlit app layout
st.title("Pothole Detection and Area Calculation")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    st.video(uploaded_file)

    # Input parameters
    real_world_width = st.number_input("Enter real-world width of the road (in meters):", min_value=0.0)
    real_world_length = st.number_input("Enter real-world length of the road (in meters):", min_value=0.0)
    cost_per_cubic_meter = st.number_input("Enter the cost of concrete per cubic meter (in Rs.):", min_value=0.0)

    # Load the YOLOv8 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace with your model path
    detection_threshold = 0.5  # IoU threshold for considering boxes as the same

    if st.button("Submit and Process Video"):
        # Initialize variables
        total_pothole_area = 0
        area_set = []
        frame_list = []
        frame_count = 0
        frame_height = 0
        frame_width = 0

        # Open the video file
        cap = cv2.VideoCapture(uploaded_file.name)

        # Display processing message
        with st.spinner("Processing the video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Capture frame dimensions
                if frame_count == 0:
                    frame_height, frame_width = frame.shape[:2]

                frame_count += 1
                results = model(frame)

                # Draw bounding boxes and calculate total area
                for bbox in results[0].boxes:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0].cpu().numpy())
                    conf = bbox.conf[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)

                    new_box = [x1, y1, x2, y2]
                    should_add = True

                    # Check if this bounding box overlaps with any existing boxes using IoU
                    for existing_box in area_set:
                        iou = calculate_iou(new_box, existing_box)
                        if iou > detection_threshold:
                            should_add = False  # Ignore this box if it overlaps too much with an existing one
                            break
                    
                    # Add the area to the set if it's a new detection
                    if should_add:
                        area_set.append(new_box)  # Store the bounding box
                        total_pothole_area += area  # Add the area to the total

                        # Draw the bounding box with a thicker line and colored background for text
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, f'Pothole {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'Pothole {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                frame_list.append(frame)

            cap.release()

        # Calculate total pothole area in real-world units only if frames were processed
        if frame_height > 0 and frame_width > 0:
            total_pothole_area_m2 = total_pothole_area * (real_world_width * real_world_length) / (frame_width * frame_height)
        else:
            total_pothole_area_m2 = 0  # Handle the case where no frames were processed

        # Provide a download button for the processed video
        out_video_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))

        for processed_frame in frame_list:
            out.write(processed_frame)

        out.release()

        # Provide a download button for the processed video
        with open(out_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_pothole_video.mp4",
                mime="video/mp4"
            )

        # Display the total pothole area
        st.write(f"Total Pothole Area Detected in Video: {total_pothole_area_m2:.2f} m²")
