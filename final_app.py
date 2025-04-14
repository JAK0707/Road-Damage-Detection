import os
# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort
import tempfile

# Initialize session state to avoid re-processing
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
    st.session_state['output_path'] = None
    st.session_state['areas'] = {}  # Store max pixel area per ID
    st.session_state['total_damage_volume_m3'] = 0  # Total damage volume in m³

# Streamlit app setup
st.title("Road Repair Cost Calculator")
st.sidebar.header("Settings")

# Sidebar inputs for user preferences
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.5, step=0.05
)
model_path = "./models/best.pt"

# Request real-world road dimensions, depth, and concrete cost
road_length_m = st.sidebar.number_input("Road Length (meters)", min_value=1.0, step=0.1)
road_width_m = st.sidebar.number_input("Road Width (meters)", min_value=1.0, step=0.1)
depth_m = st.sidebar.number_input("Pothole Depth (meters)", min_value=0.0, step=0.01)
concrete_cost_per_m3 = st.sidebar.number_input("Concrete Cost (per m³)", min_value=0.0, step=1.0)

# Upload video
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Button to process video
if st.sidebar.button("Process Video") and uploaded_video and not st.session_state['processed']:
    # Save uploaded video to temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write("Processing Video")

    # Load YOLOv8 model
    model = YOLO(model_path).to(device)

    # Initialize SORT tracker
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate pixel-to-meter scaling factors
    pixels_per_meter_width = frame_width / road_width_m
    pixels_per_meter_length = frame_height / road_length_m

    # Create output video file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    # Process video frame by frame
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, device=device)

        detections = []  # Store detections for tracking

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])  # Ensure coordinates are integers
                conf = float(result.boxes.conf[0])

                # Filter detections by confidence threshold
                if conf > confidence_threshold:
                    detections.append([x1, y1, x2, y2])

        # Track objects with SORT only if there are detections
        if len(detections) > 0:
            tracked_objects = tracker.update(np.array(detections))
        else:
            tracked_objects = np.empty((0, 5))  # Empty array for no detections

        # Draw bounding boxes and accumulate volumes
        for x1, y1, x2, y2, obj_id in tracked_objects:
            # Ensure coordinates are integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Calculate real-world dimensions
            width_m = (x2 - x1) / pixels_per_meter_width
            length_m = (y2 - y1) / pixels_per_meter_length

            # Calculate area in m²
            area_m2 = width_m * length_m

            # Calculate volume in m³ (Area * Depth)
            volume_m3 = area_m2 * depth_m

            # Track maximum volume for each ID
            if obj_id not in st.session_state['areas']:
                st.session_state['areas'][obj_id] = volume_m3
                st.session_state['total_damage_volume_m3'] += volume_m3
            else:
                # Update the total volume only if the bounding box is larger than before
                if volume_m3 > st.session_state['areas'][obj_id]:
                    st.session_state['total_damage_volume_m3'] += (volume_m3 - st.session_state['areas'][obj_id])
                    st.session_state['areas'][obj_id] = volume_m3

            # Draw bounding box in blue and thicker
            box_thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), box_thickness)

            # Prepare label with "Pothole" and confidence score
            label = f'Pothole ID {int(obj_id)}: {conf:.2f}'

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_bg_y1 = y1 - text_height - 10  # Background y-coordinate
            text_bg_y2 = y1  # Background y-coordinate
            cv2.rectangle(frame, (x1, text_bg_y1), (x1 + text_width, text_bg_y2), (255, 0, 0), cv2.FILLED)  # Text background

            # Display label on the bounding box
            cv2.putText(
                frame, label, (x1, text_bg_y1 + text_height - 2),  # Adjusted for baseline
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Display total damage volume on the video frame
        total_volume_label = f'Total Damage Volume (in meter cube): {st.session_state["total_damage_volume_m3"]:.2f}'

        cv2.putText(
            frame, total_volume_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        # Write frame to output video
        out.write(frame)

        # Update progress bar
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress_bar.progress(min(current_frame / frame_count, 1.0))

    # Release resources
    cap.release()
    out.release()

    # Save output path
    st.session_state['processed'] = True
    st.session_state['output_path'] = output_path

    st.success("Processing complete!")

# Display download button and summary after processing
if st.session_state['processed'] and st.session_state['output_path']:
    with open(st.session_state['output_path'], "rb") as file:
        st.download_button(
            label="Download Output Video",
            data=file,
            file_name="output_with_tracking.mp4",
            mime="video/mp4",
        )

    # Display area summary
    st.header("Pothole Volume Summary")
    for obj_id, volume_m3 in st.session_state['areas'].items():
        st.write(f"ID {int(obj_id)}: {volume_m3:.2f} m³")

    # Display total damage volume summary
    st.write(f"Total Damage Volume: {st.session_state['total_damage_volume_m3']:.2f} m³")
    
    # Calculate and display total repair cost
    total_cost = st.session_state['total_damage_volume_m3'] * concrete_cost_per_m3
    st.write(f"Total Repair Cost: ₹{total_cost:.2f}")
