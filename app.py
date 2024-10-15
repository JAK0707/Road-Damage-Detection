import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8 library
from PIL import Image
import time
from sort import Sort
import tempfile

model = YOLO('best.pt')

# Helper function to preprocess input image for YOLOv8
def preprocess_image(image):
    # Convert the image to RGB (if not already) for YOLOv8
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img

# Function to run the model and return segmented potholes and pixel area
def detect_potholes(image):
    input_image = preprocess_image(image)
    # Perform inference using YOLOv8 model for segmentation
    results = model(input_image)
    # Extract segmentation masks and calculate pixel area of potholes
    mask = results[0].masks  # YOLOv8 will have masks for segmentation
    # Create a blank mask to store the final segmented potholes
    combined_mask = np.zeros_like(input_image[:, :, 0])
    if results[0].masks is not None:
        # Iterate through the results to get all pothole masks
        for m in results[0].masks:
            pothole_mask = m.data.cpu().numpy()  # Convert to numpy array
            # Resize the mask to match the input image size
            pothole_mask_resized = cv2.resize(pothole_mask[0], (combined_mask.shape[1], combined_mask.shape[0]))
            combined_mask = np.maximum(combined_mask, pothole_mask_resized)  # Combine all masks
        pixel_area = np.sum(combined_mask)  # Calculate the total pixel area of potholes
        # Overlay the combined mask on the original image for visualization
        segmented_image = input_image.copy()
        segmented_image[combined_mask == 1] = [0, 255, 0]  # Color potholes green
    else:
        # If no masks are detected, use the original image and set pixel_area to 0
        segmented_image = input_image.copy()
        pixel_area = 0
    return segmented_image, pixel_area

# Streamlit Web App
st.title('Pothole Detection and Maintenance Cost Estimator')

# File upload: video/image input
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4"])

# User input for actual road width in meters
real_world_width = st.number_input("Enter the real-world width of the road in meters", min_value=1.0, step=0.1)

# Additional cost factors input
concrete_cost_per_m3 = st.number_input("Enter the cost of concrete per cubic meter (in Rs.)", min_value=0.0, step=0.1)
labour_cost = st.number_input("Enter the labour cost (in Rs.)", min_value=0.0, step=0.1)
other_costs = st.number_input("Enter any other additional costs (in Rs.)", min_value=0.0, step=0.1)

# Check if the user uploaded a file
if uploaded_file and concrete_cost_per_m3 and labour_cost and other_costs:
    # Load and process the image or video
    if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Convert to OpenCV format for YOLOv8 processing
        image_cv = np.array(image)
        # Run detection and segmentation
        segmented_image, pixel_area = detect_potholes(image_cv)
        # Display segmented image
        st.image(segmented_image, caption="Pothole Segmentation", use_column_width=True)
        # Measure road width in image (assuming it’s detected in pixels)
        image_width_pixels = image_cv.shape[1]  # Get width in pixels
        # Calculate scaling factor
        scaling_factor = real_world_width / image_width_pixels
        # Convert pixel area of potholes to real-world area
        real_world_area = pixel_area * (scaling_factor ** 2)
        st.write(f"Total Pothole Area in real-world (m²): {real_world_area:.2f}")
        # Calculate the volume of the pothole (area * depth, where depth is ~0.2 meters)
        pothole_volume = real_world_area * 0.2
        st.write(f"Estimated Volume of pothole (m³): {pothole_volume:.2f}")
        # Calculate material required and total cost
        total_material_cost = pothole_volume * concrete_cost_per_m3
        total_cost = total_material_cost + labour_cost + other_costs
        # Display cost breakdown
        st.write("### Cost Breakdown")
        st.write(f"Material Cost: Rs.{total_material_cost:.2f}")
        st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
        st.write(f"Other Costs: Rs.{other_costs:.2f}")
        st.write(f"**Total Cost: Rs.{total_cost:.2f}**")
    elif uploaded_file.name.endswith('mp4'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            video_path = temp_video_file.name
        # Display the uploaded video
        st.video(uploaded_file)
        # Open video using OpenCV
        video_cap = cv2.VideoCapture(video_path)
        total_pixel_area = 0
        frame_count = 0
        total_frame_width_pixels = 0
        # Initialize SORT tracker
        tracker = Sort()
        # For each frame in the video
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret or frame is None:
                st.warning("Failed to read frame or end of video reached.")
                break
            frame_count += 1
            total_frame_width_pixels += frame.shape[1]
            # Run pothole detection and segmentation on each frame
            results=model(frame)
            segmented_frame, pixel_area = detect_potholes(frame)
            total_pixel_area += pixel_area
            # Prepare detections for SORT tracker (x1, y1, x2, y2, confidence)
            detections = []  # You would add bounding box coordinates from the pothole detection here
            for result in results:
                if result.boxes and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy  # Get bounding boxes
                    confs = result.boxes.conf  # Get confidence scores
                    if result.masks is not None:
                        masks = result.masks.data
                        for box, mask, conf in zip(boxes, masks, confs):
                            x1, y1, x2, y2 = map(int, box)
                            conf = float(conf)
                            if conf > 0.5:
                                detections.append([x1, y1, x2, y2, conf])
                    else:
                        # If no masks, proceed with boxes and confs
                        for box, conf in zip(boxes, confs):
                            x1, y1, x2, y2 = map(int, box)
                            conf = float(conf)
                            if conf > 0.5:
                                detections.append([x1, y1, x2, y2, conf])   
            if detections:
                # Convert detections to numpy array and ensure correct shape
                detections_np = np.array(detections)
                if detections_np.ndim == 1:
                    detections_np = detections_np.reshape((1, -1))
                tracked_objects = tracker.update(np.array(detections))
                # Draw bounding boxes and tracker IDs on the frame
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = map(int, obj)
                    cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle around pothole
                    cv2.putText(segmented_frame, f'ID: {track_id}', (x1, y1 - 10),  # Add tracker ID
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                pass
            # Optionally display segmented frame with tracking info
            frame_rgb = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Segmented Frame {frame_count} (Tracked)", use_column_width=True)
            time.sleep(0.1)  # Delay for smoother frame processing display
        video_cap.release()
        average_pixel_area = total_pixel_area/frame_count
        # Calculate scaling factor (real-world meters per pixel)
        frame_width_pixels = total_frame_width_pixels/frame_count
        scaling_factor = real_world_width / frame_width_pixels
        # Convert pixel area of potholes to real-world area
        real_world_area = average_pixel_area * (scaling_factor ** 2)
        st.write(f"Average Pothole Area in real-world (m²): {real_world_area:.2f}")
        # Calculate the volume of the pothole (area * depth, where depth is ~0.2 meters)
        pothole_volume = real_world_area * 0.2  # Assuming an average depth of 0.2 meters
        st.write(f"Estimated Volume of pothole (m³): {pothole_volume:.2f}")
        # Calculate material required and total cost
        total_material_cost = pothole_volume * concrete_cost_per_m3
        total_cost = total_material_cost + labour_cost + other_costs
        # Display cost breakdown
        st.write("### Cost Breakdown")
        st.write(f"Material Cost: Rs.{total_material_cost:.2f}")
        st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
        st.write(f"Other Costs: Rs.{other_costs:.2f}")
        st.write(f"**Total Cost: Rs.{total_cost:.2f}**")