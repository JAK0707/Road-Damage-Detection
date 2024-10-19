import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8 library
from PIL import Image
import time
import tempfile
import queue
import io

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

# Function to add text (area display) to an image or frame
def display_area(image, area):
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (image.shape[1] - 300, 50)  # Top-right corner
    font_scale = 1
    font_color = (0,0,0)  # Black
    thickness = 2
    # Add text to the image
    text = f"Area: {area:.2f} m. sq."
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return image

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
        
        # Measure road width in image (assuming it’s detected in pixels)
        image_width_pixels = image_cv.shape[1]  # Get width in pixels
        # Calculate scaling factor
        scaling_factor = real_world_width / image_width_pixels
        # Convert pixel area of potholes to real-world area
        real_world_area = pixel_area * (scaling_factor ** 2)
        
        # Display the real-world area in the top-right corner of the image
        segmented_image = display_area(segmented_image, real_world_area)
        
        # Display segmented image with the area
        st.image(segmented_image, caption="Pothole Segmentation", use_column_width=True)
        
        # Convert processed image to PIL format for download
        pil_image = Image.fromarray(segmented_image)

        # Save the image to a BytesIO object for download
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)  # Reset file pointer to the beginning

        # Display download button for the processed image
        st.download_button(
            label="Download Processed Image",
            data=img_bytes,
            file_name="processed_pothole_image.png",
            mime="image/png"
        )

        # Display results
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
            
        st.video(video_path) 

        # Processing video frame by frame with incremental and cumulative area display
        total_pixel_area = 0  # Initialize cumulative area
        frame_count = 0

        # Read the video
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create a temporary file for the output video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_video_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_file.name, fourcc, fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                segmented_frame, pixel_area = detect_potholes(frame)
                total_pixel_area += pixel_area  # Accumulate the pixel area

                # Calculate real-world areas
                scaling_factor = real_world_width / frame_width
                current_real_world_area = pixel_area * (scaling_factor ** 2)
                cumulative_real_world_area = total_pixel_area * (scaling_factor ** 2)

                # Prepare the text to display
                text_current_area = f"Current Frame Area: {current_real_world_area:.2f} m. sq."
                text_cumulative_area = f"Cumulative Area: {cumulative_real_world_area:.2f} m. sq."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0,0,0)  # Black color
                thickness = 2

                # Get text sizes for positioning
                text_size_current, _ = cv2.getTextSize(text_current_area, font, font_scale, thickness)
                text_size_cumulative, _ = cv2.getTextSize(text_cumulative_area, font, font_scale, thickness)

                # Calculate positions (with padding of 10 pixels)
                text_x = frame_width - text_size_current[0] - 10  # Position for right alignment
                text_y_current = 30  # For current area, display at the top
                text_y_cumulative = text_y_current + 30  # For cumulative area, display below current

                # Add text to the frame
                cv2.putText(segmented_frame, text_current_area, (text_x, text_y_current), font, font_scale, font_color, thickness)
                cv2.putText(segmented_frame, text_cumulative_area, (text_x, text_y_cumulative), font, font_scale, font_color, thickness)

                # Write the processed frame with text
                out.write(segmented_frame)

            cap.release()
            out.release()

        # Provide the download link for the output video
        with open(output_video_file.name, "rb") as file:
            video_bytes = file.read()

        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_pothole_detection.mp4",
            mime="video/mp4"
        )

        # Display cumulative results
        st.write(f"Cumulative Pothole Area in real-world (m²): {cumulative_real_world_area:.2f}")
        
        # Calculate and display other results
        average_pixel_area = total_pixel_area / frame_count
        real_world_area = average_pixel_area * (scaling_factor ** 2)
        pothole_volume = real_world_area * 0.2
        total_material_cost = pothole_volume * concrete_cost_per_m3
        total_cost = total_material_cost + labour_cost + other_costs

        st.write(f"Average Pothole Area in real-world (m²): {real_world_area:.2f}")
        st.write(f"Estimated Volume of pothole (m³): {pothole_volume:.2f}")
        st.write(f"Material Cost: Rs.{total_material_cost:.2f}")
        st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
        st.write(f"Other Costs: Rs.{other_costs:.2f}")
        st.write(f"**Total Cost: Rs.{total_cost:.2f}**")
