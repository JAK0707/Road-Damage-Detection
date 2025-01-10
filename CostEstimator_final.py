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
import pandas as pd
from sort import Sort
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('./models/best.pt')

# Helper function to preprocess input image for YOLOv8
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img

# Function to calculate the area from the mask (minimum area)
def calculate_min_area(mask, scaling_factor):
    pixel_area = np.sum(mask)
    return pixel_area * (scaling_factor)

# Function to calculate the area from bounding boxes (maximum area)
def calculate_max_area(bbox, scaling_factor):
    width = bbox[2] - bbox[0]  # bbox format [x1, y1, x2, y2]
    height = bbox[3] - bbox[1]
    pixel_area = width * height
    return pixel_area * (scaling_factor)

# Function to run the model and return segmented potholes and pixel area
def detect_potholes(image, scaling_factor):
    input_image = preprocess_image(image)
    results = model(input_image)
    min_areas = []
    max_areas = []
    mask = results[0].masks  
    combined_mask = np.zeros_like(input_image[:, :, 0])
    if results[0].masks is not None:
        for idx, m in enumerate(results[0].masks):
            pothole_mask = m.data.cpu().numpy()
            pothole_mask_r = cv2.resize(pothole_mask[0], (combined_mask.shape[1], combined_mask.shape[0]))
            pothole_mask_resized = cv2.resize(pothole_mask[0], (input_image.shape[1], input_image.shape[0]))
            combined_mask = np.maximum(combined_mask, pothole_mask_r)
            
            # Minimum area from segmented mask
            min_area = calculate_min_area(pothole_mask_resized, scaling_factor)
            min_areas.append(min_area)

            # Maximum area from bounding box
            bbox = results[0].boxes[idx]
            max_area = calculate_max_area(bbox.xyxy[0].cpu().numpy(), scaling_factor)
            max_areas.append(max_area)
        pixel_area = np.sum(combined_mask)
        segmented_image = input_image.copy()
        segmented_image[combined_mask == 1] = [0, 255, 0]
        min_total_area = np.sum(min_areas)
        max_total_area = np.sum(max_areas)    
    else:
        min_total_area = 0
        max_total_area = 0
        segmented_image = input_image.copy()
        pixel_area = 0
    avg_total_area = (min_total_area + max_total_area) / 2
    return segmented_image, pixel_area, min_total_area, max_total_area, avg_total_area

# Function to add text (area display) to an image or frame
def display_area(image, area):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10,30)
    font_scale = 1
    font_color = (0, 0, 255)
    thickness = 2
    text = f"Damage Vol. (in m. cube): {area*0.2:.2f}"
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return image
                
# Streamlit Web App
st.title('Road Condition Analyser and Repair / Maintenance Cost Estimator')

# File upload: video/image input
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# User input for actual road width in meters
option = st.selectbox('Real world width of the road: ', ('Default - Single Lane', 'Default - Two Lane', 'Default - Three Lane', 'Default - Four Lane', 'Enter Manually'),)
real_world_width = 0
if option == 'Default - Single Lane':
    real_world_width=3.75
elif option == 'Default - Two Lane':
    real_world_width=7.25
elif option == 'Default - Three Lane':
    real_world_width=11
elif option == 'Default - Four Lane':
    real_world_width=15
else:
    real_world_width = st.number_input("Enter the real-world width of the road in meters", min_value=1.0, step=0.1) 
real_world_length = st.number_input("Enter the real-world length of the road in meters", min_value=1.0, step=0.1)
confidence_threshold = 0.5

# Additional cost factors input
concrete_cost_per_m3 = st.number_input("Enter the cost of concrete per cubic meter (in Rs.)", min_value=0.0, step=0.1)
labour_cost = st.number_input("Enter the labour cost (in Rs.)", min_value=0.0, step=0.1)
other_costs = st.number_input("Enter any other additional costs (in Rs.)", min_value=0.0, step=0.1)

# List to store area and cost data for report generation
area_list = []
cost_list = []
tempstore={}
total_damage_volume_m3 = 0

# Function to generate report
def generate_report(area_list, cost_list, labour_cost, other_costs):
    df = pd.DataFrame({
        'Area (m²)': area_list,
        'Material Cost (Rs)': cost_list
    })
    
    # Calculate statistics
    min_area = df['Area (m²)'].min()
    max_area = df['Area (m²)'].max()
    avg_area = df['Area (m²)'].mean()
    
    min_cost = df['Material Cost (Rs)'].min()
    max_cost = df['Material Cost (Rs)'].max()
    avg_cost = df['Material Cost (Rs)'].mean()
    
    total_cost_min = min_cost + labour_cost + other_costs
    total_cost_max = max_cost + labour_cost + other_costs
    total_cost_avg = avg_cost + labour_cost + other_costs
    
    # Display results
    st.write(f"### Report Summary")
    st.write(f"Minimum Area: {min_area:.2f} m²")
    st.write(f"Maximum Area: {max_area:.2f} m²")
    st.write(f"Average Area: {avg_area:.2f} m²")
    st.write(f"Minimum Material Cost: Rs.{min_cost:.2f}")
    st.write(f"Maximum Material Cost: Rs.{max_cost:.2f}")
    st.write(f"Average Material Cost: Rs.{avg_cost:.2f}")
    st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
    st.write(f"Other Costs: Rs.{other_costs:.2f}")
    st.write(f"**Total Estimated Cost (Minimum): Rs.{total_cost_min:.2f}**")
    st.write(f"**Total Estimated Cost (Maximum): Rs.{total_cost_max:.2f}**")
    st.write(f"**Total Estimated Cost (Average): Rs.{total_cost_avg:.2f}**")

    # Download report button
    report = io.BytesIO()
    with pd.ExcelWriter(report, engine='xlsxwriter') as writer:
        # df.to_excel(writer, sheet_name='Report', index=False)
        summary_df = pd.DataFrame({
            'Given': ['Material Cost', 'Labour Cost', 'Average Cost'],
            'Input': [concrete_cost_per_m3, labour_cost, other_costs],
            '':['', '', ''],
            'Statistic': ['Minimum', 'Maximum', 'Average'],
            'Area (m²)': [min_area, max_area, avg_area],
            'Material Cost (Rs)': [min_cost, max_cost, avg_cost],
            'Total Cost (Rs)': [total_cost_min, total_cost_max, total_cost_avg]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    st.download_button(
        label="Download Report",
        data=report.getvalue(),
        file_name="pothole_detection_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Check if the user uploaded a file
if st.button("Calculate"):
    if uploaded_file and concrete_cost_per_m3 and labour_cost and other_costs:
        if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_cv = np.array(image)
            image_width_pixels = image_cv.shape[1]
            image_height_pixels = image_cv.shape[0]
            scaling_factor = (real_world_width*real_world_length) / (image_width_pixels*image_height_pixels)
            segmented_image, pixel_area, min_area, max_area, avg_area = detect_potholes(image_cv, scaling_factor)
            
            segmented_image = display_area(segmented_image, avg_area)
            st.image(segmented_image, caption="Pothole Segmentation", use_column_width=True)
            
            pil_image = Image.fromarray(segmented_image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            st.download_button(
                label="Download Processed Image",
                data=img_bytes,
                file_name="processed_pothole_image.png",
                mime="image/png"
            )

            # Display results
            st.write(f"Average Pothole Area in real-world (m²): {avg_area:.2f}")
            # Calculate the volume of the pothole (area * depth, where depth is ~0.2 meters)
            pothole_volume = avg_area * 0.2
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
            
            area_list.append(min_area)
            area_list.append(max_area)
            area_list.append(avg_area)
            cost_list.append(min_area * 0.2 * concrete_cost_per_m3)
            cost_list.append(max_area * 0.2 * concrete_cost_per_m3)
            cost_list.append(total_material_cost)
            
            # Generate report button
            if st.button("Generate Report"):
                generate_report(area_list, cost_list, labour_cost, other_costs)
                
        elif uploaded_file.name.endswith(('mp4','avi', 'mov')):
            real_world_length_for_frame = st.number_input("Enter the real-world length of the captured road in meters in video", min_value=0.0, step=0.1)
            if real_world_length_for_frame != 0:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False)
                temp_video_file.write(uploaded_file.read())
                video_path = temp_video_file.name
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Load YOLOv8 model
                model = model.to(device)
                
                # Initialize SORT tracker
                tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
                    
                st.video(video_path)
                st.write("Processing Video .... ")

                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Calculate pixel-to-meter scaling factors
                pixels_per_meter_width = frame_width / real_world_width
                pixels_per_meter_length = frame_height / real_world_length_for_frame
                
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
                    
                    results=model(frame, device=device)

                    detections = []  # Store detections for tracking

                    for result in results:
                        for box in result.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box[:4])  # Ensure coordinates are integers
                            conf = float(result.boxes.conf[0])

                            # Filter detections by confidence threshold
                            if conf > confidence_threshold:
                                detections.append([x1, y1, x2, y2])

                    # Track objects with SORT
                    tracked_objects = tracker.update(np.array(detections))

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
                        volume_m3 = area_m2 * 0.2

                        # Track maximum volume for each ID
                        if obj_id not in tempstore:
                            tempstore[obj_id] = volume_m3
                            total_damage_volume_m3 += volume_m3
                        else:
                            # Update the total volume only if the bounding box is larger than before
                            if volume_m3 > tempstore[obj_id]:
                                total_damage_volume_m3 += (volume_m3 - tempstore[obj_id])
                                tempstore[obj_id] = volume_m3

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
                    total_volume_label = f'Damage Vol. (in m. cube): {total_damage_volume_m3:.2f}'

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

                st.success("Processing complete!")
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Output Video",
                        data=file,
                        file_name="output_with_tracking.mp4",
                        mime="video/mp4",
                    )
                    
                # Calculate and display other results
                real_world_area = total_damage_volume_m3/0.2
                pothole_volume = total_damage_volume_m3
                total_material_cost = pothole_volume * concrete_cost_per_m3
                total_cost = total_material_cost + labour_cost + other_costs

                st.write(f"Average Pothole Area in real-world (m²): {real_world_area:.2f}")
                st.write(f"Estimated Volume of pothole (m³): {pothole_volume:.2f}")
                st.write("### Cost Breakdown")
                st.write(f"Material Cost: Rs.{total_material_cost:.2f}")
                st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
                st.write(f"Other Costs: Rs.{other_costs:.2f}")
                st.write(f"**Total Cost: Rs.{total_cost:.2f}**")
        
                # Generate report button
                if st.button("Generate Report"):
                    area_list.append(0.9 * real_world_area)
                    area_list.append(1.1 * real_world_area)
                    area_list.append(real_world_area)
                    cost_list.append(0.9 * real_world_area * 0.2 * concrete_cost_per_m3)
                    cost_list.append(1.1 * real_world_area * 0.2 * concrete_cost_per_m3)
                    cost_list.append(total_material_cost)

                    # Call report generation
                    generate_report(area_list, cost_list, labour_cost, other_costs)
