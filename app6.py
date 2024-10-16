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

model = YOLO('best.pt')

# Helper function to preprocess input image for YOLOv8
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img

# Function to calculate the area from the mask (minimum area)
def calculate_min_area(mask, scaling_factor):
    pixel_area = np.sum(mask)
    return pixel_area * (scaling_factor ** 2)

# Function to calculate the area from bounding boxes (maximum area)
def calculate_max_area(bbox, scaling_factor):
    width = bbox[2] - bbox[0]  # bbox format [x1, y1, x2, y2]
    height = bbox[3] - bbox[1]
    pixel_area = width * height
    return pixel_area * (scaling_factor ** 2)

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
    position = (image.shape[1] - 300, 50)
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 2
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

# List to store area and cost data for report generation
area_list = []
cost_list = []

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
if uploaded_file and concrete_cost_per_m3 and labour_cost and other_costs:
    if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_cv = np.array(image)
        image_width_pixels = image_cv.shape[1]
        scaling_factor = real_world_width / image_width_pixels
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
    
    elif uploaded_file.name.endswith('mp4'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            video_path = temp_video_file.name
            
        st.video(video_path)

        total_real_area = 0
        frame_count = 0
        min_area = 0 
        max_area = 0  
        avg_area = 0

        cap = cv2.VideoCapture(video_path)
        frame_queue = queue.Queue()
        placeholder = st.empty()
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_list.append(frame)
            frame_width_pixels = frame.shape[1]
            scaling_factor = real_world_width / frame_width_pixels
            segmented_frame, pixel_area, min_a, max_a, avg_a = detect_potholes(frame, scaling_factor)
            total_real_area += avg_a
            if(avg_a>0):
                frame_count += 1
            min_area+=min_a
            max_area+=max_a
            avg_area+=avg_a

            current_real_world_area = avg_a
            cumulative_real_world_area = total_real_area

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

            
            frame_queue.put(segmented_frame)
            time.sleep(0.03)
            placeholder.image(segmented_frame, use_column_width=True)

        cap.release()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            video_output_path = temp_output.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(video_output_path, fourcc, 30, (frame_list[0].shape[1], frame_list[0].shape[0]))

            while not frame_queue.empty():
                frame = frame_queue.get()
                output_video.write(frame)

            output_video.release()

        with open(video_output_path, 'rb') as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_pothole_video.mp4",
                mime="video/mp4"
            )

        # Display cumulative results
        st.write(f"Cumulative Pothole Area in real-world (m²): {cumulative_real_world_area:.2f}")
        
        # Calculate and display other results
        real_world_area = avg_area/frame_count
        pothole_volume = real_world_area * 0.2
        total_material_cost = pothole_volume * concrete_cost_per_m3
        total_cost = total_material_cost + labour_cost + other_costs

        st.write(f"Average Pothole Area in real-world (m²): {real_world_area:.2f}")
        st.write(f"Estimated Volume of pothole (m³): {pothole_volume:.2f}")
        st.write("### Cost Breakdown")
        st.write(f"Material Cost: Rs.{total_material_cost:.2f}")
        st.write(f"Labour Cost: Rs.{labour_cost:.2f}")
        st.write(f"Other Costs: Rs.{other_costs:.2f}")
        st.write(f"**Total Cost: Rs.{total_cost:.2f}**")
        
        min_area=min_area/frame_count
        max_area=max_area/frame_count
        area_list.append(min_area)
        area_list.append(max_area)
        area_list.append(real_world_area)
        cost_list.append(min_area * 0.2 * concrete_cost_per_m3)
        cost_list.append(max_area * 0.2 * concrete_cost_per_m3)
        cost_list.append(total_material_cost)

    # Generate report button
    if st.button("Generate Report"):
        generate_report(area_list, cost_list, labour_cost, other_costs)