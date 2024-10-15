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

# Function to run the model and return segmented potholes and pixel area
def detect_potholes(image):
    input_image = preprocess_image(image)
    results = model(input_image)
    mask = results[0].masks  
    combined_mask = np.zeros_like(input_image[:, :, 0])
    if results[0].masks is not None:
        for m in results[0].masks:
            pothole_mask = m.data.cpu().numpy()
            pothole_mask_resized = cv2.resize(pothole_mask[0], (combined_mask.shape[1], combined_mask.shape[0]))
            combined_mask = np.maximum(combined_mask, pothole_mask_resized)
        pixel_area = np.sum(combined_mask)
        segmented_image = input_image.copy()
        segmented_image[combined_mask == 1] = [0, 255, 0]
    else:
        segmented_image = input_image.copy()
        pixel_area = 0
    return segmented_image, pixel_area

# Function to add text (area display) to an image or frame
def display_area(image, area):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (image.shape[1] - 300, 50)
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 2
    text = f"Area: {area:.2f} m²"
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
    
    total_cost = avg_cost + labour_cost + other_costs

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
    st.write(f"**Total Estimated Cost: Rs.{total_cost:.2f}**")

    # Download report button
    report = io.BytesIO()
    with pd.ExcelWriter(report, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Report', index=False)
        summary_df = pd.DataFrame({
            'Statistic': ['Minimum', 'Maximum', 'Average', 'Labour Cost', 'Other Costs', 'Total Estimated Cost'],
            'Area (m²)': [min_area, max_area, avg_area, '', '', ''],
            'Material Cost (Rs)': [min_cost, max_cost, avg_cost, labour_cost, other_costs, total_cost]
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
        segmented_image, pixel_area = detect_potholes(image_cv)
        
        image_width_pixels = image_cv.shape[1]
        scaling_factor = real_world_width / image_width_pixels
        real_world_area = pixel_area * (scaling_factor ** 2)
        
        segmented_image = display_area(segmented_image, real_world_area)
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
        
        area_list.append(real_world_area)
        cost_list.append(total_material_cost)
    
    elif uploaded_file.name.endswith('mp4'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            video_path = temp_video_file.name
            
        st.video(video_path)

        total_pixel_area = 0
        frame_count = 0

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
            segmented_frame, pixel_area = detect_potholes(frame)
            total_pixel_area += pixel_area
            frame_count += 1

            frame_width_pixels = frame.shape[1]
            scaling_factor = real_world_width / frame_width_pixels
            real_world_area = total_pixel_area * (scaling_factor ** 2) / frame_count

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
        average_pixel_area = total_pixel_area / frame_count
        real_world_area = average_pixel_area * (scaling_factor ** 2)
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
        
        area_list.append(real_world_area)
        cost_list.append(total_material_cost)

    # Generate report button
    if st.button("Generate Report"):
        generate_report(area_list, cost_list, labour_cost, other_costs)