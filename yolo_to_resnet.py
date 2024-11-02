import os
import shutil

# Define paths to your YOLO dataset folders
yolo_images_train_path = "E:/github/Road-Damage-Detection/Pothole_Segmentation_YOLOv8/train/images"
yolo_labels_train_path = "E:/github/Road-Damage-Detection/Pothole_Segmentation_YOLOv8/train/labels"

yolo_images_val_path = "E:/github/Road-Damage-Detection/Pothole_Segmentation_YOLOv8/val/images"
yolo_labels_val_path = "E:/github/Road-Damage-Detection/Pothole_Segmentation_YOLOv8/val/labels"

# Create new folders for ResNet-style dataset organization
resnet_dataset_path = "Dataset/resnet"
os.makedirs(os.path.join(resnet_dataset_path, 'pothole'), exist_ok=True)
os.makedirs(os.path.join(resnet_dataset_path, 'no_pothole'), exist_ok=True)

# Function to read the first class ID from the YOLO label file
def get_class_id(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        if lines:  # If there are annotations, return the first class ID
            return int(lines[0].split()[0])
        else:
            return None  # No label found

# Helper function to copy images based on class_id
def process_images(images_folder, labels_folder):
    for img_file in os.listdir(images_folder):
        # Find the corresponding label file
        label_file = os.path.join(labels_folder, img_file.replace('.jpg', '.txt'))

        if os.path.exists(label_file):
            class_id = get_class_id(label_file)

            if class_id == 1:  # Class ID 1: 'pothole'
                shutil.copy(
                    os.path.join(images_folder, img_file),
                    os.path.join(resnet_dataset_path, 'pothole')
                )
            elif class_id == 0:  # Class ID 0: 'no_pothole'
                shutil.copy(
                    os.path.join(images_folder, img_file),
                    os.path.join(resnet_dataset_path, 'no_pothole')
                )

# Process both train and validation images
process_images(yolo_images_train_path, yolo_labels_train_path)
process_images(yolo_images_val_path, yolo_labels_val_path)

print("Images successfully organized for ResNet training.")
