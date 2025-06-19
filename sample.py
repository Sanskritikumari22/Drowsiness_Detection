import os
import json
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence


# Paths to your custom dataset
images_dir = '/home/kishore-ravishankar/senior major project/Action_rec16.v1i.coco/train/'  # Change this to your images path
annotations_dir = '/home/kishore-ravishankar/senior major project/Action_rec16.v1i.coco/train/'  # Change this to your annotations path
ann_file = os.path.join(annotations_dir, '_annotations.coco.json')  # Path to the annotations file

# Load annotations from the custom JSON file
with open(ann_file, 'r') as f:
    annotations_data = json.load(f)

# Check the structure of the loaded annotations
images = annotations_data["images"]
annotations = annotations_data["annotations"]
categories = annotations_data["categories"]

# Create a dictionary for easy lookup of image metadata by image id
image_dict = {image["id"]: image for image in images}
category_dict = {category["id"]: category["name"] for category in categories}

# Function to load an image and its annotations
def load_image_and_annotations(image_id):
    image_info = image_dict.get(image_id)
    if image_info is None:
        print(f"Image ID {image_id} not found.")
        return None, None
    
    # Get image path
    img_path = os.path.join(images_dir, image_info["file_name"])
    
    # Check if image file exists
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found!")
        return None, None
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to load image '{img_path}'!")
        return None, None
    
    # Get annotations (bounding boxes) for the image
    image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
    
    return img, image_annotations

# Example: Load a sample image and its annotations
sample_image_id = images[0]["id"]  # Using the first image in the dataset
img, image_annotations = load_image_and_annotations(sample_image_id)

# If the image and annotations are loaded successfully, display them
if img is not None and image_annotations is not None:
    # Display image with bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for ann in image_annotations:
        x, y, w, h = ann["bbox"]
        category_name = category_dict[ann["category_id"]]
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
        plt.text(x, y - 10, category_name, color='r', fontsize=12)
    plt.show()
else:
    print("Failed to load the image or annotations.")
