from ultralytics import YOLO
import os

# Absolute path to your best.pt file
model_path = r"C:\Users\User\Documents\GitHub\train-yolov8-custom-dataset-step-by-step-guide\local_env\best.pt"

# Check if file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
model = YOLO(model_path)

# Path to the test image
image_path = r"C:\Users\User\Desktop\SharkTest2.jpg"

# Check if image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image not found: {image_path}")

# Run inference
results = model(image_path)

# Process results
for result in results:
    result.show()
    result.save(filename="result.jpg")
