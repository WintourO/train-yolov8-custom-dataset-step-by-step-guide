from ultralytics import YOLO
import os
import cv2
import time

# Absolute path to your best.pt file
model_path = r"C:\Users\User\Documents\GitHub\train-yolov8-custom-dataset-step-by-step-guide\local_env\best.pt"

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the YOLO model
model = YOLO(model_path)

# Path to the test video
video_path = r"C:\Users\User\Desktop\SharkTestVideo.mp4"

# Check if video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Test video not found: {video_path}")

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open video file.")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a results folder if it doesn’t exist
results_folder = os.path.dirname(video_path)
output_video_path = os.path.join(results_folder, f"SharkTestVideo_annotated_{int(time.time())}.mp4")

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Detection threshold
threshold = 0.5  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame)[0]

    # Overlay bounding boxes and confidence scores
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            text = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)  # Write frame with detections

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as: {output_video_path}")
