import os
import cv2
from ultralytics import YOLO

# Set up live stream (Replace with your drone's RTSP/HTTP stream or video device index)
drone_url = "rtsp://your-drone-ip/stream"  # Replace with actual drone stream URL
cap = cv2.VideoCapture(drone_url)  # For USB capture device, use cap = cv2.VideoCapture(1)

# Load YOLO model
model_path = r"C:\Users\User\Documents\GitHub\train-yolov8-custom-dataset-step-by-step-guide\local_env\best.pt"
model = YOLO(model_path)

threshold = 0.5  # Confidence threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No frame received from drone.")
        break

    results = model(frame)[0]  # Run YOLO detection

    # Draw bounding boxes and confidence scores
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            text = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Show the frame with detections
    cv2.imshow("Drone Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
