from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=r"C:\Users\User\Documents\GitHub\train-yolov8-custom-dataset-step-by-step-guide\local_env\config.yaml", epochs=1)

  # train the model
