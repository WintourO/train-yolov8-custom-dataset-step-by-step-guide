import ultralytics
import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(ultralytics.__version__)

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("C:\\Users\\User\\Documents\\GitHub\\train-yolov8-custom-dataset-step-by-step-guide\\runs\\detect\\train7\\weights\\best.pt")
model.to('cuda')


# Use the model
results = model.train(data=r"C:\Users\User\Documents\GitHub\train-yolov8-custom-dataset-step-by-step-guide\local_env\config.yaml", epochs=20, workers=0)

# train the model

#model = YOLO("C:\\Users\\User\\Documents\\GitHub\\train-yolov8-custom-dataset-step-by-step-guide\\runs\\detect\\train7\\weights\\best.pt")
#results = model("C:\\Users\\User\\Desktop\\SharkTest2.jpg")

#print(results)