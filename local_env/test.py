from ultralytics import YOLO
model = YOLO("C:\\Users\\User\\Documents\\GitHub\\train-yolov8-custom-dataset-step-by-step-guide\\runs\\detect\\train7\\weights\\best.pt")
results = model.predict("C:\\Users\\User\\Desktop\\SharkTest1.jpg")

print(results)