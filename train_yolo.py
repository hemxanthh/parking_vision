from ultralytics import YOLO

# Use the small YOLOv8 model as a good starting point for custom training
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for slightly more accuracy

# Train on your dataset (data.yaml must be in your project root)
model.train(data='data.yaml', epochs=50, imgsz=640)