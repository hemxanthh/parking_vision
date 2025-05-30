from ultralytics import YOLO
import cv2
import pandas as pd
import os

image_path = 'input/sample_parking.jpg'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_image_path = os.path.join(output_dir, 'yolo_parking_output.jpg')
output_csv_path = os.path.join(output_dir, 'yolo_parking_status.csv')

model = YOLO('yolov8x.pt')  # Use a larger model!

results = model(image_path)
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy().astype(int)
confidences = boxes.conf.cpu().numpy()
COCO_CLASSES = model.model.names

# Try all vehicle classes
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
vehicle_indices = [i for i, c in enumerate(class_ids) if COCO_CLASSES[c] in vehicle_classes]
num_vehicles = len(vehicle_indices)

img = cv2.imread(image_path)
for i in vehicle_indices:
    box = boxes.xyxy[i].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    label = COCO_CLASSES[class_ids[i]]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"{label} {confidences[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

cv2.imwrite(output_image_path, img)
print(f"Detected vehicles: {num_vehicles}")
print(f"Output image saved to {output_image_path}")

df = pd.DataFrame({
    'Total Number of Slots': [""],
    'Occupied Slots': [num_vehicles],
    'Available Slots': [""],
})
df.to_csv(output_csv_path, index=False)
print(f"CSV with vehicle count saved to {output_csv_path}")