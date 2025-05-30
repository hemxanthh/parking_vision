from ultralytics import YOLO
import pandas as pd

# Load your best trained weights
model = YOLO('runs/detect/train/weights/best.pt')

# Change this to your desired test image
results = model('dataset/images/sample_parking.jpg')

empty = 0
occupied = 0

for box in results[0].boxes:
    cls = int(box.cls[0])
    if cls == 0:
        empty += 1
    elif cls == 1:
        occupied += 1

total = empty + occupied

print(f"Total slots: {total}")
print(f"Occupied slots: {occupied}")
print(f"Empty slots: {empty}")

# Optionally, save image with detections
results[0].save(filename='output/my_parking_detection.jpg')

# Optionally, save to CSV
df = pd.DataFrame({
    'Total Number of Slots': [total],
    'Occupied Slots': [occupied],
    'Empty Slots': [empty]
})
df.to_csv('output/parking_status.csv', index=False)
print("Saved CSV to output/parking_status.csv")

print(results[0].boxes)