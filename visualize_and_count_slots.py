import cv2
import os

# Set file paths
image_path = "dataset/images/sample_parking.jpg"      # <-- Change to your image filename
label_path = "dataset/labels/sample_parking.txt"      # <-- Change to your label filename
output_path = "output/visualized_slots_with_stats.jpg"

# Class mapping
class_dict = {0: "empty", 1: "occupied"}
class_colors = {0: (0,255,0), 1: (0,0,255)}  # Green for empty, Red for occupied

# Load image
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Initialize counts
empty, occupied = 0, 0

# Draw boxes and count
with open(label_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])

        # Convert normalized coords to pixel coords
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)

        color = class_colors.get(cls, (255,255,255))
        label = class_dict.get(cls, str(cls))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1-4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if cls == 0:
            empty += 1
        elif cls == 1:
            occupied += 1

total = empty + occupied

# Prepare the stats string
stats_text = f"Total slots: {total} | Occupied: {occupied} | Empty: {empty}"
(font, font_scale, thickness) = (cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
(text_w, text_h), baseline = cv2.getTextSize(stats_text, font, font_scale, thickness)
# Draw a filled rectangle at the top for stats background
cv2.rectangle(img, (0,0), (w, text_h + 20), (0,0,0), -1)
# Put the stats text centered
cv2.putText(img, stats_text, (int((w - text_w)/2), text_h + 10), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, img)
print(f"Saved visualization to {output_path}")

# Show the image in a window
cv2.imshow("YOLO Slot Visualization with Stats", img)
cv2.waitKey(0)
cv2.destroyAllWindows()