import cv2
import numpy as np
import pandas as pd
import os

def preprocess_image(image_path):
    """
    Load and preprocess the parking area image.
    Steps: Grayscale, blur, and adaptive threshold.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    return image, thresh

def detect_parking_slots(thresh_img):
    """
    Detect parking slots as regions in the thresholded image.
    Returns contours which may represent slots.
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slot_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # Heuristic: filter by area and aspect ratio (tune these values for your dataset)
        if 1000 < area < 50000 and 1 < w/h < 6:
            slot_contours.append((x, y, w, h))
    return slot_contours

def classify_occupancy(image, slots):
    """
    Classify each slot as occupied or available.
    Simple method: check average brightness in slot region (tune this heuristic as needed).
    """
    occupied = []
    available = []
    for (x, y, w, h) in slots:
        slot_img = image[y:y+h, x:x+w]
        gray_slot = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray_slot)
        # Threshold to classify (tune this as needed)
        if mean_val < 100:
            occupied.append((x, y, w, h))
        else:
            available.append((x, y, w, h))
    return occupied, available

def visualize_output(image, occupied, available):
    """
    Draw rectangles for occupied (red) and available (green) slots.
    Display the counts on the image.
    """
    output = image.copy()
    for (x, y, w, h) in occupied:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in available:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(output, f"Occupied: {len(occupied)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(output, f"Available: {len(available)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return output

def export_to_csv(total, occupied, available, filename='output/parking_status.csv'):
    """
    Export the results to a CSV file with the required columns.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {
        'Total Number of Slots': [total],
        'Occupied Slots': [occupied],
        'Available Slots': [available]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def main(image_path):
    # 1. Preprocessing
    image, thresh = preprocess_image(image_path)
    # 2. Parking Slot Detection
    slots = detect_parking_slots(thresh)
    # 3. Occupancy Status Classification
    occupied, available = classify_occupancy(image, slots)
    # 4. Output Visualization
    output_img = visualize_output(image, occupied, available)
    # Save output image
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/parking_output.png', output_img)
    # 5. Export Data
    export_to_csv(len(slots), len(occupied), len(available), filename='output/parking_status.csv')
    print(f"Total: {len(slots)}, Occupied: {len(occupied)}, Available: {len(available)}")
    print("Output image saved as 'output/parking_output.png' and CSV saved as 'output/parking_status.csv'.")

if __name__ == "__main__":
    image_path = 'input/sample_parking.jpg'  # Make sure your image file is here
    main(image_path)



