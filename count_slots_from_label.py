label_file = "dataset/labels/sample_parking.txt"  # Path to your label file

empty, occupied = 0, 0

with open(label_file, "r") as f:
    for line in f:
        if line.strip():
            cls = line.split()[0]
            if cls == "0":
                empty += 1
            elif cls == "1":
                occupied += 1

total = empty + occupied

print(f"Total slots: {total}")
print(f"Occupied slots: {occupied}")
print(f"Empty slots: {empty}")

# Optional: Save to CSV
import pandas as pd
df = pd.DataFrame({
    'Total Number of Slots': [total],
    'Occupied Slots': [occupied],
    'Empty Slots': [empty]
})
df.to_csv('output/parking_status.csv', index=False)
print("Saved CSV to output/parking_status.csv")