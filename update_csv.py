import pandas as pd
import os
from PIL import Image

# Paths
test_csv_path = "GTSRB/Test.csv"
test_folder = "GTSRB/Test"

# Load the existing CSV
test_data = pd.read_csv(test_csv_path)

# Get adversarial image filenames
adversarial_images = [f for f in os.listdir(test_folder) if "fake" in f]

# Create a list to hold adversarial data
adversarial_entries = []

for img_name in adversarial_images:
    img_path = os.path.join(test_folder, img_name)
    # Get image dimensions
    with Image.open(img_path) as img:
        width, height = img.size

    # Append adversarial data
    adversarial_entries.append({
        "Width": width,
        "Height": height,
        "Roi.X1": 0,
        "Roi.Y1": 0,
        "Roi.X2": width,
        "Roi.Y2": height,
        "ClassId": 43,  # Or -1
        "Path": img_path
    })

# Convert adversarial data to DataFrame
adversarial_data = pd.DataFrame(adversarial_entries)

# Append to the existing data
updated_data = pd.concat([test_data, adversarial_data], ignore_index=True)

# Save the updated CSV
updated_csv_path = "updated_test.csv"
updated_data.to_csv(updated_csv_path, index=False)

print(f"Updated CSV saved to {updated_csv_path}")
