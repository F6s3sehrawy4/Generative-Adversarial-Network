import random

import pandas as pd
import os
from PIL import Image

# Paths
original_csv_path = "TestOG.csv"
test_folder_path = "GTSRB/Test"
adversarial_folder_path = "GTSRB/Train/43"
output_csv_path = "GTSRB/Test.csv"

# Load Original CSV
original_df = pd.read_csv(original_csv_path)

# Process Adversarial Images
adversarial_data = []
for filename in os.listdir(adversarial_folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Handle various image formats
        image_path = os.path.join(adversarial_folder_path, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            adversarial_data.append({
                "Width": width,
                "Height": height,
                "Roi.X1": 0,
                "Roi.Y1": 0,
                "Roi.X2": width,
                "Roi.Y2": height,
                #"ClassId": random.randint(0, 42),  # Placeholder for adversarial images
                "ClassId": 43,
                "Path": "Test/" + filename  # Relative path
            })
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Debugging: Output number of processed adversarial images
print(f"Processed {len(adversarial_data)} adversarial images")

# Convert Adversarial Data to DataFrame
adversarial_df = pd.DataFrame(adversarial_data)

# Combine Original and Adversarial Data
combined_df = pd.concat([original_df, adversarial_df], ignore_index=True)

# Save to New CSV
combined_df.to_csv(output_csv_path, index=False)

print(f"Combined CSV saved to {output_csv_path}")
