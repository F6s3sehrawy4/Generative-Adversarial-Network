import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset


# Custom Dataset to handle the GTSRB dataset
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['Path'])
        image = Image.open(img_path).convert("RGB")

        # Crop the image based on the ROI coordinates
        x1 = self.data.iloc[idx]['Roi.X1']
        y1 = self.data.iloc[idx]['Roi.Y1']
        x2 = self.data.iloc[idx]['Roi.X2']
        y2 = self.data.iloc[idx]['Roi.Y2']
        image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx]['ClassId']
        return image, label