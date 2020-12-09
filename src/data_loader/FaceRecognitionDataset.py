# General Python Packages
import glob
from os.path import join

# Torch Packages
from torch.utils.data import Dataset, DataLoader

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files

    # Define the constructor of this dataset object
    def __init__(self, dataset_folder: str, labels_path: str):
        """
        Args:
            dataset_folder(string): Path to the main folder containing the dataset
            labels_file(string): Full path to the labels text file
        """
        self.dataset_folder = dataset_folder
        self.labels_path = labels_path

        # Crawl every subfolder for .jpg files
        self.image_filenames = glob.glob(join(self.dataset_folder, "**/*.jpg"), recursive=True)

        self.labels = self.labels_from_txt()

    def labels_from_txt(self) -> dict:
        labels = {}
        with open(self.labels_path) as labels_file:
            # Read each line as a key value pair and save it to the labels dict
            for line in labels_file:
                (image_id, target) = line.split()
                labels[int(sub(r"\D", "", image_id))] = target

        return labels

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[Image, dict]:
        # Get the image based on the ID and resize to 500x500
        image = Image.open(self.image_filenames[idx]).resize([500, 500])

        # Get the label to the above image
        label = self.labels[idx]

        # Return the resized image and its label
        return image, label
