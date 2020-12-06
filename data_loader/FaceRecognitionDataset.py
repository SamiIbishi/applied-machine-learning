# General Python Packages
import glob
from os.path import join

# Torch Packages
from torch.utils.data import Dataset, DataLoader

class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files

    # Define the constructor of this dataset object
    def __init__(self, dataset_folder):
        """
        Args:
            dataset_folder(string): Path to the main folder containing the dataset.
        """
        self.dataset_folder = dataset_folder

        # Crawl every subfolder for .jpg files
        self.image_filenames = glob.glob(join(self.dataset_folder, "**/*.jpg"), recursive=True)

        # Load the filepaths into a list and store it
        self.read_file_paths()

        # A dict to store mapping of classes to indices since we need the classes to be numerical
        self.encode_classes()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image based on the ID and resize to 500x500
        image = Image.open(self.image_filenames[idx]).resize(500, 500)

        # Get the label to the above image

        # Return the resized image and its label
        return image, label
