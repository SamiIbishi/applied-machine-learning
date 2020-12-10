# General Python Packages
import os
import glob
import typing
from os.path import join

# Torch Packages
from torch.utils.data import Dataset

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files

    # Define the constructor of this dataset object
    def __init__(self, dataset_dir: str, labels_path: str, image_width: int = 500, image_height: int = 500):
        """
        Creates a data set object from given data set directory.

        :param dataset_dir: (Local) Path to the main folder containing the dataset
        :param labels_path: (Local) Path to the labels text file
        :param image_width:
        :param image_height:

        :return: dataset: Dataset object with resized images and their respective labels
        """

        # Paths
        self.dataset_folder = dataset_dir
        self.labels_path = labels_path

        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height

        # Crawl every subfolder for .jpg files
        self.image_filepaths = glob.glob(join(self.dataset_folder, "**/*.jpg"), recursive=True)
        self.labels = self._labels_from_txt()

    def _labels_from_txt(self) -> dict:
        labels = {}
        with open(self.labels_path) as labels_file:
            # Read each line as a key value pair and save it to the labels dict
            for line in labels_file:
                (image_id, target) = line.split()
                labels[int(sub(r"\D", "", image_id))] = target

        return labels

    def __len__(self) -> int:
        return len(self.image_filepaths)

    def __getitem__(self, idx: int) -> typing.Tuple[Image.Image, dict]:
        # Get the image based on the index and resize (default: 500x500)
        filepath = self.image_filepaths[idx]
        image_id = int(os.path.basename(filepath).split('.')[0])
        image = Image.open(filepath).resize([self.image_width, self.image_height])

        # Get the label to the above image
        label = self.labels[image_id]

        # Return the resized image and its label
        return image, label

#dataset = FaceRecognitionDataset(dataset_folder='../data/celeba_dataset/images', labels_path='../data/celeba_dataset/labels.txt')
#image, label = dataset[0]