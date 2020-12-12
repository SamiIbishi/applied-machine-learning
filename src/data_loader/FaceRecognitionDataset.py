# General Python Packages
import os
import glob
import typing
from os.path import join
import numpy as np

# Torch Packages
from typing import List, Any, Union

from torch.utils.data import Dataset

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files

    # Define the constructor of this dataset object
    def __init__(self, dataset_dir: str, labels_path: str, image_width: int = 512,
                 image_height: int = 512):
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

        # Update the image_filepaths to the pruned version
        self.pruned_filepaths = self._prune_filepaths()

        # Get the triplets of original, similar, random
        self.triplets = self._create_triplets()

    def _prune_filepaths(self) -> list:
        pruned_filepaths = []

        for filepath in self.image_filepaths:
            image_id = int(os.path.basename(filepath).split('.')[0])
            image_label = self.labels[image_id]
            # We only want to get faces that occur more than once
            if sum(value == image_label for value in self.labels.values()) > 1:
                pruned_filepaths.append(filepath)

        return pruned_filepaths

    def _create_triplets(self) -> list:
        triplets: List[List[str]] = []

        for filepath in self.pruned_filepaths:
            image_id = int(os.path.basename(filepath).split('.')[0])
            image_label = self.labels[int(os.path.basename(filepath).split('.')[0])]
            # Collect all other images with the same label as the current one
            similar_images = [image for image, label in self.labels.items() if
                              label == image_label and image != image_id]
            # Create 5 random other samples as the third image of the triplet
            for similar_image in similar_images:
                similar_filepath = self._image_id_to_filepath(filepath, similar_image)
                samples = 0
                # 5 adversary examples are created for each image pair of the same person
                while samples < 5:
                    random_id = np.random.randint(low=0, high=len(self.pruned_filepaths) - 1)
                    # We do not want to have the same person as an adversary example
                    if random_id not in similar_images and random_id != image_id:
                        random_filepath = self._image_id_to_filepath(filepath, random_id)
                        triplets.append([filepath, similar_filepath, random_filepath])
                        samples += 1

        return triplets

    def _image_id_to_filepath(self, example_filepath: str, image_id: int) -> str:
        """
        Reconstructs the filepath of image given its id.

        :param example_filepath: An example filepath of the same shape as the output filepath
        :param image_id: The id which we want to get the filepath for
        :return: str: The filepath corresponding to the image id
        """
        # Crude way to convert back to filepaths from an image ID in our setup
        example_id = os.path.basename(example_filepath).split('.')[0]
        padding_size = len(example_id)
        # Pad to have valid image_id length in filepath (leading zeros)
        image_id_string = str(image_id).zfill(padding_size)

        return example_filepath.replace(example_id, image_id_string)

    def _labels_from_txt(self) -> dict:
        labels = {}
        with open(self.labels_path) as labels_file:
            # Read each line as a key value pair and save it to the labels dict
            for line in labels_file:
                (image_id, target) = line.split()
                labels[int(sub(r"\D", "", image_id))] = target

        return labels

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> typing.Tuple[list, dict]:
        # Get the respective image triplet
        triplet = self.triplets[idx]
        # The image id is the id of the first image in the triplet
        image_id = int(os.path.basename(triplet[0]).split('.')[0])

        triplet_images = []
        for filepath in triplet:
            # Resize all 3 images, default is 512x512
            image = Image.open(filepath).resize([self.image_width, self.image_height])
            triplet_images.append(image)

        # Get the label of the original image (the first image of the triplet)
        label = self.labels[image_id]

        # Return the resized images as a list and the label of the original (first) image
        return triplet_images, label


#dataset = FaceRecognitionDataset(dataset_dir='../data/celeba_dataset/images',
#                                 labels_path='../data/celeba_dataset/labels.txt')
#images, label = dataset[100]
#
#for image in images:
#    image.show()
#print(label)
#print("length =",len(dataset))
