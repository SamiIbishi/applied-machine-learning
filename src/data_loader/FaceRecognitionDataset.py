# General Python Packages
import os
import glob
import typing
from os.path import join
import numpy as np
from typing import List, Any, Union

# Torch Packages
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms

# Matplotlib package
from matplotlib.pyplot import imshow
from matplotlib.pyplot import show

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files in celeba dataset

    def __init__(self, dataset_dir: str, image_width: int = 224,
                 image_height: int = 224):
        """
        Creates a data set object from given data set directory.

        :param dataset_dir: (Local) Path to the main folder containing the dataset
        :param image_width:
        :param image_height:

        :return: dataset: Dataset object with resized images and their respective labels
        """

        # Anchor dict
        self.anchor_dict = dict()

        # Path
        self.dataset_folder = dataset_dir

        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height

        # Get all the subfolders of the different persons
        self.image_filepaths = [f.path for f in os.scandir(self.dataset_folder) if f.is_dir()]

        self.person_dict = self._create_person_dict()

        # Method to tranform image to tensor
        self.to_tensor = transforms.ToTensor()

        # Get the triplets of original, similar, random
        self.triplets = self._create_triplets()

    def get_anchor_dict(self):
        """ External method to retrieve the generated anchor_dict"""

        return self.anchor_dict

    # TODO: advanced anchor creation
    def _choose_anchorimage(self):
        pass

    def _create_person_dict(self) -> dict:
        """ Create a dict of person ID, anchor (first index) and positives """
        person_dict = dict()
        for index, image_subfolder in enumerate(self.image_filepaths):

            person_id = int(os.path.basename(image_subfolder).split('.')[0])

            image_list = []

            image_names = os.listdir(image_subfolder)
            image_names.sort()

            # Each image is stored as a list of ID and filepath
            for image_name in image_names:
                image_id = int(os.path.basename(image_name).split('.')[0])
                image_filepath = join(self.dataset_folder, str(person_id), image_name)
                image_list.append([person_id, image_id, image_filepath])

            # For now use the first image as anchor; later use a more complex method
            anchor = image_list[0]

            # Add the anchor to the anchor_dict (only person ID and filapath needed)
            self.anchor_dict[anchor[0]] = anchor[2]

            image_list.pop(0)

            person_dict[index] = [person_id, anchor, image_list]

        return person_dict

    def _create_triplets(self) -> list:
        # This is the list of all images we sample from for negatives
        all_images = []
        for index, person in self.person_dict.items():
            all_images.append(person[1])
            for positive in person[2]:
                all_images.append(positive)

        triplets = []

        for index, person in self.person_dict.items():
            for positive in person[2]:
                negative_counter = 0
                while negative_counter < 5:
                    random_index = np.random.randint(low=0, high=len(all_images))
                    # Do not use the same person as negative
                    if all_images[random_index][0] != person[0]:
                        triplets.append([person[1], positive, all_images[random_index]])
                        negative_counter += 1
                    else:
                        pass

        return triplets

    def load_preprocessed_image(self, path: str, height: int = 224, width: int = 224) \
            -> Tensor:
        """
        :param path: filepath of the image we want to load
        :param height: height for resize operation
        :param width: width for resize operation
        :return: image converted to a Tensor
        """
        image = Image.open(path).resize([height, width])

        return self.to_tensor(image)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> typing.Tuple[list, str]:
        # Get the respective image triplet
        triplet = self.triplets[idx]

        triplet_images = []
        for image_information in triplet:
            filepath = image_information[2]
            # load all 3 images and preprocess them
            triplet_images.append(self.load_preprocessed_image(filepath, self.image_width,
                                                               self.image_height))

        # Return the resized images as a list and the label of the original (first) image
        return triplet_images, triplet[0][0]


dataset = FaceRecognitionDataset(dataset_dir='../data/celeba_dataset/images')
print(dataset.get_anchor_dict())
"""
images, label = dataset[100]

for image in images:
    imshow(image.permute(1, 2, 0))
    show()
print(label)
print("length =", len(dataset))
"""
